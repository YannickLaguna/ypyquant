"""
servicio.py — Interfaz de dominio del paper interno: crear sesiones, consultarlas y avanzarlas (sin FastAPI).

Responsabilidad:
    - `ServicioPaper(almacen, fuente, ahora)`: la interfaz que usan la API (`app/api/rutas_paper_interno.py`) y la
      CLI (`python -m ypyquant.paper`). Compone el `SimuladorPaper` con el `AlmacenPaper`.
      * `crear_sesion(receta, ...)`: congela una `Receta` (la de un experimento terminado o de una configuración),
        resuelve su universo a una lista de tickers, aplica los costes opcionales de la sesión y deja la cartera en
        efectivo con el día anterior a `fecha_alta` como último procesado. El universo `sp500_pit` (dinámico) se
        admite: se conserva como nombre y el simulador lo resuelve en cada fecha con la membresía vigente.
      * Consultas: `listar()` (una fila por sesión con métricas forward, costes, estado, «n días pendientes», frecuencia
        y fechas del último y próximo rebalanceo, para explicar cuánto tiempo lleva y cuándo se rebalancea),
        `detalle(id)` (receta, métricas, posiciones actuales, pesos objetivo, avisos), `equity`, `operaciones`,
        `pesos` y `config_espejo(id)` (la configuración del backtest espejo).
      * `pausar`, `reanudar`, `detener` (una sesión detenida no se reanuda) y `avanzar(id | todas)`.
    - Métricas forward: `evaluar(simulacion_desde_tablas(...))` con `validar=False`, la MISMA función que usa el
      backtest; el paper no calcula Sharpe ni drawdown por su cuenta. Los costes son comisiones + slippage.
    - `ControlAvance`: estado consultable de un avance en segundo plano (`en_curso`, hora, resultados, error) con
      exclusión (un solo avance a la vez) para el botón «Avanzar ahora».

Se relaciona con:
    - ypyquant.paper.{simulador, almacen, sesion, calendario} y ypyquant.estrategias (`evaluar`,
      `simulacion_desde_tablas`, `crear_slug`, `resolver_universo`, `receta_de_experimento` lee el almacén de experimentos).
    - app/api/rutas_paper_interno.py (HTTP) y ypyquant/paper/__main__.py (CLI).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from ypyquant.almacen_documentos import limpiar_para_json
from ypyquant.data.fuente_precios import FuentePrecios
from ypyquant.data.universos import Membresias, avisos_membresias, es_universo_pit, resolver_universo
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.configuraciones import crear_slug
from ypyquant.estrategias.evaluacion import evaluar
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.estrategias.receta import Receta
from ypyquant.estrategias.simulacion import simulacion_desde_tablas
from ypyquant.paper.almacen import AlmacenPaper, SesionInexistente, TablasSesion
from ypyquant.paper.calendario import ahora_nueva_york, dias_pendientes_estimados, proximo_rebalanceo
from ypyquant.paper.sesion import MODOS_EJECUCION, EstadoCartera, SesionPaper
from ypyquant.paper.simulador import ResultadoAvance, SimuladorPaper

logger = logging.getLogger(__name__)

CAPITAL_POR_DEFECTO = 100_000.0
TRANSICIONES = {"pausar": ("activa", "pausada"), "reanudar": ("pausada", "activa"),
                "detener": (("activa", "pausada"), "detenida")}


class ExperimentoNoTerminado(ValueError):
    """El experimento de origen de una sesión no ha terminado correctamente."""


class TransicionInvalida(ValueError):
    """Cambio de estado no permitido (p. ej. reanudar una sesión detenida)."""


def receta_de_experimento(almacen: AlmacenExperimentos, id_experimento: str) -> Receta:
    """
    La receta (estrategia + parámetros + entorno) de un experimento TERMINADO. `KeyError` si no existe y
    `ExperimentoNoTerminado` si está pendiente, en curso o falló.
    """
    config = almacen.leer_config(id_experimento)
    if config is None:
        raise KeyError(id_experimento)
    if almacen.estado_de(id_experimento) != "hecho":
        raise ExperimentoNoTerminado(f"El experimento {id_experimento} no ha terminado ({almacen.estado_de(id_experimento)})")
    return ConfiguracionExperimento(**config).receta()


def curva_equity(equity: pd.DataFrame) -> pd.DataFrame:
    """La equity de la sesión con `retorno` (diario) y `drawdown` (fracción bajo el máximo previo), como la del backtest."""
    curva = equity.copy()
    curva["retorno"] = curva["equity"].pct_change()
    curva["drawdown"] = curva["equity"] / curva["equity"].cummax() - 1.0
    return curva


class ServicioPaper:
    """Crea, consulta y avanza sesiones de paper interno (ver cabecera del módulo)."""

    def __init__(self, almacen: AlmacenPaper, fuente: FuentePrecios | Callable,
                 ahora: Callable[[], datetime] = ahora_nueva_york, membresias: Membresias | None = None,
                 meses_historia_pit: int = 12):
        self.almacen = almacen
        self.ahora = ahora
        self.simulador = SimuladorPaper(almacen, fuente, ahora, membresias, meses_historia_pit)

    # ---- altas ---------------------------------------------------------------------------------

    def _id_libre(self, nombre: str) -> str:
        base = crear_slug(nombre)
        candidato, n = base, 1
        while self.almacen.documentos.hay_carpeta(candidato):
            n += 1
            candidato = f"{base}-{n}"
        return candidato

    def crear_sesion(self, receta: Receta, *, nombre: str | None = None, fecha_alta: str | None = None,
                     capital_inicial: float = CAPITAL_POR_DEFECTO, modo_ejecucion: str = "apertura_siguiente",
                     comision: float | None = None, slippage: float | None = None,
                     id_experimento: str | None = None, id_configuracion: str | None = None) -> SesionPaper:
        """
        Congela `receta` en una sesión nueva (ver cabecera). `ValueError` si el modo, el capital o el universo no sirven.

        `comision` y `slippage` sustituyen a los de la receta solo en esta sesión (Alpaca no cobra comisión: así se
        compara «costes realistas» con «costes Alpaca» sin tocar la receta).
        """
        if modo_ejecucion not in MODOS_EJECUCION:
            raise ValueError(f"Modo de ejecución inválido: {modo_ejecucion!r} (válidos: {', '.join(MODOS_EJECUCION)})")
        if capital_inicial <= 0:
            raise ValueError("El capital inicial debe ser positivo")
        hoy = self.ahora().date().isoformat()
        alta = pd.Timestamp(fecha_alta).strftime("%Y-%m-%d") if fecha_alta else hoy
        if (receta.parametros.get("apalancamiento_maximo") or 1) > 1:
            raise ValueError("El paper interno aún no modela margen: la receta usa apalancamiento_maximo > 1 "
                             "(CASH negativo con intereses en el motor)")
        receta = receta.receta().normalizada()
        if isinstance(receta.universo, str):
            # `sp500_pit` se conserva como nombre: el simulador lo resuelve en cada fecha (miembros vigentes ese día)
            universo = receta.universo if es_universo_pit(receta.universo) else resolver_universo(receta.universo, inicio=alta, fin=alta)
        else:
            universo = list(receta.universo)
        cambios: dict[str, Any] = {"universo": universo}
        if comision is not None:
            cambios["comision"] = comision
        if slippage is not None:
            cambios["slippage"] = slippage
        receta = Receta(**{**receta.model_dump(), **cambios})
        nombre = nombre or f"{receta.estrategia} {alta}"
        with self.almacen.documentos.cerrojo:
            sesion = SesionPaper(id=self._id_libre(nombre), nombre=nombre, receta=receta, fecha_alta=alta,
                                 capital_inicial=capital_inicial, modo_ejecucion=modo_ejecucion,  # type: ignore[arg-type]
                                 id_experimento=id_experimento, id_configuracion=id_configuracion)
            self.almacen.crear(sesion, EstadoCartera.inicial(capital_inicial, alta))
        return sesion

    # ---- control -------------------------------------------------------------------------------

    def cambiar_estado(self, id_sesion: str, accion: str) -> SesionPaper:
        """`pausar` | `reanudar` | `detener`. `TransicionInvalida` si el estado actual no lo permite."""
        origenes, destino = TRANSICIONES[accion]
        origenes = (origenes,) if isinstance(origenes, str) else origenes
        with self.almacen.documentos.cerrojo:
            sesion = self.almacen.leer_sesion(id_sesion)
            if sesion.estado not in origenes:
                raise TransicionInvalida(f"No se puede {accion} una sesión {sesion.estado}")
            sesion.estado = destino  # type: ignore[assignment]
            self.almacen.guardar_sesion(sesion)
        return sesion

    def avanzar(self, id_sesion: str | None = None) -> list[ResultadoAvance]:
        """Avanza una sesión o, sin id, todas las activas."""
        if id_sesion is None:
            return self.simulador.avanzar_todas()
        return [self.simulador.avanzar(id_sesion)]

    def hay_sesiones_activas(self) -> bool:
        return any(s.estado == "activa" for s in self.almacen.sesiones())

    def hay_dias_pendientes(self) -> bool:
        """True si alguna sesión activa tiene días laborables cerrados sin procesar (estimación sin datos: ver `calendario`)."""
        ahora = self.ahora()
        for sesion in self.almacen.sesiones():
            if sesion.estado != "activa":
                continue
            ultima = date.fromisoformat(self.almacen.leer_estado(sesion.id).ultima_fecha_procesada)
            if dias_pendientes_estimados(ultima, ahora) > 0:
                return True
        return False

    # ---- consultas -----------------------------------------------------------------------------

    def _evaluar(self, sesion: SesionPaper, tablas: TablasSesion) -> dict[str, Any] | None:
        """Métricas forward con `evaluar`; None mientras no haya dos fechas (o si la evaluación falla)."""
        if len(tablas.equity) < 2:
            return None
        config = sesion.config_evaluacion(tablas.equity["fecha"].iloc[0], tablas.equity["fecha"].iloc[-1])
        if config is None:
            return None
        try:
            simulacion = simulacion_desde_tablas(config, curva_equity(tablas.equity), tablas.pesos, tablas.operaciones,
                                                 tablas.rotaciones)
            return evaluar(simulacion, validar=False).metricas
        except Exception as error:  # noqa: BLE001 - una métrica rota no debe tumbar la lista
            logger.warning("No se pudieron evaluar las métricas de %s: %s", sesion.id, error)
            return None

    @staticmethod
    def _costes(tablas: TablasSesion) -> dict[str, float]:
        comisiones = float(tablas.operaciones["comision"].sum()) if len(tablas.operaciones) else 0.0
        slippage = float(tablas.operaciones["slippage"].sum()) if len(tablas.operaciones) else 0.0
        return {"comisiones": comisiones, "slippage": slippage, "total": comisiones + slippage}

    def resumen(self, id_sesion: str) -> dict[str, Any]:
        """Una fila de la lista de sesiones (ver `listar`)."""
        sesion = self.almacen.leer_sesion(id_sesion)
        estado = self.almacen.leer_estado(id_sesion)
        tablas = self.almacen.leer_tablas(id_sesion, estado.ultima_fecha_procesada)
        metricas = self._evaluar(sesion, tablas) or {}
        pendientes = dias_pendientes_estimados(date.fromisoformat(estado.ultima_fecha_procesada), self.ahora())
        equity = float(tablas.equity["equity"].iloc[-1]) if len(tablas.equity) else sesion.capital_inicial
        return limpiar_para_json({
            "id": sesion.id, "nombre": sesion.nombre, "estrategia": sesion.receta.estrategia,
            "modo_ejecucion": sesion.modo_ejecucion, "estado": sesion.estado, "fecha_alta": sesion.fecha_alta,
            "capital_inicial": sesion.capital_inicial, "equity": equity, "dias": len(tablas.equity),
            "ultima_fecha": estado.ultima_fecha_procesada if len(tablas.equity) else None,
            "retorno_total": equity / sesion.capital_inicial - 1.0 if len(tablas.equity) else None,
            "sharpe": metricas.get("sharpe"), "max_drawdown": metricas.get("max_drawdown"),
            "volatilidad": metricas.get("volatilidad"), "costes": self._costes(tablas)["total"],
            "dias_pendientes": pendientes, "al_dia": pendientes == 0,
            "frecuencia_rebalanceo": sesion.receta.frecuencia_rebalanceo, "ultimo_rebalanceo": estado.ultimo_rebalanceo,
            "proximo_rebalanceo": proximo_rebalanceo(
                sesion.receta.frecuencia_rebalanceo,
                date.fromisoformat(estado.ultimo_rebalanceo) if estado.ultimo_rebalanceo else None,
                date.fromisoformat(estado.ultima_fecha_procesada)).isoformat(),
            "id_experimento": sesion.id_experimento, "id_configuracion": sesion.id_configuracion,
        })

    def listar(self) -> list[dict[str, Any]]:
        """Una fila por sesión: identidad, estado, retorno, Sharpe, drawdown, costes y días pendientes."""
        return [self.resumen(i) for i in self.almacen.ids()]

    def detalle(self, id_sesion: str) -> dict[str, Any]:
        """Todo de una sesión menos las series: receta, métricas forward, posiciones, pesos y avisos."""
        sesion = self.almacen.leer_sesion(id_sesion)
        estado = self.almacen.leer_estado(id_sesion)
        tablas = self.almacen.leer_tablas(id_sesion, estado.ultima_fecha_procesada)
        metricas = self._evaluar(sesion, tablas)
        valor_posiciones = sum(p.cantidad * estado.ultimos_cierres.get(t, p.precio_medio) for t, p in estado.posiciones.items())
        total = estado.efectivo + valor_posiciones
        posiciones = [{
            "ticker": t, "cantidad": p.cantidad, "precio_medio": p.precio_medio,
            "ultimo_cierre": estado.ultimos_cierres.get(t), "valor": p.cantidad * estado.ultimos_cierres.get(t, p.precio_medio),
            "peso": p.cantidad * estado.ultimos_cierres.get(t, p.precio_medio) / total if total > 0 else 0.0,
        } for t, p in sorted(estado.posiciones.items())]
        pesos_objetivo = []
        if len(tablas.pesos):
            ultima = tablas.pesos["fecha"].max()
            pesos_objetivo = tablas.pesos[tablas.pesos["fecha"] == ultima][["ticker", "peso"]].to_dict("records")
        resumen = self.resumen(id_sesion)
        return limpiar_para_json({
            **resumen, "sesion": sesion.model_dump(), "metricas": metricas, "costes_detalle": self._costes(tablas),
            "efectivo": estado.efectivo, "posiciones": posiciones, "pesos_objetivo": pesos_objetivo,
            "pesos_pendientes": estado.pesos_pendientes, "fecha_decision_pendiente": estado.fecha_decision_pendiente,
            "avisos": tablas.eventos, "n_operaciones": len(tablas.operaciones),
            "ejecucion_optimista": sesion.modo_ejecucion == "cierre",
            "universo_dinamico": sesion.universo_dinamico, "avisos_universo": self._avisos_universo(sesion, estado),
        })

    def _avisos_universo(self, sesion: SesionPaper, estado: EstadoCartera) -> list[str]:
        """Calidad de las membresías del universo dinámico ahora mismo (vacío con universo fijo); no se persiste."""
        if not sesion.universo_dinamico:
            return []
        try:
            return avisos_membresias(self.simulador.membresias(), estado.ultima_fecha_procesada)
        except FileNotFoundError as error:
            return [str(error)]

    @staticmethod
    def _registros(tabla: pd.DataFrame) -> list[dict[str, Any]]:
        copia = tabla.copy()
        if "fecha" in copia.columns:
            copia["fecha"] = pd.to_datetime(copia["fecha"]).dt.strftime("%Y-%m-%d")
        return limpiar_para_json(copia.replace({np.nan: None}).to_dict("records"))

    def equity(self, id_sesion: str) -> list[dict[str, Any]]:
        estado = self.almacen.leer_estado(id_sesion)
        tablas = self.almacen.leer_tablas(id_sesion, estado.ultima_fecha_procesada)
        return self._registros(curva_equity(tablas.equity)) if len(tablas.equity) else []

    def operaciones(self, id_sesion: str) -> list[dict[str, Any]]:
        estado = self.almacen.leer_estado(id_sesion)
        return self._registros(self.almacen.leer_tablas(id_sesion, estado.ultima_fecha_procesada).operaciones)

    def pesos(self, id_sesion: str) -> list[dict[str, Any]]:
        estado = self.almacen.leer_estado(id_sesion)
        return self._registros(self.almacen.leer_tablas(id_sesion, estado.ultima_fecha_procesada).pesos)

    def config_espejo(self, id_sesion: str):
        """Configuración del backtest espejo (misma receta, de `fecha_alta` a la última fecha); None si aún no hay días."""
        sesion = self.almacen.leer_sesion(id_sesion)
        estado = self.almacen.leer_estado(id_sesion)
        tablas = self.almacen.leer_tablas(id_sesion, estado.ultima_fecha_procesada)
        if len(tablas.equity) < 2:
            return None
        return sesion.config_experimento(f"{tablas.equity['fecha'].iloc[0]:%Y-%m-%d}", f"{tablas.equity['fecha'].iloc[-1]:%Y-%m-%d}")

    def existe(self, id_sesion: str) -> bool:
        try:
            return self.almacen.existe(id_sesion)
        except ValueError:
            return False

    def sesion(self, id_sesion: str) -> SesionPaper:
        """La sesión o `SesionInexistente`."""
        if not self.existe(id_sesion):
            raise SesionInexistente(id_sesion)
        return self.almacen.leer_sesion(id_sesion)


class ControlAvance:
    """
    Estado consultable de un avance en segundo plano, con exclusión: solo uno a la vez.

    `iniciar()` toma el turno (False si ya hay uno en curso); `ejecutar(funcion)` corre `funcion` (que devuelve los
    `ResultadoAvance`), guarda resultados o error y libera el turno. `estado()` es lo que expone la API.
    """

    def __init__(self) -> None:
        self._cerrojo = threading.Lock()
        self._estado: dict[str, Any] = {"en_curso": False, "inicio": None, "fin": None, "resultados": [], "error": None}

    def iniciar(self) -> bool:
        with self._cerrojo:
            if self._estado["en_curso"]:
                return False
            self._estado = {"en_curso": True, "inicio": datetime.now().isoformat(timespec="seconds"), "fin": None,
                            "resultados": [], "error": None}
            return True

    def ejecutar(self, funcion: Callable[[], list[ResultadoAvance]]) -> None:
        resultados, error = [], None
        try:
            resultados = [r.a_dict() for r in funcion()]
        except BaseException as fallo:
            logger.exception("Falló el avance del paper interno")
            error = f"{type(fallo).__name__}: {fallo}"
        with self._cerrojo:
            self._estado = {**self._estado, "en_curso": False, "fin": datetime.now().isoformat(timespec="seconds"),
                            "resultados": resultados, "error": error}

    def estado(self) -> dict[str, Any]:
        with self._cerrojo:
            return dict(self._estado)
