"""
simulador.py — Simulador incremental del paper interno: avanza las sesiones día a día con precios reales.

Responsabilidad:
    `SimuladorPaper.avanzar(id)`, `avanzar_varias(ids)` y `avanzar_todas()` procesan, en orden, cada día de
    trading CERRADO y pendiente de cada sesión (catch-up exacto tras un parón: nada depende de que el proceso
    estuviera vivo). Una descarga por ticker para todas las sesiones (unión de universos). Bucle de un día D:
      1. precios: cierre de D (y apertura de D en modo `apertura_siguiente`); un ticker sin barra usa su último
         cierre, como el motor (`BacktestEngine._get_prices_at_date`);
      2. re-base por ajuste (una vez por avance): si el cierre de la última fecha procesada, recién descargado,
         difiere del guardado más de `TOLERANCIA_REBASE` (dividendo o split: yfinance reescribe el histórico
         ajustado), se escala `cantidad /= f` y `precio_medio *= f` para que el valor no salte;
      3. ejecución: planifica con el Ejecutor (`planificar_rebalanceo` + `POLITICA_MOTOR`) y ejecuta con el
         `BrokerSimulado` aplicando la MISMA comisión y slippage que el motor (`_buy`/`_sell`);
         margen y cortos (recetas apalancadas o largo-corto): las MISMAS reglas que el motor, de una sola implementación
         (`ypyquant.execution.margen`): cargos diarios (interés del préstamo y préstamo de títulos) al empezar el día,
         capacidad de compra/venta en corto por el límite de exposición bruta, y comprobación diaria del mantenimiento
         con liquidación forzosa proporcional y un evento `llamada_margen`. Sin apalancamiento ni cortos nada de esto actúa;
      4. valoración al cierre de D -> una fila de equity; 5. decisión si toca (`toca_rebalancear`): en modo
         `cierre` se ejecuta ya al cierre de D (equivale a `run_backtest`: la fila de equity es la de antes del
         rebalanceo salvo en la asignación inicial); en `apertura_siguiente` queda como `pesos_pendientes` y se
         ejecuta a la apertura de D+1; 6. commit: tablas y, AL FINAL, `estado.json` con `ultima_fecha_procesada`.
    Solo se procesa una sesión `D` si ya es definitiva (`ultima_sesion_cerrada`) y su barra es completa
    (benchmark y al menos `COBERTURA_MINIMA` del universo; si no, se pospone el último día o se omite uno
    intermedio con aviso). Un ticker que lleva `SESIONES_PARA_LIQUIDAR` sesiones sin barra se liquida a su último
    precio con aviso (deslistado o cambio de ticker).
    Métricas forward: `evaluar(simulacion_desde_tablas(...))`, la misma función que el backtest (ver `servicio`).
    Universo `sp500_pit` en vivo: la sesión conserva el nombre y el simulador lo resuelve en CADA fecha procesada con
    los miembros vigentes ese día (point-in-time con la fecha de la barra; con la de hoy es la membresía vigente), con
    el mismo filtro que el backtest (`UniversoPuntoEnElTiempo`: miembros con historia suficiente hasta t-1). Un ticker
    que sale del índice no se vende ese día: se liquida en el siguiente rebalanceo (la estrategia ya no lo ve) y se
    anota un aviso. Se avisa una vez por sesión si las membresías tienen más de 30 días de antigüedad respecto a la
    fecha procesada o si el número de miembros vigentes se aleja de ~500 (`ypyquant.data.universos.avisos_membresias`).

Se relaciona con:
    - ypyquant.execution (ejecutor, broker_simulado, calendario): decisión de órdenes, ejecución y rebalanceo.
    - ypyquant.estrategias (registro, base: `como_funcion`, universo_dinamico) y ypyquant.data.fuente_precios (precios).
    - ypyquant.data.universos (membresías vigentes y avisos de calidad del universo `sp500_pit`).
    - ypyquant.paper.almacen (estado y tablas), .sesion (modelos) y .calendario (sesiones cerradas).
    - tests/test_paper_paridad.py: el avance día a día equivale al backtest del mismo rango (modo `cierre`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from ypyquant.almacen_documentos import limpiar_para_json
from ypyquant.data.fuente_precios import FuentePrecios, como_fuente
from ypyquant.data.universos import (
    Membresias,
    avisos_membresias,
    cargar_membresias_sp500,
    miembros_en_ventana,
    miembros_vigentes_en,
)
from ypyquant.estrategias.universo_dinamico import FiltroUniverso, UniversoPuntoEnElTiempo
from ypyquant.estrategias.registro import obtener_estrategia
from ypyquant.execution.broker_simulado import BrokerSimulado
from ypyquant.execution.calendario import toca_rebalancear
from ypyquant.execution.ejecutor import COMPRA, VENTA, planificar_rebalanceo
from ypyquant.execution.margen import (
    ReglasMargen,
    capacidad_bruta,
    capacidad_de_compra,
    cargos_diarios,
    evaluar_margen,
    factor_de_conservacion,
    politica_para,
    reglas_desde_parametros,
    valores_largo_corto,
)
from ypyquant.paper.almacen import (
    COLUMNAS_EQUITY,
    COLUMNAS_OPERACIONES,
    COLUMNAS_PESOS,
    COLUMNAS_ROTACIONES,
    AlmacenPaper,
    SesionBloqueada,
    TablasSesion,
)
from ypyquant.paper.calendario import ahora_nueva_york, dias_pendientes, ultima_sesion_cerrada
from ypyquant.paper.sesion import EstadoCartera, PosicionPaper, SesionPaper

logger = logging.getLogger(__name__)

TOLERANCIA_REBASE = 1e-6
COBERTURA_MINIMA = 0.8
SESIONES_PARA_LIQUIDAR = 5
UMBRAL_POSICION_MINIMA = 0.001  # el motor descarta lo que quede por debajo tras una venta parcial
# Claves del `estado` de una estrategia que son caché recalculable (no JSON): no se persisten
CLAVES_CACHE_ESTADO = ("modelo",)


@dataclass
class ResultadoAvance:
    """Qué hizo el avance de una sesión: días procesados, avisos y, si no avanzó, por qué (`omitida` o `error`)."""

    id_sesion: str
    dias_procesados: int = 0
    desde: str | None = None
    hasta: str | None = None
    avisos: list[str] = field(default_factory=list)
    omitida: str | None = None
    error: str | None = None

    def a_dict(self) -> dict:
        return {"id_sesion": self.id_sesion, "dias_procesados": self.dias_procesados, "desde": self.desde,
                "hasta": self.hasta, "avisos": self.avisos, "omitida": self.omitida, "error": self.error}


def estado_serializable(estado: Mapping) -> dict:
    """
    Copia JSON del `estado` de una estrategia (contrato: solo tipos JSON). Las claves de caché recalculable
    (`CLAVES_CACHE_ESTADO`) se omiten; cualquier otro valor no serializable lanza `TypeError` con la clave.
    """
    limpio = {}
    for clave, valor in estado.items():
        if clave in CLAVES_CACHE_ESTADO:
            continue
        try:
            json.dumps(limpiar_para_json(valor))
        except TypeError as error:
            raise TypeError(f"El estado de la estrategia no es JSON: clave {clave!r} ({error})") from error
        limpio[clave] = limpiar_para_json(valor)
    return limpio


def _ultimo_precio(df: pd.DataFrame, fecha: pd.Timestamp, columna: str = "Close") -> float | None:
    """Valor de `columna` el día `fecha` o, si no cotizó, el del último día anterior (None si no hay)."""
    posicion = df.index.searchsorted(fecha, side="right") - 1
    if posicion < 0:
        return None
    valor = float(df[columna].iloc[posicion])
    return valor if np.isfinite(valor) else None


def _anadir(tabla: pd.DataFrame, filas: list[dict], columnas: list[str]) -> pd.DataFrame:
    if not filas:
        return tabla
    nuevas = pd.DataFrame(filas, columns=columnas)
    return nuevas if tabla.empty else pd.concat([tabla, nuevas], ignore_index=True)


class _Decisor:
    """La estrategia de una sesión durante un avance: restaura su memoria y su generador y los vuelca al terminar."""

    def __init__(self, sesion: SesionPaper, estado: EstadoCartera, filtro_universo: FiltroUniverso | None = None):
        receta = sesion.receta
        self.estrategia = obtener_estrategia(receta.estrategia)(receta.parametros)
        self.estrategia.semilla = receta.semilla
        self.estrategia.estado = dict(estado.estado_estrategia)
        generador = np.random.default_rng(receta.semilla)
        if estado.estado_rng is not None:
            generador.bit_generator.state = estado.estado_rng
        self.estrategia.rng = generador
        self.funcion = self.estrategia.como_funcion(filtro_universo=filtro_universo)
        self.reglas = reglas_desde_parametros(self.estrategia.parametros)  # margen y cortos de la receta (inactivas por defecto)

    def pesos(self, fecha: pd.Timestamp, datos: dict[str, pd.DataFrame]) -> dict[str, float]:
        return self.funcion(fecha, datos)

    def volcar_en(self, estado: EstadoCartera) -> None:
        estado.estado_estrategia = estado_serializable(self.estrategia.estado)
        estado.estado_rng = limpiar_para_json(self.estrategia.rng.bit_generator.state)


class SimuladorPaper:
    """
    Avanza sesiones de paper interno (ver cabecera del módulo).

    Args:
        almacen: sesiones en disco. fuente: de dónde salen los precios (`FuentePrecios` o cargador suelto).
        ahora: reloj inyectable (devuelve un `datetime`); por defecto la hora de Nueva York.
        membresias: membresías del S&P 500 para `sp500_pit` (por defecto `universos/sp500_membresias.json`, leído al
            primer uso). meses_historia_pit: historia mensual que exige el filtro point-in-time (12, como el backtest).
    """

    def __init__(self, almacen: AlmacenPaper, fuente: FuentePrecios | Callable, ahora: Callable[[], datetime] = ahora_nueva_york,
                 membresias: Membresias | None = None, meses_historia_pit: int = 12):
        self.almacen = almacen
        self.fuente = como_fuente(fuente)
        self.ahora = ahora
        self._membresias = membresias
        self.meses_historia_pit = meses_historia_pit
        self._vigentes: dict[pd.Timestamp, frozenset[str]] = {}

    # ---- universo dinámico ---------------------------------------------------------------------

    def membresias(self) -> Membresias:
        """Las membresías del S&P 500 (se cargan una vez, la primera vez que hacen falta)."""
        if self._membresias is None:
            self._membresias = cargar_membresias_sp500()
        return self._membresias

    def _vigentes_en(self, dia: pd.Timestamp) -> frozenset[str]:
        """Miembros vigentes de `dia` (con caché: se consulta varias veces por día y sesión)."""
        dia = pd.Timestamp(dia).normalize()
        if dia not in self._vigentes:
            self._vigentes[dia] = frozenset(miembros_vigentes_en(self.membresias(), dia))
        return self._vigentes[dia]

    def _tickers_a_cargar(self, sesion: SesionPaper, tope: date) -> list[str]:
        """Símbolos cuyos precios hacen falta: la lista fija o, con `sp500_pit`, todo miembro entre el alta y `tope`."""
        if not sesion.universo_dinamico:
            return sesion.tickers
        return miembros_en_ventana(self.membresias(), sesion.fecha_alta, tope.isoformat())

    def _universo_del_dia(self, sesion: SesionPaper, dia: pd.Timestamp, universo: dict[str, pd.DataFrame]) -> list[str]:
        """Tickers cuya barra se espera en `dia`: todo el universo o, con `sp500_pit`, los miembros vigentes ese día."""
        if not sesion.universo_dinamico:
            return list(universo)
        vigentes = self._vigentes_en(dia)
        return [t for t in universo if t in vigentes]

    # ---- API pública ---------------------------------------------------------------------------

    def avanzar(self, id_sesion: str, hasta: date | None = None) -> ResultadoAvance:
        """Avanza UNA sesión (aunque esté pausada devuelve `omitida`)."""
        return self.avanzar_varias([id_sesion], hasta)[0]

    def avanzar_todas(self, hasta: date | None = None) -> list[ResultadoAvance]:
        """Avanza todas las sesiones activas con una sola descarga por ticker."""
        return self.avanzar_varias([s.id for s in self.almacen.sesiones() if s.estado == "activa"], hasta)

    def avanzar_varias(self, ids: Sequence[str], hasta: date | None = None) -> list[ResultadoAvance]:
        """Avanza las sesiones `ids` (cada una bajo su bloqueo). Un error en una no impide avanzar las demás."""
        ahora = self.ahora()
        tope = ultima_sesion_cerrada(ahora)
        if hasta is not None:
            tope = min(tope, hasta)
        sesiones = [self.almacen.leer_sesion(i) for i in ids]
        resultados: dict[str, ResultadoAvance] = {}
        a_avanzar = []
        for sesion in sesiones:
            if sesion.estado != "activa":
                resultados[sesion.id] = ResultadoAvance(sesion.id, omitida=sesion.estado)
            else:
                a_avanzar.append(sesion)
        datos = self._cargar_datos(a_avanzar, tope) if a_avanzar else {}
        for sesion in a_avanzar:
            try:
                resultados[sesion.id] = self._avanzar_sesion(sesion.id, datos, tope, ahora)
            except SesionBloqueada as error:
                resultados[sesion.id] = ResultadoAvance(sesion.id, omitida=f"bloqueada: {error}")
            except Exception as error:
                logger.exception("Falló el avance de la sesión %s", sesion.id)
                resultados[sesion.id] = ResultadoAvance(sesion.id, error=f"{type(error).__name__}: {error}")
        return [resultados[s.id] for s in sesiones]

    # ---- datos ---------------------------------------------------------------------------------

    def _cargar_datos(self, sesiones: Sequence[SesionPaper], tope: date) -> dict[str, pd.DataFrame]:
        """OHLCV de la unión de tickers y benchmarks: UNA petición por ticker, desde la historia más antigua que pida alguna."""
        inicio: dict[str, pd.Timestamp] = {}
        for sesion in sesiones:
            desde = pd.Timestamp(sesion.fecha_alta) - pd.Timedelta(days=sesion.receta.dias_historia_previa)
            for simbolo in [*self._tickers_a_cargar(sesion, tope), sesion.receta.benchmark]:
                inicio[simbolo] = min(inicio.get(simbolo, desde), desde)
        fin = tope.isoformat()
        try:
            self.fuente.refrescar_reciente(sorted(inicio), fin)
        except Exception as error:  # noqa: BLE001 - sin red se trabaja con lo cacheado
            logger.warning("No se pudo refrescar el tramo reciente: %s", error)

        def cargar(simbolo: str):
            try:
                return simbolo, self.fuente.ohlcv(simbolo, inicio[simbolo].strftime("%Y-%m-%d"), fin)
            except Exception as error:  # noqa: BLE001
                logger.warning("No se pudo cargar %s: %s", simbolo, error)
                return simbolo, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            cargados = dict(pool.map(cargar, sorted(inicio)))
        return {s: df for s, df in cargados.items() if df is not None and not df.empty}

    # ---- una sesión ----------------------------------------------------------------------------

    def _avanzar_sesion(self, id_sesion: str, datos: dict[str, pd.DataFrame], tope: date, ahora: datetime) -> ResultadoAvance:
        with self.almacen.bloqueo(id_sesion):
            sesion = self.almacen.leer_sesion(id_sesion)
            resultado = ResultadoAvance(id_sesion)
            if sesion.estado != "activa":
                resultado.omitida = sesion.estado
                return resultado
            estado = self.almacen.leer_estado(id_sesion)
            tablas = self.almacen.leer_tablas(id_sesion)
            if tablas.recortar(pd.Timestamp(estado.ultima_fecha_procesada)):  # restos de un día no confirmado
                self.almacen.guardar_tablas(id_sesion, tablas)
            universo = {t: datos[t] for t in self._tickers_a_cargar(sesion, tope) if t in datos}
            if not universo:
                raise RuntimeError("No hay precios de ningún ticker del universo (¿sin red y sin caché?)")
            dias = self._dias_a_procesar(sesion, estado, universo, datos.get(sesion.receta.benchmark), tope, ahora, tablas)
            if not dias:
                return resultado
            filtro = UniversoPuntoEnElTiempo(self.membresias(), self.meses_historia_pit).crear_filtro(universo) \
                if sesion.universo_dinamico else None
            decisor = _Decisor(sesion, estado, filtro)
            self._rebasar(estado, universo, tablas)
            resultado.desde = f"{dias[0]:%Y-%m-%d}"
            for dia in dias:
                eventos_antes = len(tablas.eventos)
                self._procesar_dia(sesion, estado, tablas, decisor, dia, universo, datos.get(sesion.receta.benchmark))
                decisor.volcar_en(estado)
                estado.ultima_fecha_procesada = f"{dia:%Y-%m-%d}"
                self.almacen.guardar_tablas(id_sesion, tablas)
                self.almacen.guardar_estado(id_sesion, estado)  # el commit del día, siempre lo último
                resultado.dias_procesados += 1
                resultado.hasta = estado.ultima_fecha_procesada
                resultado.avisos.extend(e["mensaje"] for e in tablas.eventos[eventos_antes:])
            return resultado

    def _dias_a_procesar(self, sesion, estado, universo, benchmark, tope: date, ahora: datetime,
                         tablas: TablasSesion) -> list[pd.Timestamp]:
        """Días con datos, cerrados y completos posteriores al último procesado (el último incompleto se pospone)."""
        union = sorted({d for df in universo.values() for d in df.index})
        candidatos = [d for d in dias_pendientes(date.fromisoformat(estado.ultima_fecha_procesada), union, ahora)
                      if d <= pd.Timestamp(tope)]
        dias = []
        for posicion, dia in enumerate(candidatos):
            esperados = self._universo_del_dia(sesion, dia, universo)
            con_barra = sum(dia in universo[t].index for t in esperados)
            completo = bool(esperados) and con_barra / len(esperados) >= COBERTURA_MINIMA \
                and (benchmark is None or dia in benchmark.index)
            if completo:
                dias.append(dia)
            elif posicion == len(candidatos) - 1:
                break  # barra parcial del último día: se pospone hasta que el proveedor la complete
            else:
                tablas.eventos.append({"fecha": f"{dia:%Y-%m-%d}", "tipo": "dia_omitido",
                                       "mensaje": f"{dia:%Y-%m-%d}: barra incompleta ({con_barra}/{len(esperados)} tickers); se omite"})
        return dias

    def _rebasar(self, estado: EstadoCartera, universo: dict[str, pd.DataFrame], tablas: TablasSesion) -> None:
        """Re-base por ajuste (ver cabecera): compara el cierre guardado de cada posición con el recién descargado."""
        fecha = pd.Timestamp(estado.ultima_fecha_procesada)
        for ticker, posicion in estado.posiciones.items():
            guardado = estado.ultimos_cierres.get(ticker)
            nuevo = _ultimo_precio(universo[ticker], fecha) if ticker in universo else None
            if not guardado or not nuevo:
                continue
            factor = nuevo / guardado
            if abs(factor - 1.0) > TOLERANCIA_REBASE:
                posicion.cantidad /= factor
                posicion.precio_medio *= factor
                estado.ultimos_cierres[ticker] = nuevo
                tablas.eventos.append({"fecha": estado.ultima_fecha_procesada, "tipo": "rebase", "mensaje": (
                    f"{ticker}: el histórico ajustado cambió (factor {factor:.6f}); se re-basa la posición para conservar su valor")})

    # ---- un día --------------------------------------------------------------------------------

    @staticmethod
    def _precios_de(universo: dict[str, pd.DataFrame], dia: pd.Timestamp) -> tuple[dict[str, float], dict[str, float]]:
        """(cierres, aperturas) de cada ticker: sin barra ese día, el último cierre; sin apertura válida, el cierre."""
        cierres, aperturas = {}, {}
        for ticker, df in universo.items():
            cierre = _ultimo_precio(df, dia)
            if cierre is None:
                continue
            cierres[ticker] = cierre
            apertura = float(df["Open"].loc[dia]) if "Open" in df.columns and dia in df.index else float("nan")
            aperturas[ticker] = apertura if np.isfinite(apertura) and apertura > 0 else cierre
        return cierres, aperturas

    @staticmethod
    def _valor(estado: EstadoCartera, precios: Mapping[str, float]) -> float:
        return estado.efectivo + sum(p.cantidad * precios[t] for t, p in estado.posiciones.items() if t in precios)

    def _procesar_dia(self, sesion: SesionPaper, estado: EstadoCartera, tablas: TablasSesion, decisor: _Decisor,
                      dia: pd.Timestamp, universo: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None) -> None:
        cierres, aperturas = self._precios_de(universo, dia)
        if sesion.universo_dinamico:
            self._avisos_universo_dinamico(estado, tablas, dia, sesion.receta.frecuencia_rebalanceo)
        self._liquidar_desaparecidos(sesion, estado, tablas, dia, universo, cierres)
        modo_cierre = sesion.modo_ejecucion == "cierre"
        frecuencia = sesion.receta.frecuencia_rebalanceo
        if decisor.reglas.activo and estado.asignacion_inicial_hecha:
            self._cobrar_cargos(estado, decisor.reglas, dia, cierres)  # como el motor: a partir del día siguiente a la asignación

        if modo_cierre:
            if not estado.asignacion_inicial_hecha:
                self._decidir_y_ejecutar(sesion, estado, tablas, decisor, dia, universo, cierres)
                fila_equity = self._valor(estado, cierres)  # la asignación inicial cuenta en la equity del primer día
            else:
                fila_equity = self._valor(estado, cierres)  # la equity de D es la de ANTES de rebalancear a su cierre
                if decisor.reglas.activo:
                    self._comprobar_margen(sesion, estado, tablas, decisor.reglas, dia, cierres, fila_equity)
                ultimo = date.fromisoformat(estado.ultimo_rebalanceo) if estado.ultimo_rebalanceo else None
                if toca_rebalancear(frecuencia, ultimo, dia.date()):
                    self._decidir_y_ejecutar(sesion, estado, tablas, decisor, dia, universo, cierres)
        else:
            if estado.pesos_pendientes is not None:
                self._ejecutar(sesion, estado, tablas, estado.pesos_pendientes, aperturas, dia, decisor.reglas)
                estado.pesos_pendientes, estado.fecha_decision_pendiente = None, None
            fila_equity = self._valor(estado, cierres)
            if decisor.reglas.activo and estado.posiciones:
                self._comprobar_margen(sesion, estado, tablas, decisor.reglas, dia, cierres, fila_equity)
            ultimo = date.fromisoformat(estado.ultimo_rebalanceo) if estado.ultimo_rebalanceo else None
            if not estado.asignacion_inicial_hecha or toca_rebalancear(frecuencia, ultimo, dia.date()):
                pesos = self._decidir(sesion, estado, tablas, decisor, dia, universo)
                if pesos is not None:
                    estado.pesos_pendientes, estado.fecha_decision_pendiente = pesos, f"{dia:%Y-%m-%d}"
                    estado.asignacion_inicial_hecha, estado.ultimo_rebalanceo = True, f"{dia:%Y-%m-%d}"

        valor_benchmark = _ultimo_precio(benchmark, dia) if benchmark is not None else None
        tablas.equity = _anadir(tablas.equity, [{
            "fecha": dia, "equity": fila_equity, "efectivo": estado.efectivo, "n_posiciones": len(estado.posiciones),
            "benchmark": valor_benchmark if valor_benchmark is not None else float("nan")}], COLUMNAS_EQUITY)
        estado.ultimos_cierres = {t: cierres[t] for t in estado.posiciones if t in cierres}

    def _avisos_universo_dinamico(self, estado: EstadoCartera, tablas: TablasSesion, dia: pd.Timestamp, frecuencia: str) -> None:
        """
        Avisos del universo `sp500_pit` al procesar `dia`: (1) cada posición que sale del índice desde el último día
        procesado (no se vende hoy: la estrategia deja de verla y se cierra en el próximo rebalanceo); (2) una vez por
        sesión, membresías antiguas o con un número de miembros anómalo.
        """
        anterior = pd.Timestamp(estado.ultima_fecha_procesada)
        vigentes_antes, vigentes_hoy = self._vigentes_en(anterior), self._vigentes_en(dia)
        for ticker in sorted(estado.posiciones):
            if ticker in vigentes_antes and ticker not in vigentes_hoy:
                tablas.eventos.append({"fecha": f"{dia:%Y-%m-%d}", "tipo": "salida_indice", "mensaje": (
                    f"{ticker}: sale del S&P 500 el {dia:%Y-%m-%d}; se mantiene hasta el próximo rebalanceo "
                    f"({frecuencia}) y entonces se liquida")})
        ya_avisados = {e.get("tipo") for e in tablas.eventos}
        for mensaje in avisos_membresias(self.membresias(), dia):
            tipo = "membresia_antigua" if "antigüedad" in mensaje else "membresia_numero"
            if tipo not in ya_avisados:
                tablas.eventos.append({"fecha": f"{dia:%Y-%m-%d}", "tipo": tipo, "mensaje": mensaje})

    def _decidir(self, sesion, estado, tablas, decisor, dia, universo) -> dict[str, float] | None:
        """Pesos objetivo de la estrategia en `dia` (None si falla: se anota el error y se reintenta el siguiente día)."""
        try:
            pesos = decisor.pesos(dia, universo)
        except Exception as error:  # noqa: BLE001 - como el motor: un día fallido no aborta la sesión
            logger.warning("Estrategia de %s falló el %s: %s", sesion.id, dia, error)
            tablas.eventos.append({"fecha": f"{dia:%Y-%m-%d}", "tipo": "error_estrategia",
                                   "mensaje": f"{dia:%Y-%m-%d}: la estrategia falló ({type(error).__name__}: {error})"})
            return None
        tablas.pesos = _anadir(tablas.pesos, [{"fecha": dia, "ticker": t, "peso": float(w)} for t, w in pesos.items()],
                               COLUMNAS_PESOS)
        return pesos

    def _decidir_y_ejecutar(self, sesion, estado, tablas, decisor, dia, universo, precios) -> None:
        pesos = self._decidir(sesion, estado, tablas, decisor, dia, universo)
        if pesos is None:
            return
        self._ejecutar(sesion, estado, tablas, pesos, precios, dia, decisor.reglas)
        estado.asignacion_inicial_hecha, estado.ultimo_rebalanceo = True, f"{dia:%Y-%m-%d}"

    def _ejecutar(self, sesion: SesionPaper, estado: EstadoCartera, tablas: TablasSesion, pesos: Mapping[str, float],
                  precios: Mapping[str, float], dia: pd.Timestamp, reglas: ReglasMargen | None = None) -> None:
        """
        Planifica con el Ejecutor (política del motor; con cortos, `politica_para`) y ejecuta con el broker simulado a `precios`.

        Traduce cada `Orden` a la cantidad exacta que calculan `_buy`/`_sell` del motor: la compra se recorta a la capacidad
        de compra (el efectivo, o el margen disponible con `reglas` activas) y no opera menos de 1; la venta vende `importe`
        al precio con slippage, limitada a la posición; una venta sin posición larga abre un corto (recortada al límite
        de exposición bruta) y una compra sobre un corto lo cubre (el cierre, entero). El broker aplica comisión y slippage.
        """
        reglas = reglas or ReglasMargen()
        comision, slippage = sesion.receta.comision, sesion.receta.slippage
        broker = BrokerSimulado(efectivo_inicial=estado.efectivo, comision=comision, slippage=slippage, precios=precios,
                                reglas_margen=reglas)
        broker.posiciones = {t: p.cantidad for t, p in estado.posiciones.items()}
        cuenta = broker.estado_cuenta()
        broker.equity_referencia = cuenta.equity  # como el motor: la capacidad se mide con el equity previo al rebalanceo
        plan = planificar_rebalanceo(pesos, precios, cuenta, politica_para(reglas))
        if cuenta.equity > 0:
            tablas.rotaciones = _anadir(tablas.rotaciones, [{"fecha": dia, "rotacion": plan.rotacion}], COLUMNAS_ROTACIONES)
        filas = []
        for orden in plan.ordenes:
            cantidad = self._cantidad_de_orden(broker, orden, precios[orden.ticker], reglas, comision, slippage)
            if cantidad is None:
                continue
            fila = self._enviar(estado, tablas, broker, orden.ticker, orden.lado, cantidad, precios[orden.ticker], comision, dia)
            if fila is not None:
                filas.append(fila)
        estado.efectivo = broker.efectivo
        tablas.operaciones = _anadir(tablas.operaciones, filas, COLUMNAS_OPERACIONES)

    @staticmethod
    def _cantidad_de_orden(broker: BrokerSimulado, orden, precio: float, reglas: ReglasMargen, comision: float,
                           slippage: float) -> float | None:
        """Cantidad a enviar de una `Orden` con las reglas del motor (None si no se opera: sin efectivo o sin capacidad)."""
        posicion = broker.posiciones.get(orden.ticker, 0.0)
        equity = broker.equity_referencia if broker.equity_referencia is not None else broker.estado_cuenta().equity
        largo, corto = valores_largo_corto(broker.posiciones, broker.precios)
        if orden.lado == VENTA:
            if posicion > 0:
                return min(orden.importe / (precio * (1 - slippage)), posicion)
            if not reglas.permite_cortos:
                return None
            importe = min(orden.importe, capacidad_bruta(equity, largo, corto, reglas))  # abrir o ampliar un corto
            return importe / (precio * (1 - slippage)) if importe >= 1.0 else None
        if posicion < 0:  # cubrir un corto: el cierre entero; si no, lo pedido (como mucho el corto)
            return -posicion if orden.motivo == "cierre" else min(orden.importe / (precio * (1 + slippage)), -posicion)
        importe = min(orden.importe, capacidad_de_compra(broker.efectivo, equity, largo, corto, reglas) / (1 + comision))
        return importe / (precio * (1 + slippage)) if importe >= 1.0 else None

    def _enviar(self, estado, tablas, broker: BrokerSimulado, ticker: str, lado: str, cantidad: float,
                precio_referencia: float, comision: float, dia: pd.Timestamp) -> dict | None:
        """Envía la orden al broker simulado y devuelve la fila de `operaciones` (None si se rechaza, con un evento)."""
        relleno = broker.consultar_orden(broker.enviar_orden(ticker, lado, cantidad))
        if relleno.estado != "filled":
            tablas.eventos.append({"fecha": f"{dia:%Y-%m-%d}", "tipo": "orden_rechazada",
                                   "mensaje": f"{dia:%Y-%m-%d}: orden de {lado} de {ticker} rechazada"})
            return None
        return self._registrar_relleno(estado, broker, relleno, precio_referencia, comision, dia)

    @staticmethod
    def _cobrar_cargos(estado: EstadoCartera, reglas: ReglasMargen, dia: pd.Timestamp, cierres: Mapping[str, float]) -> None:
        """
        Cargos del margen desde la última fecha procesada: interés del efectivo negativo y préstamo de títulos de los cortos
        (`cargos_diarios`: saldo x tasa x días / 360; el fin de semana cuenta 3). Se descuentan del efectivo.
        """
        dias = (dia.date() - date.fromisoformat(estado.ultima_fecha_procesada)).days
        _, corto = valores_largo_corto({t: p.cantidad for t, p in estado.posiciones.items()}, cierres)
        interes, prestamo = cargos_diarios(estado.efectivo, corto, dias, reglas)
        estado.efectivo -= interes + prestamo

    def _comprobar_margen(self, sesion: SesionPaper, estado: EstadoCartera, tablas: TablasSesion, reglas: ReglasMargen,
                          dia: pd.Timestamp, cierres: Mapping[str, float], equity: float) -> None:
        """
        Comprobación diaria del mantenimiento (las mismas reglas que `BacktestEngine._comprobar_margen`): si el equity cae
        bajo `mantenimiento_largo x largos + mantenimiento_corto x cortos` se reducen TODAS las posiciones proporcionalmente
        hasta el requisito inicial (con comisión y slippage) y se anota un evento `llamada_margen` con la fecha.
        """
        largo, corto = valores_largo_corto({t: p.cantidad for t, p in estado.posiciones.items()}, cierres)
        estado_margen = evaluar_margen(equity, largo, corto, reglas)
        if not estado_margen.en_llamada:
            return
        comision, slippage = sesion.receta.comision, sesion.receta.slippage
        factor = factor_de_conservacion(estado_margen.equity, estado_margen.bruta, reglas, comision + slippage)
        broker = BrokerSimulado(efectivo_inicial=estado.efectivo, comision=comision, slippage=slippage, precios=cierres,
                                reglas_margen=reglas)
        broker.posiciones = {t: p.cantidad for t, p in estado.posiciones.items()}
        filas = []
        for ticker, posicion in sorted(estado.posiciones.items()):
            precio = cierres.get(ticker)
            if precio is None:
                continue
            cantidad = posicion.cantidad
            if cantidad > 0:
                a_vender = cantidad if factor <= 0.0 else min(cantidad * precio * (1.0 - factor) / (precio * (1 - slippage)), cantidad)
                fila = self._enviar(estado, tablas, broker, ticker, VENTA, a_vender, precio, comision, dia)
            else:
                a_cubrir = -cantidad if factor <= 0.0 else min(-cantidad * precio * (1.0 - factor) / (precio * (1 + slippage)), -cantidad)
                fila = self._enviar(estado, tablas, broker, ticker, COMPRA, a_cubrir, precio, comision, dia)
            if fila is not None:
                filas.append(fila)
        estado.efectivo = broker.efectivo
        tablas.operaciones = _anadir(tablas.operaciones, filas, COLUMNAS_OPERACIONES)
        tablas.eventos.append({"fecha": f"{dia:%Y-%m-%d}", "tipo": "llamada_margen", "mensaje": (
            f"Llamada de margen {dia:%Y-%m-%d}: equity {estado_margen.equity:,.0f} por debajo del mantenimiento "
            f"{estado_margen.requisito_mantenimiento:,.0f} (exposición bruta {estado_margen.bruta:,.0f}); "
            f"liquidación forzosa: se conserva el {factor:.0%} de las posiciones.")})

    @staticmethod
    def _registrar_relleno(estado: EstadoCartera, broker: BrokerSimulado, relleno, precio_referencia: float,
                           comision: float, dia: pd.Timestamp) -> dict:
        """Actualiza la posición (con su precio medio, como el motor) y devuelve la fila de `operaciones`."""
        ticker, cantidad, precio = relleno.ticker, relleno.cantidad_ejecutada, relleno.precio_medio
        previa = estado.posiciones.get(ticker)
        if relleno.lado == COMPRA and previa is not None and previa.cantidad < 0:  # cubrir un corto
            comision_pagada = cantidad * precio * comision
            pnl = (previa.precio_medio - precio) * cantidad - comision_pagada
            slippage = cantidad * (precio - precio_referencia)
            resto = broker.posiciones.get(ticker, 0.0)
            if abs(resto) < UMBRAL_POSICION_MINIMA:  # como el motor: lo que queda por debajo se descarta
                broker.posiciones.pop(ticker, None)
                del estado.posiciones[ticker]
            else:
                previa.cantidad = resto
        elif relleno.lado == COMPRA:
            medio = precio if previa is None else (previa.cantidad * previa.precio_medio + cantidad * precio) / (previa.cantidad + cantidad)
            estado.posiciones[ticker] = PosicionPaper(cantidad=broker.posiciones[ticker], precio_medio=medio)
            comision_pagada, pnl, slippage = cantidad * precio * comision, float("nan"), cantidad * (precio - precio_referencia)
        elif previa is None or previa.cantidad <= 0:  # abrir o ampliar un corto: el resultado llega al cubrirlo
            medio = precio if previa is None else (-previa.cantidad * previa.precio_medio + cantidad * precio) / (-previa.cantidad + cantidad)
            estado.posiciones[ticker] = PosicionPaper(cantidad=broker.posiciones[ticker], precio_medio=medio)
            comision_pagada, pnl, slippage = cantidad * precio * comision, float("nan"), cantidad * (precio_referencia - precio)
        else:
            posicion = previa
            ingreso = cantidad * precio
            comision_pagada = ingreso * comision
            pnl = (precio - posicion.precio_medio) * cantidad - comision_pagada
            slippage = cantidad * (precio_referencia - precio)
            resto = broker.posiciones.get(ticker, 0.0)
            if resto < UMBRAL_POSICION_MINIMA:  # como el motor: lo que queda por debajo se descarta
                broker.posiciones.pop(ticker, None)
                del estado.posiciones[ticker]
            else:
                posicion.cantidad = resto
        return {"fecha": dia, "ticker": ticker, "lado": relleno.lado, "cantidad": cantidad, "precio": precio,
                "comision": comision_pagada, "pnl": pnl, "slippage": slippage}

    def _liquidar_desaparecidos(self, sesion, estado, tablas, dia, universo, cierres) -> None:
        """Un ticker en cartera sin barra durante `SESIONES_PARA_LIQUIDAR` sesiones seguidas se vende a su último precio."""
        for ticker in list(estado.posiciones):
            hay_barra = ticker in universo and dia in universo[ticker].index
            estado.dias_sin_barra[ticker] = 0 if hay_barra else estado.dias_sin_barra.get(ticker, 0) + 1
            if estado.dias_sin_barra[ticker] < SESIONES_PARA_LIQUIDAR or ticker not in cierres:
                continue
            posicion = estado.posiciones[ticker]
            broker = BrokerSimulado(efectivo_inicial=estado.efectivo, comision=sesion.receta.comision,
                                    slippage=sesion.receta.slippage, precios=cierres)
            broker.posiciones = {t: p.cantidad for t, p in estado.posiciones.items()}
            lado = VENTA if posicion.cantidad > 0 else COMPRA  # un corto se cubre
            relleno = broker.consultar_orden(broker.enviar_orden(ticker, lado, abs(posicion.cantidad)))
            fila = self._registrar_relleno(estado, broker, relleno, cierres[ticker], sesion.receta.comision, dia)
            estado.efectivo = broker.efectivo
            tablas.operaciones = _anadir(tablas.operaciones, [fila], COLUMNAS_OPERACIONES)
            estado.dias_sin_barra.pop(ticker, None)
            tablas.eventos.append({"fecha": f"{dia:%Y-%m-%d}", "tipo": "liquidacion", "mensaje": (
                f"{ticker}: {SESIONES_PARA_LIQUIDAR} sesiones sin barra (¿deslistado o cambio de ticker?); "
                "se liquida a su último precio conocido")})
