"""
Servicio de experimentos: la interfaz de dominio para ir de «configuración» a «backtest», sin FastAPI.

Responsabilidad:
- `ServicioExperimentos`: lanza experimentos en segundo plano (registra `pendiente`, ejecuta en el ejecutor y deja
  `en_curso` -> `hecho` | `error` con traza), reutiliza por huella (misma configuración = mismo id y mismo
  resultado) salvo `forzar`, y consulta estados.
- Lanzamientos compuestos: `lanzar_varios` (N configuraciones), `lanzar_barrido` (crea el barrido y encola sus
  combinaciones), `lanzar_comparativa` (aplica un marco a N estrategias validándolas TODAS antes de lanzar nada y
  registra la comparativa), `lanzar_backtest_de_configuracion` y `lanzar_comparativa_de_configuracion` (una
  configuración guardada con fechas por defecto de una sola regla, `marcos.resolver_fechas`: explícitas > marco enlazado >
  origen > últimos 5 años hasta el último día hábil).
- Consultas: `backtests_de_receta(configuracion)` usa el índice por receta del almacén (no recorre el disco);
  `estado_de`, `borrar` y `borrar_varios` (omite los que siguen en marcha: el ejecutor escribiría en su carpeta).
- Tareas largas de un barrido: `lanzar_permutacion` (prueba de permutación en un hilo aparte, con estado y progreso en
  `barrido.json` -> `tareas`) y `cancelar_permutacion` (cancelación cooperativa con un `threading.Event` por tarea).
- Errores: lanza las excepciones del dominio sin traducir (`KeyError` estrategia desconocida, `ValueError` /
  `pydantic.ValidationError` configuración o parámetros inválidos, `BarridoEnCurso`...). Traducirlas a códigos HTTP
  (404/409/422) es trabajo de `app/api/rutas_*.py`.
- Testeable sin FastAPI: recibe los almacenes, el cargador de precios y el ejecutor (en tests, uno síncrono).

Se relaciona con:
- ypyquant.estrategias.almacen (estado y resultados), .barridos (`AlmacenBarridos`), .marcos (`AlmacenMarcos`,
  `preparar_comparativa`, `marco_desde_configuracion`, `resolver_fechas`), .configuraciones (`ConfiguracionGuardada`),
  .receta (huella por receta) y .experimentos (`ejecutar_experimento`).
- app/api/contexto.py (lo construye y lo guarda en el contexto) y app/api/rutas_experimentos.py, rutas_barridos.py,
  rutas_marcos.py y rutas_configuraciones.py (lo llaman).
- .validacion_barridos (`prueba_permutacion_barrido`, `PruebaCancelada`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from datetime import date
from typing import Any, NamedTuple

from ypyquant.data.matriz_precios import CargadorPrecios
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.barridos import AlmacenBarridos, ConfiguracionBarrido
from ypyquant.estrategias.configuraciones import ConfiguracionGuardada
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, ejecutar_experimento
from ypyquant.estrategias.marcos import (
    AlmacenMarcos,
    MarcoComparacion,
    MarcoExistente,
    marco_desde_configuracion,
    preparar_comparativa,
    resolver_fechas,
)
from ypyquant.estrategias.validacion_barridos import PruebaCancelada, prueba_permutacion_barrido

logger = logging.getLogger(__name__)
ESTADOS_EN_MARCHA = ("pendiente", "en_curso")
ESTADOS_REUTILIZABLES = ("hecho", *ESTADOS_EN_MARCHA)


class Lanzamiento(NamedTuple):
    """Resultado de lanzar un experimento: su id y si se reutilizó uno ya calculado (o en marcha)."""

    id: str
    reutilizado: bool


class ServicioExperimentos:
    """
    Orquesta el ciclo de vida de los experimentos (ver cabecera del módulo).

    Args:
        almacen: persistencia de experimentos. cargador_precios: fuente de precios del backtest.
        ejecutor: dónde corre cada experimento (`ThreadPoolExecutor` en la API; uno síncrono en los tests).
        almacen_barridos / almacen_marcos: necesarios solo para `lanzar_barrido` y las comparativas.
        ejecutor_barridos: dónde corren las pruebas de permutación (1 a la vez: ya paralelizan por dentro); necesario
        solo para `lanzar_permutacion`.
    """

    def __init__(self, almacen: AlmacenExperimentos, cargador_precios: CargadorPrecios, ejecutor: Executor,
                 almacen_barridos: AlmacenBarridos | None = None, almacen_marcos: AlmacenMarcos | None = None,
                 ejecutor_barridos: Executor | None = None):
        self.almacen = almacen
        self.cargador_precios = cargador_precios
        self.ejecutor = ejecutor
        self.almacen_barridos = almacen_barridos
        self.almacen_marcos = almacen_marcos
        self.ejecutor_barridos = ejecutor_barridos
        self._cancelaciones: dict[tuple[str, str], threading.Event] = {}  # tareas en curso, por (id_barrido, tarea)

    # ---- experimentos ---------------------------------------------------------------------

    def _ejecutar(self, id_experimento: str, config: ConfiguracionExperimento) -> None:
        """Cuerpo del hilo: en_curso -> hecho | error (con traza). Nunca lanza."""
        try:
            self.almacen.marcar_estado(id_experimento, "en_curso")
            resultado = ejecutar_experimento(config, self.cargador_precios)
            self.almacen.guardar(id_experimento, resultado)
        except BaseException as error:
            logger.exception("Experimento %s falló", id_experimento)
            self.almacen.marcar_estado(id_experimento, "error", error)

    def lanzar(self, config: ConfiguracionExperimento, forzar: bool = False) -> Lanzamiento:
        """
        Registra y encola un experimento. Si ya existe uno con la misma huella terminado correctamente (o en
        curso) y no se fuerza, se reutiliza en vez de recalcularlo.
        """
        config = config.normalizada()
        id_experimento = config.identificador()
        if not forzar and self.almacen.existe(id_experimento):
            if self.almacen.leer(id_experimento)["estado"] in ESTADOS_REUTILIZABLES:
                return Lanzamiento(id_experimento, True)
        self.almacen.registrar(config)
        self.ejecutor.submit(self._ejecutar, id_experimento, config)
        return Lanzamiento(id_experimento, False)

    def lanzar_varios(self, configs: Sequence[ConfiguracionExperimento], forzar: bool = False) -> list[str]:
        """Lanza cada configuración (misma regla de reutilización) y devuelve sus ids."""
        return [self.lanzar(config, forzar).id for config in configs]

    def estado_de(self, id_experimento: str) -> str | None:
        """Estado del experimento (`pendiente|en_curso|hecho|error`); None si no existe. `ValueError` si el id es inválido."""
        return self.almacen.estado_de(id_experimento)

    def borrar(self, id_experimento: str) -> bool:
        """Borra el experimento; False si no existía. `ValueError` si el id es inválido."""
        return self.almacen.borrar(id_experimento)

    def borrar_varios(self, ids: Sequence[str]) -> dict[str, list[str]]:
        """
        Borra varios y dice qué pasó con cada id (sin duplicados, en orden): `borrados`, `en_marcha` (pendientes o en
        curso: no se tocan) y `no_encontrados` (incluye ids con formato inválido).
        """
        borrados, en_marcha, no_encontrados = [], [], []
        for id_experimento in dict.fromkeys(ids):
            try:
                estado = self.almacen.estado_de(id_experimento)
            except ValueError:
                estado = None
            if estado is None:
                no_encontrados.append(id_experimento)
            elif estado in ESTADOS_EN_MARCHA:
                en_marcha.append(id_experimento)
            elif self.almacen.borrar(id_experimento):
                borrados.append(id_experimento)
            else:
                no_encontrados.append(id_experimento)
        return {"borrados": borrados, "en_marcha": en_marcha, "no_encontrados": no_encontrados}

    # ---- barridos -------------------------------------------------------------------------

    def lanzar_barrido(self, config: ConfiguracionBarrido, forzar: bool = False) -> tuple[str, list[str], bool]:
        """
        Crea el barrido y encola todas sus combinaciones como experimentos. Devuelve `(id, ids, reutilizado)`.
        Si el barrido ya existía (misma configuración) se conserva con sus resultados y los experimentos se
        reutilizan; con `forzar` todo se recalcula. `BarridoEnCurso` si hay un barrido sin terminar.
        """
        id_barrido, creado = self.almacen_barridos.crear(config, forzar)
        ids = self.lanzar_varios([combinacion for combinacion, _, _ in config.combinaciones()], forzar)
        return id_barrido, ids, not creado

    # ---- tareas largas de un barrido --------------------------------------------------------

    @staticmethod
    def nombre_tarea_permutacion(modo: str) -> str:
        """Nombre de la tarea en `barrido.json` -> `tareas` (`permutacion_in_sample` | `permutacion_walk_forward`)."""
        return f"permutacion_{modo}"

    def lanzar_permutacion(self, id_barrido: str, parametros: dict[str, Any]) -> str:
        """
        Encola la prueba de permutación del barrido y devuelve el nombre de la tarea.

        `parametros` son los argumentos de `prueba_permutacion_barrido` (n_permutaciones, modo...).
        Lanza `RuntimeError` si ya hay una igual en curso.
        """
        nombre = self.nombre_tarea_permutacion(parametros["modo"])
        actual = self.almacen_barridos.tarea(id_barrido, nombre)
        if actual and actual.get("estado") in ESTADOS_EN_MARCHA:
            raise RuntimeError(f"Ya hay una {nombre} en curso para este barrido")
        self.almacen_barridos.actualizar_tarea(
            id_barrido, nombre, estado="pendiente", error=None, p_valor=None, parametros=parametros,
            progreso={"completadas": 0, "total": parametros["n_permutaciones"]})
        evento = threading.Event()
        self._cancelaciones[(id_barrido, nombre)] = evento
        self.ejecutor_barridos.submit(self._ejecutar_permutacion, id_barrido, nombre, parametros, evento)
        return nombre

    def cancelar_permutacion(self, id_barrido: str, nombre: str) -> bool:
        """Pide la cancelación (cooperativa) de la tarea en curso; False si no hay ninguna en marcha."""
        evento = self._cancelaciones.get((id_barrido, nombre))
        if evento is None:
            return False
        evento.set()
        return True

    def _ejecutar_permutacion(self, id_barrido: str, nombre: str, parametros: dict[str, Any],
                              evento: threading.Event) -> None:
        """Cuerpo del hilo de la permutación: en_curso -> hecho | cancelado | error, con progreso. Nunca lanza."""
        almacen = self.almacen_barridos
        try:
            almacen.actualizar_tarea(id_barrido, nombre, estado="en_curso")
            resultado = prueba_permutacion_barrido(
                almacen.cargar(id_barrido), self.cargador_precios, cancelar=evento,
                callback_progreso=lambda hechas, total: almacen.actualizar_tarea(
                    id_barrido, nombre, progreso={"completadas": hechas, "total": total}),
                **parametros)
            almacen.actualizar_tarea(id_barrido, nombre, estado="hecho", p_valor=resultado["p_valor"])
        except PruebaCancelada:
            almacen.actualizar_tarea(id_barrido, nombre, estado="cancelado", error=None)
        except BaseException as error:
            logger.exception("Permutación de %s falló", id_barrido)
            almacen.actualizar_tarea(id_barrido, nombre, estado="error", error=f"{type(error).__name__}: {error}")
        finally:
            self._cancelaciones.pop((id_barrido, nombre), None)

    # ---- configuraciones guardadas y marcos -----------------------------------------------

    def backtests_de_receta(self, configuracion: ConfiguracionGuardada) -> list[dict[str, Any]]:
        """Resúmenes de los experimentos cuya receta (sin fechas ni capital) es la de la configuración (índice, sin recorrer el disco)."""
        return self.almacen.resumenes_de_receta(configuracion.huella_estrategia)

    def marco_enlazado(self, configuracion: ConfiguracionGuardada) -> MarcoComparacion | None:
        """El marco al que apunta la configuración, o None si no tiene o se borró (la configuración sigue siendo válida)."""
        if not configuracion.id_marco:
            return None
        try:
            return self.almacen_marcos.leer(configuracion.id_marco)
        except (KeyError, ValueError):
            return None

    def lanzar_backtest_de_configuracion(
        self, configuracion: ConfiguracionGuardada, fecha_inicio: str | None = None, fecha_fin: str | None = None,
        capital_inicial: float | None = None, forzar: bool = False, hoy: date | None = None,
    ) -> Lanzamiento:
        """
        Lanza (o reutiliza) el backtest de la receta de `configuracion`. Fechas que falten: las de su marco enlazado
        según `marcos.resolver_fechas` (explícitas > marco enlazado > origen > 5 años hasta el último día hábil). Errores del dominio sin traducir (`KeyError` estrategia,
        `ValidationError` / `ValueError` parámetros o fechas inválidos).
        """
        inicio, fin = resolver_fechas(configuracion, self.marco_enlazado(configuracion), fecha_inicio, fecha_fin, hoy)
        return self.lanzar(configuracion.a_experimento(inicio, fin, capital_inicial), forzar)

    # ---- comparativas ---------------------------------------------------------------------

    def lanzar_comparativa(self, marco: MarcoComparacion, entradas: Sequence[Mapping[str, Any]],
                           forzar: bool = False) -> dict[str, Any]:
        """
        Aplica `marco` a cada entrada `{estrategia, parametros}` validándolas TODAS antes de lanzar nada (`ValueError`
        con los fallos de todas), lanza los experimentos normales (los ya calculados se reutilizan) y registra la
        comparativa con el informe de qué comunes se aplicaron a cada estrategia.
        """
        preparadas = preparar_comparativa(marco, entradas)
        lanzados = []
        for entrada, (config, informe) in zip(entradas, preparadas, strict=True):
            id_experimento, reutilizado = self.lanzar(config, forzar)
            lanzados.append({"id_experimento": id_experimento, "estrategia": entrada["estrategia"],
                             "parametros": entrada.get("parametros") or {}, "informe": informe.model_dump(),
                             "reutilizado": reutilizado})
        return self.almacen_marcos.registrar_comparativa(marco, lanzados)

    def lanzar_comparativa_de_configuracion(
        self, configuracion: ConfiguracionGuardada, entradas: Sequence[Mapping[str, Any]],
        fecha_inicio: str | None = None, fecha_fin: str | None = None, forzar: bool = False,
        hoy: date | None = None,
    ) -> dict[str, Any]:
        """
        Lanza el entorno de `configuracion` contra N estrategias: la convierte en un marco (id derivado de
        configuración + versión + fechas, idempotente: se crea o se reutiliza) y sigue el flujo de `lanzar_comparativa`.
        Fechas que falten: `marcos.resolver_fechas` (la misma regla que el backtest). La configuración no cambia.
        """
        inicio, fin = resolver_fechas(configuracion, self.marco_enlazado(configuracion), fecha_inicio, fecha_fin, hoy)
        marco = marco_desde_configuracion(configuracion, inicio, fin)
        try:
            self.almacen_marcos.guardar(marco)
        except MarcoExistente:
            marco = self.almacen_marcos.leer(marco.id)  # mismo id = mismo entorno: se reutiliza
        return self.lanzar_comparativa(marco, entradas, forzar)
