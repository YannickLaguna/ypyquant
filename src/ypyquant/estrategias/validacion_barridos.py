"""
Validación estadística de un barrido: ¿la mejor combinación es algo más que suerte?

Responsabilidad:
- `validar_barrido(barrido)`: elige la mejor combinación por la métrica objetivo y calcula sobre
  la matriz de retornos su Sharpe deflactado (DSR, con N = nº de combinaciones), la probabilidad
  de sobreajuste PBO-CSCV, la prueba de rachas, la superficie con meseta y la combinación de
  mejor meseta como alternativa. Guarda `validacion.json`.
- `walk_forward_barrido(barrido, ventana_entreno_dias, paso_dias)`: «¿qué habría obtenido quien
  re-optimiza la rejilla periódicamente?» (`seleccion_walk_forward`). Guarda `walk_forward.json`
  y `walk_forward_oos.parquet`.
- `prueba_permutacion_barrido(...)`: test de permutación de Masters sobre TODO el barrido (cada
  permutación de precios ejecuta todas las combinaciones y se queda con el mejor estadístico);
  los precios se cargan una vez y las permutaciones se reparten entre procesos. Guarda
  `permutacion_<modo>.json` y `.parquet` con la distribución.
- `BarridoEnCurso` (hay experimentos sin terminar) y `PruebaCancelada`.

Decisiones:
- NUNCA se toca el periodo reservado: las validaciones usan solo los experimentos del barrido
  (hasta `fecha_fin_desarrollo`) y la permutación carga precios hasta esa fecha.
- `ventana_entreno_dias` y `paso_dias` son DÍAS NATURALES (como `dias_historia_previa`); sobre la
  matriz de sesiones se convierten a filas con la densidad real (filas por día natural, ~0,69).
- Permutación `in_sample`: estadístico = mejor métrica objetivo entre las combinaciones; se
  permutan todos los precios cargados (incluida la historia previa). Permutación `walk_forward`:
  estadístico = Sharpe por periodo de la serie OOS del walk-forward sobre la matriz permutada; los
  precios se permutan solo desde `fecha_inicio + ventana_entreno_dias` (el tramo de entrenamiento
  inicial queda real) y la selección walk-forward siempre usa el Sharpe.
- Si una combinación falla sobre unos precios permutados se omite en esa permutación; si fallan
  todas, la permutación se descarta (se informa `n_validas`).
- Límites: n_permutaciones <= 1000 y aviso si combinaciones x permutaciones > 20 000 backtests
  (a ~2 s cada uno, más de 11 horas en serie).
- Procesos: Windows usa `spawn`; los workers reciben los precios una vez (initializer), importan
  las estrategias incluidas y redescubren `estrategias_usuario/` (`cargar_estrategias`). Las
  estrategias registradas en caliente solo en el proceso padre (p. ej. en tests) no existen en
  los workers: usa `procesos=1`. Con varios procesos se limitan los hilos de BLAS a 1 por worker.

Se relaciona con:
- ypyquant.estrategias.barridos (matriz, superficie, almacén) y .experimentos
  (`cargar_datos_experimento`, `ejecutar_con_datos`).
- ypyquant.validacion (sharpe, DSR, PBO, rachas, permutar_precios, seleccion_walk_forward).
- app/api/rutas_barridos.py (los expone; la permutación corre en segundo plano con progreso).
- docs/validacion.md (fórmulas e interpretación).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from ypyquant.estrategias.barridos import (
    Barrido,
    alinear_retornos,
    matriz_retornos,
    serie_retornos,
    superficie_registros,
)
from ypyquant.estrategias.experimentos import (
    DIAS_ANIO,
    ConfiguracionExperimento,
    DatosExperimento,
    _limpiar,
    cargar_datos_experimento,
    ejecutar_con_datos,
    resolver_universo,
)
from ypyquant.validacion import (
    p_valor_permutacion,
    pbo_cscv,
    permutar_precios,
    prueba_rachas,
    seleccion_walk_forward,
    sharpe,
    sharpe_deflactado,
)

logger = logging.getLogger(__name__)

MAX_PERMUTACIONES = 1000
AVISO_BACKTESTS = 20_000
BLOQUES_PBO_POR_DEFECTO = 16
FILAS_MINIMAS_BLOQUE = 10
VARIABLES_HILOS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

ModoPermutacion = Literal["in_sample", "walk_forward"]


class BarridoEnCurso(RuntimeError):
    """El barrido tiene experimentos pendientes o en curso: no se puede validar todavía."""


class PruebaCancelada(RuntimeError):
    """La prueba de permutación se canceló de forma cooperativa."""


def exigir_terminado(barrido: Barrido, minimo_combinaciones: int = 1) -> None:
    """Lanza `BarridoEnCurso` si queda algo sin terminar y `ValueError` si hay menos de `minimo_combinaciones` bien."""
    estado = barrido.almacen.estado_experimentos(barrido)
    if estado["estado"] in ("pendiente", "en_curso"):
        p = estado["progreso"]
        raise BarridoEnCurso(f"El barrido aún tiene experimentos sin terminar ({p['hechos']}/{p['total']} hechos)")
    hechos = estado["progreso"]["hechos"]
    if hechos < minimo_combinaciones:
        raise ValueError(f"Hacen falta al menos {minimo_combinaciones} experimentos terminados y hay {hechos}")


def _dias_a_filas(indice: pd.DatetimeIndex, dias: int) -> int:
    """Días naturales -> filas de la matriz, con la densidad real de sesiones (filas por día natural)."""
    if dias <= 0:
        raise ValueError("Los días deben ser positivos")
    dias_totales = max((indice[-1] - indice[0]).days, 1)
    return max(1, round(dias * (len(indice) - 1) / dias_totales))


def _combinacion(barrido: Barrido, id_experimento: str) -> dict[str, Any]:
    return next(c for c in barrido.combinaciones if c["id"] == id_experimento)


# ----------------------------------------------------------------------------
# Validación del barrido (DSR, PBO, rachas, superficie)
# ----------------------------------------------------------------------------


def _bloques_pbo(n_filas: int, n_bloques: int) -> tuple[int | None, str | None]:
    """Bloques de CSCV: los pedidos, reducidos (pares) hasta que cada bloque tenga >= 10 filas; None si < 4."""
    pedidos = n_bloques - n_bloques % 2
    bloques = pedidos
    while bloques >= 4 and n_filas // bloques < FILAS_MINIMAS_BLOQUE:
        bloques -= 2
    if bloques < 4:
        return None, f"Solo hay {n_filas} observaciones: no bastan para el PBO (mínimo ~{4 * FILAS_MINIMAS_BLOQUE})"
    if bloques != n_bloques:
        return bloques, f"PBO con {bloques} bloques en lugar de {n_bloques} (T={n_filas}, bloques pares de >= {FILAS_MINIMAS_BLOQUE} filas)"
    return bloques, None


def validar_barrido(barrido: Barrido, n_bloques: int = BLOQUES_PBO_POR_DEFECTO, semilla: int = 0,
                    relleno: Literal["recortar", "cero"] = "recortar") -> dict[str, Any]:
    """
    Valida el barrido y guarda `validacion.json`.

    Elegida = mejor combinación por la métrica objetivo. Con la matriz de retornos (N columnas):
    DSR de la elegida con N = nº de combinaciones, PBO-CSCV (`n_bloques`; se reduce y se avisa si T
    es pequeño), prueba de rachas sobre los signos de los retornos diarios de la elegida y la
    superficie con la meseta; la combinación de mejor meseta se sugiere como alternativa más robusta.

    Returns:
        dict (ver docs/ui.md): `elegida`, `dsr`, `pbo` (None si T es pequeño), `rachas`, `superficie`,
        `alternativa_meseta`, `avisos`, `n_combinaciones`, `n_observaciones`...

    Raises:
        BarridoEnCurso: hay experimentos sin terminar. ValueError: menos de 2 combinaciones terminadas.
    """
    exigir_terminado(barrido, minimo_combinaciones=2)
    matriz = matriz_retornos(barrido, relleno)
    avisos: list[str] = []
    n_pedidas = len(barrido.ids)
    if matriz.shape[1] < n_pedidas:
        avisos.append(f"Solo {matriz.shape[1]} de {n_pedidas} combinaciones terminaron bien; N del DSR = {matriz.shape[1]}")
    n_combinaciones = matriz.shape[1]
    filas = superficie_registros(barrido)
    candidatas = [f for f in filas if f["objetivo"] is not None and f["id"] in matriz.columns]
    if not candidatas:
        raise ValueError(f"Ninguna combinación tiene valor de '{barrido.config.metrica_objetivo}'")
    elegida = max(candidatas, key=lambda f: f["objetivo"])
    retornos_elegida = matriz[elegida["id"]]

    dsr = sharpe_deflactado(retornos_elegida, matriz_retornos=matriz, n_pruebas=n_combinaciones)
    dsr["sr_anualizado"] = dsr["sr"] * math.sqrt(DIAS_ANIO)

    bloques, aviso_pbo = _bloques_pbo(len(matriz), n_bloques)
    if aviso_pbo:
        avisos.append(aviso_pbo)
    pbo = None
    if bloques is not None:
        pbo = pbo_cscv(matriz, n_bloques=bloques, semilla=semilla)
        pbo["lambdas"] = [float(x) for x in pbo["lambdas"]]
        pbo["pares_is_oos"] = [[float(a), float(b)] for a, b in pbo["pares_is_oos"]]

    rachas = prueba_rachas(np.sign(retornos_elegida.to_numpy()))
    con_meseta = [f for f in filas if f["meseta"] is not None]
    mejor_meseta = max(con_meseta, key=lambda f: f["meseta"]) if con_meseta else None
    alternativa = None
    if mejor_meseta is not None:
        alternativa = {"id": mejor_meseta["id"], "etiqueta": mejor_meseta["etiqueta"], "valores": mejor_meseta["valores"],
                       "meseta": mejor_meseta["meseta"], "objetivo": mejor_meseta["objetivo"],
                       "es_la_elegida": mejor_meseta["id"] == elegida["id"]}
    resultado = _limpiar({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "metrica_objetivo": barrido.config.metrica_objetivo,
        "n_combinaciones": n_combinaciones, "n_observaciones": len(matriz),
        "fecha_inicio": matriz.index[0].strftime("%Y-%m-%d"), "fecha_fin": matriz.index[-1].strftime("%Y-%m-%d"),
        "relleno": relleno,
        "elegida": {k: elegida[k] for k in ("id", "etiqueta", "valores", "objetivo", "meseta", "n_vecinos")}
                   | {"metricas": elegida["metricas"]},
        "dsr": dsr, "pbo": pbo, "rachas": rachas,
        "superficie": filas, "alternativa_meseta": alternativa, "avisos": avisos,
    })
    barrido.almacen.guardar_resultado(barrido.id, "validacion", resultado)
    return resultado


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------


def _metricas_serie(retornos: pd.Series) -> dict[str, Any]:
    """Sharpe anualizado, CAGR, máximo drawdown y retorno total de una serie de retornos diarios."""
    r = retornos.dropna()
    if len(r) < 2:
        return {"sharpe_anualizado": None, "cagr": None, "max_drawdown": None, "retorno_total": None, "n_observaciones": len(r)}
    equity = (1.0 + r).cumprod()
    return {
        "sharpe_anualizado": sharpe(r.to_numpy(), anualizar=True),
        "cagr": float(equity.iloc[-1] ** (DIAS_ANIO / len(r)) - 1.0),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "retorno_total": float(equity.iloc[-1] - 1.0), "n_observaciones": len(r),
    }


def walk_forward_barrido(barrido: Barrido, ventana_entreno_dias: int, paso_dias: int,
                         relleno: Literal["recortar", "cero"] = "recortar") -> dict[str, Any]:
    """
    Selección walk-forward sobre la matriz de retornos del barrido; guarda `walk_forward.json` y Parquet.

    En cada reoptimización (cada `paso_dias` tras `ventana_entreno_dias`, días naturales convertidos
    a filas) se elige la combinación de mejor Sharpe en la ventana anterior y se usan sus retornos
    hacia delante. Devuelve la serie OOS (`curva`), las `elecciones` (fecha, id, etiqueta, Sharpe
    de entreno) y las métricas de la serie. Aproximación: ignora el coste de cambiar de
    combinación (ver docs/validacion.md).
    """
    exigir_terminado(barrido, minimo_combinaciones=2)
    matriz = matriz_retornos(barrido, relleno)
    ventana = _dias_a_filas(matriz.index, ventana_entreno_dias)
    paso = _dias_a_filas(matriz.index, paso_dias)
    if len(matriz) <= ventana:
        raise ValueError(f"La ventana de entrenamiento ({ventana_entreno_dias} días ≈ {ventana} sesiones) "
                         f"no cabe en las {len(matriz)} sesiones del barrido")
    serie, elecciones, valor = seleccion_walk_forward(matriz, ventana, paso)
    etiquetas = barrido.etiquetas
    serie = serie.dropna()
    equity = (1.0 + serie).cumprod()
    resultado = _limpiar({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "ventana_entreno_dias": ventana_entreno_dias, "paso_dias": paso_dias,
        "ventana_filas": ventana, "paso_filas": paso, "n_combinaciones": matriz.shape[1],
        "sharpe_periodo": valor, "metricas": _metricas_serie(serie),
        "elecciones": [{"fecha": fecha.strftime("%Y-%m-%d"), "id": fila["columna"],
                        "etiqueta": etiquetas.get(fila["columna"]), "metrica_entreno": fila["metrica_entreno"]}
                       for fecha, fila in elecciones.iterrows()],
        "curva": {"fechas": [f.strftime("%Y-%m-%d") for f in equity.index], "equity": equity.tolist()},
    })
    barrido.almacen.guardar_resultado(barrido.id, "walk_forward", resultado)
    barrido.almacen.guardar_parquet(barrido.id, "walk_forward_oos", pd.DataFrame(
        {"fecha": serie.index, "retorno": serie.to_numpy(), "equity": equity.to_numpy()}))
    return resultado


# ----------------------------------------------------------------------------
# Prueba de permutación del barrido
# ----------------------------------------------------------------------------


@dataclass
class ContextoPermutacion:
    """Todo lo que necesita una permutación (picklable: viaja una vez a cada worker)."""

    datos: DatosExperimento
    configs: list[ConfiguracionExperimento]
    modo: str
    metrica_objetivo: str
    longitud_bloque: int
    semilla: int
    inicio_permutacion: str | None
    ventana_filas: int | None
    paso_filas: int | None
    relleno: str


_CONTEXTO_WORKER: ContextoPermutacion | None = None


def _iniciar_worker(contexto: ContextoPermutacion) -> None:
    """Initializer de cada proceso: guarda el contexto y registra estrategias incluidas y de usuario."""
    global _CONTEXTO_WORKER
    from ypyquant.estrategias.registro import cargar_estrategias

    cargar_estrategias()  # importa `incluidas` y redescubre estrategias_usuario/ (spawn parte de cero)
    _CONTEXTO_WORKER = contexto


def _estadistico_permutacion(contexto: ContextoPermutacion, p: int) -> float:
    """
    Estadístico de la permutación `p`: permuta los precios (semilla + p), ejecuta todas las
    combinaciones y devuelve la mejor métrica objetivo (`in_sample`) o el Sharpe por periodo del
    walk-forward (`walk_forward`). NaN si ninguna combinación pudo ejecutarse.
    """
    permutados = permutar_precios(contexto.datos.precios, contexto.semilla + p, inicio=contexto.inicio_permutacion,
                                  longitud_bloque=contexto.longitud_bloque)
    datos = DatosExperimento(permutados, contexto.datos.benchmark, contexto.datos.ticker_benchmark, [])
    metricas: list[float] = []
    series: dict[str, pd.Series] = {}
    for config in contexto.configs:
        try:
            resultado = ejecutar_con_datos(config, datos, completo=False)
        except Exception as error:  # noqa: BLE001 - una combinación rota sobre precios permutados no aborta la prueba
            logger.debug("Combinación %s falló en la permutación %d: %s", config.identificador(), p, error)
            continue
        valor = resultado.metricas.get(contexto.metrica_objetivo)
        if valor is not None and math.isfinite(valor):
            metricas.append(valor)
        if contexto.modo == "walk_forward":
            series[config.identificador()] = serie_retornos(resultado.equity)
    if contexto.modo == "in_sample":
        return max(metricas) if metricas else float("nan")
    if len(series) < 1:
        return float("nan")
    matriz = alinear_retornos(series, contexto.relleno)  # type: ignore[arg-type]
    if len(matriz) <= contexto.ventana_filas:
        return float("nan")
    return float(seleccion_walk_forward(matriz, contexto.ventana_filas, contexto.paso_filas)[2])


def _tarea_worker(p: int) -> tuple[int, float]:
    """Cuerpo ejecutado en el proceso hijo (función de módulo: picklable con spawn)."""
    return p, _estadistico_permutacion(_CONTEXTO_WORKER, p)


@contextmanager
def _un_hilo_por_proceso(activo: bool) -> Iterator[None]:
    """Con varios procesos fija 1 hilo de BLAS/OpenMP (los hijos lo heredan al crearse) y restaura el entorno."""
    guardado = {v: os.environ.get(v) for v in VARIABLES_HILOS}
    try:
        if activo:
            for variable in VARIABLES_HILOS:
                os.environ[variable] = "1"
        yield
    finally:
        for variable, valor in guardado.items():
            if valor is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = valor


def _cargar_datos_barrido(configs: list[ConfiguracionExperimento], cargador_precios) -> DatosExperimento:
    """Carga UNA vez los precios que cubren todas las combinaciones (unión de universos, inicio más temprano)."""
    tickers = list(dict.fromkeys(t for c in configs for t in resolver_universo(c.universo)))
    inicio_carga = min(pd.Timestamp(c.fecha_inicio) - pd.Timedelta(days=c.dias_historia_previa) for c in configs)
    primero = min(configs, key=lambda c: c.fecha_inicio)
    sintetica = primero.model_copy(update={
        "universo": tickers, "fecha_fin": max(c.fecha_fin for c in configs),
        "dias_historia_previa": (pd.Timestamp(primero.fecha_inicio) - inicio_carga).days,
    })
    return cargar_datos_experimento(sintetica, cargador_precios)


def validar_parametros_permutacion(n_permutaciones: int, longitud_bloque: int, modo: str,
                                   ventana_entreno_dias: int | None, paso_dias: int | None, procesos: int) -> None:
    """Valida los parámetros de la prueba (ValueError con mensaje claro); la API lo llama antes de encolar."""
    if not 1 <= n_permutaciones <= MAX_PERMUTACIONES:
        raise ValueError(f"n_permutaciones debe estar entre 1 y {MAX_PERMUTACIONES}")
    if modo not in ("in_sample", "walk_forward"):
        raise ValueError("modo debe ser 'in_sample' o 'walk_forward'")
    if longitud_bloque < 1 or procesos < 1:
        raise ValueError("longitud_bloque y procesos deben ser >= 1")
    if modo == "walk_forward" and (not ventana_entreno_dias or not paso_dias):
        raise ValueError("El modo walk_forward necesita ventana_entreno_dias y paso_dias")


def prueba_permutacion_barrido(
    barrido: Barrido,
    cargador_precios,
    n_permutaciones: int,
    longitud_bloque: int = 1,
    modo: ModoPermutacion = "in_sample",
    ventana_entreno_dias: int | None = None,
    paso_dias: int | None = None,
    semilla: int = 0,
    procesos: int = 1,
    callback_progreso: Callable[[int, int], None] | None = None,
    cancelar: threading.Event | None = None,
    relleno: Literal["recortar", "cero"] = "recortar",
) -> dict[str, Any]:
    """
    Test de permutación de Masters sobre el barrido completo; guarda `permutacion_<modo>.json` y Parquet.

    Carga los precios una vez (hasta `fecha_fin_desarrollo`: el reservado no se toca). Por permutación
    p: `permutar_precios(datos, semilla + p, inicio, longitud_bloque)` (`in_sample`: todo lo cargado;
    `walk_forward`: desde fecha_inicio + `ventana_entreno_dias`), ejecuta TODAS las combinaciones
    (el benchmark no se permuta) y calcula el estadístico (ver cabecera del módulo). El valor real
    sale de los experimentos guardados y `p_valor_permutacion` = (1 + #{perm >= real}) / (1 + n).

    Args:
        cargador_precios: contrato `CargadorPrecios` (los precios no se guardan con el barrido).
        procesos: 1 = en serie en este proceso; > 1 = `ProcessPoolExecutor` (spawn).
        callback_progreso: `(completadas, total)`, llamado tras cada permutación.
        cancelar: `threading.Event`; si se activa se aborta con `PruebaCancelada` (a granularidad de una permutación).

    Raises:
        ValueError: parámetros inválidos (n_permutaciones fuera de 1..1000, modo, ventana/paso).
        BarridoEnCurso, PruebaCancelada.
    """
    validar_parametros_permutacion(n_permutaciones, longitud_bloque, modo, ventana_entreno_dias, paso_dias, procesos)
    exigir_terminado(barrido, minimo_combinaciones=2 if modo == "walk_forward" else 1)
    inicio = time.perf_counter()
    objetivo = barrido.config.metrica_objetivo
    configs = [c for c, _, _ in barrido.config.combinaciones()]
    n_backtests = len(configs) * n_permutaciones
    avisos: list[str] = []
    if n_backtests > AVISO_BACKTESTS:
        avisos.append(f"{n_backtests} backtests (combinaciones x permutaciones): puede tardar horas; reduce permutaciones o rejilla")
        logger.warning(avisos[-1])

    ventana_filas = paso_filas = inicio_permutacion = None
    matriz_real = matriz_retornos(barrido, relleno) if modo == "walk_forward" else None
    if modo == "walk_forward":
        ventana_filas = _dias_a_filas(matriz_real.index, ventana_entreno_dias)
        paso_filas = _dias_a_filas(matriz_real.index, paso_dias)
        if len(matriz_real) <= ventana_filas:
            raise ValueError("La ventana de entrenamiento no cabe en el periodo del barrido")
        fecha_inicio = min(pd.Timestamp(c.fecha_inicio) for c in configs)
        inicio_permutacion = (fecha_inicio + pd.Timedelta(days=ventana_entreno_dias)).strftime("%Y-%m-%d")
        valor_real = float(seleccion_walk_forward(matriz_real, ventana_filas, paso_filas)[2])
        nombre_estadistico = "sharpe_periodo_walk_forward"
    else:
        reales = [m[objetivo] for m in (barrido.almacen.experimentos.leer_metricas(i) for i in barrido.experimentos_hechos())
                  if m and m.get(objetivo) is not None]
        if not reales:
            raise ValueError(f"Ninguna combinación tiene valor de '{objetivo}'")
        valor_real = float(max(reales))
        nombre_estadistico = f"mejor_{objetivo}"
    if not math.isfinite(valor_real):
        raise ValueError("El estadístico real no está definido (¿serie constante o demasiado corta?)")

    datos = _cargar_datos_barrido(configs, cargador_precios)
    contexto = ContextoPermutacion(datos, configs, modo, objetivo, longitud_bloque, semilla, inicio_permutacion,
                                   ventana_filas, paso_filas, relleno)
    valores: dict[int, float] = {}

    def registrar(p: int, valor: float) -> None:
        valores[p] = valor
        if callback_progreso:
            callback_progreso(len(valores), n_permutaciones)

    if procesos == 1:
        for p in range(n_permutaciones):
            if cancelar is not None and cancelar.is_set():
                raise PruebaCancelada("Prueba de permutación cancelada")
            registrar(p, _estadistico_permutacion(contexto, p))
    else:
        with _un_hilo_por_proceso(True), ProcessPoolExecutor(
                max_workers=procesos, initializer=_iniciar_worker, initargs=(contexto,)) as pool:
            futuros = [pool.submit(_tarea_worker, p) for p in range(n_permutaciones)]
            try:
                for futuro in as_completed(futuros):
                    if cancelar is not None and cancelar.is_set():
                        raise PruebaCancelada("Prueba de permutación cancelada")
                    registrar(*futuro.result())
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    distribucion = np.array([valores[p] for p in range(n_permutaciones)], dtype=float)
    validas = distribucion[np.isfinite(distribucion)]
    if len(validas) < n_permutaciones:
        avisos.append(f"{n_permutaciones - len(validas)} permutaciones sin estadístico (todas las combinaciones fallaron) se descartan")
    if len(validas) == 0:
        raise RuntimeError("Ninguna permutación produjo un estadístico")
    resultado = _limpiar({
        "fecha": datetime.now().isoformat(timespec="seconds"), "modo": modo, "estadistico": nombre_estadistico,
        "metrica_objetivo": objetivo, "n_permutaciones": n_permutaciones, "n_validas": len(validas),
        "longitud_bloque": longitud_bloque, "semilla": semilla, "procesos": procesos,
        "ventana_entreno_dias": ventana_entreno_dias, "paso_dias": paso_dias, "inicio_permutacion": inicio_permutacion,
        "n_combinaciones": len(configs), "n_backtests": n_backtests, "valor_real": valor_real,
        "p_valor": p_valor_permutacion(valor_real, validas), "distribucion": validas.tolist(),
        "media_permutadas": float(validas.mean()), "p95_permutadas": float(np.percentile(validas, 95)),
        "duracion_s": round(time.perf_counter() - inicio, 2), "avisos": avisos,
    })
    barrido.almacen.guardar_resultado(barrido.id, f"permutacion_{modo}", resultado)
    barrido.almacen.guardar_parquet(barrido.id, f"permutacion_{modo}", pd.DataFrame({"estadistico": validas}))
    return resultado
