"""
Importación de los resultados del TFM original (rejilla_v2) al almacén de experimentos de la interfaz.

Responsabilidad: convertir los Parquet del TFM (`resultados_oos`, `metricas_por_celda`, `rf_mensual`,
`pesos_oos`) en experimentos «de solo lectura» con el mismo formato que los backtests normales
(`config.json`, `estado.json`, `metricas.json`, `procedencia.json`, `equity.parquet`, `pesos.parquet`,
`operaciones.parquet`), para verlos en Experimentos y compararlos en Comparar con iteraciones
posteriores. NO recalcula nada: la réplica fiel son los números originales del TFM.

Decisiones de diseño (todas quedan reflejadas en los `avisos` del experimento):
- Serie: por defecto la media de `r_neto` sobre las réplicas de cada fecha (la que da
  `sharpe_serie_promediada`); con `replica=b`, la réplica b. Es una rejilla CELDA (N, T) x estrategia.
- Fechas: el TFM etiqueta cada mes por su primer día (`2010-01-01` = retorno de enero). Aquí la fecha
  de cada punto es el último día de negociación del mes (el de SPY si se puede cargar, si no el último
  día hábil), de modo que la serie mensual coincida con los cierres de mes de los backtests diarios y
  `GET /api/comparar` (que intersecta fechas) tenga puntos comunes. Se antepone un punto base (cierre
  del mes anterior) con el capital inicial.
- Métricas: las mismas funciones que los experimentos normales (`PerformanceAnalyzer` con rf = 0 y 12
  periodos/año, `indice_ulcer`, `ratio_martin`) sobre la serie mensual importada; el alfa/beta se
  calculan aquí con periodicidad mensual porque `compare_to_benchmark` anualiza con 252. Las métricas
  originales del TFM se guardan tal cual con el prefijo `tfm_`.
- Id estable y legible: `tfm-<estrategia>-n<N>-t<T>-media` (o `-b<b>`); la estrategia del config es
  `tfm_<estrategia>`, que NO está en el registro (sin código, no se puede relanzar).

Se relaciona con:
- scripts/importar_tfm.py (CLI que llama a `importar_experimentos_tfm`).
- ypyquant.estrategias.almacen (`AlmacenExperimentos`: escribe y lee los experimentos),
  .experimentos (`ConfiguracionExperimento`, `ResultadoExperimento`), .procedencia (commit del repo)
  y .precios (cargador del benchmark SPY).
- ypyquant.portfolio.performance_metrics y ypyquant.validacion (funciones de métricas reutilizadas).
- tests/test_importar_tfm.py y la interfaz (insignia «TFM» según el prefijo `tfm_` de la estrategia).
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ypyquant.data.matriz_precios import CargadorPrecios, extraer_serie_precios
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, ResultadoExperimento
from ypyquant.estrategias.procedencia import commit_git
from ypyquant.portfolio.performance_metrics import PerformanceAnalyzer
from ypyquant.validacion import indice_ulcer, ratio_martin

PERIODOS_ANIO = 12
CAPITAL_INICIAL = 100_000.0
PREFIJO_ESTRATEGIA = "tfm_"
CELDAS_POR_DEFECTO: tuple[tuple[int, int], ...] = ((20, 12), (80, 36), (150, 120))
ESTRATEGIAS_TFM: tuple[str, ...] = (
    "mpt_muestral", "mpt_bayes_stein", "mpt_michaud", "bl_prior", "bl_momentum", "igual_ponderacion")
COSTE_TFM = 0.001  # 10 pb sobre el turnover, ex post
COLUMNAS_PESOS = ["fecha", "ticker", "peso"]
ARCHIVOS_BASE = ("resultados_oos.parquet", "metricas_por_celda.parquet", "rf_mensual.parquet")
ARCHIVO_PESOS = "pesos_oos.parquet"
# Métricas de `metricas_por_celda` que se copian tal cual (con prefijo `tfm_`) para trazabilidad
METRICAS_TFM = (
    "sharpe_media_replicas", "sharpe_serie_promediada", "sortino", "volatilidad", "retorno_anual", "max_drawdown",
    "info_ratio", "sharpe_bruto_media_replicas", "sharpe_bruto_serie_promediada", "sortino_bruto",
    "volatilidad_bruto", "retorno_anual_bruto", "max_drawdown_bruto", "info_ratio_bruto",
)

AVISO_SESGOS = (
    "Resultados del TFM original (rejilla_v2), importados y no recalculados: universo con sesgo de "
    "look-ahead/supervivencia (filtro de cobertura sobre la ventana OOS); costes 10 pb ex post; motor "
    "distinto del de los backtests de la interfaz."
)


class ErrorImportacionTfm(ValueError):
    """Datos del TFM ausentes o incoherentes (celda inexistente, archivo que falta, réplica sin pesos...)."""


# ----------------------------------------------------------------------------
# Identificadores y configuración
# ----------------------------------------------------------------------------


def identificador_tfm(estrategia: str, n: int, t: int, replica: int | None = None) -> str:
    """Id estable y legible: `tfm-mpt_bayes_stein-n80-t36-media` (o `...-b3` con una réplica concreta)."""
    return f"tfm-{estrategia}-n{n}-t{t}-{'media' if replica is None else f'b{replica}'}"


def _etiqueta(n: int, t: int, replica: int | None, n_replicas: int) -> str:
    serie = f"media {n_replicas} réplicas" if replica is None else f"réplica {replica}"
    return f"TFM original (rejilla_v2) N={n} T={t} · {serie}"


def _configuracion(estrategia: str, n: int, t: int, replica: int | None, n_replicas: int,
                   fechas: pd.DatetimeIndex) -> ConfiguracionExperimento:
    """Configuración descriptiva del experimento importado (no reproduce el cálculo: es solo ficha)."""
    inicio = fechas.min().to_period("M").start_time
    fin = fechas.max().to_period("M").end_time.normalize()
    return ConfiguracionExperimento(
        estrategia=f"{PREFIJO_ESTRATEGIA}{estrategia}",
        parametros={"N": n, "T": t, "replica": f"media_{n_replicas}" if replica is None else replica},
        universo=f"tfm:N={n}", fecha_inicio=inicio.strftime("%Y-%m-%d"), fecha_fin=fin.strftime("%Y-%m-%d"),
        capital_inicial=CAPITAL_INICIAL, comision=COSTE_TFM, slippage=0.0, frecuencia_rebalanceo="monthly",
        benchmark="SPY", etiqueta=_etiqueta(n, t, replica, n_replicas),
    )


# ----------------------------------------------------------------------------
# Lectura de los Parquet del TFM
# ----------------------------------------------------------------------------


def _sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def _commit_repo_tfm(carpeta: Path) -> str | None:
    """Commit HEAD del repo git que contiene `carpeta` (None si no es un repo o git no responde)."""
    try:
        salida = subprocess.run(["git", "-C", str(carpeta), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return salida.stdout.strip() if salida.returncode == 0 and salida.stdout.strip() else None


def celdas_disponibles(carpeta_tfm: Path) -> list[tuple[int, int]]:
    """Todas las celdas (N, T) presentes en `metricas_por_celda.parquet`, ordenadas."""
    metricas = pd.read_parquet(carpeta_tfm / "metricas_por_celda.parquet")
    claves = metricas.index.droplevel("estrategia").unique() if isinstance(metricas.index, pd.MultiIndex) else \
        pd.MultiIndex.from_frame(metricas[["N", "T"]]).unique()
    return sorted((int(n), int(t)) for n, t in claves)


def _leer_metricas_celda(carpeta_tfm: Path) -> pd.DataFrame:
    """`metricas_por_celda` con (N, T, estrategia) como índice, venga con índice o con columnas."""
    metricas = pd.read_parquet(carpeta_tfm / "metricas_por_celda.parquet")
    if not isinstance(metricas.index, pd.MultiIndex):
        metricas = metricas.set_index(["N", "T", "estrategia"])
    return metricas


def _leer_resultados(carpeta_tfm: Path, celdas: list[tuple[int, int]], replica: int | None) -> pd.DataFrame:
    """`resultados_oos` solo con las columnas y las celdas (N, T) pedidas (filtro en lectura)."""
    columnas = ["b", "N", "T", "estrategia", "fecha", "r_bruto", "turnover", "tc", "r_neto"]
    filtros: list[Any] = [("N", "in", sorted({n for n, _ in celdas}))]
    if replica is not None:
        filtros.append(("b", "==", replica))
    datos = pd.read_parquet(carpeta_tfm / "resultados_oos.parquet", columns=columnas, filters=filtros)
    pares = {(n, t) for n, t in celdas}
    return datos[[(int(n), int(t)) in pares for n, t in zip(datos["N"], datos["T"], strict=True)]]


def _pesos_replica(carpeta_tfm: Path, n: int, t: int, estrategia: str, replica: int) -> pd.DataFrame:
    """Pesos de `pesos_oos` (solo existen para las réplicas b < 20) en formato `fecha, ticker, peso`."""
    pesos = pd.read_parquet(
        carpeta_tfm / ARCHIVO_PESOS, columns=["fecha", "ticker", "w"],
        filters=[("b", "==", replica), ("N", "==", n), ("T", "==", t), ("estrategia", "==", estrategia)])
    if pesos.empty:
        raise ErrorImportacionTfm(f"No hay pesos para la réplica {replica} (solo existen para b < 20): {estrategia} N={n} T={t}")
    return pesos.rename(columns={"w": "peso"}).sort_values(["fecha", "ticker"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Serie, calendario y benchmark
# ----------------------------------------------------------------------------


def serie_de_celda(resultados: pd.DataFrame, n: int, t: int, estrategia: str, replica: int | None) -> pd.DataFrame:
    """
    Serie mensual de una celda: índice = fecha del TFM (primer día del mes) y columnas
    `r_bruto, r_neto, turnover, tc`. Sin `replica`, media de las réplicas por fecha; con ella, esa réplica.
    """
    celda = resultados[(resultados["N"] == n) & (resultados["T"] == t) & (resultados["estrategia"] == estrategia)]
    if replica is not None:
        celda = celda[celda["b"] == replica]
    if celda.empty:
        raise ErrorImportacionTfm(f"Celda sin datos: {estrategia} N={n} T={t}" + (f" b={replica}" if replica is not None else ""))
    return celda.groupby("fecha")[["r_bruto", "r_neto", "turnover", "tc"]].mean().sort_index()


def _cierres_de_mes(etiquetas: pd.DatetimeIndex, precios_benchmark: pd.Series | None) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    """
    Fecha de cierre de cada mes etiquetado (último día de negociación) y la del mes anterior al primero.

    Con precios del benchmark se usa su último día de negociación de cada mes; sin ellos, el último día hábil.
    """
    meses = etiquetas.to_period("M")
    todos = meses.insert(0, meses[0] - 1)
    if precios_benchmark is not None:
        ultimo_por_mes = precios_benchmark.index.to_series().groupby(precios_benchmark.index.to_period("M")).max()
        return pd.DatetimeIndex([ultimo_por_mes[m] for m in todos[1:]]), pd.Timestamp(ultimo_por_mes[todos[0]])
    cierres = pd.DatetimeIndex([pd.offsets.BDay().rollback(m.end_time.normalize()) for m in todos])
    return cierres[1:], cierres[0]


def _precios_benchmark(cargador: CargadorPrecios | None, ticker: str, etiquetas: pd.DatetimeIndex) -> tuple[pd.Series | None, str | None]:
    """
    Precios ajustados diarios del benchmark que cubran todos los meses necesarios (+ el mes previo).
    Devuelve `(serie, motivo)`: serie None y el motivo si no se puede cargar o no cubre el periodo.
    """
    if cargador is None:
        return None, "benchmark no solicitado"
    primero = etiquetas.min().to_period("M") - 1
    ultimo = etiquetas.max().to_period("M")
    try:
        crudo = cargador(ticker, primero.start_time.strftime("%Y-%m-%d"), ultimo.end_time.strftime("%Y-%m-%d"))
    except Exception as error:  # noqa: BLE001 - red caída, caché corrupta...: la importación sigue sin benchmark
        return None, f"error al cargar {ticker}: {type(error).__name__}: {error}"
    serie = extraer_serie_precios(crudo, ticker, avisar=False) if crudo is not None else None
    if serie is None or serie.dropna().empty:
        return None, f"sin precios de {ticker}"
    serie = serie.dropna().sort_index()
    cubiertos = set(serie.index.to_period("M"))
    faltan = [m for m in pd.period_range(primero, ultimo, freq="M") if m not in cubiertos]
    if faltan:
        return None, f"los precios de {ticker} no cubren todo el periodo (faltan {faltan[0]} ... {faltan[-1]})"
    return serie, None


def construir_equity(serie: pd.DataFrame, cierres: pd.DatetimeIndex, base: pd.Timestamp,
                     precios_benchmark: pd.Series | None) -> pd.DataFrame:
    """
    Tabla `fecha, equity, benchmark, drawdown, retorno` (como la de un backtest normal) desde `CAPITAL_INICIAL`.

    La primera fila es el punto base (`base`, capital inicial, retorno nulo). `drawdown` es la fracción
    bajo el máximo previo (negativa), igual que `drawdown_pct` del motor. El benchmark se reescala al capital.
    """
    niveles = CAPITAL_INICIAL * np.concatenate([[1.0], np.cumprod(1.0 + serie["r_neto"].to_numpy(dtype=float))])
    fechas = pd.DatetimeIndex([base, *cierres])
    equity = pd.DataFrame({
        "fecha": fechas, "equity": niveles,
        "drawdown": niveles / np.maximum.accumulate(niveles) - 1.0,
        "retorno": np.concatenate([[np.nan], serie["r_neto"].to_numpy(dtype=float)]),
    })
    if precios_benchmark is not None:
        cierre_bench = precios_benchmark.reindex(fechas).to_numpy(dtype=float)
        equity["benchmark"] = cierre_bench / cierre_bench[0] * CAPITAL_INICIAL
    else:
        equity["benchmark"] = np.nan
    return equity[["fecha", "equity", "benchmark", "drawdown", "retorno"]]


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------


def _metricas_vs_benchmark(retornos: pd.Series, retornos_bench: pd.Series) -> dict[str, float]:
    """
    Alfa (Jensen con rf = 0: CAGR - beta·CAGR_bench), beta (MCO), tracking error e information ratio
    con periodicidad mensual (`PerformanceAnalyzer.compare_to_benchmark` anualiza con 252, de ahí que no se use).
    """
    alineados = pd.DataFrame({"p": retornos, "b": retornos_bench}).dropna()
    if len(alineados) < 3:
        return {}
    p = PerformanceAnalyzer.calculate_metrics(alineados["p"], risk_free_rate=0.0, periods_per_year=PERIODOS_ANIO)
    b = PerformanceAnalyzer.calculate_metrics(alineados["b"], risk_free_rate=0.0, periods_per_year=PERIODOS_ANIO)
    varianza = float(alineados["b"].var())
    beta = float(alineados["p"].cov(alineados["b"]) / varianza) if varianza > 0 else 0.0
    diferencia = alineados["p"] - alineados["b"]
    tracking = float(diferencia.std() * np.sqrt(PERIODOS_ANIO))
    return {
        "alfa": p.annualized_return - beta * b.annualized_return, "beta": beta, "tracking_error": tracking,
        "information_ratio": (p.annualized_return - b.annualized_return) / tracking if tracking > 0 else 0.0,
        "benchmark_retorno_total": b.total_return, "benchmark_cagr": b.annualized_return,
        "benchmark_sharpe": b.sharpe_ratio,
    }


def calcular_metricas(equity: pd.DataFrame, serie: pd.DataFrame, rf_mensual: pd.Series | None,
                      metricas_tfm: dict[str, float]) -> dict[str, Any]:
    """
    Métricas con los nombres de los experimentos normales, sobre la serie mensual importada.

    `sharpe` = CAGR / volatilidad con rf = 0 (misma definición que los demás experimentos). Además:
    `sharpe_media_desv` (√12·media/desv. de los retornos, rf = 0: es la definición del `sharpe_serie_promediada`
    del TFM), `sharpe_exceso_rf` (idem sobre el exceso de la rf mensual) y las métricas del TFM con prefijo `tfm_`.
    """
    retornos = equity.set_index("fecha")["retorno"].dropna()
    base = PerformanceAnalyzer.calculate_metrics(retornos, risk_free_rate=0.0, periods_per_year=PERIODOS_ANIO)
    anios = len(retornos) / PERIODOS_ANIO
    cambios = serie["turnover"].to_numpy(dtype=float)[1:]  # el primer mes es la compra inicial
    r = retornos.to_numpy(dtype=float)
    metricas: dict[str, Any] = {
        "retorno_total": base.total_return, "cagr": base.annualized_return, "volatilidad": base.volatility,
        "sharpe": base.sharpe_ratio, "sortino": base.sortino_ratio, "calmar": base.calmar_ratio,
        "max_drawdown": float(equity["drawdown"].min()), "var_95": base.var_95, "cvar_95": base.cvar_95,
        "capital_final": float(equity["equity"].iloc[-1]),
        "turnover_medio": float(np.mean(cambios)) if len(cambios) else 0.0,
        "turnover_anual": float(np.sum(cambios) / anios) if len(cambios) and anios > 0 else 0.0,
        "n_rebalanceos": len(retornos), "n_operaciones": None, "dias": len(retornos),
        "ulcer": indice_ulcer(equity["equity"].to_numpy(dtype=float)),
        "martin": ratio_martin(r, periodos_anio=PERIODOS_ANIO),
        "sharpe_media_desv": float(np.sqrt(PERIODOS_ANIO) * r.mean() / r.std(ddof=1)) if len(r) > 2 and r.std(ddof=1) > 0 else None,
    }
    if rf_mensual is not None:
        exceso = pd.Series(r - rf_mensual.reindex(serie.index).to_numpy(dtype=float))
        if exceso.notna().all() and exceso.std(ddof=1) > 0:
            metricas["sharpe_exceso_rf"] = float(np.sqrt(PERIODOS_ANIO) * exceso.mean() / exceso.std(ddof=1))
    if equity["benchmark"].notna().all():
        retornos_bench = equity.set_index("fecha")["benchmark"].pct_change().dropna()
        metricas.update(_metricas_vs_benchmark(retornos, retornos_bench))
    metricas.update({f"tfm_{clave}": valor for clave, valor in metricas_tfm.items()})
    metricas["validacion"] = {}
    return metricas


# ----------------------------------------------------------------------------
# Construcción del experimento y escritura
# ----------------------------------------------------------------------------


def _avisos(replica: int | None, n_replicas: int, motivo_sin_benchmark: str | None) -> list[str]:
    avisos = [AVISO_SESGOS]
    if replica is None:
        avisos.append(
            f"Serie = media de r_neto sobre {n_replicas} réplicas (la que da `sharpe_serie_promediada`): promediar "
            "réplicas diversifica el ruido de selección del universo, así que su Sharpe es mayor que el de una réplica; "
            "`tfm_sharpe_media_replicas` es el Sharpe medio de las réplicas individuales. Sin pesos (la media no tiene carteras).")
    else:
        avisos.append(f"Serie de la réplica {replica} del TFM (una sola extracción del universo aleatorio).")
    avisos += [
        ("Serie mensual: la fecha de cada punto es el último día de negociación del mes (el TFM etiqueta el mes por su "
         "primer día); el primer punto es el cierre del mes anterior con el capital inicial. Sin operaciones."),
        ("`sharpe` y `volatilidad` siguen la definición de los demás experimentos (CAGR/volatilidad, rf = 0). El Sharpe del "
         "TFM (`tfm_sharpe_*`, `sharpe_media_desv`) es √12·media/desv. de los retornos mensuales NETOS sin restar la "
         "rf (comprobado numéricamente; `sharpe_exceso_rf` sí la resta). `turnover` = ½Σ|w* - w_deriva| mensual."),
    ]
    if motivo_sin_benchmark:
        avisos.append(f"Sin benchmark SPY ({motivo_sin_benchmark}): no hay alfa/beta y las fechas son el último día hábil de cada mes.")
    return avisos


def construir_resultado(
    estrategia: str, n: int, t: int, replica: int | None, resultados: pd.DataFrame, metricas_celda: pd.DataFrame,
    rf_mensual: pd.Series | None, precios_benchmark: pd.Series | None, motivo_sin_benchmark: str | None,
    carpeta_tfm: Path, procedencia_base: dict[str, Any],
) -> ResultadoExperimento:
    """Arma el `ResultadoExperimento` de una celda (N, T) y estrategia a partir de los Parquet ya leídos."""
    serie = serie_de_celda(resultados, n, t, estrategia, replica)
    n_replicas = int(resultados[(resultados["N"] == n) & (resultados["T"] == t)
                                & (resultados["estrategia"] == estrategia)]["b"].nunique())
    config = _configuracion(estrategia, n, t, replica, n_replicas, serie.index)
    cierres, base = _cierres_de_mes(serie.index, precios_benchmark)
    equity = construir_equity(serie, cierres, base, precios_benchmark)
    try:
        fila = metricas_celda.loc[(n, t, estrategia)]
        metricas_tfm = {c: float(fila[c]) for c in METRICAS_TFM if c in fila.index}
    except KeyError:
        metricas_tfm = {}
    metricas = calcular_metricas(equity, serie, rf_mensual, metricas_tfm)

    if replica is not None and replica < 20 and (carpeta_tfm / ARCHIVO_PESOS).exists():
        pesos = _pesos_replica(carpeta_tfm, n, t, estrategia, replica)
        mes_a_cierre = dict(zip(serie.index, cierres, strict=True))
        pesos["fecha"] = pesos["fecha"].map(mes_a_cierre)
        pesos = pesos.dropna(subset=["fecha"])[COLUMNAS_PESOS]
    else:
        pesos = pd.DataFrame({"fecha": pd.Series([], dtype="datetime64[ns]"), "ticker": pd.Series([], dtype=str),
                              "peso": pd.Series([], dtype=float)})
    operaciones = pd.DataFrame({
        "fecha": pd.Series([], dtype="datetime64[ns]"), "ticker": pd.Series([], dtype=str), "lado": pd.Series([], dtype=str),
        "cantidad": pd.Series([], dtype=float), "precio": pd.Series([], dtype=float),
        "comision": pd.Series([], dtype=float), "pnl": pd.Series([], dtype=float)})
    avisos = _avisos(replica, n_replicas, motivo_sin_benchmark)
    if replica is not None and replica >= 20:
        avisos.append("Sin pesos: `pesos_oos` del TFM solo guarda las réplicas b < 20.")
    return ResultadoExperimento(config, metricas, equity, pesos, operaciones, avisos, dict(procedencia_base))


def _procedencia_base(carpeta_tfm: Path, archivos: list[str]) -> dict[str, Any]:
    """Procedencia común: origen, archivos fuente con SHA-256, commit del repo `tfm` y fecha de importación."""
    ahora = datetime.now().isoformat(timespec="seconds")
    return {
        "origen": "importado del TFM", "commit": None, "arbol_sucio": None,
        "commit_tfm": _commit_repo_tfm(carpeta_tfm),
        "archivos": {nombre: {"ruta": str((carpeta_tfm / nombre).resolve()), "sha256": _sha256(carpeta_tfm / nombre)}
                     for nombre in archivos},
        "commit_importador": commit_git().get("commit"),
        "importado": ahora, "ejecutado": ahora,
    }


def _escribir(almacen: AlmacenExperimentos, id_experimento: str, resultado: ResultadoExperimento) -> None:
    """Crea la carpeta del experimento con su ficha y lo guarda como `hecho` (mismo formato que un backtest)."""
    documentos = almacen.documentos
    documentos.crear(id_experimento, reemplazar=True)  # valida el id; descarta restos de una importación interrumpida
    documentos.escribir_json(id_experimento, "config", resultado.config.model_dump())
    documentos.escribir_json(id_experimento, "estado", {"estado": "pendiente", "creado": datetime.now().isoformat(),
                                                        "actualizado": datetime.now().isoformat(), "error": None})
    almacen.guardar(id_experimento, resultado)


def importar_experimentos_tfm(
    carpeta_tfm: Path | str, almacen: AlmacenExperimentos, celdas: list[tuple[int, int]] | None = None,
    estrategias: list[str] | None = None, replica: int | None = None, forzar: bool = False,
    cargador_precios: CargadorPrecios | None = None, ticker_benchmark: str = "SPY",
) -> list[dict[str, str]]:
    """
    Importa al `almacen` los experimentos TFM de las celdas (N, T) y estrategias pedidas.

    Args:
        carpeta_tfm: carpeta `rejilla_v2` con los Parquet del TFM (solo lectura).
        celdas: lista de (N, T); por defecto `CELDAS_POR_DEFECTO`. `estrategias`: por defecto las 6 del TFM.
        replica: None = serie promediada de todas las réplicas; b = esa réplica (pesos si b < 20).
        forzar: sobrescribe los ya existentes (si no, se omiten: la importación es idempotente).
        cargador_precios: para el benchmark SPY; None = sin benchmark.

    Returns:
        Lista de `{"id", "accion"}` con `accion` en `importado | sobrescrito | omitido`.
    """
    carpeta_tfm = Path(carpeta_tfm)
    celdas = list(celdas) if celdas else list(CELDAS_POR_DEFECTO)
    estrategias = list(estrategias) if estrategias else list(ESTRATEGIAS_TFM)
    faltan = [a for a in ARCHIVOS_BASE if not (carpeta_tfm / a).exists()]
    if faltan:
        raise ErrorImportacionTfm(f"Faltan archivos del TFM en {carpeta_tfm}: {', '.join(faltan)}")
    desconocidas = [e for e in estrategias if e not in ESTRATEGIAS_TFM]
    if desconocidas:
        raise ErrorImportacionTfm(f"Estrategias desconocidas {desconocidas}; las del TFM son {list(ESTRATEGIAS_TFM)}")
    disponibles = set(celdas_disponibles(carpeta_tfm))
    ausentes = [c for c in celdas if c not in disponibles]
    if ausentes:
        raise ErrorImportacionTfm(f"Celdas (N, T) que no existen en el TFM: {ausentes}")

    acciones: list[dict[str, str]] = []
    pendientes: list[tuple[str, int, int, str]] = []
    for n, t in celdas:
        for estrategia in estrategias:
            id_experimento = identificador_tfm(estrategia, n, t, replica)
            if almacen.existe(id_experimento) and not forzar:
                acciones.append({"id": id_experimento, "accion": "omitido"})
            else:
                pendientes.append((id_experimento, n, t, estrategia))
    if not pendientes:
        return acciones

    usa_pesos = replica is not None and replica < 20 and (carpeta_tfm / ARCHIVO_PESOS).exists()
    procedencia = _procedencia_base(carpeta_tfm, [*ARCHIVOS_BASE, *([ARCHIVO_PESOS] if usa_pesos else [])])
    resultados = _leer_resultados(carpeta_tfm, [(n, t) for _, n, t, _ in pendientes], replica)
    metricas_celda = _leer_metricas_celda(carpeta_tfm)
    rf = pd.read_parquet(carpeta_tfm / "rf_mensual.parquet")["rf"]
    etiquetas = pd.DatetimeIndex(sorted(resultados["fecha"].unique()))
    precios, motivo = _precios_benchmark(cargador_precios, ticker_benchmark, etiquetas)

    for id_experimento, n, t, estrategia in pendientes:
        resultado = construir_resultado(estrategia, n, t, replica, resultados, metricas_celda, rf, precios, motivo,
                                        carpeta_tfm, procedencia)
        existia = almacen.existe(id_experimento)
        _escribir(almacen, id_experimento, resultado)
        acciones.append({"id": id_experimento, "accion": "sobrescrito" if existia else "importado"})
    return acciones
