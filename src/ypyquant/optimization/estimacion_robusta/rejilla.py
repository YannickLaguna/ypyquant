"""
Motor walk-forward de la rejilla (N, T) del TFM «Optimización robusta de portafolios».

Responsabilidad: reproducir el experimento de forma parametrizable y desacoplada de
cualquier fuente de datos. Para cada réplica b (universos anidados aleatorios), cada
ventana T, cada fecha OOS (rebalanceo mensual) y cada tamaño N:
1. elige el universo (elegibles point-in-time + prioridad aleatoria fija por réplica);
2. estima µ y Σ con retornos en la ventana de T meses previos (hasta t-1);
3. construye la cartera de cada estrategia (muestral, Bayes-Stein, Michaud, BL-prior,
   BL-momentum, 1/N) con el optimizador max-Sharpe QP;
4. mide el turnover frente a los pesos A LA DERIVA, el coste c·turnover y el retorno
   OOS bruto y neto, y registra la contingencia GMV.

Convenciones de retorno (explícitas en los nombres):
- `retornos_estimacion`: los que alimentan µ, Σ y la señal 12-1. Por defecto LOG
  excesos (`log1p(r) - log1p(rf)`); `tipo_retorno_estimacion="simple"` usa excesos
  simples `r - rf` (es lo que hizo realmente el código del TFM; necesario para paridad).
- `retornos_simples_exceso_oos`: retorno SIMPLE en exceso del mes t, con el que se evalúa
  (`r_bruto`).
- `retornos_simples_totales_deriva`: retorno SIMPLE total (sin restar rf) con el que
  evolucionan los pesos entre rebalanceos para calcular el turnover.
Coste: tc = c·turnover con turnover = ½Σ|w* - w_deriva|; r_neto = (1+r_bruto)(1-tc) - 1.

Datos inyectables: el motor consume un `DatosRejilla` (precios y volumen mensuales,
rf mensual, membresías y, opcionalmente, un `ProveedorCapitalizacion`). Cualquier
objeto con método `cargar() -> DatosRejilla` (Protocol `CargadorDatosRejilla`) sirve;
no hay dependencia de SQLite (el cargador de la BD del TFM vive en los tests de paridad).

Salida (`ResultadosRejilla`): `resultados_oos` (mismo esquema que el parquet del TFM más
`frac_contingencia`), `pesos_oos` (réplicas b < `max_replicas_con_pesos`) y
`manifest_draws`. No replica el orquestador del TFM (MLflow, checkpoints, joblib).

Se relaciona con: estimadores_mu.py, sigma.py, max_sharpe_qp.py, universo.py y
w_mercado.py (todos los bloques que orquesta); docs/migracion-v0.4.0.md (uso).
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from ypyquant.optimization.estimacion_robusta.estimadores_mu import (
    MESES_MOMENTUM,
    calcular_diagnostico_bl,
    estimar_mu_bayes_stein,
    estimar_mu_bl_momentum,
    estimar_mu_michaud,
    estimar_mu_muestral,
)
from ypyquant.optimization.estimacion_robusta.max_sharpe_qp import optimizar_max_sharpe
from ypyquant.optimization.estimacion_robusta.sigma import estimar_sigma_ledoit_wolf
from ypyquant.optimization.estimacion_robusta.universo import (
    elegibles_en_fecha,
    generar_prioridad_universo,
    seed_para,
    seleccionar_universo_anidado,
)
from ypyquant.optimization.estimacion_robusta.w_mercado import ProveedorCapitalizacion, pesos_mercado

logger = logging.getLogger(__name__)

ESTRATEGIAS_REJILLA = (
    "mpt_muestral",
    "mpt_bayes_stein",
    "mpt_michaud",
    "bl_prior",
    "bl_momentum",
    "igual_ponderacion",
)


# ============================================================================
# Configuración y datos
# ============================================================================


@dataclass
class ConfiguracionRejilla:
    """
    Parámetros del experimento (los defectos son el caso base del TFM, salvo
    `tipo_retorno_estimacion`, `cobertura_min`, `exigir_retorno_oos`, que corrigen sesgos).

    valores_n / valores_t: rejilla de tamaños de universo y meses de ventana.
    estrategias: subconjunto de `ESTRATEGIAS_REJILLA`.
    n_replicas (B): nº de universos anidados aleatorios; semilla_base: semilla de la réplica 0.
    coste_pb: coste en puntos básicos por unidad de turnover (c = coste_pb·1e-4).
    fecha_oos_ini / fecha_oos_fin: rango de fechas de rebalanceo.
    delta, tau: aversión al riesgo y escala de incertidumbre de Black-Litterman.
    replicas_michaud, metodo_bootstrap, longitud_media_bloque: remuestreo de Michaud.
    tipo_retorno_estimacion: "log" (defecto) o "simple" (paridad con el código del TFM).
    cobertura_min: fracción mínima de retornos observados en la ventana (1.0 = completa).
    exigir_retorno_oos: si True, solo son elegibles los activos con retorno en t (sesgo de
        supervivencia del TFM; por defecto False y el retorno faltante se imputa a 0 %).
    min_observaciones: filas mínimas de la ventana de estimación tras dropna.
    max_replicas_con_pesos: `pesos_oos` y `manifest_draws` solo para b < este valor.
    metodo_prioridad: "hash" (prioridad por nombre de ticker, estable) o "posicional" (esquema
        antiguo del TFM, solo para paridad con los parquet publicados).
    rng_global_legado: True reproduce el consumo del estado global de numpy del TFM
        (np.random.seed(semilla_base+b) por réplica); False usa un Generator por réplica.
    """

    valores_n: Sequence[int] = (20, 40, 80, 150)
    valores_t: Sequence[int] = (12, 24, 36, 60, 120)
    estrategias: Sequence[str] = ESTRATEGIAS_REJILLA
    n_replicas: int = 100
    semilla_base: int = 42
    coste_pb: float = 10.0
    fecha_oos_ini: str = "2010-01-01"
    fecha_oos_fin: str = "2025-12-31"
    delta: float = 2.5
    tau: float = 0.05
    replicas_michaud: int = 50
    metodo_bootstrap: Literal["iid", "bloques"] = "iid"
    longitud_media_bloque: float = 6.0
    tipo_retorno_estimacion: Literal["log", "simple"] = "log"
    cobertura_min: float = 1.0
    exigir_retorno_oos: bool = False
    min_observaciones: int = 6
    peso_maximo: float = 1.0
    frecuencia: int = 12
    max_replicas_con_pesos: int = 20
    metodo_prioridad: Literal["hash", "posicional"] = "hash"
    rng_global_legado: bool = False


@dataclass
class DatosRejilla:
    """
    Datos de entrada del motor.

    precios_mensuales: (meses x tickers) precios ajustados (un valor por mes).
    volumen_mensual: (meses x tickers) volumen, solo para el proxy de w_mkt; opcional.
    rf_mensual: tasa libre de riesgo de cada mes (fracción mensual); None = 0.
    membresias: máscara bool (fechas x tickers) o mapping ticker -> intervalos; None = sin filtro.
    proveedor_capitalizacion: capitalización real point-in-time; opcional.
    """

    precios_mensuales: pd.DataFrame
    volumen_mensual: pd.DataFrame | None = None
    rf_mensual: pd.Series | None = None
    membresias: object | None = None
    proveedor_capitalizacion: ProveedorCapitalizacion | None = None


class CargadorDatosRejilla(Protocol):
    """Contrato de un cargador de datos inyectable (SQLite, parquet, API, memoria...)."""

    def cargar(self) -> DatosRejilla:
        """Devuelve los datos listos para el motor."""
        ...


@dataclass
class ResultadosRejilla:
    """Salida del motor: tablas equivalentes a los parquet del TFM."""

    resultados_oos: pd.DataFrame = field(default_factory=pd.DataFrame)
    pesos_oos: pd.DataFrame = field(default_factory=pd.DataFrame)
    manifest_draws: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class _Panel:
    """Retornos derivados de `DatosRejilla` con nombres explícitos (uso interno)."""

    precios: pd.DataFrame
    volumen: pd.DataFrame | None
    retornos_estimacion: pd.DataFrame
    retornos_simples_exceso: pd.DataFrame
    retornos_simples_totales: pd.DataFrame


def preparar_panel(datos: DatosRejilla, tipo_retorno_estimacion: Literal["log", "simple"] = "log") -> _Panel:
    """
    Calcula los tres paneles de retornos a partir de los precios mensuales:
    simples totales (deriva), simples en exceso (evaluación) y de estimación
    (log o simple en exceso de rf, según `tipo_retorno_estimacion`).
    """
    precios = datos.precios_mensuales.where(datos.precios_mensuales > 0)
    simples = precios.pct_change(fill_method=None).dropna(how="all")
    if datos.rf_mensual is None:
        rf = pd.Series(0.0, index=simples.index)
    else:
        rf = datos.rf_mensual.reindex(simples.index).ffill().bfill().fillna(0.0)
    simples_exceso = simples.sub(rf, axis=0)
    if tipo_retorno_estimacion == "log":
        estimacion = np.log1p(simples).sub(np.log1p(rf), axis=0)
    elif tipo_retorno_estimacion == "simple":
        estimacion = simples_exceso
    else:
        raise ValueError("tipo_retorno_estimacion debe ser 'log' o 'simple'")
    return _Panel(precios, datos.volumen_mensual, estimacion, simples_exceso, simples)


# ============================================================================
# Utilidades
# ============================================================================


def _diagnosticos_pesos(pesos: np.ndarray, w_mkt: np.ndarray, w_prev: np.ndarray | None) -> dict:
    """HHI, peso máximo, nº de posiciones y distancias L1 a w_mkt y a la cartera previa (a la deriva)."""
    return {
        "hhi": float(np.sum(pesos**2)),
        "w_max": float(np.max(pesos)) if len(pesos) else np.nan,
        "n_activos_pos": int(np.sum(pesos > 1e-6)),
        "dist_w_mkt": float(np.abs(pesos - w_mkt).sum()),
        "dist_w_prev": float(np.abs(pesos - w_prev).sum()) if w_prev is not None else np.nan,
    }


def calcular_turnover(
    pesos_nuevos: pd.Series, pesos_previos: pd.Series, retornos_totales_previos: pd.Series
) -> tuple[float, pd.Series]:
    """
    Turnover ½Σ|w* - w_deriva| frente a los pesos a la deriva.

    La deriva evoluciona los pesos previos con el retorno SIMPLE TOTAL del periodo que
    acaba de cerrar: w_deriva ∝ w_prev·(1+r), renormalizado. Los activos que salen o
    entran se tratan con peso 0 en el otro lado.

    Retorna (turnover, w_deriva reindexada a la unión de activos).
    """
    bruta = pesos_previos * (1.0 + retornos_totales_previos.reindex(pesos_previos.index).fillna(0.0))
    denominador = float(bruta.sum())
    deriva = bruta / denominador if abs(denominador) > 1e-8 else bruta
    union = sorted(set(pesos_previos.index) | set(pesos_nuevos.index))
    deriva_u = deriva.reindex(union, fill_value=0.0)
    nuevos_u = pesos_nuevos.reindex(union, fill_value=0.0)
    return 0.5 * float(np.abs(nuevos_u.values - deriva_u.values).sum()), deriva_u


def calcular_retorno_neto(retorno_bruto: float, turnover: float, coste_pb: float) -> tuple[float, float]:
    """(tc, r_neto) con tc = c·turnover y r_neto = (1+r_bruto)(1-tc) - 1; NaN si falta algún dato."""
    if not (np.isfinite(retorno_bruto) and np.isfinite(turnover)):
        return np.nan, np.nan
    tc = coste_pb * 1e-4 * turnover
    return tc, (1.0 + retorno_bruto) * (1.0 - tc) - 1.0


# ============================================================================
# Núcleo: una réplica
# ============================================================================


def ejecutar_replica(
    b: int,
    panel: _Panel,
    config: ConfiguracionRejilla,
    funcion_elegibles: Callable[[pd.Timestamp, int], pd.Index],
    proveedor_capitalizacion: ProveedorCapitalizacion | None = None,
    guardar_pesos: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Procesa la réplica `b` sobre toda la rejilla (T, fecha, N, estrategia).

    `funcion_elegibles(fecha, T)` devuelve los tickers elegibles en la fecha de rebalanceo
    (típicamente `elegibles_en_fecha`; inyectable para reproducir filtros del TFM).
    Devuelve `(filas_resultados, filas_pesos)` como listas de dicts.
    """
    if config.rng_global_legado:
        np.random.seed(seed_para(b, config.semilla_base))
        rng = None
    else:
        rng = np.random.default_rng(seed_para(b, config.semilla_base))

    retornos = panel.retornos_simples_totales
    prioridad = generar_prioridad_universo(
        retornos.columns, b=b, semilla_base=config.semilla_base, metodo=config.metodo_prioridad
    )
    fechas_oos = retornos.loc[config.fecha_oos_ini : config.fecha_oos_fin].index
    frec = config.frecuencia

    filas_resultados: list[dict] = []
    filas_pesos: list[dict] = []

    for t_meses in config.valores_t:
        estado_previo: dict[tuple[int, str], dict] = {}

        for fecha_oos in fechas_oos:
            fecha_ini = fecha_oos - pd.DateOffset(months=t_meses)
            fecha_fin = fecha_oos - pd.DateOffset(months=1)  # t-1: último dato permitido

            ret_total_t = panel.retornos_simples_totales.loc[fecha_oos]
            ret_exceso_t = panel.retornos_simples_exceso.loc[fecha_oos]

            elegibles = pd.Index(funcion_elegibles(fecha_oos, t_meses))
            if config.exigir_retorno_oos:
                elegibles = elegibles.intersection(ret_total_t.dropna().index)

            for n in config.valores_n:
                activos_n = seleccionar_universo_anidado(elegibles, prioridad, n)
                claves_n = [(n, e) for e in config.estrategias]
                if not activos_n:
                    for clave in claves_n:
                        estado_previo.pop(clave, None)
                    continue

                ret_is = panel.retornos_estimacion.loc[fecha_ini:fecha_fin, activos_n].dropna()
                if len(ret_is) < config.min_observaciones:
                    for clave in claves_n:
                        estado_previo.pop(clave, None)
                    continue

                sigma = estimar_sigma_ledoit_wolf(ret_is, frec)
                sigma_np = sigma.values
                w_mkt, _ = pesos_mercado(
                    activos_n, fecha_fin, proveedor_capitalizacion, panel.precios, panel.volumen
                )
                ret_oos_n = ret_exceso_t[activos_n].fillna(0.0).values  # simple en exceso: evaluación
                ret_deriva_n = ret_total_t[activos_n].fillna(0.0).values  # simple total: deriva de pesos

                for estrategia in config.estrategias:
                    t0 = time.perf_counter()
                    pesos, degenerate, solver_status = None, False, "n/a"
                    frac_contingencia = np.nan
                    diag_bl = {"conf_aislada": np.nan, "conf_conjunta": np.nan, "dispersion_omega": np.nan}
                    try:
                        if estrategia == "igual_ponderacion":
                            pesos = np.full(n, 1.0 / n)

                        elif estrategia == "bl_prior":
                            # Σ⁻¹Π = δ·w_mkt exacto (sin views): w_mkt directo, sin optimizar.
                            pesos = w_mkt.copy()
                            solver_status = "shortcut_w_mkt"

                        elif estrategia == "mpt_michaud":
                            pesos, frac_contingencia = estimar_mu_michaud(
                                ret_is, n_replicas=config.replicas_michaud, rf=0.0,
                                peso_maximo=config.peso_maximo, devolver_diagnostico=True,
                                metodo_bootstrap=config.metodo_bootstrap,
                                longitud_media_bloque=config.longitud_media_bloque,
                                rng=rng, frecuencia=frec,
                            )
                            # Regla de mayoría: degenerate si >50 % de las réplicas usaron la GMV.
                            degenerate = bool(frac_contingencia > 0.5) if np.isfinite(frac_contingencia) else False
                            solver_status = "michaud_bootstrap"

                        elif estrategia == "bl_momentum":
                            # Señal 12-1 por FECHA (t-12..t-2), no por posición sobre ret_is.
                            ventana_signal = panel.retornos_estimacion.loc[
                                fecha_fin - pd.DateOffset(months=MESES_MOMENTUM - 1) : fecha_fin - pd.DateOffset(months=1),
                                activos_n,
                            ].dropna()
                            mu_bl, m_post = estimar_mu_bl_momentum(
                                ventana_signal, sigma, w_mkt, config.delta, config.tau, frec
                            )
                            diag_bl = calcular_diagnostico_bl(
                                ventana_signal, sigma, w_mkt, config.delta, config.tau, frec
                            )
                            pesos, degenerate, solver_status = optimizar_max_sharpe(
                                mu_bl, sigma_np + m_post, rf=0.0, peso_maximo=config.peso_maximo
                            )

                        else:
                            if estrategia == "mpt_bayes_stein":
                                mu = estimar_mu_bayes_stein(ret_is, sigma, frec, si_t_menor_n2="ignorar")
                            elif estrategia == "mpt_muestral":
                                mu = estimar_mu_muestral(ret_is, frec)
                            else:
                                raise ValueError(f"Estrategia desconocida: {estrategia}")
                            pesos, degenerate, solver_status = optimizar_max_sharpe(
                                mu, sigma_np, rf=0.0, peso_maximo=config.peso_maximo
                            )

                        r_bruto = float((pesos * ret_oos_n).sum())
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("[b=%s N=%s T=%s %s] %s: %s", b, n, t_meses, fecha_oos.date(), estrategia, exc)
                        pesos, degenerate, solver_status, r_bruto = None, True, "error", np.nan

                    tiempo_ms = (time.perf_counter() - t0) * 1000.0
                    clave = (n, estrategia)
                    previo = estado_previo.get(clave)

                    turnover = np.nan
                    diag = {"hhi": np.nan, "w_max": np.nan, "n_activos_pos": np.nan,
                            "dist_w_mkt": np.nan, "dist_w_prev": np.nan}
                    if pesos is not None:
                        w_prev_diag = None
                        if previo is not None:
                            turnover, deriva_u = calcular_turnover(
                                pd.Series(pesos, index=activos_n),
                                pd.Series(previo["pesos"], index=previo["activos"]),
                                pd.Series(previo["ret"], index=previo["activos"]),
                            )
                            w_prev_diag = deriva_u.reindex(activos_n, fill_value=0.0).values
                        else:
                            turnover = 0.0
                        diag = _diagnosticos_pesos(pesos, w_mkt, w_prev_diag)

                    tc, r_neto = calcular_retorno_neto(r_bruto, turnover, config.coste_pb)

                    filas_resultados.append({
                        "b": b, "N": n, "T": t_meses, "estrategia": estrategia, "fecha": fecha_oos,
                        "r_bruto": r_bruto, "turnover": turnover, "tc": tc, "r_neto": r_neto,
                        "n_activos_pos": diag["n_activos_pos"], "hhi": diag["hhi"], "w_max": diag["w_max"],
                        "dist_w_prev": diag["dist_w_prev"], "dist_w_mkt": diag["dist_w_mkt"],
                        "degenerate": bool(degenerate), "solver_status": solver_status,
                        "tiempo_ms": tiempo_ms,
                        "conf_aislada": diag_bl["conf_aislada"], "conf_conjunta": diag_bl["conf_conjunta"],
                        "dispersion_omega": diag_bl["dispersion_omega"],
                        "frac_contingencia": frac_contingencia,
                    })

                    if guardar_pesos and pesos is not None:
                        filas_pesos.extend(
                            {"b": b, "N": n, "T": t_meses, "estrategia": estrategia,
                             "fecha": fecha_oos, "ticker": ticker, "w": float(w)}
                            for ticker, w in zip(activos_n, pesos, strict=True) if w > 1e-6
                        )

                    if pesos is not None and np.isfinite(r_bruto):
                        estado_previo[clave] = {"activos": activos_n, "pesos": pesos, "ret": ret_deriva_n}
                    else:
                        estado_previo.pop(clave, None)

    return filas_resultados, filas_pesos


def generar_manifest(
    b: int, tickers, semilla_base: int = 42, metodo_prioridad: Literal["hash", "posicional"] = "hash"
) -> list[dict]:
    """Prioridad u_i completa de la réplica b ordenada: reconstruye el universo de cualquier (N, T, fecha)."""
    prioridad = generar_prioridad_universo(tickers, b=b, semilla_base=semilla_base, metodo=metodo_prioridad)
    orden = prioridad.sort_values(kind="stable")
    return [
        {"b": b, "semilla": seed_para(b, semilla_base), "rank": i, "ticker": tk, "prioridad": float(u)}
        for i, (tk, u) in enumerate(orden.items())
    ]


# ============================================================================
# API pública
# ============================================================================


def ejecutar_rejilla(
    datos: DatosRejilla | CargadorDatosRejilla,
    config: ConfiguracionRejilla | None = None,
    funcion_elegibles: Callable[[pd.Timestamp, int], pd.Index] | None = None,
    replicas: Sequence[int] | None = None,
) -> ResultadosRejilla:
    """
    Ejecuta la rejilla (secuencial) y devuelve las tablas de resultados.

    Parámetros
    ----------
    datos : `DatosRejilla` o un cargador con método `cargar()`.
    config : `ConfiguracionRejilla` (caso base del TFM si se omite).
    funcion_elegibles : `(fecha, T) -> Index`; por defecto `elegibles_en_fecha` point-in-time
        con `config.cobertura_min`. Inyectable para reproducir otros criterios.
    replicas : índices b a ejecutar (por defecto `range(config.n_replicas)`).
    """
    config = config or ConfiguracionRejilla()
    if not isinstance(datos, DatosRejilla):
        datos = datos.cargar()
    desconocidas = set(config.estrategias) - set(ESTRATEGIAS_REJILLA)
    if desconocidas:
        raise ValueError(f"Estrategias desconocidas: {sorted(desconocidas)}")

    panel = preparar_panel(datos, config.tipo_retorno_estimacion)
    if funcion_elegibles is None:
        cache: dict[tuple[pd.Timestamp, int], pd.Index] = {}

        def funcion_elegibles(fecha: pd.Timestamp, t_meses: int) -> pd.Index:
            clave = (fecha, t_meses)
            if clave not in cache:
                cache[clave] = elegibles_en_fecha(
                    panel.precios, datos.membresias, fecha, t_meses, config.cobertura_min
                )
            return cache[clave]

    todas_filas: list[dict] = []
    todos_pesos: list[dict] = []
    manifest: list[dict] = []
    for b in replicas if replicas is not None else range(config.n_replicas):
        con_pesos = b < config.max_replicas_con_pesos
        filas, pesos = ejecutar_replica(
            b, panel, config, funcion_elegibles, datos.proveedor_capitalizacion, guardar_pesos=con_pesos
        )
        todas_filas.extend(filas)
        todos_pesos.extend(pesos)
        if con_pesos:
            manifest.extend(generar_manifest(
                    b, panel.retornos_simples_totales.columns, config.semilla_base, config.metodo_prioridad
                )
            )

    return ResultadosRejilla(
        resultados_oos=pd.DataFrame(todas_filas),
        pesos_oos=pd.DataFrame(todos_pesos),
        manifest_draws=pd.DataFrame(manifest),
    )
