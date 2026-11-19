"""
Estadísticos de validación de estrategias (funciones puras, numpy/pandas/scipy).

Responsabilidad:
- Sharpe (por periodo o anualizado), Sharpe máximo esperado bajo N pruebas, Sharpe probabilístico
  (PSR) y Sharpe deflactado (DSR): Bailey y López de Prado (2014), JPM 40(5).
- Probabilidad de sobreajuste del backtest por CSCV (PBO): Bailey, Borwein, López de Prado y Zhu (2017).
- Prueba de rachas de Wald-Wolfowitz, índice Ulcer, ratio de Martin.
- p-valor de un test de permutación con corrección +1.

Todas las funciones reciben retornos (Series/ndarray) o una matriz de retornos diarios
(DataFrame índice=fechas, columnas=configuraciones de un barrido). No leen datos, no tocan el
motor de backtest ni la API.

Se relaciona con:
- ypyquant.validacion.permutacion (genera los valores permutados que consume `p_valor_permutacion`).
- ypyquant.validacion.walk_forward (usa `sharpe` como métrica por defecto).
- ypyquant.estrategias.experimentos (los barridos producirán la matriz de retornos; la conexión
  se hará en un paso posterior mediante `registrar_validador`).
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

GAMMA_EULER_MASCHERONI = 0.5772156649
DIAS_ANIO = 252
MAX_COMBINACIONES_CSCV = 5000

# Una métrica recibe un ndarray 2D (filas=observaciones, columnas=configuraciones)
# y devuelve un valor por columna.
Metrica = Callable[[np.ndarray], np.ndarray]


def _a_float_array(valores: Any) -> np.ndarray:
    """Convierte Series/DataFrame/lista a ndarray float (conserva 1D o 2D)."""
    return np.asarray(valores, dtype=float)


def sharpe(retornos: Any, periodos_anio: int = DIAS_ANIO, anualizar: bool = False) -> Any:
    """
    Ratio de Sharpe = media / desviación típica muestral (ddof=1), sin tasa libre de riesgo.

    Por defecto es el Sharpe POR PERIODO (el que exigen PSR/DSR). Con `anualizar=True` se
    multiplica por sqrt(periodos_anio). Ignora NaN. Con desviación nula o < 2 datos devuelve NaN.

    Args:
        retornos: Series/lista/ndarray 1D (devuelve float), ndarray 2D (devuelve ndarray por
            columna) o DataFrame (devuelve Series por columna).
    """
    es_df = isinstance(retornos, pd.DataFrame)
    columnas = retornos.columns if es_df else None
    matriz = _a_float_array(retornos)
    unidimensional = matriz.ndim == 1
    if unidimensional:
        matriz = matriz[:, None]
    n_validos = np.sum(~np.isnan(matriz), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columnas todo NaN -> resultado NaN
        media = np.nanmean(matriz, axis=0)
        desviacion = np.nanstd(matriz, axis=0, ddof=1)
        resultado = np.where((n_validos >= 2) & (desviacion > 1e-15), media / desviacion, np.nan)
    if anualizar:
        resultado = resultado * math.sqrt(periodos_anio)
    if es_df:
        return pd.Series(resultado, index=columnas)
    return float(resultado[0]) if unidimensional else resultado


def sharpe_maximo_esperado(varianza_sharpes: float, n_pruebas: int) -> float:
    """
    Sharpe máximo esperado entre N pruebas independientes sin habilidad (SR0 de Bailey y López de Prado):

        E[max SR] = sqrt(V) · ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))

    con γ = 0.5772156649 (Euler-Mascheroni) y V la varianza de los Sharpes (por periodo) entre
    las pruebas. Con N <= 1 devuelve 0 (no hay selección).
    """
    if n_pruebas <= 1 or not np.isfinite(varianza_sharpes) or varianza_sharpes <= 0:
        return 0.0
    gamma = GAMMA_EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_pruebas)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_pruebas * math.e))
    return float(math.sqrt(varianza_sharpes) * ((1.0 - gamma) * z1 + gamma * z2))


def _momentos(retornos: np.ndarray) -> tuple[float, float, float, int]:
    """Devuelve (sharpe por periodo, asimetría, curtosis NO en exceso, T) de una serie sin NaN."""
    x = retornos[~np.isnan(retornos)]
    n = len(x)
    sr = sharpe(x)
    if n < 3:
        return sr, float("nan"), float("nan"), n
    return sr, float(stats.skew(x)), float(stats.kurtosis(x, fisher=False)), n


def _psr_desde_momentos(sr: float, sr_ref: float, n: int, asimetria: float, curtosis: float) -> float:
    """PSR = Φ((SR̂ − SR*)·sqrt(T−1) / sqrt(1 − γ3·SR̂ + (γ4−1)/4·SR̂²))."""
    if not np.isfinite(sr) or not np.isfinite(asimetria) or n < 3:
        return float("nan")
    radicando = 1.0 - asimetria * sr + (curtosis - 1.0) / 4.0 * sr**2
    if radicando <= 0:
        return float("nan")
    z = (sr - sr_ref) * math.sqrt(n - 1) / math.sqrt(radicando)
    return float(stats.norm.cdf(z))


def sharpe_probabilistico(retornos: Any, sharpe_referencia: float = 0.0) -> float:
    """
    Sharpe probabilístico (PSR): probabilidad de que el Sharpe verdadero supere `sharpe_referencia`
    teniendo en cuenta longitud de la muestra, asimetría y curtosis:

        PSR = Φ((SR̂ − SR*)·sqrt(T−1) / sqrt(1 − γ3·SR̂ + (γ4−1)/4·SR̂²))

    Todo en unidades POR PERIODO (no anualizado), también `sharpe_referencia`. γ3 es la
    asimetría y γ4 la curtosis NO en exceso (normal = 3), momentos muestrales (sesgados).
    """
    sr, asimetria, curtosis, n = _momentos(_a_float_array(retornos))
    return _psr_desde_momentos(sr, float(sharpe_referencia), n, asimetria, curtosis)


def sharpe_deflactado(
    retornos_elegida: Any,
    sharpes_pruebas: Sequence[float] | None = None,
    matriz_retornos: pd.DataFrame | None = None,
    n_pruebas: int | None = None,
) -> dict[str, float]:
    """
    Sharpe deflactado (DSR): PSR con SR* = `sharpe_maximo_esperado(V, N)`, donde V es la varianza
    (ddof=1) de los Sharpes por periodo de las N pruebas del barrido y N su número.

    Pasa los Sharpes por periodo de las pruebas (`sharpes_pruebas`) o la matriz de retornos del
    barrido (`matriz_retornos`, se calculan por columna). `n_pruebas` permite fijar un N
    distinto (p. ej. el número efectivo de pruebas independientes).

    Returns:
        dict con dsr, sr (por periodo de la elegida), sr_umbral (SR*), n_pruebas, T, asimetria, curtosis.
        Con N = 1, SR* = 0 y el DSR coincide con PSR(0).

    Referencia: Bailey y López de Prado (2014), «The Deflated Sharpe Ratio», JPM 40(5).
    """
    if (sharpes_pruebas is None) == (matriz_retornos is None):
        raise ValueError("Indica exactamente uno: sharpes_pruebas o matriz_retornos")
    if matriz_retornos is not None:
        sharpes = np.asarray(sharpe(matriz_retornos), dtype=float)
    else:
        sharpes = np.asarray(sharpes_pruebas, dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    n = int(n_pruebas) if n_pruebas is not None else len(sharpes)
    varianza = float(np.var(sharpes, ddof=1)) if len(sharpes) >= 2 else 0.0
    sr_umbral = sharpe_maximo_esperado(varianza, n)
    sr, asimetria, curtosis, t = _momentos(_a_float_array(retornos_elegida))
    return {
        "dsr": _psr_desde_momentos(sr, sr_umbral, t, asimetria, curtosis),
        "sr": sr,
        "sr_umbral": sr_umbral,
        "n_pruebas": n,
        "T": t,
        "asimetria": asimetria,
        "curtosis": curtosis,
    }


def pbo_cscv(
    matriz_retornos: pd.DataFrame | np.ndarray,
    n_bloques: int = 16,
    metrica: Metrica = sharpe,
    max_combinaciones: int = MAX_COMBINACIONES_CSCV,
    semilla: int = 0,
) -> dict[str, Any]:
    """
    Probabilidad de sobreajuste del backtest (PBO) por validación cruzada simétrica combinatoria (CSCV).

    Las T filas se parten en S = `n_bloques` bloques contiguos de igual tamaño (S par; si T no es
    múltiplo de S se descartan las primeras T mod S filas). Para cada una de las C(S, S/2)
    combinaciones: IS = bloques elegidos, OOS = su complemento; n* = argmax de la métrica en IS;
    ω̄ = rango relativo de n* en OOS = rango/(N+1); λ = ln(ω̄/(1−ω̄)). PBO = proporción de λ ≤ 0.
    Si hay más de `max_combinaciones` se muestrean al azar (sin repetición, con `semilla`).

    Args:
        matriz_retornos: T×N (N >= 2 configuraciones), sin NaN (rellena antes con 0 si procede).
        metrica: callable(ndarray 2D) -> ndarray por columna. Por defecto, Sharpe por periodo.

    Returns:
        dict con pbo, lambdas (lista), pares_is_oos (lista de (métrica IS de n*, métrica OOS de n*),
        para el gráfico de degradación), pendiente (regresión OOS~IS; negativa = degradación fuerte),
        prob_perdida_oos (proporción con métrica OOS de n* < 0), n_combinaciones, n_bloques,
        n_configuraciones.

    Referencia: Bailey, Borwein, López de Prado y Zhu (2017), «The Probability of Backtest
    Overfitting», Journal of Computational Finance 20(4).
    """
    matriz = _a_float_array(matriz_retornos)
    if matriz.ndim != 2 or matriz.shape[1] < 2:
        raise ValueError("matriz_retornos debe ser 2D con al menos 2 columnas")
    if n_bloques < 2 or n_bloques % 2:
        raise ValueError("n_bloques debe ser par y >= 2")
    if np.isnan(matriz).any():
        raise ValueError("matriz_retornos contiene NaN; rellénalos antes (p. ej. con 0)")
    filas, n_config = matriz.shape
    if filas < n_bloques:
        raise ValueError("Hay menos filas que bloques")
    tamano = filas // n_bloques
    matriz = matriz[filas - tamano * n_bloques :]
    bloques = [matriz[i * tamano : (i + 1) * tamano] for i in range(n_bloques)]

    mitad = n_bloques // 2
    total = math.comb(n_bloques, mitad)
    if total <= max_combinaciones:
        combinaciones = list(itertools.combinations(range(n_bloques), mitad))
    else:
        rng = np.random.default_rng(semilla)
        vistas: set[tuple[int, ...]] = set()
        while len(vistas) < max_combinaciones:
            vistas.add(tuple(sorted(rng.choice(n_bloques, size=mitad, replace=False).tolist())))
        combinaciones = sorted(vistas)

    lambdas: list[float] = []
    pares: list[tuple[float, float]] = []
    for comb in combinaciones:
        resto = [i for i in range(n_bloques) if i not in comb]
        m_is = np.asarray(metrica(np.vstack([bloques[i] for i in comb])), dtype=float)
        m_oos = np.asarray(metrica(np.vstack([bloques[i] for i in resto])), dtype=float)
        m_is = np.where(np.isnan(m_is), -np.inf, m_is)
        m_oos = np.where(np.isnan(m_oos), -np.inf, m_oos)
        n_estrella = int(np.argmax(m_is))
        rango = stats.rankdata(m_oos)[n_estrella]
        omega = rango / (n_config + 1.0)
        lambdas.append(float(math.log(omega / (1.0 - omega))))
        pares.append((float(m_is[n_estrella]), float(m_oos[n_estrella])))

    arreglo_lambdas = np.asarray(lambdas)
    es = np.array([p[0] for p in pares])
    oos = np.array([p[1] for p in pares])
    finitos = np.isfinite(es) & np.isfinite(oos)
    if finitos.sum() >= 3 and np.ptp(es[finitos]) > 0:
        pendiente = float(stats.linregress(es[finitos], oos[finitos]).slope)
    else:
        pendiente = float("nan")
    return {
        "pbo": float(np.mean(arreglo_lambdas <= 0)),
        "lambdas": lambdas,
        "pares_is_oos": pares,
        "pendiente": pendiente,
        "prob_perdida_oos": float(np.mean(oos < 0)),
        "n_combinaciones": len(combinaciones),
        "n_bloques": n_bloques,
        "n_configuraciones": n_config,
    }


def prueba_rachas(signos: Any) -> dict[str, float]:
    """
    Prueba de rachas de Wald-Wolfowitz sobre una secuencia de signos (los ceros se descartan).

    R = número de rachas; μ = 2·n₊·n₋/n + 1; σ² = 2·n₊·n₋·(2·n₊·n₋ − n) / (n²·(n−1));
    Z = (R − μ)/σ; p bilateral = 2·(1 − Φ(|Z|)). Z < 0 indica rachas de menos (persistencia:
    los signos se agrupan); Z > 0, alternancia excesiva.

    Returns:
        dict con rachas, mu, sigma, z, p_valor, n_positivos, n_negativos. NaN si falta algún signo.
    """
    datos = _a_float_array(signos)
    datos = datos[~np.isnan(datos)]
    datos = datos[datos != 0]
    positivos = datos > 0
    n = len(datos)
    n_pos = int(positivos.sum())
    n_neg = n - n_pos
    nan = float("nan")
    if n_pos == 0 or n_neg == 0 or n < 3:
        return {"rachas": 1 if n else 0, "mu": nan, "sigma": nan, "z": nan, "p_valor": nan,
                "n_positivos": n_pos, "n_negativos": n_neg}
    rachas = 1 + int(np.sum(positivos[1:] != positivos[:-1]))
    mu = 2.0 * n_pos * n_neg / n + 1.0
    varianza = 2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n) / (n**2 * (n - 1))
    if varianza <= 0:
        return {"rachas": rachas, "mu": mu, "sigma": 0.0, "z": nan, "p_valor": nan,
                "n_positivos": n_pos, "n_negativos": n_neg}
    sigma = math.sqrt(varianza)
    z = (rachas - mu) / sigma
    return {"rachas": rachas, "mu": mu, "sigma": sigma, "z": float(z),
            "p_valor": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "n_positivos": n_pos, "n_negativos": n_neg}


def indice_ulcer(equity: Any) -> float:
    """
    Índice Ulcer = sqrt(media(drawdown²)), con el drawdown en % respecto al máximo previo
    (drawdown% = 100·(equity/máximo acumulado − 1)). Penaliza la profundidad y la duración
    de los drawdowns. `equity` es la curva de capital (positiva).
    """
    valores = _a_float_array(equity)
    valores = valores[~np.isnan(valores)]
    if len(valores) == 0:
        return float("nan")
    maximo = np.maximum.accumulate(valores)
    drawdown = 100.0 * (valores / maximo - 1.0)
    return float(math.sqrt(np.mean(drawdown**2)))


def ratio_martin(retornos: Any, periodos_anio: int = DIAS_ANIO) -> float:
    """
    Ratio de Martin = retorno anualizado (%) / índice Ulcer (%).

    Retorno anualizado geométrico = (Π(1+r))^(periodos_anio/T) − 1; el Ulcer se calcula sobre la
    equity acumulada, que parte de 1 (el capital inicial cuenta como punto de la curva).
    Sin drawdowns (Ulcer = 0) devuelve NaN.
    """
    r = _a_float_array(retornos)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    ulcer = indice_ulcer(equity)
    if not np.isfinite(ulcer) or ulcer <= 0 or equity[-1] <= 0:
        return float("nan")
    anual = equity[-1] ** (periodos_anio / len(r)) - 1.0
    return float(100.0 * anual / ulcer)


def p_valor_permutacion(valor_real: float, valores_permutados: Any) -> float:
    """
    p-valor unilateral de un test de permutación: (1 + #{perm ≥ real}) / (1 + n).

    La corrección +1 cuenta el dato real como una permutación más: el p-valor nunca es 0
    (mínimo 1/(n+1)), que es lo correcto para un test de Monte Carlo y evita declarar
    significativo lo que solo es «ninguna permutación lo superó» con pocas permutaciones.
    Ignora NaN en los permutados.
    """
    permutados = _a_float_array(valores_permutados)
    permutados = permutados[~np.isnan(permutados)]
    return float((1 + np.sum(permutados >= valor_real)) / (1 + len(permutados)))
