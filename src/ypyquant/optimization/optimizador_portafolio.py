"""
Optimizador de portafolios: MPT, PMPT, frontera eficiente, objetivos financieros y modelos.

Origen: la lógica de cálculo de GrowSave `app/services/optimizacion/optimizacion_ev.py`
(PortfolioOptimizationService), separada de lo que es de la aplicación (sesión de BD, ORM
de carteras, ModelDataProvider, tasa libre de riesgo descargada).

Responsabilidades:
- Funciones puras sobre una matriz de retornos diarios (columnas = tickers):
  · Markowitz: optimizar_max_sharpe_scipy / _pypfopt, optimizar_min_volatilidad,
    optimizar_para_retorno_objetivo, optimizar_risk_parity, optimizar_portafolio,
    calcular_frontera_eficiente, optimizar_volatilidad_objetivo, metricas_cartera.
  · CML: ajustar_portafolio_a_volatilidad (cartera tangente + liquidez).
  · Objetivos financieros: calcular_retorno_objetivo_desde_metas, validar_retorno_objetivo.
  · PMPT: calcular_semivarianza, calcular_cvar, calcular_ratio_sortino, optimizar_pmpt.
  · Modelos modulares: crear_modelo_personalizado, modelos_disponibles.
- `PortfolioOptimizationService`: la clase de GrowSave con las mismas firmas (tickers,
  start_date, end_date, ...), que carga precios con un `cargador_precios` inyectado y delega
  en las funciones puras. GrowSave la subclasifica para añadir `db`, get_risk_free_rate,
  get_portfolio_metrics (ORM) y los datos externos de optimize_advanced
  (gancho `_obtener_datos_modelo`).

Convenciones (las de producción): medias y covarianzas anualizadas con 252 días; el Sharpe
de las funciones Markowitz es retorno / volatilidad (sin restar r_f), salvo donde se indica.
Los pesos se etiquetan con las columnas de la matriz de retornos (si el preprocesador
descarta un ticker por baja calidad, no aparece en los pesos).

Se relaciona con:
- ypyquant.optimization.portfolio_models (PortfolioOptimizationModel y componentes).
- ypyquant.optimization.presets (obtener_preset, listar_presets, obtener_detalles_preset).
- ypyquant.data.matriz_precios (carga de precios y retornos).
- ypyquant.utils.serializacion (resultados serializables a JSON).
"""

import hashlib
import json
import logging
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize

from ypyquant.data.matriz_precios import CargadorPrecios, calcular_matriz_retornos, construir_matriz_precios
from ypyquant.optimization.data_preprocessing import (
    DiasTrading,
    Periodo,
    TipoRetorno,
    anualizar_covarianza,
    anualizar_retorno_medio,
    anualizar_volatilidad,
)
from ypyquant.optimization.portfolio_models import (
    BayesSteinReturnsModel,
    BlackLittermanReturnsModel,
    BLMomentumReturnsModel,
    BLPriorReturnsModel,
    CAPMReturnsModel,
    ExponentialCovarianceModel,
    HistoricalReturnsModel,
    LedoitWolfShrinkageModel,
    MarkowitzOptimizer,
    MaxSharpeQPOptimizer,
    MichaudOptimizer,
    OptimizerType,
    PortfolioOptimizationModel,
    ReturnModelType,
    RiskModelType,
    RiskParityOptimizer,
    RobustoCajaOptimizer,
    RobustoElipsoideOptimizer,
    SampleCovarianceModel,
    SemiCovarianceModel,
)
from ypyquant.utils.serializacion import convertir_numpy_a_nativo

logger = logging.getLogger(__name__)

try:
    from pypfopt import EfficientFrontier
    PYPFOPT_AVAILABLE = True
except ImportError:  # pragma: no cover - PyPortfolioOpt es dependencia del paquete
    PYPFOPT_AVAILABLE = False
    logger.warning("PyPortfolioOpt no instalado: se usa el optimizador scipy de reserva")

_DIAS = 252  # Días de trading por año


# ============================================================================
# UTILIDADES
# ============================================================================


def _limites(n_activos: int, restricciones: dict | None, minimo: float = 0.0) -> tuple:
    """Límites (minimo, max_weight) por activo; max_weight = 1 si no se indica."""
    max_peso = restricciones.get('max_weight', 1.0) if restricciones else 1.0
    return tuple((minimo, max_peso) for _ in range(n_activos))


def _pesos_iguales(n_activos: int) -> np.ndarray:
    """Vector 1/n."""
    return np.array([1 / n_activos] * n_activos)


def metricas_cartera(pesos: np.ndarray, medias_anuales: pd.Series, cov_anual: pd.DataFrame) -> tuple[float, float, float]:
    """
    Retorno esperado, volatilidad y Sharpe (sin r_f) de una cartera.

    Args:
        pesos: vector de pesos alineado con medias_anuales / cov_anual.
        medias_anuales: retornos medios anualizados.
        cov_anual: covarianza anualizada.
    """
    pesos = np.asarray(pesos, dtype=float)
    retorno = float(np.sum(medias_anuales * pesos))
    volatilidad = float(np.sqrt(np.dot(pesos.T, np.dot(cov_anual, pesos))))
    sharpe = retorno / volatilidad if volatilidad > 0 else 0
    return retorno, volatilidad, float(sharpe)


def _pesos_a_dict(columnas, pesos: np.ndarray) -> dict[str, float]:
    """{ticker: peso} con floats nativos."""
    return {ticker: float(w) for ticker, w in zip(columnas, pesos, strict=True)}


# ============================================================================
# MARKOWITZ
# ============================================================================


def optimizar_max_sharpe_scipy(retornos_df: pd.DataFrame, restricciones: dict | None = None) -> np.ndarray:
    """Máximo Sharpe (retorno/vol, sin r_f) con SLSQP, pesos en [0, max_weight] y suma 1."""
    medias = retornos_df.mean()
    cov = retornos_df.cov()
    n = len(retornos_df.columns)

    def sharpe_negativo(pesos):
        retorno = np.sum(medias * pesos) * _DIAS
        vol = np.sqrt(np.dot(pesos.T, np.dot(cov * _DIAS, pesos)))
        return -(retorno / vol)

    resultado = minimize(
        sharpe_negativo, _pesos_iguales(n), method='SLSQP', bounds=_limites(n, restricciones),
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
    )
    return resultado.x


def optimizar_min_volatilidad(retornos_df: pd.DataFrame, restricciones: dict | None = None) -> np.ndarray:
    """Mínima volatilidad anualizada con SLSQP, pesos en [0, max_weight] y suma 1."""
    cov = retornos_df.cov()
    n = len(retornos_df.columns)

    def volatilidad(pesos):
        return np.sqrt(np.dot(pesos.T, np.dot(cov * _DIAS, pesos)))

    resultado = minimize(
        volatilidad, _pesos_iguales(n), method='SLSQP', bounds=_limites(n, restricciones),
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
    )
    return resultado.x


def optimizar_para_retorno_objetivo(
    medias_anuales: pd.Series, cov_anual: pd.DataFrame, retorno_objetivo: float
) -> np.ndarray:
    """
    Mínima volatilidad con retorno esperado = retorno_objetivo (pesos en [0, 1], suma 1).
    Si el optimizador no converge devuelve pesos iguales (comportamiento de producción).
    """
    n = len(medias_anuales)

    def volatilidad(pesos):
        return np.sqrt(np.dot(pesos.T, np.dot(cov_anual, pesos)))

    resultado = minimize(
        volatilidad, _pesos_iguales(n), method='SLSQP', bounds=tuple((0, 1) for _ in range(n)),
        constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(medias_anuales * w) - retorno_objetivo},
        ],
        options={'maxiter': 1000},
    )
    if not resultado.success:
        return _pesos_iguales(n)
    return resultado.x


def optimizar_max_sharpe_pypfopt(
    medias_anuales: pd.Series, cov_anual: pd.DataFrame, restricciones: dict | None = None
) -> np.ndarray:
    """Máximo Sharpe con PyPortfolioOpt (r_f por defecto de pypfopt) y pesos limpiados."""
    ef = EfficientFrontier(medias_anuales, cov_anual)
    if restricciones and 'max_weight' in restricciones:
        max_peso = restricciones['max_weight']
        ef.add_constraint(lambda w: w <= max_peso)
    ef.max_sharpe()
    limpios = ef.clean_weights()
    return np.array([limpios[col] for col in medias_anuales.index])


def optimizar_risk_parity(cov_anual: pd.DataFrame, restricciones: dict | None = None) -> np.ndarray:
    """Paridad de riesgo: contribuciones al riesgo iguales (mínimos cuadrados, SLSQP)."""
    n = len(cov_anual)

    def objetivo(pesos):
        vol = np.sqrt(np.dot(pesos.T, np.dot(cov_anual, pesos)))
        contribucion = pesos * (np.dot(cov_anual, pesos) / vol)
        return np.sum((contribucion - vol / n) ** 2)

    resultado = minimize(
        objetivo, _pesos_iguales(n), method='SLSQP', bounds=_limites(n, restricciones),
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
    )
    return resultado.x


def optimizar_portafolio(
    retornos_df: pd.DataFrame, objective: str = 'max_sharpe', constraints: dict | None = None
) -> dict:
    """
    Optimiza según `objective`: 'max_sharpe' (pypfopt si está, si no scipy), 'min_vol',
    'equal_weight' o 'risk_parity'.

    Returns:
        {weights, expected_return, volatility, sharpe_ratio} (anualizados; Sharpe sin r_f).
    """
    medias = retornos_df.mean() * _DIAS
    cov = retornos_df.cov() * _DIAS
    n = len(retornos_df.columns)

    if objective == 'max_sharpe':
        pesos = (
            optimizar_max_sharpe_pypfopt(medias, cov, constraints) if PYPFOPT_AVAILABLE
            else optimizar_max_sharpe_scipy(retornos_df, constraints)
        )
    elif objective == 'min_vol':
        pesos = optimizar_min_volatilidad(retornos_df, constraints)
    elif objective == 'equal_weight':
        pesos = _pesos_iguales(n)
    elif objective == 'risk_parity':
        pesos = optimizar_risk_parity(cov, constraints)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    retorno, vol, sharpe = metricas_cartera(pesos, medias, cov)
    return {
        'weights': _pesos_a_dict(retornos_df.columns, pesos),
        'expected_return': retorno,
        'volatility': vol,
        'sharpe_ratio': sharpe,
    }


def calcular_frontera_eficiente(retornos_df: pd.DataFrame, num_points: int = 50) -> list[dict]:
    """
    Frontera eficiente: `num_points` retornos objetivo equiespaciados entre la menor y la
    mayor media anualizada; para cada uno, cartera de mínima volatilidad.

    Returns:
        [{expected_return, volatility, sharpe_ratio, weights, is_optimal}], con is_optimal en
        el punto de máximo Sharpe.
    """
    medias = retornos_df.mean() * _DIAS
    cov = retornos_df.cov() * _DIAS
    puntos = []
    for objetivo in np.linspace(medias.min(), medias.max(), num_points):
        try:
            pesos = optimizar_para_retorno_objetivo(medias, cov, objetivo)
        except Exception as exc:  # noqa: BLE001 — un punto fallido no invalida la frontera
            logger.warning(f"Error en el punto de frontera con retorno {objetivo}: {exc}")
            continue
        retorno, vol, sharpe = metricas_cartera(pesos, medias, cov)
        puntos.append({
            'expected_return': retorno,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'weights': _pesos_a_dict(retornos_df.columns, pesos),
            'is_optimal': False,
        })
    if puntos:
        mejor = max(range(len(puntos)), key=lambda i: puntos[i]['sharpe_ratio'])
        puntos[mejor]['is_optimal'] = True
    return puntos


def optimizar_volatilidad_objetivo(
    retornos_df: pd.DataFrame, target_volatility: float, allow_leverage: bool = False
) -> dict:
    """
    Máximo retorno con volatilidad <= target_volatility. Sin apalancamiento: pesos [0, 1] y
    suma 1; con apalancamiento: pesos [-2, 2] sin restricción de suma.

    Raises:
        ValueError si el optimizador no converge.
    """
    medias = retornos_df.mean() * _DIAS
    cov = retornos_df.cov() * _DIAS
    n = len(retornos_df.columns)

    def restriccion_vol(pesos):
        return target_volatility - np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))

    if allow_leverage:
        restricciones = [{'type': 'ineq', 'fun': restriccion_vol}]
        limites = tuple((-2, 2) for _ in range(n))
    else:
        restricciones = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, {'type': 'ineq', 'fun': restriccion_vol}]
        limites = tuple((0, 1) for _ in range(n))

    resultado = minimize(
        lambda w: -np.sum(medias * w), _pesos_iguales(n), method='SLSQP',
        bounds=limites, constraints=restricciones, options={'maxiter': 1000},
    )
    if not resultado.success:
        raise ValueError("Could not optimize for target volatility. Try adjusting target.")

    retorno, vol, sharpe = metricas_cartera(resultado.x, medias, cov)
    return {
        'weights': _pesos_a_dict(retornos_df.columns, resultado.x),
        'expected_return': retorno,
        'volatility': vol,
        'sharpe_ratio': sharpe,
        'target_volatility': target_volatility,
        'volatility_achieved': bool(abs(vol - target_volatility) < 0.001),
    }


def ajustar_portafolio_a_volatilidad(
    tangent_weights: dict[str, float],
    tangent_vol: float,
    target_vol: float,
    tangent_return: float,
    risk_free_rate: float,
) -> dict:
    """
    Capital Market Line: mezcla la cartera tangente con liquidez para alcanzar target_vol.
    w_tangente = σ_obj / σ_tangente; E[R] = w·E[R_t] + (1 − w)·r_f.

    Raises:
        ValueError si target_vol > tangent_vol (apalancamiento no permitido).
    """
    if target_vol > tangent_vol:
        raise ValueError(
            f"Target volatility ({target_vol:.2%}) mayor que tangent portfolio volatility "
            f"({tangent_vol:.2%}). Apalancamiento no permitido. Use target_vol <= {tangent_vol:.2%}"
        )
    fraccion = target_vol / tangent_vol if tangent_vol > 0 else 0
    peso_liquidez = 1.0 - fraccion
    return convertir_numpy_a_nativo({
        'weights': {t: w * fraccion for t, w in tangent_weights.items()},
        'cash_weight': float(peso_liquidez),
        'expected_return': float(fraccion * tangent_return + peso_liquidez * risk_free_rate),
        'expected_volatility': float(target_vol),
        'includes_cash': True,
    })


# ============================================================================
# OBJETIVOS FINANCIEROS
# ============================================================================


def calcular_retorno_objetivo_desde_metas(
    capital_inicial: float, aportacion_mensual: float, meses: int, objetivo_final: float
) -> dict:
    """
    Tasa anual necesaria para llegar a `objetivo_final` con FV = PV(1+r)^n + PMT·((1+r)^n − 1)/r
    (r mensual, resuelta con fsolve desde varios puntos iniciales).

    Returns:
        {target_return_annual, target_return_monthly, total_aportado, ganancia_necesaria,
         feasibility_warning, validation_errors}
    """
    errores = []
    if capital_inicial < 0:
        errores.append("capital_inicial no puede ser negativo")
    if aportacion_mensual < 0:
        errores.append("aportacion_mensual no puede ser negativa")
    if meses <= 0:
        errores.append("meses debe ser positivo")
    if objetivo_final <= 0:
        errores.append("objetivo_final debe ser positivo")
    if errores:
        return {
            'target_return_annual': None, 'target_return_monthly': None, 'total_aportado': None,
            'ganancia_necesaria': None, 'feasibility_warning': None, 'validation_errors': errores,
        }

    total_aportado = capital_inicial + aportacion_mensual * meses
    ganancia_necesaria = objetivo_final - total_aportado
    if ganancia_necesaria <= 0:
        return {
            'target_return_annual': 0.0, 'target_return_monthly': 0.0,
            'total_aportado': float(total_aportado), 'ganancia_necesaria': float(ganancia_necesaria),
            'feasibility_warning': 'No se necesita retorno de inversión para alcanzar el objetivo',
            'validation_errors': [],
        }

    def ecuacion_valor_futuro(r):
        if r == 0:
            return capital_inicial + aportacion_mensual * meses - objetivo_final
        valor = capital_inicial * (1 + r) ** meses + aportacion_mensual * (((1 + r) ** meses - 1) / r)
        return valor - objetivo_final

    r_mensual = None
    for inicial in (0.005, 0.01, 0.02, 0.001):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solucion = fsolve(ecuacion_valor_futuro, inicial, full_output=True)
        except Exception:  # noqa: BLE001, S112 — se prueba el siguiente punto inicial
            continue
        if solucion[2] == 1:
            r_mensual = solucion[0][0]
            if r_mensual > -0.5:
                break
    if r_mensual is None:
        error = "No se pudo encontrar solución numérica"
        return {
            'target_return_annual': None, 'target_return_monthly': None,
            'total_aportado': float(total_aportado), 'ganancia_necesaria': float(ganancia_necesaria),
            'feasibility_warning': f'No se pudo calcular tasa de retorno necesaria: {error}',
            'validation_errors': [error],
        }

    r_anual = (1 + r_mensual) ** 12 - 1
    aviso = None
    if r_anual > 0.30:
        aviso = (f"⚠️ OBJETIVO IRREAL: Se requiere {r_anual:.1%} anual. Retornos >30% anuales son "
                 f"extremadamente improbables para carteras diversificadas. Considere aumentar aportaciones o plazo.")
    elif r_anual > 0.25:
        aviso = (f"⚠️ OBJETIVO MUY AGRESIVO: Se requiere {r_anual:.1%} anual. Retornos >25% anuales "
                 f"requieren riesgo muy alto. Revise si es realista para su perfil.")
    elif r_anual > 0.15:
        aviso = (f"⚠️ OBJETIVO AGRESIVO: Se requiere {r_anual:.1%} anual. "
                 f"Este retorno requiere asumir riesgo significativo.")
    elif r_anual < -0.05:
        aviso = (f"⚠️ OBJETIVO CON MARGEN: Puede alcanzar su objetivo incluso con pérdidas de hasta "
                 f"{abs(r_anual):.1%} anual. Considere estrategias más conservadoras.")

    return {
        'target_return_annual': float(r_anual),
        'target_return_monthly': float(r_mensual),
        'total_aportado': float(total_aportado),
        'ganancia_necesaria': float(ganancia_necesaria),
        'feasibility_warning': aviso,
        'validation_errors': [],
    }


def validar_retorno_objetivo(target_return: float, retornos_df: pd.DataFrame, context: str = "optimización") -> dict:
    """
    Comprueba que un retorno objetivo anual sea realista (>30 % es error; >25 %, >15 %,
    < −10 % y comparaciones con el histórico son avisos).

    Returns:
        {is_valid, warnings, errors, historical_comparison}
    """
    avisos, errores = [], []
    medias = retornos_df.mean() * _DIAS
    media_hist, max_hist, min_hist = medias.mean(), medias.max(), medias.min()

    if target_return > 0.30:
        errores.append(f"Target return {target_return:.1%} es irreal (>30% anual). No proceder con {context}.")
    elif target_return > 0.25:
        avisos.append(f"Target return {target_return:.1%} es muy agresivo (>25% anual). Revise expectativas.")
    elif target_return > 0.15:
        avisos.append(f"Target return {target_return:.1%} es agresivo. Requerirá riesgo significativo.")
    if target_return < -0.10:
        avisos.append(f"Target return {target_return:.1%} es negativo. ¿Está seguro de esperar pérdidas?")

    if target_return > max_hist * 1.5:
        avisos.append(
            f"Target return {target_return:.1%} es {target_return / max_hist:.1f}x el mejor activo histórico "
            f"({max_hist:.1%}). Muy improbable de alcanzar."
        )
    elif target_return > media_hist * 2:
        avisos.append(
            f"Target return {target_return:.1%} es 2x el promedio histórico ({media_hist:.1%}). "
            f"Considere si es realista."
        )

    return {
        'is_valid': len(errores) == 0,
        'warnings': avisos,
        'errors': errores,
        'historical_comparison': {
            'target_return': float(target_return),
            'historical_avg': float(media_hist),
            'historical_max': float(max_hist),
            'historical_min': float(min_hist),
            'target_vs_avg_ratio': float(target_return / media_hist) if media_hist != 0 else None,
        },
    }


# ============================================================================
# PMPT
# ============================================================================


def calcular_semivarianza(
    retornos_df: pd.DataFrame, target_return: float = 0.0, annualized_target: bool = True
) -> pd.DataFrame:
    """
    Matriz de semicovarianza anualizada respecto al MAR: los retornos por encima del umbral
    diario se ponen a 0 y se calcula la covarianza × 252.
    """
    umbral = target_return / _DIAS if annualized_target else target_return
    bajistas = retornos_df.copy()
    bajistas[bajistas > umbral] = 0
    return bajistas.cov() * _DIAS


def calcular_cvar(
    retornos_df: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.95,
    target_return: float | None = None,
    time_horizon_days: int | None = None,
) -> dict:
    """
    VaR/CVaR históricos de la cartera; con target_return, shortfall respecto al objetivo.
    Escalado √252 (anual) y √horizonte (si se da time_horizon_days).

    Returns:
        {cvar, var, cvar_horizon, shortfall_risk, interpretation, tail_mean, confidence_level}
    """
    retornos_cartera = (retornos_df * weights).sum(axis=1)
    if target_return is not None:
        retornos_cartera = retornos_cartera - target_return / _DIAS

    var = np.percentile(retornos_cartera, (1 - confidence_level) * 100)
    cola = retornos_cartera[retornos_cartera <= var]
    cvar_diario = var if len(cola) == 0 else cola.mean()

    cvar_anual = float(cvar_diario * np.sqrt(_DIAS))
    var_anual = float(var * np.sqrt(_DIAS))
    cvar_horizonte = float(cvar_diario * np.sqrt(time_horizon_days)) if time_horizon_days is not None else None

    cola_pct = (1 - confidence_level) * 100
    if target_return is not None:
        interpretacion = (
            f"Shortfall Risk (CVaR): En el peor {cola_pct:.0f}% de escenarios, el retorno quedará "
            f"{abs(cvar_anual):.2%} por debajo del objetivo {target_return:.2%} anual."
        )
    else:
        interpretacion = (
            f"CVaR: En el peor {cola_pct:.0f}% de escenarios, la pérdida promedio será de {abs(cvar_anual):.2%} anual."
        )
    if time_horizon_days is not None:
        interpretacion += f" En un horizonte de {time_horizon_days / _DIAS:.1f} años: {abs(cvar_horizonte):.2%}."

    return {
        'cvar': cvar_anual,
        'var': var_anual,
        'cvar_horizon': cvar_horizonte,
        'shortfall_risk': bool(target_return is not None),
        'interpretation': interpretacion,
        'tail_mean': float(cvar_diario * _DIAS),
        'confidence_level': confidence_level,
    }


def calcular_ratio_sortino(
    retornos_df: pd.DataFrame, weights: np.ndarray, target_return: float = 0.0, risk_free_rate: float = 0.02
) -> float:
    """
    Sortino = (media anual − r_f) / desviación bajista anual. `target_return` es el umbral
    DIARIO con el que se comparan los retornos diarios (los de arriba cuentan como 0).
    """
    retornos_cartera = (retornos_df * weights).sum(axis=1)
    media_anual = retornos_cartera.mean() * _DIAS
    bajistas = retornos_cartera.copy()
    bajistas[bajistas > target_return] = 0
    desviacion = bajistas.std() * np.sqrt(_DIAS)
    if desviacion == 0:
        return 0.0
    return float((media_anual - risk_free_rate) / desviacion)


def optimizar_pmpt(
    retornos_df: pd.DataFrame,
    objective: str = 'min_semivariance',
    target_return: float | None = None,
    confidence_level: float = 0.95,
    constraints: dict | None = None,
    capital_inicial: float | None = None,
    aportacion_mensual: float | None = None,
    meses: int | None = None,
    objetivo_final: float | None = None,
    validate_target: bool = True,
) -> dict:
    """
    Optimización Post-Modern Portfolio Theory: 'min_semivariance', 'max_sortino' o 'min_cvar'.

    El MAR (target_return anual) puede darse o calcularse desde objetivos financieros
    (capital_inicial, aportacion_mensual, meses, objetivo_final); si no hay nada, 0.0. Si no
    converge se devuelven pesos iguales (optimization_converged=False).

    Raises:
        ValueError si los objetivos son inválidos, el MAR no es realista o el objetivo no existe.
    """
    n = len(retornos_df.columns)

    info_objetivo = None
    if target_return is None:
        if all(p is not None for p in (capital_inicial, aportacion_mensual, meses, objetivo_final)):
            info_objetivo = calcular_retorno_objetivo_desde_metas(capital_inicial, aportacion_mensual, meses, objetivo_final)
            if info_objetivo['validation_errors']:
                raise ValueError(f"Errores en parámetros de objetivos: {info_objetivo['validation_errors']}")
            target_return = info_objetivo['target_return_annual']
            logger.info(f"[PMPT] Target return calculado desde objetivos: {target_return:.2%} anual")
            if info_objetivo['feasibility_warning']:
                logger.info(f"[PMPT] {info_objetivo['feasibility_warning']}")
        else:
            target_return = 0.0

    validacion = None
    if validate_target and target_return != 0.0:
        validacion = validar_retorno_objetivo(target_return, retornos_df, context=f"optimización PMPT ({objective})")
        for aviso in validacion['warnings']:
            logger.warning(f"[PMPT] {aviso}")
        if not validacion['is_valid']:
            raise ValueError(
                f"Target return {target_return:.2%} no es válido. Ajuste expectativas o parámetros de objetivos."
            )

    horizonte = int((meses / 12) * _DIAS) if meses is not None else None
    umbral_diario = target_return / _DIAS

    if objective == 'min_semivariance':
        semivar = calcular_semivarianza(retornos_df, target_return)

        def funcion_objetivo(pesos):
            return np.sqrt(np.dot(pesos.T, np.dot(semivar, pesos)))
    elif objective == 'max_sortino':
        def funcion_objetivo(pesos):
            return -calcular_ratio_sortino(retornos_df, pesos, umbral_diario)
    elif objective == 'min_cvar':
        def funcion_objetivo(pesos):
            return -calcular_cvar(
                retornos_df, pesos, confidence_level, target_return=target_return, time_horizon_days=horizonte
            )['cvar']
    else:
        raise ValueError(f"Unknown PMPT objective: {objective}")

    limites = tuple((0, constraints['max_weight']) for _ in range(n)) if constraints and 'max_weight' in constraints \
        else tuple((0, 1) for _ in range(n))
    iniciales = _pesos_iguales(n)
    resultado = minimize(
        funcion_objetivo, iniciales, method='SLSQP', bounds=limites,
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}], options={'maxiter': 1000},
    )
    if not resultado.success:
        logger.warning("[PMPT] La optimización no convergió; se usan pesos iguales")
    pesos = resultado.x if resultado.success else iniciales

    medias = retornos_df.mean() * _DIAS
    cov = retornos_df.cov() * _DIAS
    semivar = calcular_semivarianza(retornos_df, target_return)
    retorno, vol, sharpe = metricas_cartera(pesos, medias, cov)
    vol_bajista = float(np.sqrt(np.dot(pesos.T, np.dot(semivar, pesos))))

    return {
        'weights': _pesos_a_dict(retornos_df.columns, pesos),
        'expected_return': retorno,
        'volatility': vol,
        'downside_volatility': vol_bajista,
        'sharpe_ratio': sharpe,
        'sortino_ratio': float(calcular_ratio_sortino(retornos_df, pesos, umbral_diario)),
        'cvar_info': calcular_cvar(
            retornos_df, pesos, confidence_level, target_return=target_return, time_horizon_days=horizonte
        ),
        'optimization_method': objective,
        'target_return': float(target_return),
        'target_return_info': info_objetivo,
        'validation': validacion,
        'optimization_converged': bool(resultado.success),
    }


# ============================================================================
# MODELOS MODULARES
# ============================================================================


# Tipos que `crear_modelo_personalizado` sabe construir (única fuente de verdad de `modelos_disponibles`).
_RETORNOS_CONSTRUIBLES = (
    ReturnModelType.HISTORICAL_MEAN, ReturnModelType.HISTORICAL_MEDIAN, ReturnModelType.CAPM,
    ReturnModelType.BLACK_LITTERMAN, ReturnModelType.BAYES_STEIN, ReturnModelType.BL_PRIOR,
    ReturnModelType.BL_MOMENTUM,
)
_RIESGO_CONSTRUIBLES = (
    RiskModelType.SAMPLE_COVARIANCE, RiskModelType.LEDOIT_WOLF, RiskModelType.EXPONENTIAL_WEIGHTED,
    RiskModelType.SEMI_COVARIANCE,
)
_OPTIMIZADORES_CONSTRUIBLES = (
    OptimizerType.MAX_SHARPE, OptimizerType.MIN_VOLATILITY, OptimizerType.RISK_PARITY,
    OptimizerType.MAX_SHARPE_QP, OptimizerType.MICHAUD, OptimizerType.EFFICIENT_RISK,
    OptimizerType.ROBUSTO_ELIPSOIDE, OptimizerType.ROBUSTO_CAJA,
)

# Parámetros de configuración admitidos por cada optimizador (el resto se descarta con un aviso).
_PARAMETROS_OPTIMIZADOR = {
    OptimizerType.MAX_SHARPE: ('risk_free_rate',),
    OptimizerType.MIN_VOLATILITY: ('risk_free_rate',),
    OptimizerType.RISK_PARITY: (),
    OptimizerType.MAX_SHARPE_QP: ('risk_free_rate', 'penalizacion_turnover', 'volatilidad_objetivo'),
    OptimizerType.MICHAUD: (
        'n_replicas', 'metodo_bootstrap', 'longitud_media_bloque', 'frequency', 'risk_free_rate', 'seed',
        'volatilidad_objetivo',
    ),
    # Máxima rentabilidad con σ <= target_volatility: fija el riesgo con cualquier modelo de µ
    OptimizerType.EFFICIENT_RISK: ('risk_free_rate', 'target_volatility'),
    # Incertidumbre en µ (v0.5.0): elipsoide (kappa/confianza/tipo_theta) y caja (z/confianza)
    OptimizerType.ROBUSTO_ELIPSOIDE: (
        'kappa', 'confianza', 'tipo_theta', 'aversion', 'volatilidad_objetivo', 'frequency', 'risk_free_rate',
        'n_observaciones',
    ),
    OptimizerType.ROBUSTO_CAJA: (
        'z', 'confianza', 'aversion', 'volatilidad_objetivo', 'frequency', 'risk_free_rate', 'n_observaciones',
    ),
}


def crear_modelo_personalizado(
    name: str,
    returns_model_type: str,
    risk_model_type: str,
    optimizer_type: str,
    returns_config: dict | None = None,
    risk_config: dict | None = None,
    optimizer_config: dict | None = None,
    description: str = "",
) -> PortfolioOptimizationModel:
    """
    Construye un PortfolioOptimizationModel a partir de los tipos (ReturnModelType,
    RiskModelType, OptimizerType) y sus configuraciones.

    Raises:
        ValueError si algún tipo no es válido.
    """
    returns_config = returns_config or {}
    risk_config = risk_config or {}
    optimizer_config = optimizer_config or {}

    if returns_model_type == ReturnModelType.HISTORICAL_MEAN:
        modelo_retornos = HistoricalReturnsModel(method="mean", **returns_config)
    elif returns_model_type == ReturnModelType.HISTORICAL_MEDIAN:
        modelo_retornos = HistoricalReturnsModel(method="median", **returns_config)
    elif returns_model_type == ReturnModelType.CAPM:
        modelo_retornos = CAPMReturnsModel(**returns_config)
    elif returns_model_type == ReturnModelType.BLACK_LITTERMAN:
        modelo_retornos = BlackLittermanReturnsModel(**returns_config)
    elif returns_model_type == ReturnModelType.BAYES_STEIN:
        modelo_retornos = BayesSteinReturnsModel(**returns_config)
    elif returns_model_type == ReturnModelType.BL_PRIOR:
        modelo_retornos = BLPriorReturnsModel(**returns_config)
    elif returns_model_type == ReturnModelType.BL_MOMENTUM:
        modelo_retornos = BLMomentumReturnsModel(**returns_config)
    else:
        raise ValueError(f"Unknown returns model type: {returns_model_type}")

    if risk_model_type == RiskModelType.SAMPLE_COVARIANCE:
        modelo_riesgo = SampleCovarianceModel(**risk_config)
    elif risk_model_type == RiskModelType.LEDOIT_WOLF:
        modelo_riesgo = LedoitWolfShrinkageModel(**risk_config)
    elif risk_model_type == RiskModelType.EXPONENTIAL_WEIGHTED:
        modelo_riesgo = ExponentialCovarianceModel(**risk_config)
    elif risk_model_type == RiskModelType.SEMI_COVARIANCE:
        modelo_riesgo = SemiCovarianceModel(**risk_config)
    else:
        raise ValueError(f"Unknown risk model type: {risk_model_type}")

    # Cada optimizador admite su propio conjunto de parámetros (RiskParity ninguno)
    try:
        permitidos = _PARAMETROS_OPTIMIZADOR.get(OptimizerType(optimizer_type), ())
    except ValueError:
        permitidos = ()
    parametros = {k: v for k, v in optimizer_config.items() if k in permitidos}
    descartados = sorted(set(optimizer_config) - set(parametros))
    if descartados:
        logger.warning("[crear_modelo_personalizado] Parámetros de optimizador ignorados: %s", descartados)
    if optimizer_type == OptimizerType.MAX_SHARPE:
        optimizador = MarkowitzOptimizer(objective="max_sharpe", **parametros)
    elif optimizer_type == OptimizerType.MIN_VOLATILITY:
        optimizador = MarkowitzOptimizer(objective="min_volatility", **parametros)
    elif optimizer_type == OptimizerType.EFFICIENT_RISK:
        optimizador = MarkowitzOptimizer(objective="efficient_risk", **parametros)
    elif optimizer_type == OptimizerType.RISK_PARITY:
        optimizador = RiskParityOptimizer()
    elif optimizer_type == OptimizerType.MAX_SHARPE_QP:
        optimizador = MaxSharpeQPOptimizer(**parametros)
    elif optimizer_type == OptimizerType.MICHAUD:
        optimizador = MichaudOptimizer(**parametros)
    elif optimizer_type == OptimizerType.ROBUSTO_ELIPSOIDE:
        optimizador = RobustoElipsoideOptimizer(**parametros)
    elif optimizer_type == OptimizerType.ROBUSTO_CAJA:
        optimizador = RobustoCajaOptimizer(**parametros)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return PortfolioOptimizationModel(
        name=name, returns_model=modelo_retornos, risk_model=modelo_riesgo,
        optimizer=optimizador, description=description,
    )


def modelos_disponibles() -> dict[str, list[str]]:
    """Tipos de modelo de retornos, de riesgo, optimizadores y modelos preconfigurados."""
    return {
        'returns_models': [e.value for e in _RETORNOS_CONSTRUIBLES],
        'risk_models': [e.value for e in _RIESGO_CONSTRUIBLES],
        'optimizers': [e.value for e in _OPTIMIZADORES_CONSTRUIBLES],
        'preconfigured_models': ['Classic Markowitz', 'Robust Risk Parity', 'CAPM Max Sharpe'],
    }


# ============================================================================
# SERVICIO (API de GrowSave sin BD)
# ============================================================================


class PortfolioOptimizationService:
    """
    Servicio de optimización con las firmas de GrowSave: recibe tickers y fechas, carga los
    precios con `cargador_precios` (ticker, inicio, fin) -> OHLCV, calcula retornos con el
    preprocesador centralizado (log por defecto, 95 % de datos válidos) y delega en las
    funciones puras de este módulo.

    Ganchos para la aplicación:
        _obtener_datos_modelo(model, start_date, end_date) -> dict de datos externos que
            necesita el modelo (mercado para CAPM, factores...). Por defecto {}.
        _get_cached_frontier / _store_frontier_results: caché de frontera (desactivada).
    """

    CACHE_TTL_HOURS = 6
    DIAS_TRADING_ANUAL = DiasTrading.ANUAL

    def __init__(
        self,
        cargador_precios: CargadorPrecios | None = None,
        tipo_retorno: Literal["log", "simple"] = "log",
    ):
        self.cargador_precios = cargador_precios
        self.tipo_retorno = tipo_retorno
        self._tipo_retorno_enum = TipoRetorno(tipo_retorno)

    # --- Datos ---------------------------------------------------------------------

    def _cargar_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        if self.cargador_precios is None:
            raise RuntimeError("PortfolioOptimizationService necesita un cargador_precios para cargar datos")
        return self.cargador_precios(ticker, start_date, end_date)

    def _get_prices_matrix(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Matriz de precios ajustados (ValueError si ningún ticker tiene datos)."""
        if self.cargador_precios is None:
            raise RuntimeError("PortfolioOptimizationService necesita un cargador_precios para cargar datos")
        return construir_matriz_precios(tickers, start_date, end_date, self._cargar_ohlcv, estricto=True)

    def _get_returns_matrix(
        self, tickers: list[str], start_date: str, end_date: str,
        tipo_retorno: Literal["log", "simple"] | None = None,
    ) -> pd.DataFrame:
        """Matriz de retornos diarios limpia (ProcesadorDatos, 95 % de datos válidos)."""
        precios = self._get_prices_matrix(tickers, start_date, end_date)
        return calcular_matriz_retornos(
            precios, tipo_retorno=tipo_retorno or self.tipo_retorno, ratio_minimo_validos=0.95, estricto=True
        )

    def _anualizar_media(self, media_retornos: pd.Series) -> pd.Series:
        return anualizar_retorno_medio(media_retornos, tipo_retorno=self._tipo_retorno_enum, periodo_origen=Periodo.DIARIO)

    def _anualizar_volatilidad(self, volatilidad: pd.Series) -> pd.Series:
        return anualizar_volatilidad(volatilidad, periodo_origen=Periodo.DIARIO)

    def _anualizar_covarianza(self, matriz_cov: pd.DataFrame) -> pd.DataFrame:
        return anualizar_covarianza(matriz_cov, periodo_origen=Periodo.DIARIO)

    @staticmethod
    def _calculate_tickers_hash(tickers: list[str], start_date: str, end_date: str) -> str:
        """Hash MD5 de (tickers ordenados, fechas) para claves de caché."""
        texto = json.dumps({'tickers': sorted(tickers), 'start': start_date, 'end': end_date}, sort_keys=True)
        return hashlib.md5(texto.encode()).hexdigest()  # clave de caché, no seguridad

    # --- Ganchos -------------------------------------------------------------------

    def _get_cached_frontier(self, tickers_hash: str) -> list[dict] | None:
        """Caché de frontera desactivada: siempre se recalcula."""
        return None

    def _store_frontier_results(self, frontier_points: list[dict], tickers_hash: str, start_date: str, end_date: str):
        """Caché de frontera desactivada: no se guarda nada."""

    def _obtener_datos_modelo(self, model: PortfolioOptimizationModel, start_date: str, end_date: str) -> dict:
        """Datos externos que necesita el modelo (la aplicación lo sobrescribe)."""
        return {}

    # --- Markowitz -----------------------------------------------------------------

    def adjust_portfolio_to_volatility(
        self, tangent_weights: dict[str, float], tangent_vol: float, target_vol: float,
        tangent_return: float, risk_free_rate: float,
    ) -> dict:
        """Ver ajustar_portafolio_a_volatilidad."""
        return ajustar_portafolio_a_volatilidad(tangent_weights, tangent_vol, target_vol, tangent_return, risk_free_rate)

    def _optimize_max_sharpe_fallback(self, returns_df: pd.DataFrame, constraints: dict | None = None) -> np.ndarray:
        return optimizar_max_sharpe_scipy(returns_df, constraints)

    def _optimize_min_volatility_fallback(self, returns_df: pd.DataFrame, constraints: dict | None = None) -> np.ndarray:
        return optimizar_min_volatilidad(returns_df, constraints)

    def _optimize_for_target_return(
        self, returns_df: pd.DataFrame, mean_returns: pd.Series, cov_matrix: pd.DataFrame, target_return: float
    ) -> np.ndarray:
        return optimizar_para_retorno_objetivo(mean_returns, cov_matrix, target_return)

    def _optimize_max_sharpe_pypfopt(
        self, returns_df: pd.DataFrame, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
        constraints: dict | None = None,
    ) -> np.ndarray:
        return optimizar_max_sharpe_pypfopt(mean_returns, cov_matrix, constraints)

    def _optimize_risk_parity(
        self, returns_df: pd.DataFrame, cov_matrix: pd.DataFrame, constraints: dict | None = None
    ) -> np.ndarray:
        return optimizar_risk_parity(cov_matrix, constraints)

    def calculate_efficient_frontier(
        self, tickers: list[str], start_date: str, end_date: str, num_points: int = 50, use_cache: bool = True
    ) -> list[dict]:
        """Ver calcular_frontera_eficiente (con caché opcional vía ganchos)."""
        clave = self._calculate_tickers_hash(tickers, start_date, end_date)
        if use_cache:
            en_cache = self._get_cached_frontier(clave)
            if en_cache:
                return en_cache
        puntos = calcular_frontera_eficiente(self._get_returns_matrix(tickers, start_date, end_date), num_points)
        if puntos and use_cache:
            self._store_frontier_results(puntos, clave, start_date, end_date)
        return puntos

    def optimize_portfolio(
        self, tickers: list[str], start_date: str, end_date: str,
        objective: str = 'max_sharpe', constraints: dict | None = None,
    ) -> dict:
        """Ver optimizar_portafolio."""
        return optimizar_portafolio(self._get_returns_matrix(tickers, start_date, end_date), objective, constraints)

    def calcular_metricas_cartera(
        self, tickers: list[str], weights: np.ndarray, start_date: str, end_date: str
    ) -> dict:
        """
        Retorno, volatilidad y Sharpe (sin r_f) de unos pesos dados sobre el histórico.
        Es la parte de cálculo de GrowSave get_portfolio_metrics (que lee los pesos del ORM).
        """
        retornos = self._get_returns_matrix(tickers, start_date, end_date)
        pesos = pd.Series(np.asarray(weights, dtype=float), index=tickers).reindex(retornos.columns).fillna(0.0)
        retorno, vol, sharpe = metricas_cartera(pesos.values, retornos.mean() * _DIAS, retornos.cov() * _DIAS)
        return {
            'weights': {t: float(w) for t, w in zip(tickers, weights, strict=True)},
            'expected_return': retorno,
            'volatility': vol,
            'sharpe_ratio': sharpe,
        }

    def optimize_target_volatility(
        self, tickers: list[str], start_date: str, end_date: str,
        target_volatility: float, allow_leverage: bool = False,
    ) -> dict:
        """Ver optimizar_volatilidad_objetivo."""
        return optimizar_volatilidad_objetivo(
            self._get_returns_matrix(tickers, start_date, end_date), target_volatility, allow_leverage
        )

    # --- Objetivos y PMPT ------------------------------------------------------------

    calculate_target_return_from_goals = staticmethod(calcular_retorno_objetivo_desde_metas)
    validate_target_return = staticmethod(validar_retorno_objetivo)

    def calculate_semivariance(
        self, returns_df: pd.DataFrame, target_return: float = 0.0, annualized_target: bool = True
    ) -> pd.DataFrame:
        return calcular_semivarianza(returns_df, target_return, annualized_target)

    def calculate_cvar(
        self, returns_df: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95,
        target_return: float | None = None, time_horizon_days: int | None = None,
    ) -> dict:
        return calcular_cvar(returns_df, weights, confidence_level, target_return, time_horizon_days)

    def calculate_sortino_ratio(
        self, returns_df: pd.DataFrame, weights: np.ndarray, target_return: float = 0.0, risk_free_rate: float = 0.02
    ) -> float:
        return calcular_ratio_sortino(returns_df, weights, target_return, risk_free_rate)

    def optimize_pmpt(
        self, tickers: list[str], start_date: str, end_date: str, objective: str = 'min_semivariance',
        target_return: float | None = None, confidence_level: float = 0.95, constraints: dict | None = None,
        capital_inicial: float | None = None, aportacion_mensual: float | None = None,
        meses: int | None = None, objetivo_final: float | None = None, validate_target: bool = True,
    ) -> dict:
        """Ver optimizar_pmpt."""
        return optimizar_pmpt(
            self._get_returns_matrix(tickers, start_date, end_date), objective, target_return, confidence_level,
            constraints, capital_inicial, aportacion_mensual, meses, objetivo_final, validate_target,
        )

    # --- Modelos modulares y presets -----------------------------------------------

    def create_custom_model(
        self, name: str, returns_model_type: str, risk_model_type: str, optimizer_type: str,
        returns_config: dict | None = None, risk_config: dict | None = None,
        optimizer_config: dict | None = None, description: str = "",
    ) -> PortfolioOptimizationModel:
        """Ver crear_modelo_personalizado."""
        return crear_modelo_personalizado(
            name, returns_model_type, risk_model_type, optimizer_type,
            returns_config, risk_config, optimizer_config, description,
        )

    def optimize_advanced(
        self, tickers: list[str], start_date: str, end_date: str,
        model: PortfolioOptimizationModel, constraints: dict | None = None, **kwargs,
    ) -> dict:
        """
        Ejecuta un PortfolioOptimizationModel. Los datos externos que pida el modelo los da
        `_obtener_datos_modelo` (los kwargs explícitos tienen prioridad).
        """
        retornos = self._get_returns_matrix(tickers, start_date, end_date)
        for clave, valor in self._obtener_datos_modelo(model, start_date, end_date).items():
            kwargs.setdefault(clave, valor)
        logger.info(f"[OptimizationService] Ejecutando optimización con {model.name}...")
        resultado = model.optimize(retornos, constraints, **kwargs)
        resultado['tickers'] = tickers
        resultado['date_range'] = {'start': start_date, 'end': end_date, 'days': len(retornos)}
        return convertir_numpy_a_nativo(resultado)

    def compare_models(
        self, tickers: list[str], start_date: str, end_date: str,
        models: list[PortfolioOptimizationModel], constraints: dict | None = None, **kwargs,
    ) -> dict[str, dict]:
        """Optimiza con cada modelo y añade 'rank' por Sharpe a los que no fallan."""
        resultados: dict[str, dict] = {}
        for modelo in models:
            try:
                resultados[modelo.name] = self.optimize_advanced(
                    tickers=tickers, start_date=start_date, end_date=end_date,
                    model=modelo, constraints=constraints, **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 — un modelo fallido no aborta la comparación
                logger.warning(f"Error optimizando con {modelo.name}: {exc}")
                resultados[modelo.name] = {'error': str(exc), 'model_info': modelo.to_dict()}
        validos = {n: r for n, r in resultados.items() if 'error' not in r}
        for rango, (nombre, _) in enumerate(
            sorted(validos.items(), key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True), 1
        ):
            resultados[nombre]['rank'] = rango
        return resultados

    def get_available_models(self) -> dict[str, list[str]]:
        return modelos_disponibles()

    def get_preconfigured_model(self, name: str):
        """Preset por nombre (PortfolioOptimizationModel o dict de pipeline ML)."""
        from ypyquant.optimization.presets import obtener_preset
        return obtener_preset(name)

    def list_available_presets(self) -> dict[str, str]:
        from ypyquant.optimization.presets import listar_presets
        return listar_presets()

    def get_preset_details(self, preset_name: str) -> dict:
        from ypyquant.optimization.presets import obtener_detalles_preset
        return obtener_detalles_preset(preset_name)

    def optimize_with_backtesting(
        self, tickers: list[str], start_date: str, end_date: str, model: PortfolioOptimizationModel,
        rebalance_frequency: str = "quarterly", constraints: dict | None = None, **kwargs,
    ) -> dict:
        """Optimización simple con una marca de backtesting pendiente (como en producción)."""
        resultado = self.optimize_advanced(
            tickers=tickers, start_date=start_date, end_date=end_date,
            model=model, constraints=constraints, **kwargs,
        )
        resultado['backtesting'] = {
            'enabled': False,
            'rebalance_frequency': rebalance_frequency,
            'note': 'Full backtesting implementation pending',
        }
        return resultado


# Tipo de la función de datos externos que puede pasar la aplicación al subclasificar
ProveedorDatosModelo = Callable[[PortfolioOptimizationModel, str, str], dict[str, Any]]
