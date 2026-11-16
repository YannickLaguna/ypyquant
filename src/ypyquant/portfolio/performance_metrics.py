"""
Métricas de performance para portafolios: ratios, drawdown y comparación con benchmark.

Clases principales:
- PerformanceMetrics (dataclass): contenedor de resultados.
- PerformanceAnalyzer: métodos estáticos de cálculo.
  · calculate_metrics(): retorno anualizado, vol, Sharpe, Sortino, Calmar, VaR, CVaR.
  · compare_to_benchmark(): beta (scipy.stats.linregress + SE + p-valor),
    alpha Jensen anualizado, tracking error, information ratio, up/downside capture.
  · rolling_metrics(): métricas en ventana deslizante.

Relacionado con: walk_forward.py (consume compare_to_benchmark), monte_carlo.py
(calculate_var_quantlib usa MonteCarloRiesgo como backend).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas completas de performance"""
    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk
    volatility: float  # Anualizada
    downside_deviation: float
    max_drawdown: float
    avg_drawdown: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Other
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR


class PerformanceAnalyzer:
    """
    Analizador de performance de portafolios.

    Calcula métricas avanzadas de riesgo y retorno.
    """

    @staticmethod
    def calculate_metrics(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> PerformanceMetrics:
        """
        Métricas completas de performance.

        I: Series de retornos diarios, rf anualizada, períodos/año.
        T: Calcula retorno total, vol, drawdown, Sharpe, Sortino, Calmar, Omega, VaR, CVaR.
        O: PerformanceMetrics dataclass con todas las métricas anualizadas.
        """
        # Limpiar NaN
        returns = returns.dropna()

        if len(returns) == 0:
            return PerformanceAnalyzer._empty_metrics()

        # Total return
        total_return = (1 + returns).prod() - 1

        # Annualized return
        n_periods = len(returns)
        years = n_periods / periods_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Cumulative return
        cumulative_return = total_return

        # Volatility (anualizada)
        volatility = returns.std() * np.sqrt(periods_per_year)

        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)

        # Drawdowns
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max

        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

        # Sharpe ratio
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Sortino ratio
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Omega ratio
        threshold = risk_free_rate / periods_per_year
        omega_ratio = PerformanceAnalyzer._calculate_omega_ratio(returns, threshold)

        # Skewness & Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # VaR & CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95
        )

    @staticmethod
    def _calculate_omega_ratio(returns: pd.Series, threshold: float) -> float:
        """Calcula Omega ratio."""
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())

        return gains / losses if losses > 0 else float('inf')

    @staticmethod
    def _empty_metrics() -> PerformanceMetrics:
        """Retorna métricas vacías."""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, cumulative_return=0,
            volatility=0, downside_deviation=0, max_drawdown=0, avg_drawdown=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, omega_ratio=0,
            skewness=0, kurtosis=0, var_95=0, cvar_95=0
        )

    @staticmethod
    def compare_to_benchmark(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Comparación portafolio vs benchmark.

        I: Series de retornos diarios de portafolio y benchmark, rf anualizada.
        T: Alinea fechas → beta vía scipy.stats.linregress (slope, SE, p-valor)
           → alpha Jensen anualizado (intercept * 252) → tracking error, IR,
           upside/downside capture.
        O: Dict con portfolio, benchmark (PerformanceMetrics), beta, beta_pvalue,
           beta_stderr, alpha, tracking_error, information_ratio, upside/downside_capture.
        """
        # Alinear fechas
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if aligned.empty:
            return {'error': 'No hay datos alineados'}

        # Calcular métricas para ambos
        portfolio_metrics = PerformanceAnalyzer.calculate_metrics(aligned['portfolio'])
        benchmark_metrics = PerformanceAnalyzer.calculate_metrics(aligned['benchmark'])

        # Tracking error
        tracking_diff = aligned['portfolio'] - aligned['benchmark']
        tracking_error = tracking_diff.std() * np.sqrt(252)

        # Information ratio
        excess_return = portfolio_metrics.annualized_return - benchmark_metrics.annualized_return
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

        # Beta con regresión lineal (scipy.stats.linregress devuelve SE y p-valor)
        from scipy import stats as _sp_stats
        slope, intercept, r_val, p_val, std_err = _sp_stats.linregress(
            aligned['benchmark'].values, aligned['portfolio'].values
        )
        beta = float(slope)
        beta_pvalue = float(p_val)
        beta_stderr = float(std_err)

        # Alpha de Jensen (anualizado): intercept diario * 252
        alpha = float(intercept) * 252

        # Upside/Downside capture
        up_months = aligned[aligned['benchmark'] > 0]
        down_months = aligned[aligned['benchmark'] < 0]

        upside_capture = (up_months['portfolio'].mean() / up_months['benchmark'].mean()
                         if len(up_months) > 0 and up_months['benchmark'].mean() != 0 else 0)

        downside_capture = (down_months['portfolio'].mean() / down_months['benchmark'].mean()
                           if len(down_months) > 0 and down_months['benchmark'].mean() != 0 else 0)

        return {
            'portfolio': portfolio_metrics,
            'benchmark': benchmark_metrics,
            'excess_return': excess_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'beta_pvalue': beta_pvalue,
            'beta_stderr': beta_stderr,
            'alpha': alpha,
            'upside_capture': upside_capture,
            'downside_capture': downside_capture,
        }

    @staticmethod
    def rolling_metrics(
        returns: pd.Series,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Métricas en ventana deslizante.

        I: Series de retornos diarios, window (default 60 días ≈ 3 meses).
        T: Rolling mean (retorno), std (vol), Sharpe y drawdown dentro de la ventana.
        O: DataFrame con columnas [return, volatility, sharpe, drawdown], sin NaN.
        """
        rolling = pd.DataFrame(index=returns.index)

        # Rolling returns
        rolling['return'] = returns.rolling(window).mean() * 252

        # Rolling volatility
        rolling['volatility'] = returns.rolling(window).std() * np.sqrt(252)

        # Rolling Sharpe
        rolling['sharpe'] = rolling['return'] / rolling['volatility']

        # Rolling drawdown
        cum_returns = (1 + returns).cumprod()
        rolling['drawdown'] = cum_returns.rolling(window).apply(
            lambda x: ((x - x.max()) / x.max()).min() if len(x) > 0 else 0,
            raw=False
        )

        return rolling.dropna()


def calculate_var_quantlib(
    portafolio: Dict[str, float],
    retornos_df: pd.DataFrame,
    horizonte_dias: int = 252,
    num_simulaciones: int = 10000,
    nivel_confianza: float = 0.95,
    usar_quantlib: bool = True
) -> Dict:
    """
    Calcula VaR y CVaR usando simulaciones Monte Carlo con QuantLib.

    Esta función mejora el cálculo básico de VaR/CVaR usando simulaciones
    estocásticas en lugar de asumir normalidad de retornos.

    Args:
        portafolio: Diccionario {ticker: peso} con pesos del portafolio
        retornos_df: DataFrame con retornos históricos de los activos
        horizonte_dias: Días a futuro para proyección (default: 252 = 1 año)
        num_simulaciones: Número de simulaciones Monte Carlo (default: 10,000)
        nivel_confianza: Nivel de confianza para VaR (default: 0.95)
        usar_quantlib: Si True, usa QuantLib; si False, usa método básico

    Returns:
        Diccionario con:
        - var_95: Value at Risk al 95% de confianza
        - cvar_95: Conditional VaR (Expected Shortfall)
        - peor_escenario: Peor pérdida simulada
        - mejor_escenario: Mejor ganancia simulada
        - percentiles: Distribución de percentiles
        - trayectorias_muestra: Trayectorias para visualización
        - metodo: 'quantlib' o 'basico'

    Example:
        >>> portafolio = {'AAPL': 0.5, 'MSFT': 0.5}
        >>> retornos = pd.DataFrame(...)  # Retornos históricos
        >>> resultado = calculate_var_quantlib(portafolio, retornos)
        >>> print(f"VaR 95%: {resultado['var_95']:.2%}")
        >>> print(f"CVaR 95%: {resultado['cvar_95']:.2%}")
    """
    if not usar_quantlib:
        # Fallback: método básico usando cuantiles históricos
        logger.info("Usando método básico de VaR/CVaR (sin QuantLib)")

        # Calcular retornos del portafolio
        tickers = list(portafolio.keys())
        pesos = np.array([portafolio[t] for t in tickers])

        # Filtrar solo los tickers que están en retornos_df
        tickers_disponibles = [t for t in tickers if t in retornos_df.columns]
        if len(tickers_disponibles) < len(tickers):
            logger.warning(f"Algunos tickers no encontrados en retornos: {set(tickers) - set(tickers_disponibles)}")

        retornos_portafolio = (retornos_df[tickers_disponibles] * pesos[:len(tickers_disponibles)]).sum(axis=1)

        # VaR y CVaR históricos
        var = retornos_portafolio.quantile(1 - nivel_confianza)
        cvar = retornos_portafolio[retornos_portafolio <= var].mean()

        return {
            f'var_{int(nivel_confianza * 100)}': float(-var),  # Negativo porque VaR se reporta como pérdida
            f'cvar_{int(nivel_confianza * 100)}': float(-cvar),
            'peor_escenario': float(-retornos_portafolio.min()),
            'mejor_escenario': float(-retornos_portafolio.max()),
            'metodo': 'basico',
            'num_simulaciones': 0,
            'horizonte_dias': horizonte_dias
        }

    # Método avanzado: Monte Carlo con QuantLib
    try:
        from ypyquant.portfolio.monte_carlo import MonteCarloRiesgo

        logger.info(f"Calculando VaR/CVaR con Monte Carlo ({num_simulaciones:,} simulaciones)")

        # Preparar datos para Monte Carlo
        tickers = list(portafolio.keys())

        # Calcular estadísticas de retornos históricos
        retornos_esperados = {}
        volatilidades = {}

        for ticker in tickers:
            if ticker in retornos_df.columns:
                retornos_ticker = retornos_df[ticker].dropna()
                # Anualizar retorno esperado y volatilidad
                retornos_esperados[ticker] = retornos_ticker.mean() * 252
                volatilidades[ticker] = retornos_ticker.std() * np.sqrt(252)
            else:
                logger.warning(f"Ticker {ticker} no encontrado en retornos, usando valores por defecto")
                retornos_esperados[ticker] = 0.10
                volatilidades[ticker] = 0.20

        # Calcular matriz de correlación
        tickers_disponibles = [t for t in tickers if t in retornos_df.columns]
        if len(tickers_disponibles) > 1:
            matriz_correlacion = retornos_df[tickers_disponibles].corr()
        else:
            matriz_correlacion = None

        # Ejecutar simulación Monte Carlo
        mc = MonteCarloRiesgo()
        resultado = mc.calcular_var_cvar(
            portafolio=portafolio,
            retornos_esperados=retornos_esperados,
            volatilidades=volatilidades,
            horizonte_dias=horizonte_dias,
            num_simulaciones=num_simulaciones,
            nivel_confianza=nivel_confianza,
            matriz_correlacion=matriz_correlacion
        )

        resultado['metodo'] = 'quantlib'
        return resultado

    except ImportError:
        logger.warning("QuantLib no disponible, usando método básico")
        return calculate_var_quantlib(
            portafolio=portafolio,
            retornos_df=retornos_df,
            horizonte_dias=horizonte_dias,
            num_simulaciones=num_simulaciones,
            nivel_confianza=nivel_confianza,
            usar_quantlib=False
        )
    except Exception as e:
        logger.error(f"Error en cálculo Monte Carlo: {e}, usando método básico")
        return calculate_var_quantlib(
            portafolio=portafolio,
            retornos_df=retornos_df,
            horizonte_dias=horizonte_dias,
            num_simulaciones=num_simulaciones,
            nivel_confianza=nivel_confianza,
            usar_quantlib=False
        )


def format_performance_metrics(metrics: PerformanceMetrics) -> str:
    """Formatea métricas para display."""
    return f"""
Performance Metrics:
  📈 Returns:
     Total Return:        {metrics.total_return:+.2%}
     Annualized Return:   {metrics.annualized_return:+.2%}

  ⚠️ Risk:
     Volatility:          {metrics.volatility:.2%}
     Max Drawdown:        {metrics.max_drawdown:.2%}
     Avg Drawdown:        {metrics.avg_drawdown:.2%}

  📊 Risk-Adjusted:
     Sharpe Ratio:        {metrics.sharpe_ratio:.2f}
     Sortino Ratio:       {metrics.sortino_ratio:.2f}
     Calmar Ratio:        {metrics.calmar_ratio:.2f}
     Omega Ratio:         {metrics.omega_ratio:.2f}

  📉 Distribution:
     Skewness:            {metrics.skewness:.2f}
     Kurtosis:            {metrics.kurtosis:.2f}
     VaR (95%):           {metrics.var_95:.2%}
     CVaR (95%):          {metrics.cvar_95:.2%}
    """.strip()
