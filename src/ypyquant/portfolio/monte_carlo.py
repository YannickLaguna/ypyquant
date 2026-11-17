"""
Simulación Monte Carlo para riesgo de portafolios (GBM + correlaciones).

Responsabilidad: calcular VaR/CVaR del portafolio mediante simulación de
trayectorias de precios con Movimiento Browniano Geométrico.

Flujo principal (MonteCarloRiesgo.calcular_var_cvar):
  Parámetros por activo → ruido correlacionado (Cholesky) → trayectorias GBM
  → retornos del portafolio → VaR y CVaR empíricos.

Si QuantLib está disponible se usa su generador de paths (mayor calidad numérica);
si no, el fallback NumPy aplica descomposición de Cholesky sobre la matriz de
correlación para preservar la estructura de dependencia entre activos.

Relacionado con:
  - portfolio/performance_metrics.py  (calculate_var_quantlib lo usa como backend)
"""

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import QuantLib as ql
    QUANTLIB_DISPONIBLE = True
except ImportError:
    QUANTLIB_DISPONIBLE = False
    warnings.warn("QuantLib no disponible. Las simulaciones Monte Carlo estarán limitadas.")


class MonteCarloRiesgo:
    """
    Simulador GBM multi-activo con estructura de correlación.

    simular_trayectorias_gbm(): un activo a la vez, vía QuantLib o NumPy.
    calcular_var_cvar(): portafolio completo con correlaciones (Cholesky en fallback NumPy).
    stress_test(): repite calcular_var_cvar bajo escenarios de retorno/volatilidad ajustados.
    """

    def __init__(self):
        """Inicializa el simulador Monte Carlo."""
        if not QUANTLIB_DISPONIBLE:
            warnings.warn(
                "QuantLib no está disponible. "
                "Las simulaciones usarán NumPy estándar en lugar de QuantLib."
            )

    def simular_trayectorias_gbm(
        self,
        precio_inicial: float,
        drift: float,
        volatilidad: float,
        horizonte_dias: int,
        num_simulaciones: int,
        pasos_por_dia: int = 1,
        seed: int | None = None
    ) -> np.ndarray:
        """
        GBM para un único activo: dS/S = μ dt + σ dW.

        I: precio_inicial, drift (μ anual), volatilidad (σ anual), horizonte_dias,
           num_simulaciones, seed.
        T: QuantLib GaussianPathGenerator si disponible; fallback NumPy log-normal.
        O: np.ndarray (num_simulaciones × num_pasos+1) con trayectorias de precios.
        """
        if seed is not None:
            np.random.seed(seed)

        num_pasos = horizonte_dias * pasos_por_dia
        dt = 1.0 / (252.0 * pasos_por_dia)  # Fracción de año

        if QUANTLIB_DISPONIBLE:
            # Usar QuantLib para simulación de alta calidad
            try:
                # Crear proceso GBM en QuantLib
                spot_handle = ql.QuoteHandle(ql.SimpleQuote(precio_inicial))
                risk_free_handle = ql.YieldTermStructureHandle(
                    ql.FlatForward(0, ql.NullCalendar(), drift, ql.Actual365Fixed())
                )
                volatility_handle = ql.BlackVolTermStructureHandle(
                    ql.BlackConstantVol(0, ql.NullCalendar(), volatilidad, ql.Actual365Fixed())
                )

                proceso = ql.BlackScholesMertonProcess(
                    spot_handle,
                    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), 0.0, ql.Actual365Fixed())),
                    risk_free_handle,
                    volatility_handle
                )

                # Generar números aleatorios
                rng = ql.GaussianRandomSequenceGenerator(
                    ql.UniformRandomSequenceGenerator(num_pasos, ql.UniformRandomGenerator(seed if seed else 42))
                )

                # Generar paths usando PathGenerator
                path_generator = ql.GaussianPathGenerator(
                    proceso,
                    horizonte_dias / 252.0,  # Tiempo en años
                    num_pasos,
                    rng,
                    False  # brownianBridge
                )

                # Recolectar trayectorias
                paths = np.zeros((num_simulaciones, num_pasos + 1))
                for i in range(num_simulaciones):
                    sample_path = path_generator.next()
                    path = sample_path.value()
                    # Convertir QuantLib Path a numpy array
                    paths[i, :] = np.array([path.at(j) for j in range(path.length())])

                return paths

            except Exception as e:
                warnings.warn(f"Error usando QuantLib, fallback a NumPy: {e}")

        # Fallback: Simulación estándar con NumPy
        # dS/S = μ dt + σ sqrt(dt) * Z, donde Z ~ N(0,1)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((num_simulaciones, num_pasos))

        # Calcular retornos log-normales
        drift_ajustado = (drift - 0.5 * volatilidad ** 2) * dt
        difusion = volatilidad * np.sqrt(dt) * Z

        # Retornos acumulados
        log_returns = drift_ajustado + difusion
        cumsum_returns = np.cumsum(log_returns, axis=1)

        # Trayectorias de precios
        paths = precio_inicial * np.exp(cumsum_returns)

        # Agregar precio inicial al inicio
        paths = np.column_stack([np.full(num_simulaciones, precio_inicial), paths])

        return paths

    def calcular_var_cvar(
        self,
        portafolio: dict[str, float],
        retornos_esperados: dict[str, float],
        volatilidades: dict[str, float],
        horizonte_dias: int = 252,
        num_simulaciones: int = 10000,
        nivel_confianza: float = 0.95,
        matriz_correlacion: pd.DataFrame | None = None,
        valor_inicial: float = 1.0,
        seed: int | None = None
    ) -> dict:
        """
        VaR y CVaR del portafolio completo con correlaciones.

        I: portafolio ({ticker: peso}), μ y σ anualizados por ticker, horizonte,
           matriz_correlacion (opcional), nivel_confianza, seed.
        T: Genera ruido correlacionado (Cholesky sobre matriz_correlacion) →
           trayectorias GBM multi-activo → retornos del portafolio → percentiles.
           Con QuantLib: paths por activo sin correlación cruzada (activos independientes).
        O: Dict con var_XX, cvar_XX, percentiles, trayectorias_muestra.
        """
        if seed is not None:
            np.random.seed(seed)

        # Validar portafolio
        tickers = list(portafolio.keys())
        pesos = np.array([portafolio[t] for t in tickers])

        if not np.isclose(pesos.sum(), 1.0, atol=1e-4):
            warnings.warn(f"Pesos del portafolio suman {pesos.sum():.4f}, normalizando a 1.0")
            pesos = pesos / pesos.sum()

        # Matriz de correlación por defecto (identidad)
        if matriz_correlacion is None:
            matriz_correlacion = pd.DataFrame(
                np.eye(len(tickers)),
                index=tickers,
                columns=tickers
            )

        # Simular trayectorias multi-activo con correlaciones (Cholesky)
        num_pasos = horizonte_dias
        n_activos = len(tickers)
        dt = 1.0 / 252.0

        drifts = np.array([retornos_esperados.get(t, 0.0) for t in tickers])
        vols = np.array([volatilidades.get(t, 0.0) for t in tickers])

        if QUANTLIB_DISPONIBLE:
            # Generar trayectorias activo por activo con QuantLib (sin correlación cruzada)
            trayectorias_activos = {}
            for ticker in tickers:
                trayectorias_activos[ticker] = self.simular_trayectorias_gbm(
                    precio_inicial=1.0,
                    drift=retornos_esperados.get(ticker, 0.0),
                    volatilidad=volatilidades.get(ticker, 0.0),
                    horizonte_dias=horizonte_dias,
                    num_simulaciones=num_simulaciones,
                    seed=seed,
                )
            trayectorias_portafolio = np.zeros((num_simulaciones, num_pasos + 1))
            for i, ticker in enumerate(tickers):
                trayectorias_portafolio += pesos[i] * trayectorias_activos[ticker]
        else:
            # Fallback NumPy: generar ruido correlacionado con Cholesky
            rng_mc = np.random.default_rng(seed)
            # Z shape: (n_sims, n_dias, n_activos) — shocks estándar independientes
            Z = rng_mc.standard_normal((num_simulaciones, num_pasos, n_activos))

            corr_mat = matriz_correlacion.loc[tickers, tickers].values
            try:
                L = np.linalg.cholesky(corr_mat)
                # Aplicar Cholesky: correlacionar la última dimensión
                Z_flat = Z.reshape(num_simulaciones * num_pasos, n_activos)
                Z_corr = (L @ Z_flat.T).T
                Z = Z_corr.reshape(num_simulaciones, num_pasos, n_activos)
            except np.linalg.LinAlgError:
                logger.warning("[MonteCarloRiesgo] Cholesky falló en matriz de correlación, usando activos independientes")

            # GBM log-normal para cada activo
            drift_adj = (drifts - 0.5 * vols ** 2) * dt  # shape (n_activos,)
            difusion = vols * np.sqrt(dt) * Z             # shape (n_sims, n_dias, n_activos)
            log_ret = drift_adj + difusion
            cum_log = np.cumsum(log_ret, axis=1)          # (n_sims, n_dias, n_activos)
            precios = np.exp(cum_log)                      # precios normalizados desde 1.0

            # Portafolio: precio_inicial de cada activo = 1.0
            trayectorias_portafolio = np.zeros((num_simulaciones, num_pasos + 1))
            trayectorias_portafolio[:, 0] = 1.0
            trayectorias_portafolio[:, 1:] = (precios * pesos[np.newaxis, np.newaxis, :]).sum(axis=2)

        # Calcular retornos finales
        valor_final = trayectorias_portafolio[:, -1] * valor_inicial
        valor_inicial_array = np.full(num_simulaciones, valor_inicial)
        retornos_finales = (valor_final - valor_inicial_array) / valor_inicial_array

        # Calcular pérdidas (negativo de retornos)
        perdidas = -retornos_finales

        # VaR: percentil al nivel de confianza
        alpha = 1 - nivel_confianza
        var = np.percentile(perdidas, nivel_confianza * 100)

        # CVaR: pérdida esperada dado que se supera VaR
        cvar = np.mean(perdidas[perdidas >= var])

        # Estadísticas adicionales
        peor_escenario = np.max(perdidas)
        mejor_escenario = np.min(perdidas)

        percentiles = {
            'p01': np.percentile(perdidas, 1),
            'p05': np.percentile(perdidas, 5),
            'p10': np.percentile(perdidas, 10),
            'p50': np.percentile(perdidas, 50),
            'p90': np.percentile(perdidas, 90),
            'p95': np.percentile(perdidas, 95),
            'p99': np.percentile(perdidas, 99),
        }

        # Seleccionar algunas trayectorias para visualización
        indices_muestra = np.linspace(0, num_simulaciones - 1, min(100, num_simulaciones), dtype=int)
        trayectorias_muestra = trayectorias_portafolio[indices_muestra, :].tolist()

        return {
            f'var_{int(nivel_confianza * 100)}': float(var),
            f'cvar_{int(nivel_confianza * 100)}': float(cvar),
            'peor_escenario': float(peor_escenario),
            'mejor_escenario': float(mejor_escenario),
            'media_perdidas': float(np.mean(perdidas)),
            'std_perdidas': float(np.std(perdidas)),
            'percentiles': percentiles,
            'distribuciones': perdidas.tolist(),
            'trayectorias_muestra': trayectorias_muestra,
            'num_simulaciones': num_simulaciones,
            'horizonte_dias': horizonte_dias,
            'nivel_confianza': nivel_confianza
        }

    def stress_test(
        self,
        portafolio: dict[str, float],
        retornos_esperados: dict[str, float],
        volatilidades: dict[str, float],
        escenarios: list[dict[str, float]],
        horizonte_dias: int = 252,
        num_simulaciones: int = 1000
    ) -> dict:
        """
        Realiza stress testing del portafolio bajo diferentes escenarios.

        Args:
            portafolio: Pesos del portafolio
            retornos_esperados: Retornos base esperados
            volatilidades: Volatilidades base
            escenarios: Lista de diccionarios con ajustes de retorno/volatilidad
                       Ejemplo: [
                           {'nombre': 'Crisis 2008', 'ajuste_retorno': -0.30, 'ajuste_vol': 2.0},
                           {'nombre': 'Bull Market', 'ajuste_retorno': 0.15, 'ajuste_vol': 0.7}
                       ]
            horizonte_dias: Horizonte de simulación
            num_simulaciones: Simulaciones por escenario

        Returns:
            Diccionario con resultados por escenario

        Example:
            >>> mc = MonteCarloRiesgo()
            >>> escenarios = [
            ...     {'nombre': 'Crisis', 'ajuste_retorno': -0.40, 'ajuste_vol': 2.5},
            ...     {'nombre': 'Normal', 'ajuste_retorno': 0.0, 'ajuste_vol': 1.0}
            ... ]
            >>> resultados = mc.stress_test(portafolio, retornos, vols, escenarios)
        """
        resultados = {}

        for escenario in escenarios:
            nombre = escenario.get('nombre', 'Sin nombre')
            ajuste_retorno = escenario.get('ajuste_retorno', 0.0)
            ajuste_vol = escenario.get('ajuste_vol', 1.0)

            # Ajustar parámetros según escenario
            retornos_ajustados = {
                t: r + ajuste_retorno for t, r in retornos_esperados.items()
            }
            volatilidades_ajustadas = {
                t: v * ajuste_vol for t, v in volatilidades.items()
            }

            # Calcular VaR/CVaR bajo este escenario
            resultado = self.calcular_var_cvar(
                portafolio=portafolio,
                retornos_esperados=retornos_ajustados,
                volatilidades=volatilidades_ajustadas,
                horizonte_dias=horizonte_dias,
                num_simulaciones=num_simulaciones
            )

            resultados[nombre] = resultado

        return resultados
