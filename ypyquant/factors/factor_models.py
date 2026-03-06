"""
Modelos de factores para estimar μ: CAPM y Fama-French (FF4/FF5/FF5+MOM).

Cada modelo sigue el mismo contrato:
  fit(retornos, datos_factor) → estima cargas factoriales por OLS-HC3.
  calculate_expected_returns() → proyecta μ anualizado con las primas de factor.

Relacionado con: french_data_fetcher.py (provee datos FF), portfolio_models.py
(consume μ), walk_forward.py (orquesta fit + calculate).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER OLS COMPARTIDO
# ============================================================================

def _fit_ols_regresion(
    y_excess: np.ndarray,
    X: np.ndarray,
    nombres_factores: List[str],
) -> Dict:
    """
    OLS con errores HC3 compartido por CAPMModel y FamaFrenchModel.

    I: y_excess (R_i - RF), X (regresores con constante), nombres_factores.
    T: sm.OLS(...).fit(cov_type='HC3') — errores robustos a heterocedasticidad.
    O: Dict con params, bse, pvalues, rsquared, rsquared_adj.
    """
    modelo = sm.OLS(y_excess, X).fit(cov_type='HC3', use_t=True)
    return {
        'params': modelo.params,
        'bse': modelo.bse,
        'pvalues': modelo.pvalues,
        'rsquared': modelo.rsquared,
        'rsquared_adj': modelo.rsquared_adj,
    }


# ============================================================================
# CLASE BASE ABSTRACTA PARA FACTORES
# ============================================================================

class Factor(ABC):
    """
    Clase base abstracta para cualquier factor.
    """

    def __init__(self, name: str):
        self.name = name
        self.metadata = {}

    @abstractmethod
    def calculate(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calcula el factor para los tickers dados.

        Returns:
            DataFrame con índice = fechas, columnas = tickers
        """
        pass

    def to_dict(self) -> Dict:
        """Serializa configuración del factor"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'metadata': self.metadata
        }


# ============================================================================
# MODELO CAPM (Capital Asset Pricing Model)
# ============================================================================

class CAPMModel:
    """
    CAPM: E[Ri] = Rf + βi·(E[Rm] - Rf).

    fit():
      I: retornos de activos + retornos del mercado (Series).
      T: OLS-HC3 de excess returns contra el exceso de mercado.
      O: self (almacena betas, alphas, SE, p-valores, R²).

    calculate_expected_returns():
      I: ninguno (usa betas y market_premium del fit).
      T: E[Ri] = Rf + βi·prima; tasas dinámicas por QuantLib si disponibles.
      O: Series con μ anualizado por ticker.
    """

    def __init__(
        self,
        market_ticker: str = "^GSPC",
        risk_free_rate: Optional[float] = None,
        frequency: int = 252,
        usar_tasas_dinamicas: bool = True,
        horizonte_inversion_años: float = 5.0
    ):
        """
        Inicializa el modelo CAPM.

        Args:
            market_ticker: Ticker del índice de mercado (default: S&P 500)
            risk_free_rate: Tasa libre de riesgo fija. Si None y usar_tasas_dinamicas=False, usar 0.02
            frequency: Frecuencia de datos (252 para diario)
            usar_tasas_dinamicas: Si True, usa curvas de QuantLib cuando disponible
            horizonte_inversion_años: Horizonte de inversión para tasas dinámicas
        """
        self.market_ticker = market_ticker
        self.frequency = frequency
        self.usar_tasas_dinamicas = usar_tasas_dinamicas
        self.horizonte_inversion_años = horizonte_inversion_años
        self.betas = {}
        self.alphas = {}
        self.r_squared = {}
        self.market_premium = 0.0

        # Atributos de inferencia (se rellenan en fit())
        self.r_squared_adj: Dict[str, float] = {}
        self.betas_se: Dict[str, float] = {}
        self.alphas_se: Dict[str, float] = {}
        self.pvalues_beta: Dict[str, float] = {}
        self.pvalues_alpha: Dict[str, float] = {}

        # Configurar tasa libre de riesgo
        self._servicio_tasas = None
        if usar_tasas_dinamicas:
            try:
                from quant.data.risk_free_rates import TasaLibreRiesgoDinamica
                self._servicio_tasas = TasaLibreRiesgoDinamica(actualizar_curva=True)
                self.risk_free_rate = self._servicio_tasas.obtener_tasa_por_plazo(horizonte_inversion_años)
                logger.debug(f"[CAPM] Usando tasa dinámica {horizonte_inversion_años}Y: {self.risk_free_rate:.4f}")
            except Exception as e:
                logger.warning(f"[CAPM] No se pudo usar QuantLib: {e}. Usando tasa fija.")
                self.risk_free_rate = risk_free_rate if risk_free_rate is not None else 0.02
                self.usar_tasas_dinamicas = False
        else:
            self.risk_free_rate = risk_free_rate if risk_free_rate is not None else 0.02

    def fit(
        self,
        returns_df: pd.DataFrame,
        market_returns: pd.Series
    ) -> 'CAPMModel':
        """
        Estima betas y alphas de Jensen para cada activo por MCO con errores
        robustos a heterocedasticidad (HC3, corrección de White).

        Especificación por series temporales (retornos diarios en exceso):
            R_i,t - RF_t = α_i + β_i * (R_m,t - RF_t) + ε_i,t

        La RF diaria se obtiene dividiendo self.risk_free_rate (anual) por
        self.frequency. Los parámetros se almacenan tanto para predicción como
        para inferencia estadística (errores estándar, p-valores).

        Args:
            returns_df: DataFrame con retornos diarios brutos de activos
                        (columnas = tickers). La RF se substrae internamente.
            market_returns: Series con retornos diarios brutos del mercado.
                            La RF se substrae internamente.

        Returns:
            self (para method chaining)
        """
        # Alinear fechas
        aligned_data = returns_df.join(market_returns.rename('market'), how='inner')

        if aligned_data.empty:
            raise ValueError(
                "No hay fechas comunes entre retornos de activos y retornos del mercado. "
                "No se puede calibrar el modelo CAPM."
            )

        market_col = aligned_data['market']
        assets_df = aligned_data.drop('market', axis=1)

        # RF diaria para construir excess returns
        rf_diario = self.risk_free_rate / self.frequency
        mkt_excess = market_col - rf_diario  # R_m,t - RF_t

        # Prima de mercado anualizada: E[R_m - RF] * 252
        self.market_premium = float(mkt_excess.mean() * self.frequency)

        # Regresor con constante: [1, R_m,t - RF_t]
        X = sm.add_constant(mkt_excess.values, has_constant='add')

        for ticker in assets_df.columns:
            y_excess = (assets_df[ticker] - rf_diario).values  # R_i,t - RF_t
            res = _fit_ols_regresion(y_excess, X, nombres_factores=['beta'])

            self.alphas[ticker] = float(res['params'][0])
            self.betas[ticker] = float(res['params'][1])
            self.alphas_se[ticker] = float(res['bse'][0])
            self.betas_se[ticker] = float(res['bse'][1])
            self.pvalues_alpha[ticker] = float(res['pvalues'][0])
            self.pvalues_beta[ticker] = float(res['pvalues'][1])
            self.r_squared[ticker] = float(res['rsquared'])
            self.r_squared_adj[ticker] = float(res['rsquared_adj'])

        return self

    def calculate_expected_returns(self) -> pd.Series:
        """
        Calcula retornos esperados usando CAPM con tasas dinámicas.

        Si usar_tasas_dinamicas=True, cada activo usa la tasa apropiada
        según sus características (value/growth/neutral).

        Returns:
            Series con retornos esperados anualizados
        """
        if not self.betas:
            raise RuntimeError("Llamar fit() antes de calculate_expected_returns()")

        expected_returns = {}

        for ticker, beta in self.betas.items():
            # Obtener tasa libre de riesgo apropiada para este activo
            if self.usar_tasas_dinamicas and self._servicio_tasas is not None:
                try:
                    rf_activo = self._servicio_tasas.obtener_tasa_para_activo(ticker, beta)
                except Exception:
                    rf_activo = self.risk_free_rate
            else:
                rf_activo = self.risk_free_rate

            # E[Ri] = Rf(activo) + βi * (E[Rm] - Rf(mercado))
            expected_returns[ticker] = rf_activo + beta * self.market_premium

        return pd.Series(expected_returns)

    def get_security_market_line(self, beta_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Calcula la Security Market Line (SML).

        Args:
            beta_range: Rango de betas (min, max). Si None, usa rango de los activos.

        Returns:
            DataFrame con betas y retornos esperados para graficar SML
        """
        if beta_range is None:
            beta_min = min(self.betas.values()) if self.betas else 0.0
            beta_max = max(self.betas.values()) if self.betas else 2.0
            beta_range = (beta_min, beta_max)

        betas = np.linspace(beta_range[0], beta_range[1], 100)
        expected_returns = self.risk_free_rate + betas * self.market_premium

        return pd.DataFrame({
            'beta': betas,
            'expected_return': expected_returns
        })

    def to_dict(self) -> Dict:
        """Serializa el modelo incluyendo parámetros de inferencia HC3."""
        return {
            'type': 'CAPM',
            'market_ticker': self.market_ticker,
            'risk_free_rate': self.risk_free_rate,
            'market_premium': self.market_premium,
            'betas': self.betas,
            'alphas': self.alphas,
            'r_squared': self.r_squared,
            'r_squared_adj': self.r_squared_adj,
            'betas_se': self.betas_se,
            'alphas_se': self.alphas_se,
            'pvalues_beta': self.pvalues_beta,
            'pvalues_alpha': self.pvalues_alpha,
        }

    def to_json(self) -> str:
        """Exporta a JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CAPMModel':
        """Carga desde diccionario, compatible con versiones previas sin HC3."""
        model = cls(
            market_ticker=data['market_ticker'],
            risk_free_rate=data['risk_free_rate']
        )
        model.market_premium = data['market_premium']
        model.betas = data['betas']
        model.alphas = data['alphas']
        model.r_squared = data['r_squared']
        model.r_squared_adj = data.get('r_squared_adj', {})
        model.betas_se = data.get('betas_se', {})
        model.alphas_se = data.get('alphas_se', {})
        model.pvalues_beta = data.get('pvalues_beta', {})
        model.pvalues_alpha = data.get('pvalues_alpha', {})
        return model


# ============================================================================
# MODELO FAMA-FRENCH GENÉRICO (configurable: 4, 5 o 6 factores)
# ============================================================================

# Configuraciones predefinidas de factores
FACTORES_FF4 = ['MKT', 'SMB', 'HML', 'UMD']
FACTORES_FF5 = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
FACTORES_FF5_MOM = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']


class FamaFrenchModel:
    """
    Modelo Fama-French genérico (FF4, FF5, FF5+MOM configurables).

    Regresión: R_i - RF = α_i + Σ β_j·F_j + ε_i

    fit():
      I: retornos de activos (DataFrame) + factor_data (DataFrame con columnas MKT, SMB…).
      T: OLS-HC3 por activo; extrae RF de factor_data si está disponible.
      O: self (almacena factor_loadings, alphas, primas, SE, p-valores, R²).

    calculate_expected_returns():
      I: ninguno (usa factor_loadings y factor_premiums del fit).
      T: E[Ri] = RF + Σ(β_j · E[F_j]).
      O: Series con μ anualizado por ticker.
    """

    def __init__(
        self,
        factores: Optional[List[str]] = None,
        data_source: str = "manual",
        risk_free_rate: Optional[float] = None,
        frequency: int = 252
    ):
        """
        Inicializa el modelo Fama-French.

        Args:
            factores: Lista de nombres de factores a usar.
                      Por defecto FF5+MOM (6 factores).
                      Opciones predefinidas: FACTORES_FF4, FACTORES_FF5, FACTORES_FF5_MOM.
            data_source: Fuente de datos ('manual' o 'french_library')
            risk_free_rate: Tasa libre de riesgo anualizada. Si None, usar fallback 0.02.
            frequency: Frecuencia de datos (252 para diario)
        """
        self.factores = factores if factores is not None else list(FACTORES_FF5_MOM)
        self.data_source = data_source
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else 0.02
        self.frequency = frequency

        # Resultados de la regresión
        self.factor_loadings: Dict[str, Dict[str, float]] = {}
        self.alphas: Dict[str, float] = {}
        self.r_squared: Dict[str, float] = {}
        self.factor_premiums: Dict[str, float] = {}

        # Atributos de inferencia estadística (se rellenan en fit())
        self.r_squared_adj: Dict[str, float] = {}
        self.alphas_se: Dict[str, float] = {}
        self.pvalues_alpha: Dict[str, float] = {}
        self.factor_loadings_se: Dict[str, Dict[str, float]] = {}
        self.pvalues_loadings: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        returns_df: pd.DataFrame,
        factor_data: pd.DataFrame
    ) -> 'FamaFrenchModel':
        """
        Calibra los factor loadings (betas) para cada activo mediante MCO con errores
        robustos a heterocedasticidad (HC3, corrección de White).

        Especificación por series temporales (retornos diarios en exceso):
            R_i,t - RF_t = α_i + β_MKT·MKT_t + β_SMB·SMB_t + … + ε_i,t

        La RF diaria se extrae de factor_data['RF'] si está disponible
        (Kenneth French Library la incluye). Fallback: self.risk_free_rate / self.frequency.
        MKT ya llega como excess return (Rm - Rf) desde la French Library.

        Los parámetros se almacenan para predicción e inferencia estadística
        (errores estándar, p-valores, R² y R² ajustado).

        Args:
            returns_df: DataFrame con retornos diarios brutos de activos
                        (columnas = tickers). La RF se substrae internamente.
            factor_data: DataFrame con columnas correspondientes a self.factores
                         (y opcionalmente 'RF'). Columnas extra se ignoran;
                         las requeridas deben estar presentes.

        Returns:
            self (method chaining)
        """
        # Verificar columnas requeridas
        faltantes = [f for f in self.factores if f not in factor_data.columns]
        if faltantes:
            raise ValueError(
                f"factor_data no contiene los factores requeridos: {faltantes}. "
                f"Columnas disponibles: {list(factor_data.columns)}"
            )

        # Extraer RF antes de filtrar columnas de factor_data
        rf_disponible = 'RF' in factor_data.columns

        # Alinear fechas entre retornos y factores
        aligned_data = returns_df.join(factor_data[self.factores], how='inner')

        if aligned_data.empty:
            raise ValueError(
                "No hay fechas comunes entre retornos de activos y datos de factores. "
                f"No se puede calibrar el modelo Fama-French ({len(self.factores)} factores)."
            )

        factores_df = aligned_data[self.factores]
        activos_df = aligned_data.drop(columns=self.factores)

        # RF diaria alineada al periodo de estimación
        if rf_disponible:
            rf_diario_serie = (
                factor_data['RF']
                .reindex(aligned_data.index)
                .fillna(self.risk_free_rate / self.frequency)
            )
        else:
            rf_diario_serie = pd.Series(
                self.risk_free_rate / self.frequency, index=aligned_data.index
            )

        # Calcular primas de riesgo de factores (media anualizada)
        for factor in self.factores:
            self.factor_premiums[factor] = float(factores_df[factor].mean() * self.frequency)

        # Regresor con constante: [1, F1, F2, ..., Fk]
        X = sm.add_constant(factores_df.values, has_constant='add')

        for ticker in activos_df.columns:
            y_excess = (activos_df[ticker] - rf_diario_serie).values  # R_i,t - RF_t
            res = _fit_ols_regresion(y_excess, X, nombres_factores=self.factores)

            self.alphas[ticker] = float(res['params'][0])
            self.factor_loadings[ticker] = {
                factor: float(coef)
                for factor, coef in zip(self.factores, res['params'][1:])
            }
            self.r_squared[ticker] = float(res['rsquared'])
            self.r_squared_adj[ticker] = float(res['rsquared_adj'])
            self.alphas_se[ticker] = float(res['bse'][0])
            self.pvalues_alpha[ticker] = float(res['pvalues'][0])
            self.factor_loadings_se[ticker] = {
                factor: float(se)
                for factor, se in zip(self.factores, res['bse'][1:])
            }
            self.pvalues_loadings[ticker] = {
                factor: float(pv)
                for factor, pv in zip(self.factores, res['pvalues'][1:])
            }

        return self

    def calculate_expected_returns(self) -> pd.Series:
        """
        Calcula retornos esperados: E[R_i] = RF + Σ(β_j · E[F_j]).

        Returns:
            Series con retornos esperados anualizados por ticker
        """
        if not self.factor_loadings:
            raise RuntimeError("Llamar fit() antes de calculate_expected_returns()")

        retornos_esperados = {}

        for ticker, loadings in self.factor_loadings.items():
            retorno = self.risk_free_rate
            for factor, beta in loadings.items():
                retorno += beta * self.factor_premiums.get(factor, 0.0)
            retornos_esperados[ticker] = retorno

        return pd.Series(retornos_esperados)

    def calculate_factor_attribution(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcula la atribución de retorno a cada factor para un portafolio dado.

        Args:
            weights: Pesos del portafolio {ticker: peso}

        Returns:
            Dict con la contribución de cada factor y del alpha al retorno total
        """
        atribucion = {factor: 0.0 for factor in self.factores}
        atribucion['Alpha'] = 0.0

        for ticker, peso in weights.items():
            if ticker not in self.factor_loadings:
                continue

            loadings = self.factor_loadings[ticker]
            for factor, beta in loadings.items():
                prima = self.factor_premiums.get(factor, 0.0)
                atribucion[factor] += peso * beta * prima

            atribucion['Alpha'] += peso * self.alphas.get(ticker, 0.0) * self.frequency

        return atribucion

    def to_dict(self) -> Dict:
        """Serializa el modelo a diccionario incluyendo parámetros de inferencia HC3."""
        return {
            'type': 'FamaFrenchModel',
            'factores': self.factores,
            'data_source': self.data_source,
            'risk_free_rate': self.risk_free_rate,
            'factor_premiums': self.factor_premiums,
            'factor_loadings': self.factor_loadings,
            'alphas': self.alphas,
            'r_squared': self.r_squared,
            'r_squared_adj': self.r_squared_adj,
            'alphas_se': self.alphas_se,
            'pvalues_alpha': self.pvalues_alpha,
            'factor_loadings_se': self.factor_loadings_se,
            'pvalues_loadings': self.pvalues_loadings,
        }

    def to_json(self) -> str:
        """Exporta a JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> 'FamaFrenchModel':
        """Reconstruye el modelo desde un diccionario serializado"""
        modelo = cls(
            factores=data.get('factores', FACTORES_FF5_MOM),
            data_source=data.get('data_source', 'manual'),
            risk_free_rate=data.get('risk_free_rate')
        )
        modelo.factor_premiums = data.get('factor_premiums', {})
        modelo.factor_loadings = data.get('factor_loadings', {})
        modelo.alphas = data.get('alphas', {})
        modelo.r_squared = data.get('r_squared', {})
        modelo.r_squared_adj = data.get('r_squared_adj', {})
        modelo.alphas_se = data.get('alphas_se', {})
        modelo.pvalues_alpha = data.get('pvalues_alpha', {})
        modelo.factor_loadings_se = data.get('factor_loadings_se', {})
        modelo.pvalues_loadings = data.get('pvalues_loadings', {})
        return modelo


# Alias de compatibilidad con código existente
FamaFrench4FactorModel = FamaFrenchModel




# ============================================================================
# UTILIDADES PARA FACTORES
# ============================================================================

def load_fama_french_data_from_csv(
    csv_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Carga datos de Fama-French desde archivo CSV.

    Args:
        csv_path: Path al archivo CSV
        start_date: Filtrar desde esta fecha
        end_date: Filtrar hasta esta fecha

    Returns:
        DataFrame con factores
    """
    # TODO: Implementar lectura desde CSV de Kenneth French Library
    # Format esperado: Date, Mkt-RF, SMB, HML, Mom (WML/UMD)
    raise NotImplementedError("CSV loading not implemented yet. Use generate_mock_factor_data() for testing.")
