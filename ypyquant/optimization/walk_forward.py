"""
Motor Walk-Forward para Experimento Factorial de Optimización de Portafolios.
=============================================================================

Responsabilidad:
- Orquestar evaluaciones walk-forward mensuales con rolling window.
- Definir y ejecutar los 21 experimentos del diseño factorial (2 etapas + benchmarks).
- Calcular métricas out-of-sample agregadas y Jensen's alpha.

Se relaciona con:
- portfolio_models.py: optimizadores (Markowitz, PMPTMinCVaR, PMPTMaxSortino).
- factor_models.py: FamaFrenchModel para estimación de μ con FF5+MOM.
- data_preprocessing.py: ProcesadorDatos para cálculo de retornos.
- performance_metrics.py: PerformanceAnalyzer para métricas agregadas.
- french_data_fetcher.py: FrenchDataFetcher para factores y tasa RF.
- fundamentals_fetcher.py: get_latest_fundamentals para screening.
- black_litterman.py: BlackLittermanQuantLib para views BL.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ypyquant.optimization.data_preprocessing import obtener_retornos_desde_precios

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS — ejes del diseño factorial
# ============================================================================

class MetodoEstimacionMu(str, Enum):
    """Método para estimar el vector de retornos esperados (μ)."""
    NINGUNO = "ninguno"           # Etapa 1: γ=∞, solo optimización de riesgo
    MUESTRA = "muestra"           # Media histórica
    CAPM = "capm"                 # CAPM: E[R_i] = Rf + β_i*(E[R_m] - Rf)
    FF5_MOM = "ff5_mom"           # Fama-French 5 factores + Momentum



class MetodoEstimacionSigma(str, Enum):
    """Método para estimar la matriz de covarianza (Σ)."""
    MUESTRA = "muestra"           # Covarianza muestral
    LEDOIT_WOLF = "ledoit_wolf"   # Shrinkage de Ledoit-Wolf


class MetodoRiesgo(str, Enum):
    """Función objetivo de optimización."""
    MPT_MIN_VAR = "mpt_min_var"             # Markowitz mínima varianza
    MPT_MAX_SHARPE = "mpt_max_sharpe"       # Markowitz máximo Sharpe
    PMPT_MIN_SEMIVAR = "pmpt_min_semivar"   # PMPT mínima semi-varianza (downside risk)
    PMPT_MIN_CVAR = "pmpt_min_cvar"         # PMPT mínimo CVaR
    PMPT_MAX_SORTINO = "pmpt_max_sortino"   # PMPT máximo Sortino


class MetodoRegularizacion(str, Enum):
    """Método de regularización del vector μ o de pesos."""
    NINGUNA = "ninguna"
    KAN_ZHOU = "kan_zhou"                     # Shrinkage Kan-Zhou con extensión Fabozzi
    RESAMPLED_EFFICIENCY = "resampled_efficiency"  # Michaud & Michaud (bootstrap de pesos)


class MetodoBL(str, Enum):
    """Método de generación de views para Black-Litterman."""
    NINGUNO = "ninguno"
    CAPM = "capm"                 # Views implícitas CAPM
    FF5 = "ff5"                   # Views desde modelo FF5



class TipoExperimento(str, Enum):
    """Clasificación del experimento dentro del diseño factorial."""
    ETAPA_1 = "etapa_1"           # Solo riesgo (γ=∞)
    ETAPA_2A = "etapa_2a"         # Retornos esperados + regularización
    ETAPA_2B = "etapa_2b"         # Black-Litterman
    BENCHMARK_EQUAL = "benchmark_equal"   # Pesos iguales (1/N)
    BENCHMARK_SP500 = "benchmark_sp500"   # S&P 500


# ============================================================================
# DATACLASSES de configuración y resultados
# ============================================================================

@dataclass
class ConfigWalkForward:
    """
    Parámetros globales del motor walk-forward.
    Compartidos por todos los experimentos.
    """
    ventana_entrenamiento_meses: int = 12
    paso_avance_meses: int = 1
    fecha_inicio_oos: str = "2010-01-01"
    fecha_fin_oos: str = "2025-01-01"
    top_n_activos: int = 50
    # Distribución por cuartiles de Sharpe (evita sesgo de optimización)
    n_q1: int = 15                       # Cuartil superior (mejores Sharpe)
    n_centrales: int = 20                # Cuartiles centrales (Q2-Q3)
    n_q4: int = 15                       # Cuartil inferior (peores Sharpe)
    ratio_datos_minimo: float = 0.95     # 95% de datos completos requerido
    tipo_retorno: str = "log"            # "log" o "simple"
    peso_maximo: float = 0.10            # Peso máximo por activo
    nivel_confianza_cvar: float = 0.95
    directorio_checkpoints: Optional[str] = None
    semilla: int = 42


@dataclass
class ConfigExperimento:
    """
    Configuración de 1 de los 21 experimentos del diseño factorial.
    """
    id_experimento: str
    nombre: str
    tipo: TipoExperimento
    metodo_sigma: MetodoEstimacionSigma = MetodoEstimacionSigma.LEDOIT_WOLF
    metodo_mu: MetodoEstimacionMu = MetodoEstimacionMu.NINGUNO
    metodo_riesgo: MetodoRiesgo = MetodoRiesgo.MPT_MIN_VAR
    regularizacion: MetodoRegularizacion = MetodoRegularizacion.NINGUNA
    metodo_bl: MetodoBL = MetodoBL.NINGUNO


@dataclass
class ResultadoVentana:
    """Resultado de una única ventana OOS (1 mes)."""
    fecha_inicio_train: pd.Timestamp
    fecha_fin_train: pd.Timestamp
    fecha_inicio_oos: pd.Timestamp
    fecha_fin_oos: pd.Timestamp
    pesos: Dict[str, float]
    retorno_oos: float
    tickers_seleccionados: List[str]
    optimizacion_convergente: bool = True
    ventana_fallida: bool = False
    tiempo_segundos: float = 0.0


@dataclass
class ResultadoExperimento:
    """Resultado completo de un experimento walk-forward."""
    config: ConfigExperimento
    ventanas: List[ResultadoVentana] = field(default_factory=list)
    retornos_oos_mensuales: Optional[pd.Series] = None
    curva_equity: Optional[pd.Series] = None
    pesos_historicos: Optional[pd.DataFrame] = None
    metricas: Dict = field(default_factory=dict)
    alpha_jensen: Optional[float] = None
    alpha_tstat: Optional[float] = None
    alpha_pvalue: Optional[float] = None


# ============================================================================
# CLASE PRINCIPAL — WalkForwardEngine
# ============================================================================

class WalkForwardEngine:
    """
    Motor de walk-forward mensual con rolling window.

    Ejecuta un loop de evaluación progresiva:
    1. Para cada mes OOS, entrena con los últimos N meses.
    2. Aplica screening, estimación de μ/Σ, regularización y optimización.
    3. Registra retornos OOS y pesos.
    4. Agrega métricas finales y calcula Jensen's alpha.

    Se relaciona con todos los módulos de optimizacion/ y con performance_metrics.py.
    """

    def __init__(
        self,
        config: ConfigWalkForward,
        db_session=None,
        mlflow_enabled: bool = False,
        mlflow_experiment_name: str = "walk_forward_factorial",
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """
        Args:
            config: Parámetros globales del walk-forward
            db_session: Sesión SQLAlchemy (opcional, para screening con fundamentals)
            mlflow_enabled: Si True, activa tracking MLflow (requiere mlflow>=2.10).
            mlflow_experiment_name: Nombre del experimento MLflow.
            mlflow_tracking_uri: URI del tracking store (None → default de mlflow).
        """
        self.config = config
        self.db_session = db_session
        np.random.seed(config.semilla)

        from ypyquant.optimization.mlflow_tracker import MLflowTracker
        self._tracker = MLflowTracker(
            enabled=mlflow_enabled,
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_tracking_uri,
        )

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def ejecutar_experimento(
        self,
        config_exp: ConfigExperimento,
        precios_universo: pd.DataFrame,
        factores_ff: Optional[pd.DataFrame] = None,
        retornos_mercado: Optional[pd.Series] = None
    ) -> ResultadoExperimento:
        """
        Ejecuta un experimento walk-forward completo.

        Args:
            config_exp: Configuración del experimento (1 de 24)
            precios_universo: DataFrame de precios diarios (columnas=tickers, índice=fechas).
                              Debe cubrir desde antes de fecha_inicio_oos - ventana_train hasta fecha_fin_oos.
            factores_ff: DataFrame con columnas FF (MKT, SMB, HML, RMW, CMA, MOM, RF).
                         Requerido si metodo_mu=FF5_MOM o metodo_bl=FF5.
            retornos_mercado: Series con retornos diarios del mercado (ej: S&P 500).
                              Requerido para Jensen's alpha y para metodo_bl=CAPM.

        Returns:
            ResultadoExperimento con ventanas, retornos OOS y métricas agregadas
        """
        logger.info(f"[WF] Iniciando experimento: {config_exp.nombre} ({config_exp.id_experimento})")
        self._tracker.start_run(config_exp)

        # Extraer RF de los factores FF si están disponibles
        serie_rf = self._extraer_risk_free_rate(factores_ff)

        # Intentar cargar checkpoint
        resultado = self._cargar_checkpoint(config_exp.id_experimento)
        ventanas_completadas = set()
        if resultado is not None and resultado.ventanas:
            ventanas_completadas = {
                v.fecha_inicio_oos.isoformat() for v in resultado.ventanas
            }
            logger.info(f"[WF] Checkpoint cargado: {len(resultado.ventanas)} ventanas previas")
        else:
            resultado = ResultadoExperimento(config=config_exp)

        # Generar ventanas temporales validadas contra el rango real de datos
        fecha_min_datos = precios_universo.index.min()
        fecha_max_datos = precios_universo.index.max()
        ventanas = self._generar_ventanas(fecha_min_datos, fecha_max_datos)
        logger.info(f"[WF] Total ventanas a procesar: {len(ventanas)}")

        for i, (train_start, train_end, oos_start, oos_end) in enumerate(ventanas):
            # Saltar ventanas ya procesadas (checkpoint)
            if oos_start.isoformat() in ventanas_completadas:
                continue

            t0 = time.time()

            try:
                resultado_ventana = self._procesar_ventana(
                    config_exp=config_exp,
                    precios=precios_universo,
                    train_start=train_start,
                    train_end=train_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    factores_ff=factores_ff,
                    serie_rf=serie_rf,
                    retornos_mercado=retornos_mercado
                )
                resultado_ventana.tiempo_segundos = time.time() - t0
                resultado.ventanas.append(resultado_ventana)
                self._tracker.log_window(resultado_ventana, step=i)

            except Exception as e:
                logger.warning(
                    f"[WF] Error en ventana {oos_start.date()} → {oos_end.date()}: {e}"
                )
                # Registrar ventana fallida — se excluye de métricas agregadas
                resultado.ventanas.append(ResultadoVentana(
                    fecha_inicio_train=train_start,
                    fecha_fin_train=train_end,
                    fecha_inicio_oos=oos_start,
                    fecha_fin_oos=oos_end,
                    pesos={},
                    retorno_oos=np.nan,
                    tickers_seleccionados=[],
                    optimizacion_convergente=False,
                    ventana_fallida=True,
                    tiempo_segundos=time.time() - t0
                ))

            # Checkpoint cada 5 ventanas
            if (i + 1) % 5 == 0:
                self._guardar_checkpoint(resultado, config_exp.id_experimento)

        # Agregar métricas finales
        resultado = self._agregar_resultados(resultado, retornos_mercado, serie_rf)

        # Checkpoint final
        self._guardar_checkpoint(resultado, config_exp.id_experimento)

        # MLflow: métricas agregadas + equity curve artifact
        self._tracker.end_run(resultado)

        logger.info(
            f"[WF] Experimento {config_exp.id_experimento} completado: "
            f"{len(resultado.ventanas)} ventanas, "
            f"Sharpe OOS={resultado.metricas.get('sharpe_ratio', 'N/A')}"
        )

        return resultado

    # ------------------------------------------------------------------
    # Procesamiento de una ventana individual
    # ------------------------------------------------------------------

    def _procesar_ventana(
        self,
        config_exp: ConfigExperimento,
        precios: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
        factores_ff: Optional[pd.DataFrame],
        serie_rf: Optional[pd.Series],
        retornos_mercado: Optional[pd.Series]
    ) -> ResultadoVentana:
        """Procesa una única ventana: screening → estimación → optimización → retorno OOS."""

        # Rama especial para S&P 500: retorno directo del mercado, sin screening ni optimización
        if config_exp.tipo == TipoExperimento.BENCHMARK_SP500:
            return self._procesar_ventana_sp500(
                retornos_mercado, oos_start, oos_end, train_start, train_end
            )

        # 1. Screening: seleccionar activos por cuartiles
        tickers = self._ejecutar_screening(precios, train_start, train_end)

        # 2. Extraer retornos de entrenamiento y validar suficiencia
        precios_train = precios.loc[train_start:train_end, tickers]
        retornos_train = obtener_retornos_desde_precios(
            precios_train, tipo_retorno=self.config.tipo_retorno
        )
        if retornos_train.empty or len(retornos_train) < 20:
            raise ValueError(
                f"Retornos de entrenamiento insuficientes: {len(retornos_train)} filas "
                f"(mínimo 20) para ventana {train_start.date()} → {train_end.date()}"
            )

        # 3. Estimar covarianza
        sigma = self._estimar_covarianza(config_exp, retornos_train)

        # 4. Estimar retornos esperados
        rf_periodo = self._obtener_rf_periodo(serie_rf, train_start, train_end)
        mu = self._estimar_retornos_esperados(
            config_exp, retornos_train, factores_ff,
            retornos_mercado, sigma, tickers, rf_periodo
        )

        # 5. Regularización
        if config_exp.regularizacion == MetodoRegularizacion.KAN_ZHOU and mu is not None:
            mu = self._regularizar_mu(mu, sigma, len(retornos_train))

        # 6. Optimizar (Resampled Efficiency actúa a nivel de pesos, no de μ)
        if config_exp.regularizacion == MetodoRegularizacion.RESAMPLED_EFFICIENCY and mu is not None:
            resultado_opt = self._optimizar_con_resampled_efficiency(
                config_exp, mu, sigma, tickers, retornos_train, rf_periodo
            )
        else:
            resultado_opt = self._optimizar(config_exp, mu, sigma, tickers, retornos_train, rf_periodo)
        pesos = resultado_opt.get('weights', {})
        convergente = resultado_opt.get('optimizer_metadata', {}).get('success', True)

        # 7. Calcular retorno OOS
        precios_oos = precios.loc[oos_start:oos_end, tickers]
        retornos_oos = obtener_retornos_desde_precios(
            precios_oos, tipo_retorno=self.config.tipo_retorno
        )
        retorno_mes = self._calcular_retorno_oos(pesos, retornos_oos)

        return ResultadoVentana(
            fecha_inicio_train=train_start,
            fecha_fin_train=train_end,
            fecha_inicio_oos=oos_start,
            fecha_fin_oos=oos_end,
            pesos=pesos,
            retorno_oos=retorno_mes,
            tickers_seleccionados=tickers,
            optimizacion_convergente=convergente
        )

    def _procesar_ventana_sp500(
        self,
        retornos_mercado: Optional[pd.Series],
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp
    ) -> ResultadoVentana:
        """
        Calcula el retorno OOS del S&P 500 directamente desde retornos_mercado.

        No requiere screening ni optimización. Para log-retornos suma los retornos
        diarios del periodo OOS; para retornos simples compone multiplicativamente.

        Args:
            retornos_mercado: Series con log-retornos (o simples) diarios del ^GSPC
            oos_start: Inicio del periodo out-of-sample
            oos_end: Fin del periodo out-of-sample
            train_start: Inicio del entrenamiento (solo para el ResultadoVentana)
            train_end: Fin del entrenamiento (solo para el ResultadoVentana)

        Returns:
            ResultadoVentana con retorno_oos del mercado y pesos vacíos
        """
        if retornos_mercado is None or retornos_mercado.empty:
            logger.warning("[WF] SP500: retornos_mercado es None o vacío, retorno OOS = 0")
            retorno_mes = 0.0
        else:
            retornos_oos = retornos_mercado.loc[oos_start:oos_end]
            if retornos_oos.empty:
                logger.warning(
                    f"[WF] SP500: sin datos OOS en {oos_start.date()}→{oos_end.date()}"
                )
                retorno_mes = 0.0
            elif self.config.tipo_retorno == "log":
                # Log-retornos son aditivos intra-periodo
                retorno_mes = float(retornos_oos.sum())
            else:
                # Retornos simples son multiplicativos
                retorno_mes = float((1 + retornos_oos).prod() - 1)

        return ResultadoVentana(
            fecha_inicio_train=train_start,
            fecha_fin_train=train_end,
            fecha_inicio_oos=oos_start,
            fecha_fin_oos=oos_end,
            pesos={},
            retorno_oos=retorno_mes,
            tickers_seleccionados=[],
            optimizacion_convergente=True
        )

    # ------------------------------------------------------------------
    # Generación de ventanas temporales
    # ------------------------------------------------------------------

    def _generar_ventanas(
        self,
        fecha_min_datos: Optional[pd.Timestamp] = None,
        fecha_max_datos: Optional[pd.Timestamp] = None
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Genera tuplas (train_start, train_end, oos_start, oos_end) para cada paso.

        La fecha oos_start avanza mensualmente desde fecha_inicio_oos.
        train_start = oos_start - ventana_entrenamiento_meses meses.
        oos_end = siguiente inicio OOS (o fecha_fin_oos).

        Valida cada ventana contra el rango real de datos disponibles:
        descarta ventanas cuyo período de entrenamiento o OOS caiga fuera
        del rango [fecha_min_datos, fecha_max_datos]. Esto evita edge cases
        en la primera/última ventana cuando los datos no cubren exactamente
        el rango configurado.

        Args:
            fecha_min_datos: Primera fecha con datos disponibles. Si es None
                             no se valida el límite inferior.
            fecha_max_datos: Última fecha con datos disponibles. Si es None
                             no se valida el límite superior.

        Returns:
            Lista de tuplas (train_start, train_end, oos_start, oos_end)
        """
        inicio_oos = pd.Timestamp(self.config.fecha_inicio_oos)
        fin_oos = pd.Timestamp(self.config.fecha_fin_oos)
        ventana_train = self.config.ventana_entrenamiento_meses

        # Generar fechas de inicio OOS mensuales
        fechas_oos = pd.date_range(start=inicio_oos, end=fin_oos, freq='MS')

        ventanas = []
        descartadas = 0
        for i in range(len(fechas_oos) - 1):
            oos_start = fechas_oos[i]
            oos_end = fechas_oos[i + 1] - pd.Timedelta(days=1)
            train_start = oos_start - pd.DateOffset(months=ventana_train)
            train_end = oos_start - pd.Timedelta(days=1)

            # Validar contra rango real de datos
            if fecha_min_datos is not None and train_start < fecha_min_datos:
                descartadas += 1
                continue
            if fecha_max_datos is not None and oos_end > fecha_max_datos:
                descartadas += 1
                continue

            ventanas.append((train_start, train_end, oos_start, oos_end))

        if descartadas > 0:
            logger.info(
                f"[WF] {descartadas} ventana(s) descartada(s) por caer fuera "
                f"del rango de datos [{fecha_min_datos}, {fecha_max_datos}]"
            )

        return ventanas

    # ------------------------------------------------------------------
    # Screening de activos
    # ------------------------------------------------------------------

    def _ejecutar_screening(
        self,
        precios: pd.DataFrame,
        fecha_inicio_train: pd.Timestamp,
        fecha_fin_train: pd.Timestamp
    ) -> List[str]:
        """
        Filtra y rankea activos para la ventana de entrenamiento.

        Selección distribuida por cuartiles de Sharpe para evitar sesgo de
        optimización (el efecto "caballo ganador"). Si solo se eligen los
        top-N por Sharpe, se sobreajusta a rendimiento pasado.

        Pipeline:
        1. Filtrar tickers con >= 95% datos completos en la ventana.
        2. Ranking por Sharpe histórico (media/std diarios, anualizado).
        3. Selección por cuartiles:
           - n_q1 del Q1 (mejores Sharpe)
           - n_centrales del Q2-Q3 (valores centrales, uniformemente espaciados)
           - n_q4 del Q4 (peores Sharpe)

        Args:
            precios: DataFrame completo de precios (columnas=tickers, índice=fechas)
            fecha_inicio_train: Inicio de la ventana de entrenamiento
            fecha_fin_train: Fin de la ventana de entrenamiento

        Returns:
            Lista con los tickers seleccionados por cuartiles
        """
        precios_ventana = precios.loc[fecha_inicio_train:fecha_fin_train]

        # 1. Filtrar tickers con datos completos
        ratio_datos = precios_ventana.notna().mean()
        tickers_completos = ratio_datos[
            ratio_datos >= self.config.ratio_datos_minimo
        ].index.tolist()

        if not tickers_completos:
            logger.warning("[WF] Screening: ningún ticker con datos completos")
            return []

        # 2. Calcular Sharpe histórico para ranking
        retornos = obtener_retornos_desde_precios(
            precios_ventana[tickers_completos], tipo_retorno=self.config.tipo_retorno
        )
        if retornos.empty:
            return tickers_completos[:self.config.top_n_activos]

        media = retornos.mean()
        std = retornos.std()
        sharpe = (media / std.replace(0, np.nan)) * np.sqrt(252)
        sharpe = sharpe.dropna().sort_values(ascending=False)

        # 3. Selección distribuida por cuartiles
        return self._seleccionar_por_cuartiles(sharpe)

    def _seleccionar_por_cuartiles(self, sharpe_ranking: pd.Series) -> List[str]:
        """
        Selecciona activos distribuidos por cuartiles de Sharpe.

        Evita el sesgo del "caballo ganador" al incluir activos de todo el
        espectro riesgo-retorno. Los activos centrales se eligen uniformemente
        espaciados dentro de Q2-Q3 para maximizar representatividad.

        Args:
            sharpe_ranking: Serie de Sharpe ordenada descendentemente

        Returns:
            Lista de tickers seleccionados (n_q1 + n_centrales + n_q4)
        """
        n_total = len(sharpe_ranking)
        n_q1 = self.config.n_q1
        n_centrales = self.config.n_centrales
        n_q4 = self.config.n_q4

        if n_total <= n_q1 + n_centrales + n_q4:
            return sharpe_ranking.index.tolist()

        # Fronteras de cuartiles
        frontera_q1 = int(n_total * 0.25)
        frontera_q4 = int(n_total * 0.75)

        tickers_q1 = sharpe_ranking.iloc[:frontera_q1].index.tolist()
        tickers_centrales = sharpe_ranking.iloc[frontera_q1:frontera_q4].index.tolist()
        tickers_q4 = sharpe_ranking.iloc[frontera_q4:].index.tolist()

        # Seleccionar de cada grupo
        selec_q1 = tickers_q1[:min(n_q1, len(tickers_q1))]
        selec_q4 = tickers_q4[:min(n_q4, len(tickers_q4))]

        # Para centrales: selección uniformemente espaciada
        if len(tickers_centrales) <= n_centrales:
            selec_centrales = tickers_centrales
        else:
            indices = np.linspace(
                0, len(tickers_centrales) - 1, n_centrales, dtype=int
            )
            selec_centrales = [tickers_centrales[i] for i in indices]

        seleccionados = selec_q1 + selec_centrales + selec_q4

        logger.debug(
            f"[WF] Screening cuartiles: Q1={len(selec_q1)}, "
            f"Q2-Q3={len(selec_centrales)}, Q4={len(selec_q4)} "
            f"(de {n_total} candidatos)"
        )

        return seleccionados

    # ------------------------------------------------------------------
    # Estimación de covarianza (Σ)
    # ------------------------------------------------------------------

    def _estimar_covarianza(
        self, config_exp: ConfigExperimento, retornos_train: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estima la matriz de covarianza según el método configurado.

        Args:
            config_exp: Configuración del experimento
            retornos_train: Retornos diarios de entrenamiento

        Returns:
            DataFrame con la matriz de covarianza anualizada
        """
        from ypyquant.optimization.portfolio_models import (
            SampleCovarianceModel, LedoitWolfShrinkageModel
        )

        if config_exp.metodo_sigma == MetodoEstimacionSigma.MUESTRA:
            return SampleCovarianceModel(frequency=252).calculate(retornos_train)
        else:
            return LedoitWolfShrinkageModel(frequency=252).calculate(retornos_train)

    # ------------------------------------------------------------------
    # Estimación de retornos esperados (μ)
    # ------------------------------------------------------------------

    def _estimar_retornos_esperados(
        self,
        config_exp: ConfigExperimento,
        retornos_train: pd.DataFrame,
        factores_ff: Optional[pd.DataFrame],
        retornos_mercado: Optional[pd.Series],
        sigma: pd.DataFrame,
        tickers: List[str],
        rf: float
    ) -> Optional[pd.Series]:
        """
        Estima retornos esperados según el método configurado.

        Devuelve None si metodo_mu=NINGUNO (Etapa 1: solo optimización de riesgo).

        Si metodo_bl != NINGUNO, combina con Black-Litterman sobre las views generadas.

        Args:
            config_exp: Configuración del experimento
            retornos_train: Retornos diarios de entrenamiento
            factores_ff: Datos de factores FF (con RF)
            retornos_mercado: Retornos diarios del mercado
            sigma: Matriz de covarianza estimada
            tickers: Lista de tickers en el universo actual
            rf: Tasa libre de riesgo del período

        Returns:
            Series con retornos esperados anualizados, o None
        """
        # --- Estimación base de μ ---
        mu = None

        if config_exp.metodo_mu == MetodoEstimacionMu.NINGUNO:
            # Etapa 1: no se estima μ
            pass

        elif config_exp.metodo_mu == MetodoEstimacionMu.MUESTRA:
            from ypyquant.optimization.portfolio_models import HistoricalReturnsModel
            mu = HistoricalReturnsModel(method="mean", frequency=252).calculate(retornos_train)

        elif config_exp.metodo_mu == MetodoEstimacionMu.CAPM:
            if retornos_mercado is None:
                raise ValueError("Retornos del mercado requeridos para metodo_mu=CAPM")
            from ypyquant.factors.factor_models import CAPMModel
            modelo_capm = CAPMModel(
                risk_free_rate=rf,
                frequency=252,
                usar_tasas_dinamicas=False
            )
            # Alinear retornos del mercado a la ventana de entrenamiento
            mercado_train = retornos_mercado.loc[
                retornos_train.index[0]:retornos_train.index[-1]
            ]
            modelo_capm.fit(retornos_train, mercado_train)
            mu = modelo_capm.calculate_expected_returns()

        elif config_exp.metodo_mu == MetodoEstimacionMu.FF5_MOM:
            if factores_ff is None:
                raise ValueError("Factores FF requeridos para metodo_mu=FF5_MOM")
            from ypyquant.factors.factor_models import FamaFrenchModel, FACTORES_FF5_MOM
            modelo_ff = FamaFrenchModel(
                factores=FACTORES_FF5_MOM,
                risk_free_rate=rf,
                frequency=252
            )
            # Filtrar factores al período de entrenamiento
            factores_periodo = factores_ff.loc[
                retornos_train.index[0]:retornos_train.index[-1]
            ]

            # Validar cobertura temporal de los factores
            self._validar_cobertura_factores(
                factores_periodo, retornos_train, "FF5_MOM._estimar_retornos_esperados"
            )

            cols_factores = [c for c in FACTORES_FF5_MOM if c in factores_periodo.columns]
            modelo_ff.factores = cols_factores
            modelo_ff.fit(retornos_train, factores_periodo)
            mu = modelo_ff.calculate_expected_returns()

        # --- Black-Litterman (Etapa 2b) ---
        if config_exp.metodo_bl != MetodoBL.NINGUNO and mu is not None:
            mu = self._aplicar_black_litterman(
                config_exp, mu, sigma, retornos_train,
                factores_ff, retornos_mercado, tickers, rf
            )

        return mu

    def _aplicar_black_litterman(
        self,
        config_exp: ConfigExperimento,
        mu_views: pd.Series,
        sigma: pd.DataFrame,
        retornos_train: pd.DataFrame,
        factores_ff: Optional[pd.DataFrame],
        retornos_mercado: Optional[pd.Series],
        tickers: List[str],
        rf: float
    ) -> pd.Series:
        """
        Aplica Black-Litterman combinando el equilibrio de mercado con views generadas.

        Las views se generan según metodo_bl (CAPM o FF5).
        El modelo BL combina el prior de equilibrio con estas views.

        Returns:
            Series con retornos esperados posteriores (BL posterior)
        """
        # Validar cobertura de factores FF si las views provienen de FF5
        if config_exp.metodo_bl == MetodoBL.FF5 and factores_ff is not None:
            factores_periodo_bl = factores_ff.loc[
                retornos_train.index[0]:retornos_train.index[-1]
            ]
            self._validar_cobertura_factores(
                factores_periodo_bl, retornos_train, "BL._aplicar_black_litterman"
            )

        try:
            from ypyquant.optimization.black_litterman import BlackLittermanQuantLib, crear_views_desde_dict

            # Pesos de mercado: proxy con pesos iguales (simplificación)
            pesos_mercado = pd.Series(1.0 / len(tickers), index=tickers)

            # Generar views absolutas desde μ estimado
            views_list = []
            for ticker in mu_views.index:
                if ticker in tickers:
                    views_list.append({
                        'tipo': 'absolute',
                        'activos': [ticker],
                        'valor_esperado': float(mu_views[ticker]),
                        'confianza': 0.5
                    })

            if not views_list:
                return mu_views

            views_objects = crear_views_desde_dict(views_list)

            bl = BlackLittermanQuantLib(
                risk_aversion=2.5,
                tau=0.025,
                metodo_omega='idzorek'
            )

            # Calcular prior de equilibrio
            prior = bl.calcular_retornos_equilibrio(sigma, pesos_mercado)

            # Calcular posterior
            posterior = bl.calcular_posterior(
                cov_matrix=sigma,
                market_weights=pesos_mercado,
                views=views_objects,
                prior=prior
            )

            return posterior

        except ImportError as e:
            logger.warning(f"[WF] Módulo Black-Litterman no instalado: {e}. Usando views directas.")
            return mu_views

        except np.linalg.LinAlgError as e:
            logger.warning(
                f"[WF] Error numérico en Black-Litterman (matriz singular o mal condicionada): {e}. "
                "Fallback a views directas."
            )
            return mu_views

        except Exception as e:
            logger.error(f"[WF] Error inesperado en Black-Litterman: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Regularización Kan-Zhou (con extensión Fabozzi)
    # ------------------------------------------------------------------

    def _regularizar_mu(
        self, mu: pd.Series, sigma: pd.DataFrame, T: int
    ) -> pd.Series:
        """
        Aplica shrinkage de Kan-Zhou al vector de retornos esperados.

        Fórmula:
            α = (N + 2) / ((N + 2) + T · (μ̂ - μ_gmv)' Σ⁻¹ (μ̂ - μ_gmv))
            μ_shrunk = (1 - α) · μ̂ + α · μ_gmv

        Extensión Fabozzi:
            ŵ_s = (1 - ν) · (1/N) · 1_N + ν · ŵ_KZ
            donde ν se calibra para controlar concentración.

        Args:
            mu: Vector de retornos esperados sin regularizar
            sigma: Matriz de covarianza
            T: Número de observaciones de entrenamiento

        Returns:
            Vector de retornos esperados regularizado
        """
        N = len(mu)
        tickers = mu.index

        try:
            sigma_inv = pd.DataFrame(
                np.linalg.inv(sigma.values),
                index=sigma.index, columns=sigma.columns
            )
        except np.linalg.LinAlgError:
            logger.warning("[WF] Matriz singular en Kan-Zhou, retornando μ sin regularizar")
            return mu

        # Portafolio de mínima varianza global (GMV)
        unos = np.ones(N)
        w_gmv = sigma_inv.values @ unos / (unos @ sigma_inv.values @ unos)
        mu_gmv = float(mu.values @ w_gmv)

        # Shrinkage intensity α
        diff = mu.values - mu_gmv
        cuadratica = float(diff @ sigma_inv.values @ diff)
        alpha = (N + 2) / ((N + 2) + T * cuadratica) if cuadratica > 0 else 1.0
        alpha = min(max(alpha, 0.0), 1.0)

        # μ regularizado
        mu_shrunk = (1 - alpha) * mu.values + alpha * mu_gmv

        # Extensión Fabozzi: mezcla con pesos iguales para evitar concentración
        nu = 1.0 - alpha  # A mayor alpha (más shrinkage), más peso a 1/N
        mu_final = (1 - nu) * (np.ones(N) / N * mu.values.mean()) + nu * mu_shrunk

        return pd.Series(mu_final, index=tickers)

    # ------------------------------------------------------------------
    # Optimización
    # ------------------------------------------------------------------

    def _optimizar(
        self,
        config_exp: ConfigExperimento,
        mu: Optional[pd.Series],
        sigma: pd.DataFrame,
        tickers: List[str],
        retornos_train: pd.DataFrame,
        rf: float
    ) -> Dict:
        """
        Ejecuta la optimización de portafolio según el método de riesgo configurado.

        Args:
            config_exp: Configuración del experimento
            mu: Retornos esperados (None para Etapa 1 con min_var)
            sigma: Matriz de covarianza
            tickers: Lista de tickers
            retornos_train: Retornos diarios de entrenamiento (para PMPT)
            rf: Tasa libre de riesgo del período

        Returns:
            Dict con 'weights' y metadata del optimizador
        """
        from ypyquant.optimization.portfolio_models import (
            MarkowitzOptimizer, PMPTMinCVaROptimizer, PMPTMaxSortinoOptimizer
        )

        restricciones = {'max_weight': self.config.peso_maximo}

        # Para min_var no se necesita μ; generar dummy si es None
        if mu is None:
            mu_dummy = pd.Series(0.0, index=tickers)
        else:
            mu_dummy = mu

        # Benchmarks especiales
        if config_exp.tipo == TipoExperimento.BENCHMARK_EQUAL:
            n = len(tickers)
            pesos_iguales = {t: 1.0 / n for t in tickers}
            return {'weights': pesos_iguales, 'optimizer_metadata': {'success': True}}

        if config_exp.tipo == TipoExperimento.BENCHMARK_SP500:
            # Retorno del mercado se usa directamente, no hay pesos por ticker
            return {'weights': {}, 'optimizer_metadata': {'success': True}}

        # Optimizadores
        if config_exp.metodo_riesgo == MetodoRiesgo.MPT_MIN_VAR:
            optimizador = MarkowitzOptimizer(objective="min_volatility", risk_free_rate=rf)
            return optimizador.optimize(mu_dummy, sigma, restricciones)

        elif config_exp.metodo_riesgo == MetodoRiesgo.MPT_MAX_SHARPE:
            if mu is None:
                raise ValueError("MPT_MAX_SHARPE requiere μ estimado (metodo_mu != NINGUNO)")
            optimizador = MarkowitzOptimizer(objective="max_sharpe", risk_free_rate=rf)
            return optimizador.optimize(mu, sigma, restricciones)

        elif config_exp.metodo_riesgo == MetodoRiesgo.PMPT_MIN_SEMIVAR:
            # Mínima semi-varianza: min w'Σ_down w usando solo retornos debajo de τ=rf
            # τ = rf diario (rf viene anualizado → dividir por 252)
            rf_diario = rf / 252
            if config_exp.metodo_sigma == MetodoEstimacionSigma.LEDOIT_WOLF:
                from ypyquant.optimization.portfolio_models import LedoitWolfSemiCovarianceModel
                semi_cov = LedoitWolfSemiCovarianceModel(
                    target_return=rf_diario, frequency=252
                ).calculate(retornos_train)
            else:
                from ypyquant.optimization.portfolio_models import SemiCovarianceModel
                semi_cov = SemiCovarianceModel(
                    target_return=rf_diario, frequency=252
                ).calculate(retornos_train)
            optimizador = MarkowitzOptimizer(objective="min_volatility", risk_free_rate=rf)
            return optimizador.optimize(mu_dummy, semi_cov, restricciones)

        elif config_exp.metodo_riesgo == MetodoRiesgo.PMPT_MIN_CVAR:
            optimizador = PMPTMinCVaROptimizer(
                nivel_confianza=self.config.nivel_confianza_cvar,
                risk_free_rate=rf
            )
            return optimizador.optimize(
                mu_dummy, sigma, restricciones,
                retornos_historicos=retornos_train
            )

        elif config_exp.metodo_riesgo == MetodoRiesgo.PMPT_MAX_SORTINO:
            if mu is None:
                raise ValueError("PMPT_MAX_SORTINO requiere μ estimado (metodo_mu != NINGUNO)")
            # Calcular semi-covarianza Σ^down (igual que PMPT_MIN_SEMIVAR)
            rf_diario = rf / 252
            if config_exp.metodo_sigma == MetodoEstimacionSigma.LEDOIT_WOLF:
                from ypyquant.optimization.portfolio_models import LedoitWolfSemiCovarianceModel
                semi_cov = LedoitWolfSemiCovarianceModel(
                    target_return=rf_diario, frequency=252
                ).calculate(retornos_train)
            else:
                from ypyquant.optimization.portfolio_models import SemiCovarianceModel
                semi_cov = SemiCovarianceModel(
                    target_return=rf_diario, frequency=252
                ).calculate(retornos_train)
            optimizador = PMPTMaxSortinoOptimizer(
                risk_free_rate=rf,
                target_return=rf_diario
            )
            return optimizador.optimize(mu, semi_cov, restricciones)

        else:
            raise ValueError(f"Método de riesgo no soportado: {config_exp.metodo_riesgo}")

    # ------------------------------------------------------------------
    # Resampled Efficiency (Michaud & Michaud)
    # ------------------------------------------------------------------

    def _optimizar_con_resampled_efficiency(
        self,
        config_exp: ConfigExperimento,
        mu: pd.Series,
        sigma: pd.DataFrame,
        tickers: List[str],
        retornos_train: pd.DataFrame,
        rf: float,
        n_simulaciones: int = 100
    ) -> Dict:
        """
        Regularización por Resampled Efficiency (Michaud & Michaud, 1998).

        En vez de aplicar shrinkage al vector μ, perturba (μ, Σ) N veces
        mediante bootstrap paramétrico y promedia los pesos óptimos resultantes.
        Esto reduce el error de estimación sin alterar la función objetivo.

        Algoritmo:
        1. Para i en 1..N:
           a. μ̂_i ~ N(μ̂, Σ/T)            — perturbación del vector de retornos
           b. Σ̂_i ~ Wishart(T-1, Σ/(T-1))  — perturbación de la covarianza
           c. Resolver optimización estándar con (μ̂_i, Σ̂_i) → pesos_i
           d. Descartar si SLSQP no convergió (success=False)
        2. Retornar pesos_promedio = mean(pesos exitosos)
        3. Early stopping si pesos promedio estabilizan (norma de cambio < 1e-4)

        Args:
            config_exp: Configuración del experimento
            mu: Vector de retornos esperados estimado
            sigma: Matriz de covarianza estimada
            tickers: Lista de tickers
            retornos_train: Retornos diarios de entrenamiento (para PMPT)
            rf: Tasa libre de riesgo
            n_simulaciones: Número de muestras bootstrap (default 100)

        Returns:
            Dict con pesos promediados y metadata
        """
        from scipy.stats import wishart

        t0 = time.time()
        T = len(retornos_train)
        N = len(tickers)
        mu_vals = mu.values
        sigma_vals = sigma.values

        # Contadores de resultados
        pesos_acumulados = np.zeros(N)
        simulaciones_exitosas = 0
        fallos_convergencia = 0
        fallos_excepcion = 0

        # Escala para Wishart: Σ/(T-1)
        escala_wishart = sigma_vals / (T - 1)

        # Asegurar que la escala sea definida positiva para Wishart
        try:
            np.linalg.cholesky(escala_wishart)
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(escala_wishart)
            min_eig = eigvals.min()
            if min_eig <= 0:
                escala_wishart += (-min_eig + 1e-8) * np.eye(N)

        rng = np.random.default_rng(self.config.semilla)

        for i in range(n_simulaciones):
            try:
                # a. Perturbar μ ~ N(μ̂, Σ/T)
                mu_perturbado = rng.multivariate_normal(mu_vals, sigma_vals / T)
                mu_perturbado_series = pd.Series(mu_perturbado, index=tickers)

                # b. Perturbar Σ ~ Wishart(T-1, Σ/(T-1))
                sigma_perturbado = wishart.rvs(
                    df=T - 1, scale=escala_wishart, random_state=rng
                )
                sigma_perturbado_df = pd.DataFrame(
                    sigma_perturbado, index=tickers, columns=tickers
                )

                # c. Optimizar con parámetros perturbados
                resultado_i = self._optimizar(
                    config_exp, mu_perturbado_series, sigma_perturbado_df,
                    tickers, retornos_train, rf
                )

                # d. Filtrar optimizaciones que no convergieron
                exito_opt = resultado_i.get('optimizer_metadata', {}).get('success', True)
                if not exito_opt:
                    fallos_convergencia += 1
                    continue

                pesos_i = np.array([resultado_i['weights'].get(t, 0.0) for t in tickers])
                pesos_acumulados += pesos_i
                simulaciones_exitosas += 1

                # Early stopping: tras 50 simulaciones exitosas, verificar estabilidad
                if simulaciones_exitosas >= 50 and simulaciones_exitosas % 10 == 0:
                    pesos_actual = pesos_acumulados / simulaciones_exitosas
                    pesos_actual_norm = pesos_actual / pesos_actual.sum() if pesos_actual.sum() > 0 else pesos_actual
                    # Comparar con promedio sin la última contribución
                    pesos_prev = (pesos_acumulados - pesos_i) / (simulaciones_exitosas - 1)
                    pesos_prev_norm = pesos_prev / pesos_prev.sum() if pesos_prev.sum() > 0 else pesos_prev
                    cambio_norma = np.linalg.norm(pesos_actual_norm - pesos_prev_norm)
                    if cambio_norma < 1e-4:
                        logger.info(
                            f"[WF] Resampled early stop en sim {i}: "
                            f"cambio norma={cambio_norma:.2e} < 1e-4"
                        )
                        break

            except Exception as e:
                fallos_excepcion += 1
                if fallos_excepcion <= 3:
                    logger.warning(
                        f"[WF] Resampled sim {i}: {type(e).__name__}: {e}"
                    )
                continue

        if simulaciones_exitosas == 0:
            logger.warning(
                "[WF] Resampled Efficiency: 0 simulaciones exitosas, "
                "fallback a optimización estándar"
            )
            return self._optimizar(config_exp, mu, sigma, tickers, retornos_train, rf)

        # Promediar pesos y renormalizar
        pesos_promedio = pesos_acumulados / simulaciones_exitosas
        pesos_promedio = np.maximum(pesos_promedio, 0)
        suma = pesos_promedio.sum()
        if suma > 0:
            pesos_promedio /= suma

        tiempo_total = time.time() - t0
        logger.info(
            f"[WF] Resampled Efficiency: {simulaciones_exitosas}/{n_simulaciones} exitosas, "
            f"{fallos_convergencia} no convergieron, {fallos_excepcion} excepciones "
            f"({tiempo_total:.1f}s)"
        )

        return {
            'weights': {t: float(w) for t, w in zip(tickers, pesos_promedio)},
            'optimizer_metadata': {
                'success': True,
                'method': 'resampled_efficiency',
                'simulaciones_exitosas': simulaciones_exitosas,
                'n_simulaciones': n_simulaciones,
                'fallos_convergencia': fallos_convergencia,
                'fallos_excepcion': fallos_excepcion,
                'tiempo_segundos': tiempo_total,
            }
        }

    # ------------------------------------------------------------------
    # Cálculo de retornos OOS
    # ------------------------------------------------------------------

    def _calcular_retorno_oos(
        self, pesos: Dict[str, float], retornos_oos: pd.DataFrame
    ) -> float:
        """
        Calcula el retorno out-of-sample del portafolio para un mes.

        Para log-retornos: sum(w_i * sum(r_diarios_i)) — aditivos intra-período.
        Para retornos simples: sum(w_i * (prod(1 + r_diarios_i) - 1)).

        Args:
            pesos: Dict con pesos por ticker
            retornos_oos: DataFrame con retornos diarios del período OOS

        Returns:
            Retorno del portafolio en el mes
        """
        if not pesos or retornos_oos.empty:
            return 0.0

        tickers_comunes = [t for t in pesos if t in retornos_oos.columns]
        if not tickers_comunes:
            return 0.0

        if self.config.tipo_retorno == "log":
            # Log-retornos son aditivos: retorno del mes = suma de retornos diarios
            retorno = sum(
                pesos[t] * retornos_oos[t].sum()
                for t in tickers_comunes
            )
        else:
            # Retornos simples son multiplicativos
            retorno = sum(
                pesos[t] * ((1 + retornos_oos[t]).prod() - 1)
                for t in tickers_comunes
            )

        return float(retorno)

    # ------------------------------------------------------------------
    # Agregación de resultados y Jensen's alpha
    # ------------------------------------------------------------------

    def _agregar_resultados(
        self,
        resultado: ResultadoExperimento,
        retornos_mercado: Optional[pd.Series],
        serie_rf: Optional[pd.Series]
    ) -> ResultadoExperimento:
        """
        Calcula métricas agregadas: serie OOS, curva de equity, Jensen's alpha.

        Args:
            resultado: ResultadoExperimento con ventanas procesadas
            retornos_mercado: Retornos diarios del mercado (para benchmark)
            serie_rf: Serie con tasa RF diaria

        Returns:
            ResultadoExperimento enriquecido con métricas
        """
        if not resultado.ventanas:
            return resultado

        # Filtrar ventanas fallidas para que no contaminen las métricas.
        # getattr para compatibilidad con checkpoints antiguos sin el campo ventana_fallida
        ventanas_validas = [
            v for v in resultado.ventanas
            if not getattr(v, 'ventana_fallida', False)
        ]
        n_fallidas = len(resultado.ventanas) - len(ventanas_validas)
        if n_fallidas > 0:
            logger.warning(
                f"[WF] {n_fallidas} ventana(s) fallida(s) excluida(s) de las métricas "
                f"({len(ventanas_validas)} válidas de {len(resultado.ventanas)} totales)"
            )

        if not ventanas_validas:
            logger.warning("[WF] Todas las ventanas fallaron, no se calculan métricas")
            return resultado

        # Construir serie de retornos OOS mensuales (solo ventanas válidas)
        fechas = [v.fecha_inicio_oos for v in ventanas_validas]
        retornos = [v.retorno_oos for v in ventanas_validas]
        resultado.retornos_oos_mensuales = pd.Series(retornos, index=pd.DatetimeIndex(fechas))

        # Curva de equity (base 1)
        if self.config.tipo_retorno == "log":
            resultado.curva_equity = np.exp(resultado.retornos_oos_mensuales.cumsum())
        else:
            resultado.curva_equity = (1 + resultado.retornos_oos_mensuales).cumprod()

        # Pesos históricos (solo ventanas válidas con pesos reales)
        registros_pesos = []
        for v in ventanas_validas:
            fila = {'fecha': v.fecha_inicio_oos, **v.pesos}
            registros_pesos.append(fila)
        resultado.pesos_historicos = pd.DataFrame(registros_pesos).set_index('fecha')

        # Métricas con PerformanceAnalyzer
        rf_media = 0.02  # fallback
        if serie_rf is not None and not serie_rf.empty:
            rf_media = float(serie_rf.mean() * 252)

        try:
            from ypyquant.portfolio.performance_metrics import PerformanceAnalyzer
            metricas_obj = PerformanceAnalyzer.calculate_metrics(
                resultado.retornos_oos_mensuales,
                risk_free_rate=rf_media,
                periods_per_year=12
            )
            resultado.metricas = {
                'retorno_anualizado': metricas_obj.annualized_return,
                'volatilidad': metricas_obj.volatility,
                'sharpe_ratio': metricas_obj.sharpe_ratio,
                'sortino_ratio': metricas_obj.sortino_ratio,
                'max_drawdown': metricas_obj.max_drawdown,
                'var_95': metricas_obj.var_95,
                'cvar_95': metricas_obj.cvar_95,
                'calmar_ratio': metricas_obj.calmar_ratio,
            }

            # Tracking error vs benchmark
            if retornos_mercado is not None:
                retornos_mercado_mensuales = retornos_mercado.resample('MS').sum()
                benchmark_alineado = retornos_mercado_mensuales.reindex(
                    resultado.retornos_oos_mensuales.index
                ).dropna()
                portafolio_alineado = resultado.retornos_oos_mensuales.reindex(
                    benchmark_alineado.index
                ).dropna()

                if len(portafolio_alineado) > 2 and len(benchmark_alineado) > 2:
                    comparacion = PerformanceAnalyzer.compare_to_benchmark(
                        portafolio_alineado, benchmark_alineado,
                        risk_free_rate=rf_media
                    )
                    resultado.metricas['tracking_error'] = comparacion.get('tracking_error')
                    resultado.metricas['beta'] = comparacion.get('beta')
                    resultado.metricas['information_ratio'] = comparacion.get('information_ratio')

        except (ImportError, Exception) as e:
            logger.warning(f"[WF] PerformanceAnalyzer no disponible: {e}. Métricas manuales.")
            ret_mensuales = resultado.retornos_oos_mensuales
            resultado.metricas = {
                'retorno_anualizado': float(ret_mensuales.mean() * 12),
                'volatilidad': float(ret_mensuales.std() * np.sqrt(12)),
                'sharpe_ratio': float(
                    (ret_mensuales.mean() * 12 - rf_media)
                    / (ret_mensuales.std() * np.sqrt(12))
                ) if ret_mensuales.std() > 0 else 0.0,
                'max_drawdown': float(
                    (resultado.curva_equity / resultado.curva_equity.cummax() - 1).min()
                ),
            }

        # Jensen's alpha via regresión OLS
        self._calcular_alpha_jensen(resultado, retornos_mercado, serie_rf)

        return resultado

    def _calcular_alpha_jensen(
        self,
        resultado: ResultadoExperimento,
        retornos_mercado: Optional[pd.Series],
        serie_rf: Optional[pd.Series]
    ) -> None:
        """
        Calcula Jensen's alpha: α = R_p - RF - β(R_m - RF).

        Estima α y β mediante regresión OLS de los excess returns mensuales
        del portafolio contra los excess returns del mercado.

        Modifica resultado in-place.
        """
        if retornos_mercado is None or resultado.retornos_oos_mensuales is None:
            return

        try:
            import statsmodels.api as sm

            # Retornos mensuales del mercado
            retornos_mkt_mensuales = retornos_mercado.resample('MS').sum()

            # RF mensual
            if serie_rf is not None and not serie_rf.empty:
                rf_mensual = serie_rf.resample('MS').mean()
            else:
                rf_mensual = pd.Series(0.02 / 12, index=retornos_mkt_mensuales.index)

            # Alinear series
            datos = pd.DataFrame({
                'r_portafolio': resultado.retornos_oos_mensuales,
                'r_mercado': retornos_mkt_mensuales,
                'rf': rf_mensual
            }).dropna()

            if len(datos) < 6:
                return

            # Excess returns
            y = datos['r_portafolio'] - datos['rf']
            X = sm.add_constant(datos['r_mercado'] - datos['rf'])

            modelo = sm.OLS(y, X).fit()

            resultado.alpha_jensen = float(modelo.params.iloc[0])
            resultado.alpha_tstat = float(modelo.tvalues.iloc[0])
            resultado.alpha_pvalue = float(modelo.pvalues.iloc[0])

            # Rolling beta (ventana 12 meses) como metadata de estabilidad
            if len(datos) >= 12:
                try:
                    from statsmodels.regression.rolling import RollingOLS as _RollingOLS
                    rolling_res = _RollingOLS(y, X, window=12).fit()
                    betas_rolling = rolling_res.params.iloc[:, 1].dropna()
                    resultado.metricas['rolling_beta_mean'] = float(betas_rolling.mean())
                    resultado.metricas['rolling_beta_std'] = float(betas_rolling.std())
                except Exception as exc_rolling:
                    logger.debug(f"[WF] Rolling beta no disponible: {exc_rolling}")

        except ImportError:
            # Fallback con scipy
            try:
                from scipy import stats as sp_stats

                retornos_mkt_mensuales = retornos_mercado.resample('MS').sum()
                rf_media_mensual = 0.02 / 12
                if serie_rf is not None and not serie_rf.empty:
                    rf_media_mensual = float(serie_rf.mean())

                datos = pd.DataFrame({
                    'r_p': resultado.retornos_oos_mensuales,
                    'r_m': retornos_mkt_mensuales,
                }).dropna()

                if len(datos) < 6:
                    return

                y = datos['r_p'] - rf_media_mensual
                x = datos['r_m'] - rf_media_mensual

                slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

                resultado.alpha_jensen = float(intercept)
                resultado.alpha_tstat = float(intercept / std_err) if std_err > 0 else 0.0
                resultado.alpha_pvalue = float(p_value)

            except Exception as e:
                logger.warning(f"[WF] No se pudo calcular Jensen's alpha: {e}")

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _validar_cobertura_factores(
        self,
        factores_periodo: pd.DataFrame,
        retornos_train: pd.DataFrame,
        contexto: str
    ) -> None:
        """
        Verifica que los datos de factores FF cubran adecuadamente el periodo de entrenamiento.

        Lanza ValueError si no hay datos de factores. Emite warning si la cobertura
        es inferior al 80% de los días de entrenamiento.

        Args:
            factores_periodo: DataFrame de factores filtrado al periodo de entrenamiento
            retornos_train: Retornos diarios de entrenamiento (referencia de cobertura)
            contexto: Cadena descriptiva para los mensajes de log (ej: "FF5_MOM._estimar")
        """
        if factores_periodo.empty:
            raise ValueError(
                f"[WF] {contexto}: factores FF vacíos para el periodo "
                f"{retornos_train.index[0].date()}→{retornos_train.index[-1].date()}. "
                "Kenneth French no cubre este rango."
            )

        cobertura = len(factores_periodo) / len(retornos_train)
        if cobertura < 0.80:
            logger.warning(
                f"[WF] {contexto}: cobertura baja de factores FF = {cobertura:.1%} "
                f"({len(factores_periodo)}/{len(retornos_train)} días) en periodo "
                f"{retornos_train.index[0].date()}→{retornos_train.index[-1].date()}"
            )

    def _extraer_risk_free_rate(
        self, factores_ff: Optional[pd.DataFrame]
    ) -> Optional[pd.Series]:
        """
        Extrae la serie RF de los datos Fama-French.

        La columna RF contiene la tasa diaria del T-Bill en términos decimales.
        """
        if factores_ff is not None and 'RF' in factores_ff.columns:
            return factores_ff['RF']
        return None

    def _obtener_rf_periodo(
        self, serie_rf: Optional[pd.Series],
        fecha_inicio: pd.Timestamp,
        fecha_fin: pd.Timestamp
    ) -> float:
        """
        Obtiene la tasa RF anualizada promedio para un período de entrenamiento.

        Si no hay datos de RF, usa fallback de 0.02 (2%).
        """
        if serie_rf is None:
            return 0.02

        rf_periodo = serie_rf.loc[fecha_inicio:fecha_fin]
        if rf_periodo.empty:
            return 0.02

        # RF en datos FF está en términos diarios → anualizar
        return float(rf_periodo.mean() * 252)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _guardar_checkpoint(self, resultado: ResultadoExperimento, id_exp: str) -> None:
        """Guarda estado parcial del experimento a disco (pickle)."""
        if self.config.directorio_checkpoints is None:
            return

        try:
            directorio = Path(self.config.directorio_checkpoints)
            directorio.mkdir(parents=True, exist_ok=True)
            ruta = directorio / f"checkpoint_{id_exp}.pkl"
            with open(ruta, 'wb') as f:
                pickle.dump(resultado, f)
            logger.debug(f"[WF] Checkpoint guardado: {ruta}")
        except Exception as e:
            logger.warning(f"[WF] Error guardando checkpoint: {e}")

    def _cargar_checkpoint(self, id_exp: str) -> Optional[ResultadoExperimento]:
        """Carga estado parcial desde disco si existe."""
        if self.config.directorio_checkpoints is None:
            return None

        ruta = Path(self.config.directorio_checkpoints) / f"checkpoint_{id_exp}.pkl"
        if not ruta.exists():
            return None

        try:
            with open(ruta, 'rb') as f:
                resultado = pickle.load(f)
            logger.info(f"[WF] Checkpoint cargado desde {ruta}")
            return resultado
        except Exception as e:
            logger.warning(f"[WF] Error cargando checkpoint: {e}")
            return None

    # ------------------------------------------------------------------
    # Ejecución masiva
    # ------------------------------------------------------------------

    def ejecutar_todos(
        self,
        experimentos: List[ConfigExperimento],
        precios_universo: pd.DataFrame,
        factores_ff: Optional[pd.DataFrame] = None,
        retornos_mercado: Optional[pd.Series] = None
    ) -> Dict[str, ResultadoExperimento]:
        """
        Ejecuta todos los experimentos secuencialmente.

        Args:
            experimentos: Lista de configuraciones de experimentos
            precios_universo: DataFrame de precios
            factores_ff: DataFrame de factores FF5+MOM+RF
            retornos_mercado: Retornos del mercado (S&P 500)

        Returns:
            Dict {id_experimento: ResultadoExperimento}
        """
        resultados = {}
        for i, exp in enumerate(experimentos, 1):
            logger.info(f"[WF] === Experimento {i}/{len(experimentos)}: {exp.nombre} ===")
            try:
                resultado = self.ejecutar_experimento(
                    exp, precios_universo, factores_ff, retornos_mercado
                )
                resultados[exp.id_experimento] = resultado
            except Exception as e:
                logger.error(f"[WF] Error fatal en {exp.id_experimento}: {e}")
                resultados[exp.id_experimento] = ResultadoExperimento(config=exp)

        return resultados

    @staticmethod
    def generar_tabla_comparativa(
        resultados: Dict[str, ResultadoExperimento]
    ) -> pd.DataFrame:
        """
        Genera tabla resumen comparando todos los experimentos.

        Returns:
            DataFrame con una fila por experimento y columnas de métricas
        """
        filas = []
        for id_exp, res in resultados.items():
            fila = {
                'id': id_exp,
                'nombre': res.config.nombre,
                'tipo': res.config.tipo.value,
                'n_ventanas': len(res.ventanas),
                'alpha_jensen': res.alpha_jensen,
                'alpha_pvalue': res.alpha_pvalue,
            }
            fila.update(res.metricas)
            filas.append(fila)

        tabla = pd.DataFrame(filas)
        if 'sharpe_ratio' in tabla.columns:
            tabla = tabla.sort_values('sharpe_ratio', ascending=False)
        return tabla


# ============================================================================
# FÁBRICA DE EXPERIMENTOS — diseño factorial completo
# ============================================================================

class FabricaExperimentos:
    """
    Genera las configuraciones del diseño factorial según la pauta experimental.

    Diseño:
    - Etapa 1: 2(Σ) × 2(riesgo: min_var, min_semivar) + 1(CVaR) = 5 experimentos
    - Etapa 2a: 3(μ: muestra, CAPM, FF5+MOM) × 2(reg: ninguna, resampled) × 2(riesgo) = 12
    - Etapa 2b: 1(BL: FF5) × 2(riesgo) = 2 experimentos
    - Benchmarks: 2 (Equal Weight + S&P 500)
    Total: 21
    """

    @staticmethod
    def crear_todos() -> List[ConfigExperimento]:
        """
        Genera la lista completa de configuraciones experimentales.

        Returns:
            Lista de ConfigExperimento (21 experimentos)
        """
        experimentos = []
        contador = 1

        # ============================
        # ETAPA 1: solo riesgo (γ=∞, inversor completamente averso)
        # 2(Σ) × 2(riesgo: min_var, min_semivar) + 1(min_cvar con LW) = 5
        #
        # Se evalúa el efecto aislado de la estimación de Σ comparando
        # muestra vs Ledoit-Wolf en min_var y min_semivar.
        # CVaR es empírico (no depende de Σ), se ejecuta una sola vez con LW.
        # ============================
        riesgos_dependientes_sigma = [
            MetodoRiesgo.MPT_MIN_VAR,       # Mínima varianza global (MPT)
            MetodoRiesgo.PMPT_MIN_SEMIVAR,   # Mínima semi-varianza (PMPT)
        ]
        for sigma in MetodoEstimacionSigma:
            for riesgo in riesgos_dependientes_sigma:
                exp = ConfigExperimento(
                    id_experimento=f"E1_{contador:02d}",
                    nombre=f"Etapa1 Σ={sigma.value} Riesgo={riesgo.value}",
                    tipo=TipoExperimento.ETAPA_1,
                    metodo_sigma=sigma,
                    metodo_mu=MetodoEstimacionMu.NINGUNO,
                    metodo_riesgo=riesgo,
                    regularizacion=MetodoRegularizacion.NINGUNA,
                    metodo_bl=MetodoBL.NINGUNO
                )
                experimentos.append(exp)
                contador += 1

        # CVaR empírico: no depende de Σ → una sola ejecución con LW por defecto
        exp = ConfigExperimento(
            id_experimento=f"E1_{contador:02d}",
            nombre="Etapa1 Riesgo=pmpt_min_cvar",
            tipo=TipoExperimento.ETAPA_1,
            metodo_sigma=MetodoEstimacionSigma.LEDOIT_WOLF,
            metodo_mu=MetodoEstimacionMu.NINGUNO,
            metodo_riesgo=MetodoRiesgo.PMPT_MIN_CVAR,
            regularizacion=MetodoRegularizacion.NINGUNA,
            metodo_bl=MetodoBL.NINGUNO
        )
        experimentos.append(exp)
        contador += 1

        # ============================
        # ETAPA 2a: marco frecuentista (μ estimado + regularización)
        # 3 μ × 2 reg (ninguna, resampled_efficiency) × 2 riesgo = 12
        #
        # Se fija el Σ ganador de Etapa 1 (Ledoit-Wolf por defecto).
        # μ estimado por: media muestral, CAPM (OLS), FF5+Momentum.
        # Cada estimador se prueba sin/con regularización Resampled Efficiency.
        # ============================
        metodos_mu_2a = [
            MetodoEstimacionMu.MUESTRA,      # Media histórica (benchmark)
            MetodoEstimacionMu.CAPM,          # CAPM por regresión OLS
            MetodoEstimacionMu.FF5_MOM,       # Fama-French 5 + Momentum
        ]
        regularizaciones_2a = [
            MetodoRegularizacion.NINGUNA,
            MetodoRegularizacion.RESAMPLED_EFFICIENCY,
        ]
        for mu_met in metodos_mu_2a:
            for reg in regularizaciones_2a:
                for riesgo in [MetodoRiesgo.MPT_MAX_SHARPE, MetodoRiesgo.PMPT_MAX_SORTINO]:
                    exp = ConfigExperimento(
                        id_experimento=f"E2A_{contador:02d}",
                        nombre=f"Etapa2a μ={mu_met.value} Reg={reg.value} Riesgo={riesgo.value}",
                        tipo=TipoExperimento.ETAPA_2A,
                        metodo_sigma=MetodoEstimacionSigma.LEDOIT_WOLF,
                        metodo_mu=mu_met,
                        metodo_riesgo=riesgo,
                        regularizacion=reg,
                        metodo_bl=MetodoBL.NINGUNO
                    )
                    experimentos.append(exp)
                    contador += 1

        # ============================
        # ETAPA 2b: marco bayesiano (Black-Litterman)
        # 1 BL(FF5) × 2 riesgo = 2
        #
        # Las views se construyen con FF5+MOM (no CAPM, ya incluido en la
        # distribución prior Π=δΣw_mkt; no muestral, no aporta estructura).
        # BL actúa como regularización bayesiana: no se aplica Kan-Zhou.
        # ============================
        for riesgo in [MetodoRiesgo.MPT_MAX_SHARPE, MetodoRiesgo.PMPT_MAX_SORTINO]:
            exp = ConfigExperimento(
                id_experimento=f"E2B_{contador:02d}",
                nombre=f"Etapa2b BL=ff5 Riesgo={riesgo.value}",
                tipo=TipoExperimento.ETAPA_2B,
                metodo_sigma=MetodoEstimacionSigma.LEDOIT_WOLF,
                metodo_mu=MetodoEstimacionMu.FF5_MOM,
                metodo_riesgo=riesgo,
                regularizacion=MetodoRegularizacion.NINGUNA,
                metodo_bl=MetodoBL.FF5
            )
            experimentos.append(exp)
            contador += 1

        # ============================
        # BENCHMARKS
        # ============================
        experimentos.append(ConfigExperimento(
            id_experimento=f"BM_{contador:02d}",
            nombre="Benchmark Equal Weight (1/N)",
            tipo=TipoExperimento.BENCHMARK_EQUAL,
            metodo_sigma=MetodoEstimacionSigma.MUESTRA,
            metodo_mu=MetodoEstimacionMu.NINGUNO,
            metodo_riesgo=MetodoRiesgo.MPT_MIN_VAR,
        ))
        contador += 1

        experimentos.append(ConfigExperimento(
            id_experimento=f"BM_{contador:02d}",
            nombre="Benchmark S&P 500",
            tipo=TipoExperimento.BENCHMARK_SP500,
            metodo_sigma=MetodoEstimacionSigma.MUESTRA,
            metodo_mu=MetodoEstimacionMu.NINGUNO,
            metodo_riesgo=MetodoRiesgo.MPT_MIN_VAR,
        ))

        logger.info(f"[Fábrica] Generados {len(experimentos)} experimentos")
        return experimentos
