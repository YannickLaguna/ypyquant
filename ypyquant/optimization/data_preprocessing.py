"""
Módulo de Preprocesamiento de Datos para Optimización de Portafolios
=====================================================================

Este módulo es la FUENTE CENTRALIZADA para el cálculo de retornos en todo el sistema.
Cualquier módulo que necesite calcular retornos DEBE usar este preprocesador.

Responsabilidades:
1. Calcular retornos logarítmicos o simples (centralizado)
2. Calcular retornos para múltiples períodos (diario, semanal, mensual, semestral)
3. Normalizar datos (estandarización, escalado)
4. Limpiar datos (NaN, infinitos, outliers)
5. Validar calidad de datos
6. Proveer funciones de anualización correctas según tipo de retorno

Flujo de uso:
    precios_df → ProcesadorDatos → RetornosMultiPeriodo → Optimización/Análisis

IMPORTANTE - Consistencia de Retornos:
    - Log-retornos: Aditivos temporalmente, anualización: mean * 252
    - Retornos simples: Multiplicativos, anualización: (1 + mean)^252 - 1
    - NUNCA mezclar tipos de retornos en el mismo análisis
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal, Dict, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES DE TRADING
# =============================================================================

class DiasTrading:
    """
    Constantes para días de trading por período.
    Basado en mercados estadounidenses (~252 días/año).
    """
    DIARIO = 1
    SEMANAL = 5          # 5 días de trading por semana
    MENSUAL = 21         # ~21 días de trading por mes
    TRIMESTRAL = 63      # ~63 días de trading por trimestre
    SEMESTRAL = 126      # ~126 días de trading por semestre
    ANUAL = 252          # ~252 días de trading por año


class TipoRetorno(str, Enum):
    """Tipos de retornos soportados."""
    LOG = "log"
    SIMPLE = "simple"


class Periodo(str, Enum):
    """Períodos de retorno disponibles."""
    DIARIO = "diario"
    SEMANAL = "semanal"
    MENSUAL = "mensual"
    SEMESTRAL = "semestral"


# Mapeo de período a días de trading
DIAS_POR_PERIODO: Dict[Periodo, int] = {
    Periodo.DIARIO: DiasTrading.DIARIO,
    Periodo.SEMANAL: DiasTrading.SEMANAL,
    Periodo.MENSUAL: DiasTrading.MENSUAL,
    Periodo.SEMESTRAL: DiasTrading.SEMESTRAL,
}

# Factor de anualización por período (cuántos períodos hay en un año)
FACTOR_ANUALIZACION: Dict[Periodo, float] = {
    Periodo.DIARIO: DiasTrading.ANUAL,           # 252 días/año
    Periodo.SEMANAL: DiasTrading.ANUAL / DiasTrading.SEMANAL,    # ~50.4 semanas/año
    Periodo.MENSUAL: DiasTrading.ANUAL / DiasTrading.MENSUAL,    # 12 meses/año
    Periodo.SEMESTRAL: DiasTrading.ANUAL / DiasTrading.SEMESTRAL,  # 2 semestres/año
}


# =============================================================================
# CONTENEDOR DE RETORNOS MULTI-PERÍODO
# =============================================================================

@dataclass
class RetornosMultiPeriodo:
    """
    Contenedor para retornos calculados en múltiples períodos temporales.

    Almacena retornos diarios, semanales, mensuales y semestrales,
    junto con metadata del procesamiento.

    Attributes:
        tipo_retorno: 'log' o 'simple' - tipo de retornos calculados
        diario: DataFrame con retornos diarios (principal para optimización)
        semanal: DataFrame con retornos semanales
        mensual: DataFrame con retornos mensuales
        semestral: DataFrame con retornos semestrales
        precios_originales: DataFrame con precios originales (referencia)
        metadata: Diccionario con información del procesamiento

    Ejemplo:
        >>> procesador = ProcesadorDatos(tipo_retorno="log")
        >>> retornos = procesador.procesar_completo(precios_df)
        >>> retornos.diario  # Para optimización
        >>> retornos.mensual  # Para análisis de largo plazo
        >>> retornos.anualizar_media(retornos.diario.mean())  # Media anualizada
    """
    tipo_retorno: TipoRetorno
    diario: pd.DataFrame
    semanal: pd.DataFrame
    mensual: pd.DataFrame
    semestral: pd.DataFrame
    precios_originales: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def obtener_retornos(self, periodo: Periodo) -> pd.DataFrame:
        """
        Obtiene retornos para un período específico.

        Args:
            periodo: Período deseado (Periodo.DIARIO, etc.)

        Returns:
            DataFrame con retornos del período solicitado
        """
        mapeo = {
            Periodo.DIARIO: self.diario,
            Periodo.SEMANAL: self.semanal,
            Periodo.MENSUAL: self.mensual,
            Periodo.SEMESTRAL: self.semestral,
        }
        return mapeo[periodo]

    def anualizar_media(
        self,
        media_retornos: pd.Series,
        periodo_origen: Periodo = Periodo.DIARIO
    ) -> pd.Series:
        """
        Anualiza la media de retornos usando la fórmula correcta según el tipo.

        Para log-retornos: media_anual = media_periodo * factor
        Para simples: media_anual = (1 + media_periodo)^factor - 1

        Args:
            media_retornos: Serie con media de retornos por ticker
            periodo_origen: Período de los retornos de entrada

        Returns:
            Serie con medias anualizadas
        """
        factor = FACTOR_ANUALIZACION[periodo_origen]

        if self.tipo_retorno == TipoRetorno.LOG:
            # Log-retornos: anualización aditiva
            return media_retornos * factor
        else:
            # Retornos simples: anualización compuesta
            return (1 + media_retornos) ** factor - 1

    def anualizar_volatilidad(
        self,
        volatilidad: pd.Series,
        periodo_origen: Periodo = Periodo.DIARIO
    ) -> pd.Series:
        """
        Anualiza la volatilidad (desviación estándar).

        La volatilidad se anualiza con raíz cuadrada del factor temporal
        para ambos tipos de retornos (asumiendo independencia).

        Args:
            volatilidad: Serie con volatilidad por ticker
            periodo_origen: Período de los retornos de entrada

        Returns:
            Serie con volatilidades anualizadas
        """
        factor = FACTOR_ANUALIZACION[periodo_origen]
        return volatilidad * np.sqrt(factor)

    def anualizar_covarianza(
        self,
        matriz_cov: pd.DataFrame,
        periodo_origen: Periodo = Periodo.DIARIO
    ) -> pd.DataFrame:
        """
        Anualiza una matriz de covarianza.

        La covarianza se anualiza multiplicando por el factor temporal
        (misma regla para log y simples).

        Args:
            matriz_cov: Matriz de covarianza
            periodo_origen: Período de los retornos de entrada

        Returns:
            Matriz de covarianza anualizada
        """
        factor = FACTOR_ANUALIZACION[periodo_origen]
        return matriz_cov * factor

    def calcular_estadisticas_anualizadas(
        self,
        periodo: Periodo = Periodo.DIARIO
    ) -> Dict[str, Any]:
        """
        Calcula estadísticas anualizadas completas para un período.

        Args:
            periodo: Período de origen para los cálculos

        Returns:
            Dict con media, volatilidad, covarianza y correlación anualizadas
        """
        retornos = self.obtener_retornos(periodo)

        return {
            'media_anualizada': self.anualizar_media(retornos.mean(), periodo),
            'volatilidad_anualizada': self.anualizar_volatilidad(retornos.std(), periodo),
            'covarianza_anualizada': self.anualizar_covarianza(retornos.cov(), periodo),
            'correlacion': retornos.corr(),  # Correlación no se anualiza
            'n_observaciones': len(retornos),
            'periodo': periodo.value,
            'tipo_retorno': self.tipo_retorno.value
        }


# =============================================================================
# PROCESADOR DE DATOS PRINCIPAL
# =============================================================================

class ProcesadorDatos:
    """
    Procesador centralizado de datos históricos para optimización de portafolios.

    Esta clase es el ÚNICO punto de entrada para calcular retornos en el sistema.
    Transforma precios históricos en matrices de retornos limpias y consistentes.

    Características:
        - Calcula retornos logarítmicos o simples
        - Soporta múltiples períodos temporales
        - Limpia datos (NaN, infinitos, outliers)
        - Valida calidad de datos
        - Provee funciones de anualización correctas

    Ejemplo básico:
        >>> procesador = ProcesadorDatos(tipo_retorno="log")
        >>> retornos_diarios, metadata = procesador.procesar(precios_df)

    Ejemplo multi-período:
        >>> procesador = ProcesadorDatos(tipo_retorno="log")
        >>> retornos = procesador.procesar_completo(precios_df)
        >>> retornos.diario   # Para optimización
        >>> retornos.mensual  # Para análisis
    """

    def __init__(
        self,
        tipo_retorno: Literal["log", "simple"] = "log",
        metodo_llenar_na: Literal["ffill", "bfill", "drop", "zero"] = "ffill",
        remover_outliers: bool = True,
        umbral_std_outliers: float = 5.0,
        ratio_minimo_validos: float = 0.95
    ):
        """
        Inicializa el preprocesador.

        Args:
            tipo_retorno: Tipo de retornos a calcular:
                - 'log': Retornos logarítmicos (RECOMENDADO para optimización)
                         r_t = ln(P_t / P_{t-1})
                         Ventajas: aditivos, más estables, distribución más normal
                - 'simple': Retornos simples/aritméticos
                           r_t = (P_t - P_{t-1}) / P_{t-1}
                           Ventajas: interpretación directa como porcentaje
            metodo_llenar_na: Método para manejar NaN ('ffill', 'bfill', 'drop', 'zero')
            remover_outliers: Si remover outliers extremos mediante winsorización
            umbral_std_outliers: Número de desviaciones estándar para considerar outlier
            ratio_minimo_validos: Ratio mínimo de datos válidos por ticker (0.95 = 95%)
        """
        self.tipo_retorno = TipoRetorno(tipo_retorno)
        self.metodo_llenar_na = metodo_llenar_na
        self.remover_outliers = remover_outliers
        self.umbral_std_outliers = umbral_std_outliers
        self.ratio_minimo_validos = ratio_minimo_validos

        self._reset_metadata()

    def _reset_metadata(self) -> None:
        """Reinicia la metadata para un nuevo procesamiento."""
        self.metadata = {
            'fecha_procesamiento': None,
            'tipo_retorno': self.tipo_retorno.value,
            'n_tickers_entrada': 0,
            'n_tickers_salida': 0,
            'n_observaciones_entrada': 0,
            'n_observaciones_salida': 0,
            'tickers_removidos': [],
            'outliers_removidos': 0,
            'advertencias': [],
            'periodos_calculados': []
        }

    def procesar(
        self,
        precios_df: pd.DataFrame,
        validar: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Procesa datos de precios y retorna matriz de retornos DIARIOS limpia.

        Este método mantiene compatibilidad con el código existente.
        Para retornos multi-período, usar procesar_completo().

        Args:
            precios_df: DataFrame con precios históricos (columnas = tickers, índice = fechas)
            validar: Si validar calidad de datos

        Returns:
            Tuple[pd.DataFrame, Dict]: (retornos_diarios limpios, metadata del procesamiento)

        Raises:
            ValueError: Si los datos no cumplen requisitos mínimos
        """
        self._reset_metadata()
        self.metadata['fecha_procesamiento'] = datetime.now().isoformat()
        self.metadata['n_tickers_entrada'] = len(precios_df.columns)
        self.metadata['n_observaciones_entrada'] = len(precios_df)

        logger.info(f"🔧 Iniciando preprocessing: {len(precios_df.columns)} tickers, "
                   f"{len(precios_df)} observaciones, tipo: {self.tipo_retorno.value}")

        # 1. Validar entrada
        if validar:
            self._validar_entrada(precios_df)

        # 2. Calcular retornos diarios
        retornos_df = self._calcular_retornos(precios_df, periodo=Periodo.DIARIO)

        # 3. Limpiar NaN
        retornos_df = self._manejar_valores_faltantes(retornos_df)

        # 4. Remover outliers extremos
        if self.remover_outliers:
            retornos_df = self._remover_outliers(retornos_df)

        # 5. Validar calidad de datos por ticker
        retornos_df = self._validar_calidad_ticker(retornos_df)

        # 6. Verificar infinitos
        retornos_df = self._verificar_infinitos(retornos_df)

        # 7. Metadata final
        self.metadata['n_tickers_salida'] = len(retornos_df.columns)
        self.metadata['n_observaciones_salida'] = len(retornos_df)
        self.metadata['periodos_calculados'] = [Periodo.DIARIO.value]

        logger.info(f"✅ Preprocessing completado: {len(retornos_df.columns)} tickers, "
                   f"{len(retornos_df)} observaciones")

        if self.metadata['tickers_removidos']:
            logger.warning(f"⚠️ Tickers removidos por baja calidad: {self.metadata['tickers_removidos']}")

        return retornos_df, self.metadata.copy()

    def procesar_completo(
        self,
        precios_df: pd.DataFrame,
        validar: bool = True,
        periodos: Optional[list] = None
    ) -> RetornosMultiPeriodo:
        """
        Procesa datos de precios y retorna retornos para TODOS los períodos.

        Calcula retornos diarios, semanales, mensuales y semestrales
        de forma consistente con el mismo tipo de retorno.

        Args:
            precios_df: DataFrame con precios históricos (columnas = tickers, índice = fechas)
            validar: Si validar calidad de datos
            periodos: Lista de períodos a calcular. Por defecto todos.

        Returns:
            RetornosMultiPeriodo con retornos para cada período

        Raises:
            ValueError: Si los datos no cumplen requisitos mínimos

        Ejemplo:
            >>> procesador = ProcesadorDatos(tipo_retorno="log")
            >>> retornos = procesador.procesar_completo(precios_df)
            >>>
            >>> # Usar retornos diarios para optimización
            >>> media_anual = retornos.anualizar_media(retornos.diario.mean())
            >>> vol_anual = retornos.anualizar_volatilidad(retornos.diario.std())
            >>> cov_anual = retornos.anualizar_covarianza(retornos.diario.cov())
        """
        if periodos is None:
            periodos = list(Periodo)

        self._reset_metadata()
        self.metadata['fecha_procesamiento'] = datetime.now().isoformat()
        self.metadata['n_tickers_entrada'] = len(precios_df.columns)
        self.metadata['n_observaciones_entrada'] = len(precios_df)

        logger.info(f"🔧 Iniciando preprocessing multi-período: {len(precios_df.columns)} tickers, "
                   f"{len(precios_df)} observaciones, tipo: {self.tipo_retorno.value}")

        # 1. Validar entrada
        if validar:
            self._validar_entrada(precios_df)

        # 2. Guardar precios originales (referencia)
        precios_limpios = precios_df.copy()

        # 3. Calcular retornos para cada período
        retornos_por_periodo = {}

        for periodo in periodos:
            logger.info(f"📊 Calculando retornos {periodo.value}...")
            retornos = self._calcular_retornos(precios_limpios, periodo=periodo)
            retornos = self._manejar_valores_faltantes(retornos)

            if self.remover_outliers:
                retornos = self._remover_outliers(retornos)

            retornos = self._verificar_infinitos(retornos)
            retornos_por_periodo[periodo] = retornos

            logger.info(f"   → {periodo.value}: {len(retornos)} observaciones")

        # 4. Validar calidad de tickers (basado en retornos diarios)
        if Periodo.DIARIO in retornos_por_periodo:
            retornos_diarios_validados = self._validar_calidad_ticker(
                retornos_por_periodo[Periodo.DIARIO]
            )
            tickers_validos = retornos_diarios_validados.columns.tolist()

            # Filtrar otros períodos para mantener mismos tickers
            for periodo in retornos_por_periodo:
                retornos_por_periodo[periodo] = retornos_por_periodo[periodo][
                    [t for t in retornos_por_periodo[periodo].columns if t in tickers_validos]
                ]

        # 5. Metadata final
        self.metadata['n_tickers_salida'] = len(tickers_validos) if 'tickers_validos' in dir() else len(precios_df.columns)
        self.metadata['n_observaciones_salida'] = len(retornos_por_periodo.get(Periodo.DIARIO, pd.DataFrame()))
        self.metadata['periodos_calculados'] = [p.value for p in periodos]

        logger.info(f"✅ Preprocessing multi-período completado: {self.metadata['n_tickers_salida']} tickers")

        # 6. Crear objeto RetornosMultiPeriodo
        return RetornosMultiPeriodo(
            tipo_retorno=self.tipo_retorno,
            diario=retornos_por_periodo.get(Periodo.DIARIO, pd.DataFrame()),
            semanal=retornos_por_periodo.get(Periodo.SEMANAL, pd.DataFrame()),
            mensual=retornos_por_periodo.get(Periodo.MENSUAL, pd.DataFrame()),
            semestral=retornos_por_periodo.get(Periodo.SEMESTRAL, pd.DataFrame()),
            precios_originales=precios_limpios,
            metadata=self.metadata.copy()
        )

    def _validar_entrada(self, precios_df: pd.DataFrame) -> None:
        """Valida que los datos de entrada sean correctos."""
        if precios_df.empty:
            raise ValueError("DataFrame de precios está vacío")

        if len(precios_df) < 20:
            raise ValueError(f"Datos insuficientes: {len(precios_df)} observaciones (mínimo 20)")

        if len(precios_df.columns) < 2:
            raise ValueError(f"Tickers insuficientes: {len(precios_df.columns)} (mínimo 2)")

        # Verificar que son precios positivos
        if (precios_df <= 0).any().any():
            n_negativos = (precios_df <= 0).sum().sum()
            self.metadata['advertencias'].append(f"Se encontraron {n_negativos} precios negativos o cero")
            logger.warning(f"⚠️ {n_negativos} precios negativos o cero detectados")

    def _calcular_retornos(
        self,
        precios_df: pd.DataFrame,
        periodo: Periodo = Periodo.DIARIO
    ) -> pd.DataFrame:
        """
        Calcula retornos logarítmicos o simples para un período específico.

        Retornos logarítmicos (recomendado):
            r_t = ln(P_t / P_{t-n})
            donde n = días del período

        Retornos simples:
            r_t = (P_t - P_{t-n}) / P_{t-n}

        Args:
            precios_df: DataFrame con precios
            periodo: Período para el cálculo (DIARIO, SEMANAL, MENSUAL, SEMESTRAL)

        Returns:
            DataFrame con retornos del período especificado
        """
        dias = DIAS_POR_PERIODO[periodo]

        logger.debug(f"📊 Calculando retornos {periodo.value} (shift={dias}), tipo: {self.tipo_retorno.value}")

        if self.tipo_retorno == TipoRetorno.LOG:
            # Retornos logarítmicos (más estables, aditivos temporalmente)
            # r = ln(P_t / P_{t-n})
            retornos_df = np.log(precios_df / precios_df.shift(dias))
        else:
            # Retornos simples/aritméticos
            # r = (P_t - P_{t-n}) / P_{t-n}
            retornos_df = precios_df.pct_change(periods=dias)

        # Remover primeras filas (NaN por el shift)
        retornos_df = retornos_df.iloc[dias:]

        return retornos_df

    def _manejar_valores_faltantes(self, retornos_df: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores faltantes (NaN)."""
        n_na = retornos_df.isna().sum().sum()

        if n_na == 0:
            return retornos_df

        logger.debug(f"🔧 Manejando {n_na} valores faltantes con método: {self.metodo_llenar_na}")

        if self.metodo_llenar_na == "ffill":
            retornos_df = retornos_df.ffill().bfill()
        elif self.metodo_llenar_na == "bfill":
            retornos_df = retornos_df.bfill().ffill()
        elif self.metodo_llenar_na == "drop":
            longitud_inicial = len(retornos_df)
            retornos_df = retornos_df.dropna()
            removidos = longitud_inicial - len(retornos_df)
            if removidos > 0:
                self.metadata['advertencias'].append(f"Se removieron {removidos} filas con NaN")
        elif self.metodo_llenar_na == "zero":
            retornos_df = retornos_df.fillna(0)

        # Verificar que no queden NaN
        na_restantes = retornos_df.isna().sum().sum()
        if na_restantes > 0:
            logger.warning(f"⚠️ Quedan {na_restantes} NaN después de limpieza, removiendo...")
            retornos_df = retornos_df.dropna()

        return retornos_df

    def _remover_outliers(self, retornos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remueve outliers extremos mediante winsorización.

        Outlier: |r_t - media| > umbral * std(r)
        Los outliers se reemplazan con el valor del límite (winsorización).
        """
        conteo_outliers = 0

        for col in retornos_df.columns:
            media = retornos_df[col].mean()
            std = retornos_df[col].std()

            if std == 0:
                continue

            limite_superior = media + self.umbral_std_outliers * std
            limite_inferior = media - self.umbral_std_outliers * std

            outliers = ((retornos_df[col] > limite_superior) |
                       (retornos_df[col] < limite_inferior)).sum()
            conteo_outliers += outliers

            retornos_df[col] = retornos_df[col].clip(lower=limite_inferior, upper=limite_superior)

        self.metadata['outliers_removidos'] = conteo_outliers

        if conteo_outliers > 0:
            logger.debug(f"🔧 {conteo_outliers} outliers extremos winsorized")

        return retornos_df

    def _validar_calidad_ticker(self, retornos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida calidad de datos por ticker y remueve tickers de baja calidad.

        Criterios:
        - Ratio mínimo de datos válidos
        - Varianza no cero
        """
        tickers_removidos = []

        for ticker in retornos_df.columns:
            datos_ticker = retornos_df[ticker]

            ratio_validos = datos_ticker.notna().sum() / len(datos_ticker)
            if ratio_validos < self.ratio_minimo_validos:
                logger.warning(f"⚠️ {ticker}: ratio de datos válidos {ratio_validos:.2%} < {self.ratio_minimo_validos:.2%}")
                tickers_removidos.append(ticker)
                continue

            if datos_ticker.std() == 0:
                logger.warning(f"⚠️ {ticker}: varianza cero (activo sin movimiento)")
                tickers_removidos.append(ticker)
                continue

        if tickers_removidos:
            retornos_df = retornos_df.drop(columns=tickers_removidos)
            self.metadata['tickers_removidos'] = tickers_removidos
            logger.warning(f"🗑️ Removidos {len(tickers_removidos)} tickers de baja calidad")

        return retornos_df

    def _verificar_infinitos(self, retornos_df: pd.DataFrame) -> pd.DataFrame:
        """Verifica y maneja valores infinitos."""
        n_inf = np.isinf(retornos_df.values).sum()

        if n_inf > 0:
            logger.error(f"❌ {n_inf} valores infinitos detectados!")
            self.metadata['advertencias'].append(f"{n_inf} valores infinitos detectados")
            retornos_df = retornos_df.replace([np.inf, -np.inf], np.nan)
            retornos_df = self._manejar_valores_faltantes(retornos_df)

        return retornos_df

    def obtener_resumen(self) -> Dict[str, Any]:
        """Retorna resumen del preprocessing realizado."""
        return {
            'configuracion': {
                'tipo_retorno': self.tipo_retorno.value,
                'metodo_llenar_na': self.metodo_llenar_na,
                'remover_outliers': self.remover_outliers,
                'umbral_std_outliers': self.umbral_std_outliers,
                'ratio_minimo_validos': self.ratio_minimo_validos
            },
            'resultados': self.metadata
        }


# =============================================================================
# FUNCIONES DE ANUALIZACIÓN (STANDALONE)
# =============================================================================

def anualizar_retorno_medio(
    media_retornos: pd.Series,
    tipo_retorno: TipoRetorno = TipoRetorno.LOG,
    periodo_origen: Periodo = Periodo.DIARIO
) -> pd.Series:
    """
    Anualiza la media de retornos usando la fórmula correcta según el tipo.

    IMPORTANTE: Usar la fórmula correcta según el tipo de retorno:
        - Log-retornos: media_anual = media_periodo * factor (aditivo)
        - Simples: media_anual = (1 + media_periodo)^factor - 1 (compuesto)

    Args:
        media_retornos: Serie con media de retornos por ticker
        tipo_retorno: Tipo de retornos (LOG o SIMPLE)
        periodo_origen: Período de los retornos de entrada

    Returns:
        Serie con medias anualizadas

    Ejemplo:
        >>> media_diaria = retornos_diarios.mean()
        >>> media_anual = anualizar_retorno_medio(media_diaria, TipoRetorno.LOG)
    """
    factor = FACTOR_ANUALIZACION[periodo_origen]

    if tipo_retorno == TipoRetorno.LOG:
        return media_retornos * factor
    else:
        return (1 + media_retornos) ** factor - 1


def anualizar_volatilidad(
    volatilidad: pd.Series,
    periodo_origen: Periodo = Periodo.DIARIO
) -> pd.Series:
    """
    Anualiza la volatilidad (desviación estándar).

    La volatilidad se anualiza con raíz cuadrada del factor temporal
    (asumiendo retornos i.i.d., válido para ambos tipos).

    Args:
        volatilidad: Serie con volatilidad por ticker
        periodo_origen: Período de los retornos de entrada

    Returns:
        Serie con volatilidades anualizadas
    """
    factor = FACTOR_ANUALIZACION[periodo_origen]
    return volatilidad * np.sqrt(factor)


def anualizar_covarianza(
    matriz_cov: pd.DataFrame,
    periodo_origen: Periodo = Periodo.DIARIO
) -> pd.DataFrame:
    """
    Anualiza una matriz de covarianza.

    La covarianza se anualiza multiplicando por el factor temporal
    (válido para ambos tipos de retornos).

    Args:
        matriz_cov: Matriz de covarianza
        periodo_origen: Período de los retornos de entrada

    Returns:
        Matriz de covarianza anualizada
    """
    factor = FACTOR_ANUALIZACION[periodo_origen]
    return matriz_cov * factor


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

def preprocesar_para_optimizacion(
    precios_df: pd.DataFrame,
    tipo_retorno: Literal["log", "simple"] = "log",
    remover_outliers: bool = True
) -> pd.DataFrame:
    """
    Función de conveniencia para preprocesar datos con configuración estándar.
    Retorna solo retornos diarios (compatibilidad hacia atrás).

    Args:
        precios_df: DataFrame con precios históricos
        tipo_retorno: Tipo de retornos ('log' o 'simple')
        remover_outliers: Si remover outliers extremos

    Returns:
        DataFrame con retornos diarios limpios

    Ejemplo:
        >>> precios = pd.DataFrame(...)
        >>> retornos = preprocesar_para_optimizacion(precios, tipo_retorno='log')
    """
    procesador = ProcesadorDatos(
        tipo_retorno=tipo_retorno,
        metodo_llenar_na="ffill",
        remover_outliers=remover_outliers,
        umbral_std_outliers=5.0,
        ratio_minimo_validos=0.95
    )

    retornos_df, metadata = procesador.procesar(precios_df)

    if metadata['advertencias']:
        logger.warning(f"Advertencias durante preprocessing: {metadata['advertencias']}")

    return retornos_df


def preprocesar_multiperiodo(
    precios_df: pd.DataFrame,
    tipo_retorno: Literal["log", "simple"] = "log",
    remover_outliers: bool = True
) -> RetornosMultiPeriodo:
    """
    Función de conveniencia para preprocesar datos con múltiples períodos.

    Args:
        precios_df: DataFrame con precios históricos
        tipo_retorno: Tipo de retornos ('log' o 'simple')
        remover_outliers: Si remover outliers extremos

    Returns:
        RetornosMultiPeriodo con retornos para todos los períodos

    Ejemplo:
        >>> precios = pd.DataFrame(...)
        >>> retornos = preprocesar_multiperiodo(precios, tipo_retorno='log')
        >>> retornos.diario    # Para optimización
        >>> retornos.mensual   # Para análisis
        >>>
        >>> # Estadísticas anualizadas
        >>> stats = retornos.calcular_estadisticas_anualizadas()
    """
    procesador = ProcesadorDatos(
        tipo_retorno=tipo_retorno,
        metodo_llenar_na="ffill",
        remover_outliers=remover_outliers,
        umbral_std_outliers=5.0,
        ratio_minimo_validos=0.95
    )

    return procesador.procesar_completo(precios_df)


def normalizar_retornos(
    retornos_df: pd.DataFrame,
    metodo: Literal["estandarizar", "minmax", "ninguno"] = "ninguno"
) -> pd.DataFrame:
    """
    Normaliza retornos (opcional, para algunos modelos ML).

    Args:
        retornos_df: DataFrame con retornos
        metodo: Método de normalización
            - 'estandarizar': (r - media) / std  (z-score)
            - 'minmax': (r - min) / (max - min)  (escala 0-1)
            - 'ninguno': Sin normalización

    Returns:
        DataFrame normalizado

    Nota:
        La mayoría de modelos de optimización NO requieren normalización.
        Solo usar para modelos ML que lo necesiten.
    """
    if metodo == "ninguno":
        return retornos_df
    elif metodo == "estandarizar":
        return (retornos_df - retornos_df.mean()) / retornos_df.std()
    elif metodo == "minmax":
        return (retornos_df - retornos_df.min()) / (retornos_df.max() - retornos_df.min())
    else:
        raise ValueError(f"Método de normalización inválido: {metodo}")


# =============================================================================
# FUNCIÓN PARA OBTENER RETORNOS DESDE PRECIOS (USO GENERAL)
# =============================================================================

def obtener_retornos_desde_precios(
    precios_df: pd.DataFrame,
    tipo_retorno: Literal["log", "simple"] = "log",
    periodo: Literal["diario", "semanal", "mensual", "semestral"] = "diario",
    limpiar: bool = True
) -> pd.DataFrame:
    """
    Función unificada para obtener retornos desde precios.

    Esta función debe ser usada por TODOS los módulos que necesiten
    calcular retornos para garantizar consistencia en todo el sistema.

    Args:
        precios_df: DataFrame con precios (columnas=tickers, índice=fechas)
        tipo_retorno: 'log' (recomendado) o 'simple'
        periodo: Período de los retornos
        limpiar: Si aplicar limpieza de datos (NaN, outliers)

    Returns:
        DataFrame con retornos limpios del período especificado

    Ejemplo:
        >>> # En cualquier módulo:
        >>> from ypyquant.optimization.data_preprocessing import obtener_retornos_desde_precios
        >>> retornos = obtener_retornos_desde_precios(precios, tipo_retorno="log")
    """
    periodo_enum = Periodo(periodo)

    procesador = ProcesadorDatos(
        tipo_retorno=tipo_retorno,
        remover_outliers=limpiar,
        metodo_llenar_na="ffill" if limpiar else "drop"
    )

    if periodo_enum == Periodo.DIARIO:
        retornos, _ = procesador.procesar(precios_df, validar=limpiar)
        return retornos
    else:
        retornos_multi = procesador.procesar_completo(precios_df, validar=limpiar)
        return retornos_multi.obtener_retornos(periodo_enum)
