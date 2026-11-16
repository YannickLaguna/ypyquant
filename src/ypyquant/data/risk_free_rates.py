"""
Servicio para obtener tasas libres de riesgo dinámicas por plazo.

Responsabilidad: encapsular la lógica de obtención de tasas RF adecuadas
para distintos horizontes y tipos de activos, usando curvas del Tesoro USA.

Relacionado con:
  - data/yield_curves.py      (CurvaRendimientoTesoro — motor de tasas)
  - data/quantlib_utils.py    (tiene_quantlib)
  - factors/factor_models.py  (usa TasaLibreRiesgoDinamica como rf en CAPM/FF)
  - optimization/portfolio_models.py (usa TasaLibreRiesgoDinamica)
"""

from typing import Dict, Optional, Literal
from datetime import datetime, date
import warnings
import pandas as pd

from ypyquant.data.yield_curves import CurvaRendimientoTesoro
from ypyquant.data.quantlib_utils import tiene_quantlib


class TasaLibreRiesgoDinamica:
    """
    Servicio para obtener tasas libres de riesgo dinámicas basadas en
    curvas de rendimiento del Tesoro USA.

    Este servicio permite obtener tasas apropiadas según:
    - Plazo de inversión (3M, 1Y, 5Y, 10Y, etc.)
    - Tipo de activo (value, growth, neutral)
    - Características del portafolio

    Attributes:
        curva: CurvaRendimientoTesoro con la curva bootstrapped
        cache: Cache de tasas calculadas
        fecha_actualizacion: Última fecha de actualización de la curva

    Example:
        >>> servicio = TasaLibreRiesgoDinamica()
        >>> # Tasa para horizonte de 5 años
        >>> tasa_5y = servicio.obtener_tasa_por_plazo(5.0)
        >>> print(f"Tasa 5Y: {tasa_5y:.2%}")
        Tasa 5Y: 4.35%
    """

    def __init__(
        self,
        actualizar_curva: bool = True,
        fecha: Optional[str | datetime | date] = None
    ):
        """
        Inicializa el servicio de tasas libres de riesgo.

        Args:
            actualizar_curva: Si True, construye la curva inmediatamente
            fecha: Fecha para construir la curva. Si None, usa fecha actual.
        """
        self.curva = CurvaRendimientoTesoro()
        self.cache = {}
        self.fecha_actualizacion = None
        self._tasa_fallback = 0.02  # 2% fallback conservador

        if actualizar_curva:
            try:
                self.actualizar_curva(fecha)
            except Exception as e:
                warnings.warn(
                    f"No se pudo construir la curva de rendimiento: {e}. "
                    f"Usando tasa fallback de {self._tasa_fallback:.2%}"
                )

    def actualizar_curva(
        self,
        fecha: Optional[str | datetime | date] = None,
        tasas_manuales: Optional[Dict[str, float]] = None
    ) -> 'TasaLibreRiesgoDinamica':
        """
        Actualiza la curva de rendimiento con datos de mercado recientes.

        Args:
            fecha: Fecha para la cual construir la curva
            tasas_manuales: Tasas manuales opcionales (para testing)

        Returns:
            self (para method chaining)

        Example:
            >>> servicio = TasaLibreRiesgoDinamica(actualizar_curva=False)
            >>> servicio.actualizar_curva('2025-11-18')
        """
        try:
            self.curva.construir_desde_mercado(fecha, tasas_manuales)

            # Guardar fecha de actualización
            if fecha is None:
                self.fecha_actualizacion = date.today()
            elif isinstance(fecha, str):
                self.fecha_actualizacion = datetime.strptime(fecha, '%Y-%m-%d').date()
            elif isinstance(fecha, datetime):
                self.fecha_actualizacion = fecha.date()
            else:
                self.fecha_actualizacion = fecha

            # Limpiar cache
            self.cache = {}

            print(f"[TasaLibreRiesgo] Curva actualizada para {self.fecha_actualizacion}")

        except Exception as e:
            warnings.warn(f"Error actualizando curva: {e}")
            raise

        return self

    def obtener_tasa_por_plazo(
        self,
        años: float,
        usar_cache: bool = True
    ) -> float:
        """
        Obtiene la tasa libre de riesgo para un plazo específico.

        Args:
            años: Plazo de inversión en años (puede ser fraccionario)
            usar_cache: Si True, usa cache de tasas calculadas

        Returns:
            Tasa anualizada como decimal (ej: 0.045 para 4.5%)

        Raises:
            ValueError: Si la curva no está construida

        Example:
            >>> servicio = TasaLibreRiesgoDinamica()
            >>> tasa_3m = servicio.obtener_tasa_por_plazo(0.25)  # 3 meses
            >>> tasa_10y = servicio.obtener_tasa_por_plazo(10.0)  # 10 años
        """
        # Verificar cache
        if usar_cache and años in self.cache:
            return self.cache[años]

        # Intentar obtener de la curva
        try:
            if self.curva.curva_yield is None:
                raise ValueError("La curva no ha sido construida")

            tasa = self.curva.obtener_tasa(años)

            # Guardar en cache
            if usar_cache:
                self.cache[años] = tasa

            return tasa

        except Exception as e:
            warnings.warn(
                f"No se pudo obtener tasa para plazo {años} años: {e}. "
                f"Usando fallback de {self._tasa_fallback:.2%}"
            )
            return self._tasa_fallback

    def obtener_tasa_para_activo(
        self,
        ticker: str,
        beta: Optional[float] = None,
        tipo_activo: Optional[Literal['value', 'growth', 'neutral']] = None
    ) -> float:
        """
        Obtiene la tasa libre de riesgo apropiada según las características
        del activo.

        La lógica es:
        - Acciones value (bajo crecimiento): horizonte largo → tasa 10Y
        - Acciones growth (alto crecimiento): horizonte corto → tasa 2Y
        - Acciones neutral/balanced: horizonte medio → tasa 5Y

        Args:
            ticker: Símbolo del activo
            beta: Beta del activo (opcional, para ajustes futuros)
            tipo_activo: Tipo de activo ('value', 'growth', 'neutral')
                        Si None, se infiere automáticamente

        Returns:
            Tasa libre de riesgo apropiada

        Example:
            >>> servicio = TasaLibreRiesgoDinamica()
            >>> # Para acción growth tech
            >>> tasa_growth = servicio.obtener_tasa_para_activo('NVDA', tipo_activo='growth')
            >>> # Para acción value
            >>> tasa_value = servicio.obtener_tasa_para_activo('XOM', tipo_activo='value')
        """
        # Si no se especifica tipo, inferir del ticker
        if tipo_activo is None:
            tipo_activo = self._inferir_tipo_activo(ticker, beta)

        # Mapeo de tipo de activo a plazo
        plazos_map = {
            'value': 10.0,      # Value stocks: horizonte largo
            'neutral': 5.0,     # Balanced: horizonte medio
            'growth': 2.0       # Growth stocks: horizonte corto
        }

        plazo = plazos_map.get(tipo_activo, 5.0)  # Default: 5 años

        return self.obtener_tasa_por_plazo(plazo)

    def _inferir_tipo_activo(
        self,
        ticker: str,
        beta: Optional[float]
    ) -> Literal['value', 'growth', 'neutral']:
        """
        Infiere el tipo de activo basándose en el ticker y beta.

        Esta es una implementación simplificada. En producción se podría
        usar datos fundamentales (P/E, P/B, crecimiento, etc.)

        Args:
            ticker: Símbolo del activo
            beta: Beta del activo (si está disponible)

        Returns:
            Tipo de activo inferido
        """
        # Sectores tech típicamente growth
        tickers_growth = [
            'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'GOOG', 'META',
            'NFLX', 'AMD', 'ADBE', 'CRM', 'SHOP'
        ]

        # Sectores tradicionales típicamente value
        tickers_value = [
            'XOM', 'CVX', 'JPM', 'BAC', 'WFC', 'C',
            'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'T', 'VZ'
        ]

        ticker_upper = ticker.upper()

        if ticker_upper in tickers_growth:
            return 'growth'
        elif ticker_upper in tickers_value:
            return 'value'
        else:
            # Si tenemos beta, usar como heurística
            if beta is not None:
                if beta > 1.3:
                    return 'growth'  # Alta volatilidad sugiere growth
                elif beta < 0.8:
                    return 'value'   # Baja volatilidad sugiere value

            return 'neutral'  # Default

    def obtener_curva_completa(
        self,
        plazos: Optional[list[float]] = None
    ) -> pd.DataFrame:
        """
        Obtiene tasas para múltiples plazos (útil para análisis).

        Args:
            plazos: Lista de plazos en años. Si None, usa plazos estándar.

        Returns:
            DataFrame con columnas 'plazo' y 'tasa'

        Example:
            >>> servicio = TasaLibreRiesgoDinamica()
            >>> df = servicio.obtener_curva_completa()
            >>> print(df)
               plazo     tasa
            0   0.25  0.0450
            1   0.50  0.0460
            2   1.00  0.0475
            ...
        """
        if plazos is None:
            # Plazos estándar del Tesoro
            plazos = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]

        tasas = [self.obtener_tasa_por_plazo(plazo) for plazo in plazos]

        return pd.DataFrame({
            'plazo': plazos,
            'tasa': tasas
        })

    def obtener_tasa_promedio_portafolio(
        self,
        tickers: list[str],
        pesos: Optional[Dict[str, float]] = None,
        betas: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calcula una tasa libre de riesgo ponderada para un portafolio.

        Cada activo tiene su propia tasa según sus características,
        y se calcula el promedio ponderado.

        Args:
            tickers: Lista de símbolos en el portafolio
            pesos: Pesos de cada activo. Si None, asume equal-weight.
            betas: Betas de cada activo (opcional)

        Returns:
            Tasa libre de riesgo promedio del portafolio

        Example:
            >>> servicio = TasaLibreRiesgoDinamica()
            >>> tickers = ['AAPL', 'MSFT', 'XOM']
            >>> pesos = {'AAPL': 0.4, 'MSFT': 0.4, 'XOM': 0.2}
            >>> tasa_portafolio = servicio.obtener_tasa_promedio_portafolio(tickers, pesos)
        """
        # Si no hay pesos, usar equal-weight
        if pesos is None:
            peso_igual = 1.0 / len(tickers)
            pesos = {ticker: peso_igual for ticker in tickers}

        # Si no hay betas, usar None para cada ticker
        if betas is None:
            betas = {}

        tasa_ponderada = 0.0

        for ticker in tickers:
            peso = pesos.get(ticker, 0.0)
            beta = betas.get(ticker, None)

            tasa_ticker = self.obtener_tasa_para_activo(ticker, beta)
            tasa_ponderada += peso * tasa_ticker

        return tasa_ponderada

    def obtener_tasa_o_fallback(
        self,
        años: float,
        fallback: Optional[float] = None
    ) -> float:
        """
        Obtiene tasa por plazo con fallback personalizado si hay error.

        Args:
            años: Plazo en años
            fallback: Tasa de fallback. Si None, usa self._tasa_fallback

        Returns:
            Tasa obtenida o fallback
        """
        try:
            return self.obtener_tasa_por_plazo(años)
        except Exception:
            return fallback if fallback is not None else self._tasa_fallback

    def esta_disponible(self) -> bool:
        """
        Verifica si el servicio está disponible y funcional.

        Returns:
            True si la curva está construida y funcional
        """
        return (
            tiene_quantlib() and
            self.curva.curva_yield is not None and
            self.fecha_actualizacion is not None
        )

    def to_dict(self) -> Dict:
        """
        Serializa información del servicio a diccionario.

        Returns:
            Diccionario con metadata
        """
        return {
            'disponible': self.esta_disponible(),
            'fecha_actualizacion': self.fecha_actualizacion.isoformat() if self.fecha_actualizacion else None,
            'tasa_fallback': self._tasa_fallback,
            'num_tasas_en_cache': len(self.cache),
            'curva_info': self.curva.to_dict()
        }

    def __repr__(self):
        if self.esta_disponible():
            return f"TasaLibreRiesgoDinamica(fecha={self.fecha_actualizacion}, disponible=True)"
        else:
            return "TasaLibreRiesgoDinamica(disponible=False, usando fallback)"
