"""
Construcción de curvas de rendimiento del Tesoro USA usando QuantLib.

Responsabilidad: construir curvas zero-coupon bootstrapped desde tasas de mercado
reales (Yahoo Finance) o manuales, y proveer tasas e interpolaciones para cualquier plazo.

Relacionado con:
  - data/quantlib_utils.py    (helpers QuantLib: fechas, calendarios, períodos)
  - data/risk_free_rates.py   (usa CurvaRendimientoTesoro como motor de tasas RF)
  - portfolio/bond_pricing.py (usa CurvaRendimientoTesoro para pricing con curva de mercado)
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import warnings

from quant.data.quantlib_utils import (
    tiene_quantlib,
    convertir_fecha_a_ql,
    convertir_ql_a_fecha,
    crear_calendario_usa,
    obtener_day_count_convention,
    crear_quote_handle,
    obtener_periodo,
    configurar_fecha_evaluacion
)


class CurvaRendimientoTesoro:
    """
    Construye y gestiona curvas de rendimiento del Tesoro USA usando QuantLib.

    La curva se construye mediante bootstrap desde tasas de mercado de T-Bills,
    T-Notes y T-Bonds, proporcionando tasas zero-coupon interpoladas para
    cualquier plazo.

    Attributes:
        curva_yield: Curva de rendimiento QuantLib bootstrapped
        fecha_construccion: Fecha para la cual se construyó la curva
        tasas_mercado: Diccionario con tasas de mercado usadas

    Example:
        >>> curva = CurvaRendimientoTesoro()
        >>> curva.construir_desde_mercado('2025-11-18')
        >>> tasa_5y = curva.obtener_tasa(5.0)  # Tasa 5 años
        >>> print(f"Tasa 5Y: {tasa_5y:.2%}")
        Tasa 5Y: 4.35%
    """

    def __init__(self):
        """
        Inicializa la curva de rendimiento.
        """
        if not tiene_quantlib():
            warnings.warn(
                "QuantLib no está disponible. "
                "Las funcionalidades de curvas de rendimiento no estarán disponibles."
            )

        self.curva_yield = None
        self.fecha_construccion = None
        self.tasas_mercado = {}
        self.calendario = None
        self.day_count = None

        # Inicializar si QuantLib disponible
        if tiene_quantlib():
            self.calendario = crear_calendario_usa()
            self.day_count = obtener_day_count_convention('Actual/Actual')

    def construir_desde_mercado(
        self,
        fecha: str | datetime | date | None = None,
        tasas_manuales: Optional[Dict[str, float]] = None
    ) -> 'CurvaRendimientoTesoro':
        """
        Construye la curva de rendimiento mediante bootstrap desde datos de mercado.

        Args:
            fecha: Fecha de valoración. Si None, usa fecha actual.
            tasas_manuales: Diccionario opcional con tasas manuales
                           en lugar de descargarlas. Formato:
                           {'3M': 0.0450, '1Y': 0.0475, ...}

        Returns:
            self (para method chaining)

        Raises:
            ImportError: Si QuantLib no está disponible
            ValueError: Si no hay suficientes datos para construir la curva

        Example:
            >>> curva = CurvaRendimientoTesoro()
            >>> curva.construir_desde_mercado('2025-11-18')
            >>> # O con tasas manuales:
            >>> tasas = {'3M': 0.045, '1Y': 0.047, '10Y': 0.042}
            >>> curva.construir_desde_mercado(tasas_manuales=tasas)
        """
        if not tiene_quantlib():
            raise ImportError(
                "QuantLib no está disponible. "
                "Instalar con: pip install QuantLib"
            )

        import QuantLib as ql

        # Configurar fecha de evaluación
        if fecha is None:
            fecha = date.today()
        configurar_fecha_evaluacion(fecha)

        fecha_ql = convertir_fecha_a_ql(fecha)
        self.fecha_construccion = fecha if isinstance(fecha, date) else fecha.date() if isinstance(fecha, datetime) else datetime.strptime(fecha, '%Y-%m-%d').date()

        # Obtener tasas de mercado
        if tasas_manuales:
            self.tasas_mercado = tasas_manuales
        else:
            self.tasas_mercado = self.descargar_tasas_tesoro(fecha)

        # Validar que tenemos suficientes datos
        if len(self.tasas_mercado) < 3:
            raise ValueError(
                f"No hay suficientes datos de mercado para construir la curva. "
                f"Se encontraron {len(self.tasas_mercado)} tasas, se requieren al menos 3."
            )

        # Crear helpers para el bootstrap
        helpers = []

        # T-Bills (depósitos hasta 1 año)
        plazos_depositos = ['1M', '3M', '6M', '1Y']
        for plazo in plazos_depositos:
            if plazo in self.tasas_mercado:
                tasa = self.tasas_mercado[plazo]
                periodo = obtener_periodo(plazo)

                helper = ql.DepositRateHelper(
                    crear_quote_handle(tasa),
                    periodo,
                    2,  # settlement days
                    self.calendario,
                    ql.ModifiedFollowing,
                    False,  # end of month
                    self.day_count
                )
                helpers.append(helper)

        # T-Notes y T-Bonds (swap helpers para plazos >= 2 años)
        plazos_largos = ['2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
        for plazo in plazos_largos:
            if plazo in self.tasas_mercado:
                tasa = self.tasas_mercado[plazo]
                periodo = obtener_periodo(plazo)

                # Para T-Notes/Bonds usamos SwapRateHelper
                # (aproximación razonable para curva de Tesoro)
                try:
                    helper = ql.SwapRateHelper(
                        crear_quote_handle(tasa),
                        periodo,
                        self.calendario,
                        ql.Annual,  # frecuencia de cupón
                        ql.Unadjusted,
                        self.day_count,
                        ql.Euribor6M()  # index (no usado en realidad para Tesoro)
                    )
                    helpers.append(helper)
                except Exception as e:
                    warnings.warn(f"No se pudo crear helper para {plazo}: {e}")

        if not helpers:
            raise ValueError("No se pudieron crear helpers para el bootstrap")

        # Bootstrap de la curva
        try:
            self.curva_yield = ql.PiecewiseLinearZero(
                fecha_ql,
                helpers,
                self.day_count
            )

            # Habilitar extrapolación para plazos fuera del rango
            self.curva_yield.enableExtrapolation()

        except Exception as e:
            raise ValueError(f"Error en bootstrap de curva: {e}")

        return self

    def descargar_tasas_tesoro(
        self,
        fecha: str | datetime | date
    ) -> Dict[str, float]:
        """
        Descarga tasas del Tesoro USA desde Yahoo Finance.

        Args:
            fecha: Fecha para la cual descargar las tasas

        Returns:
            Diccionario con tasas por plazo {'3M': 0.045, '10Y': 0.042, ...}

        Note:
            Usa símbolos de Yahoo Finance:
            - ^IRX: 13-week T-Bill (3 meses)
            - ^FVX: 5-year T-Note
            - ^TNX: 10-year T-Note
            - ^TYX: 30-year T-Bond
        """
        import yfinance as yf

        # Mapeo de plazos a símbolos de Yahoo Finance
        simbolos_yahoo = {
            '3M': '^IRX',    # 13-week T-Bill
            '6M': '^IRX',    # Aproximación (no hay símbolo específico)
            '1Y': '^IRX',    # Aproximación
            '2Y': '^2YY',    # Puede no estar disponible
            '5Y': '^FVX',    # 5-year T-Note
            '10Y': '^TNX',   # 10-year T-Note
            '30Y': '^TYX'    # 30-year T-Bond
        }

        tasas = {}

        # Convertir fecha a string si es necesario
        if isinstance(fecha, (datetime, date)):
            fecha_str = fecha.strftime('%Y-%m-%d')
        else:
            fecha_str = fecha

        print(f"[CurvaRendimiento] Descargando tasas del Tesoro para {fecha_str}...")

        for plazo, simbolo in simbolos_yahoo.items():
            try:
                ticker = yf.Ticker(simbolo)

                # Obtener datos de los últimos 5 días (para asegurar que hay datos)
                fecha_fin = datetime.strptime(fecha_str, '%Y-%m-%d') if isinstance(fecha_str, str) else fecha
                fecha_inicio = fecha_fin - timedelta(days=5)

                hist = ticker.history(
                    start=fecha_inicio.strftime('%Y-%m-%d'),
                    end=(fecha_fin + timedelta(days=1)).strftime('%Y-%m-%d')
                )

                if not hist.empty:
                    # Los índices de Yahoo están en porcentaje
                    tasa_porcentaje = hist['Close'].iloc[-1]
                    tasas[plazo] = tasa_porcentaje / 100.0
                    print(f"   {plazo}: {tasas[plazo]:.4f} ({tasas[plazo]*100:.2f}%)")
                else:
                    # Fallback
                    tasa_fallback = self._fallback_tasa(plazo)
                    tasas[plazo] = tasa_fallback
                    print(f"   {plazo}: {tasa_fallback:.4f} (fallback)")

            except Exception as e:
                print(f"   Error descargando {simbolo} para {plazo}: {e}")
                tasa_fallback = self._fallback_tasa(plazo)
                tasas[plazo] = tasa_fallback
                print(f"   {plazo}: {tasa_fallback:.4f} (fallback por error)")

        # Si no se descargó nada, usar todos los fallbacks
        if not tasas:
            print("   [WARNING] No se pudieron descargar tasas. Usando todos los fallbacks.")
            for plazo in simbolos_yahoo.keys():
                tasas[plazo] = self._fallback_tasa(plazo)

        return tasas

    def _fallback_tasa(self, plazo: str) -> float:
        """
        Retorna tasas de fallback conservadoras.

        Estas tasas son aproximaciones razonables cuando no se pueden
        descargar datos de mercado reales.

        Args:
            plazo: Plazo ('3M', '1Y', '10Y', etc.)

        Returns:
            Tasa anualizada como decimal
        """
        fallbacks = {
            '1M': 0.0440,
            '3M': 0.0450,
            '6M': 0.0460,
            '1Y': 0.0470,
            '2Y': 0.0475,
            '3Y': 0.0465,
            '5Y': 0.0450,
            '7Y': 0.0440,
            '10Y': 0.0420,
            '20Y': 0.0430,
            '30Y': 0.0440
        }
        return fallbacks.get(plazo, 0.0400)

    def obtener_tasa(self, años: float) -> float:
        """
        Obtiene la tasa zero-coupon para un plazo específico.

        La tasa se obtiene mediante interpolación de la curva bootstrapped.

        Args:
            años: Plazo en años (puede ser fraccionario, ej: 2.5 para 2.5 años)

        Returns:
            Tasa anualizada como decimal (ej: 0.045 para 4.5%)

        Raises:
            ValueError: Si la curva no ha sido construida
            ImportError: Si QuantLib no está disponible

        Example:
            >>> curva = CurvaRendimientoTesoro()
            >>> curva.construir_desde_mercado()
            >>> tasa_2_5y = curva.obtener_tasa(2.5)
            >>> print(f"Tasa 2.5Y: {tasa_2_5y:.4f}")
            Tasa 2.5Y: 0.0465
        """
        if not tiene_quantlib():
            raise ImportError("QuantLib no está disponible")

        if self.curva_yield is None:
            raise ValueError(
                "La curva no ha sido construida. "
                "Llamar primero a construir_desde_mercado()"
            )

        import QuantLib as ql

        # Calcular fecha futura
        fecha_futura = self.curva_yield.referenceDate() + ql.Period(int(años * 12), ql.Months)

        # Obtener tasa zero-coupon
        tasa = self.curva_yield.zeroRate(
            fecha_futura,
            self.day_count,
            ql.Continuous  # Compounding continuo
        ).rate()

        return float(tasa)

    def obtener_tasas_rango(
        self,
        años_min: float = 0.25,
        años_max: float = 30.0,
        num_puntos: int = 100
    ) -> pd.DataFrame:
        """
        Obtiene tasas para un rango de plazos (útil para graficar).

        Args:
            años_min: Plazo mínimo en años
            años_max: Plazo máximo en años
            num_puntos: Número de puntos a calcular

        Returns:
            DataFrame con columnas 'años' y 'tasa'

        Example:
            >>> curva = CurvaRendimientoTesoro()
            >>> curva.construir_desde_mercado()
            >>> df = curva.obtener_tasas_rango(0.25, 10, 50)
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(df['años'], df['tasa'])
            >>> plt.show()
        """
        if self.curva_yield is None:
            raise ValueError("La curva no ha sido construida")

        años_array = np.linspace(años_min, años_max, num_puntos)
        tasas_array = [self.obtener_tasa(años) for años in años_array]

        return pd.DataFrame({
            'años': años_array,
            'tasa': tasas_array
        })

    def obtener_forward_rate(
        self,
        años_inicio: float,
        años_fin: float
    ) -> float:
        """
        Calcula la tasa forward entre dos plazos.

        La tasa forward es la tasa implícita para un período futuro,
        derivada de las tasas spot actuales.

        Args:
            años_inicio: Inicio del período forward (en años desde hoy)
            años_fin: Fin del período forward (en años desde hoy)

        Returns:
            Tasa forward anualizada

        Example:
            >>> curva = CurvaRendimientoTesoro()
            >>> curva.construir_desde_mercado()
            >>> # Tasa forward para préstamo 1Y que empieza en 2 años
            >>> fwd_rate = curva.obtener_forward_rate(2.0, 3.0)
            >>> print(f"Forward 2Y-3Y: {fwd_rate:.2%}")
            Forward 2Y-3Y: 4.25%
        """
        if self.curva_yield is None:
            raise ValueError("La curva no ha sido construida")

        import QuantLib as ql

        fecha_inicio = self.curva_yield.referenceDate() + ql.Period(int(años_inicio * 12), ql.Months)
        fecha_fin = self.curva_yield.referenceDate() + ql.Period(int(años_fin * 12), ql.Months)

        tasa = self.curva_yield.forwardRate(
            fecha_inicio,
            fecha_fin,
            self.day_count,
            ql.Continuous
        ).rate()

        return float(tasa)

    def obtener_discount_factor(self, años: float) -> float:
        """
        Calcula el factor de descuento para un plazo dado.

        El factor de descuento es el valor presente de 1 unidad monetaria
        a recibir en el futuro.

        Args:
            años: Plazo en años

        Returns:
            Factor de descuento (0 < df <= 1)

        Example:
            >>> curva = CurvaRendimientoTesoro()
            >>> curva.construir_desde_mercado()
            >>> df_10y = curva.obtener_discount_factor(10.0)
            >>> print(f"$1 en 10 años vale hoy: ${df_10y:.4f}")
            $1 en 10 años vale hoy: $0.6598
        """
        if self.curva_yield is None:
            raise ValueError("La curva no ha sido construida")

        import QuantLib as ql

        fecha_futura = self.curva_yield.referenceDate() + ql.Period(int(años * 12), ql.Months)
        df = self.curva_yield.discount(fecha_futura)

        return float(df)

    def to_dict(self) -> Dict:
        """
        Serializa la información de la curva a diccionario.

        Returns:
            Diccionario con metadata de la curva
        """
        return {
            'fecha_construccion': self.fecha_construccion.isoformat() if self.fecha_construccion else None,
            'tasas_mercado': self.tasas_mercado,
            'tiene_curva': self.curva_yield is not None
        }

    def __repr__(self):
        if self.curva_yield is None:
            return "CurvaRendimientoTesoro(no construida)"
        else:
            return f"CurvaRendimientoTesoro(fecha={self.fecha_construccion}, plazos={len(self.tasas_mercado)})"
