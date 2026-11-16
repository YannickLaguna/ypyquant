"""
Pricing de bonos del Tesoro USA usando QuantLib.

Responsabilidad: calcular precio, YTM, duration, convexity y DV01 de
instrumentos de renta fija del Tesoro (T-Bills, T-Notes, T-Bonds).

Relacionado con:
  - data/quantlib_utils.py  (helpers QuantLib: fechas, calendarios, convenciones)
  - data/yield_curves.py    (CurvaRendimientoTesoro — usado en price_treasury_from_yield_curve)
"""

from typing import Optional, Dict, Literal
from datetime import date, datetime, timedelta
import pandas as pd
import warnings

from ypyquant.data.quantlib_utils import (
    tiene_quantlib,
    convertir_fecha_a_ql,
    crear_calendario_usa,
    obtener_day_count_convention,
    configurar_fecha_evaluacion
)


class BondPricer:
    """
    Pricer de bonos del Tesoro USA usando QuantLib.

    Soporta pricing completo de T-Bills, T-Notes y T-Bonds con todas
    las métricas estándar de renta fija.
    """

    def __init__(self):
        """Inicializa el pricer de bonos."""
        if not tiene_quantlib():
            raise ImportError(
                "QuantLib no está disponible. "
                "Instalar con: pip install QuantLib"
            )

        self.calendario = crear_calendario_usa()
        self.day_count = obtener_day_count_convention('Actual/Actual')

    def price_tbill(
        self,
        vencimiento: str | date,
        tasa_descuento: float,
        valor_nominal: float = 100.0,
        fecha_valoracion: Optional[str | date] = None
    ) -> Dict:
        """
        Calcula el precio de un T-Bill (bono de descuento sin cupón).

        Los T-Bills se venden con descuento y pagan el valor nominal al vencimiento.

        Args:
            vencimiento: Fecha de vencimiento 'YYYY-MM-DD' o date
            tasa_descuento: Tasa de descuento anual (decimal, ej: 0.045 para 4.5%)
            valor_nominal: Valor nominal del bono (default: 100)
            fecha_valoracion: Fecha de valoración. Si None, usa fecha actual.

        Returns:
            Dict con precio, YTM, días al vencimiento, etc.

        Example:
            >>> pricer = BondPricer()
            >>> resultado = pricer.price_tbill('2025-06-15', 0.045, 100.0)
            >>> print(f"Precio: ${resultado['precio']:.2f}")
            Precio: $97.50
        """
        import QuantLib as ql

        # Configurar fecha de evaluación
        if fecha_valoracion is None:
            fecha_valoracion = date.today()
        configurar_fecha_evaluacion(fecha_valoracion)

        fecha_val_ql = convertir_fecha_a_ql(fecha_valoracion)
        fecha_vto_ql = convertir_fecha_a_ql(vencimiento)

        # Días al vencimiento
        dias_vto = (fecha_vto_ql - fecha_val_ql)

        # Precio de T-Bill: P = FV * (1 - r * (días/360))
        # Usamos convención 360 días para T-Bills
        años_vto = dias_vto / 360.0
        precio = valor_nominal * (1 - tasa_descuento * años_vto)

        # YTM (Yield to Maturity) equivalente
        if años_vto > 0:
            ytm = (valor_nominal - precio) / precio / años_vto
        else:
            ytm = 0.0

        return {
            'precio': float(precio),
            'valor_nominal': float(valor_nominal),
            'tasa_descuento': float(tasa_descuento),
            'ytm': float(ytm),
            'dias_vencimiento': int(dias_vto),
            'años_vencimiento': float(años_vto),
            'fecha_valoracion': fecha_valoracion.isoformat() if isinstance(fecha_valoracion, date) else fecha_valoracion,
            'fecha_vencimiento': vencimiento.isoformat() if isinstance(vencimiento, date) else vencimiento,
            'tipo': 'T-Bill'
        }

    def price_bond_coupon(
        self,
        vencimiento: str | date,
        cupon_anual: float,
        frecuencia_cupon: int = 2,
        ytm: Optional[float] = None,
        precio: Optional[float] = None,
        valor_nominal: float = 100.0,
        fecha_valoracion: Optional[str | date] = None
    ) -> Dict:
        """
        Calcula precio o YTM de un bono con cupón (T-Note o T-Bond).

        Debe proporcionar YTM (para calcular precio) o precio (para calcular YTM).

        Args:
            vencimiento: Fecha de vencimiento
            cupon_anual: Tasa de cupón anual (decimal, ej: 0.04 para 4%)
            frecuencia_cupon: Pagos por año (2=semestral, 4=trimestral, 12=mensual)
            ytm: Yield to Maturity. Si None, se calcula desde precio.
            precio: Precio del bono. Si None, se calcula desde YTM.
            valor_nominal: Valor nominal (default: 100)
            fecha_valoracion: Fecha de valoración

        Returns:
            Dict con precio, YTM, duration, convexity, etc.

        Example:
            >>> pricer = BondPricer()
            >>> # Calcular precio desde YTM
            >>> resultado = pricer.price_bond_coupon(
            ...     vencimiento='2030-05-15',
            ...     cupon_anual=0.04,
            ...     ytm=0.045
            ... )
            >>> print(f"Precio: ${resultado['precio']:.2f}")
            Precio: $96.23
        """
        import QuantLib as ql

        if ytm is None and precio is None:
            raise ValueError("Debe proporcionar ytm o precio")

        # Configurar fecha de evaluación
        if fecha_valoracion is None:
            fecha_valoracion = date.today()
        configurar_fecha_evaluacion(fecha_valoracion)

        fecha_val_ql = convertir_fecha_a_ql(fecha_valoracion)
        fecha_vto_ql = convertir_fecha_a_ql(vencimiento)

        # Mapeo de frecuencia
        freq_map = {
            1: ql.Annual,
            2: ql.Semiannual,
            4: ql.Quarterly,
            12: ql.Monthly
        }
        frecuencia_ql = freq_map.get(frecuencia_cupon, ql.Semiannual)

        # Crear schedule de cupones
        schedule = ql.Schedule(
            fecha_val_ql,
            fecha_vto_ql,
            ql.Period(frecuencia_ql),
            self.calendario,
            ql.Unadjusted,
            ql.Unadjusted,
            ql.DateGeneration.Backward,
            False
        )

        # Crear bono
        bond = ql.FixedRateBond(
            2,  # settlement days
            valor_nominal,
            schedule,
            [cupon_anual],
            self.day_count
        )

        # Configurar pricing engine
        if ytm is not None:
            # Calcular precio desde YTM
            bond_engine = ql.DiscountingBondEngine(
                ql.YieldTermStructureHandle(
                    ql.FlatForward(fecha_val_ql, ytm, self.day_count, ql.Compounded, frecuencia_ql)
                )
            )
            bond.setPricingEngine(bond_engine)

            precio_clean = bond.cleanPrice()
            precio_dirty = bond.dirtyPrice()
            ytm_calculado = ytm
        else:
            # Calcular YTM desde precio
            precio_clean = precio
            ytm_calculado = bond.bondYield(
                precio,
                self.day_count,
                ql.Compounded,
                frecuencia_ql
            )

            # Reconfigurar con YTM calculado para obtener dirty price y métricas
            bond_engine = ql.DiscountingBondEngine(
                ql.YieldTermStructureHandle(
                    ql.FlatForward(fecha_val_ql, ytm_calculado, self.day_count, ql.Compounded, frecuencia_ql)
                )
            )
            bond.setPricingEngine(bond_engine)
            precio_dirty = bond.dirtyPrice()

        # Calcular duration y convexity
        duration_macaulay = ql.BondFunctions.duration(
            bond,
            ytm_calculado,
            self.day_count,
            ql.Compounded,
            frecuencia_ql,
            ql.Duration.Macaulay
        )

        duration_modified = ql.BondFunctions.duration(
            bond,
            ytm_calculado,
            self.day_count,
            ql.Compounded,
            frecuencia_ql,
            ql.Duration.Modified
        )

        convexity = ql.BondFunctions.convexity(
            bond,
            ytm_calculado,
            self.day_count,
            ql.Compounded,
            frecuencia_ql
        )

        # DV01: cambio en precio por 1 bp de cambio en YTM
        # DV01 ≈ -Modified Duration * Precio / 10000
        dv01 = -duration_modified * precio_clean / 10000.0

        # Accrued interest
        accrued = precio_dirty - precio_clean

        # Años al vencimiento
        años_vto = duration_macaulay / frecuencia_cupon if frecuencia_cupon > 0 else 0

        # Determinar tipo de bono
        if años_vto <= 2:
            tipo_bono = 'T-Note (2Y)'
        elif años_vto <= 10:
            tipo_bono = 'T-Note'
        elif años_vto <= 20:
            tipo_bono = 'T-Bond (20Y)'
        else:
            tipo_bono = 'T-Bond (30Y)'

        return {
            'precio_clean': float(precio_clean),
            'precio_dirty': float(precio_dirty),
            'accrued_interest': float(accrued),
            'valor_nominal': float(valor_nominal),
            'cupon_anual': float(cupon_anual),
            'frecuencia_cupon': int(frecuencia_cupon),
            'ytm': float(ytm_calculado),
            'duration_macaulay': float(duration_macaulay),
            'duration_modified': float(duration_modified),
            'convexity': float(convexity),
            'dv01': float(dv01),
            'años_vencimiento': float(años_vto),
            'fecha_valoracion': fecha_valoracion.isoformat() if isinstance(fecha_valoracion, date) else fecha_valoracion,
            'fecha_vencimiento': vencimiento.isoformat() if isinstance(vencimiento, date) else vencimiento,
            'tipo': tipo_bono
        }

    def price_treasury_from_yield_curve(
        self,
        vencimiento: str | date,
        cupon_anual: float,
        frecuencia_cupon: int = 2,
        valor_nominal: float = 100.0,
        fecha_valoracion: Optional[str | date] = None
    ) -> Dict:
        """
        Calcula precio de un bono usando la curva de rendimiento actual.

        Este método descarga la curva de rendimiento del Tesoro y usa
        la tasa spot apropiada para el plazo del bono.

        Args:
            vencimiento: Fecha de vencimiento
            cupon_anual: Tasa de cupón anual
            frecuencia_cupon: Pagos por año
            valor_nominal: Valor nominal
            fecha_valoracion: Fecha de valoración

        Returns:
            Dict con pricing completo usando curva de mercado
        """
        from ypyquant.data.yield_curves import CurvaRendimientoTesoro

        # Configurar fecha
        if fecha_valoracion is None:
            fecha_valoracion = date.today()

        # Construir curva de rendimiento
        curva = CurvaRendimientoTesoro()
        curva.construir_desde_mercado(fecha_valoracion)

        # Calcular años al vencimiento
        if isinstance(vencimiento, str):
            vto_date = datetime.strptime(vencimiento, '%Y-%m-%d').date()
        else:
            vto_date = vencimiento

        if isinstance(fecha_valoracion, str):
            val_date = datetime.strptime(fecha_valoracion, '%Y-%m-%d').date()
        else:
            val_date = fecha_valoracion

        años_vto = (vto_date - val_date).days / 365.25

        # Obtener YTM de la curva
        ytm_mercado = curva.obtener_tasa(años_vto)

        # Price con YTM de mercado
        if años_vto < 1.0:
            # T-Bill
            resultado = self.price_tbill(
                vencimiento=vencimiento,
                tasa_descuento=ytm_mercado,
                valor_nominal=valor_nominal,
                fecha_valoracion=fecha_valoracion
            )
            resultado['ytm_mercado'] = ytm_mercado
            resultado['fuente_ytm'] = 'curva_tesoro'
        else:
            # T-Note/T-Bond
            resultado = self.price_bond_coupon(
                vencimiento=vencimiento,
                cupon_anual=cupon_anual,
                frecuencia_cupon=frecuencia_cupon,
                ytm=ytm_mercado,
                valor_nominal=valor_nominal,
                fecha_valoracion=fecha_valoracion
            )
            resultado['ytm_mercado'] = ytm_mercado
            resultado['fuente_ytm'] = 'curva_tesoro'

        return resultado

    def calcular_metricas_portafolio_bonos(
        self,
        bonos: list[Dict],
        pesos: Optional[list[float]] = None
    ) -> Dict:
        """
        Calcula métricas agregadas de un portafolio de bonos.

        Args:
            bonos: Lista de dicts con resultados de pricing
            pesos: Pesos de cada bono. Si None, equal-weight.

        Returns:
            Dict con duration, convexity, ytm promedio ponderados

        Example:
            >>> pricer = BondPricer()
            >>> bono1 = pricer.price_bond_coupon('2027-05-15', 0.04, ytm=0.045)
            >>> bono2 = pricer.price_bond_coupon('2030-11-15', 0.035, ytm=0.042)
            >>> metricas = pricer.calcular_metricas_portafolio_bonos(
            ...     [bono1, bono2],
            ...     pesos=[0.6, 0.4]
            ... )
        """
        if not bonos:
            raise ValueError("Lista de bonos vacía")

        if pesos is None:
            pesos = [1.0 / len(bonos)] * len(bonos)

        if len(pesos) != len(bonos):
            raise ValueError("Número de pesos debe coincidir con número de bonos")

        # Normalizar pesos
        suma_pesos = sum(pesos)
        pesos_norm = [p / suma_pesos for p in pesos]

        # Calcular promedios ponderados
        duration_portfolio = 0.0
        convexity_portfolio = 0.0
        ytm_portfolio = 0.0
        dv01_portfolio = 0.0
        valor_total = 0.0

        for bono, peso in zip(bonos, pesos_norm):
            # Obtener precio (clean si disponible, sino precio directo)
            precio_bono = bono.get('precio_clean', bono.get('precio', 0.0))
            valor_bono = precio_bono * peso

            # Duration ponderado por valor de mercado
            duration_portfolio += bono.get('duration_modified', 0.0) * valor_bono
            convexity_portfolio += bono.get('convexity', 0.0) * valor_bono
            ytm_portfolio += bono.get('ytm', 0.0) * valor_bono
            dv01_portfolio += bono.get('dv01', 0.0) * peso

            valor_total += valor_bono

        # Normalizar por valor total
        if valor_total > 0:
            duration_portfolio /= valor_total
            convexity_portfolio /= valor_total
            ytm_portfolio /= valor_total

        return {
            'duration_modified': float(duration_portfolio),
            'convexity': float(convexity_portfolio),
            'ytm_promedio': float(ytm_portfolio),
            'dv01': float(dv01_portfolio),
            'valor_total': float(valor_total),
            'num_bonos': len(bonos),
            'pesos': pesos_norm
        }

    def to_dataframe(self, resultados: list[Dict]) -> pd.DataFrame:
        """
        Convierte lista de resultados de pricing a DataFrame.

        Args:
            resultados: Lista de dicts con resultados de pricing

        Returns:
            DataFrame con todos los bonos y sus métricas
        """
        return pd.DataFrame(resultados)
