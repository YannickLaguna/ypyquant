"""
Ejemplo de estrategia de señales: ruptura de canal con filtro de RSI y tamaño por volatilidad.

Responsabilidad: segunda plantilla de `EstrategiaSenales`, con indicadores más variados que el cruce de
medias: usa un canal de precios (máximo de las últimas N sesiones), el RSI como filtro de sobrecompra y el
tamaño inverso a la volatilidad que ofrece la base (`ponderacion="inversa_volatilidad"` por defecto).

Regla: entra cuando el cierre supera el máximo de las `ventana_canal` sesiones anteriores (ruptura) y el
RSI no está en sobrecompra; sale cuando el cierre pierde el mínimo de las `ventana_salida` sesiones
anteriores (canal de salida más corto) o por stop. Entre candidatos elige la mayor ruptura relativa.
Los canales usan `shift(1)`: se compara con el máximo/mínimo ANTERIORES, así que no hay look-ahead
(se comprueba en tests/test_senales_ejemplos.py).

Se relaciona con: ypyquant.estrategias.senales (base, stops y tamaño), el registro (descubrimiento
automático), docs/estrategias_senales.md (guía) y estrategias_usuario/cruce_medias.py (plantilla mínima).
"""

from typing import Literal

from pydantic import Field, model_validator

from ypyquant.estrategias import registrar_estrategia
from ypyquant.estrategias.base import MARGEN_SESIONES
from ypyquant.estrategias.senales import EstrategiaSenales, ParametrosSenales


def calcular_rsi(cierres, ventana):
    """RSI de Wilder (media exponencial de alfa 1/ventana) de cada columna; 0-100, NaN sin historia."""
    cambio = cierres.diff()
    ganancia = cambio.clip(lower=0.0).ewm(alpha=1.0 / ventana, adjust=False, min_periods=ventana).mean()
    perdida = (-cambio.clip(upper=0.0)).ewm(alpha=1.0 / ventana, adjust=False, min_periods=ventana).mean()
    fuerza = ganancia / perdida
    return 100.0 - 100.0 / (1.0 + fuerza)


@registrar_estrategia
class CanalRsiVolatilidad(EstrategiaSenales):
    """Ruptura de máximos con filtro de RSI y tamaño inversamente proporcional a la volatilidad."""

    nombre = "canal_rsi_volatilidad"
    descripcion = ("Compra rupturas de máximos con RSI no sobrecomprado y tamaño por volatilidad; "
                   "sale al perder el canal corto o por stop (ejemplo de señales).")
    universo_sugerido = "sp500_pit"

    class Parametros(ParametrosSenales):
        """Parámetros de la base (con tamaño por volatilidad y trailing por defecto) más los del canal y el RSI."""

        ponderacion: Literal["iguales", "inversa_volatilidad"] = "inversa_volatilidad"
        stop_trailing: float | None = Field(0.15, gt=0, lt=1, description="Sale si cae esta fracción desde el máximo")
        ventana_canal: int = Field(55, ge=5, le=400, description="Sesiones del canal de entrada (máximo previo)")
        ventana_salida: int = Field(20, ge=2, le=200, description="Sesiones del canal de salida (mínimo previo)")
        ventana_rsi: int = Field(14, ge=2, le=100, description="Sesiones del RSI")
        rsi_maximo: float = Field(80.0, gt=0, le=100, description="No entra si el RSI supera este valor (sobrecompra)")

        @model_validator(mode="after")
        def _salida_menor_que_entrada(self):
            if self.ventana_salida >= self.ventana_canal:
                raise ValueError("ventana_salida debe ser menor que ventana_canal")
            return self

    def indicadores(self, cierres, datos):
        """Máximo y mínimo de las sesiones PREVIAS (shift(1)) y RSI de Wilder."""
        p = self.parametros
        return {
            "maximo_previo": cierres.rolling(p.ventana_canal).max().shift(1),
            "minimo_previo": cierres.rolling(p.ventana_salida).min().shift(1),
            "rsi": calcular_rsi(cierres, p.ventana_rsi),
        }

    def senal_entrada(self, cierres, indicadores):
        """Cierre por encima del máximo previo y RSI por debajo de `rsi_maximo`."""
        ruptura = cierres > indicadores["maximo_previo"]
        return (ruptura & (indicadores["rsi"] < self.parametros.rsi_maximo)).fillna(False)

    def senal_salida(self, cierres, indicadores):
        """Cierre por debajo del mínimo previo del canal corto."""
        return (cierres < indicadores["minimo_previo"]).fillna(False)

    def puntuacion(self, cierres, indicadores):
        """Entre candidatos, la mayor ruptura relativa sobre el máximo previo."""
        return cierres / indicadores["maximo_previo"] - 1.0

    def sesiones_necesarias(self) -> int:
        """Historia mínima: el canal más largo (o el RSI, si fuera mayor) más el margen estándar."""
        p = self.parametros
        return max(p.ventana_canal, p.ventana_rsi * 3) + MARGEN_SESIONES
