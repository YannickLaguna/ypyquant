"""
Puente entre las señales heredadas (`ypyquant.signals`) y el registro de estrategias.

Responsabilidad: `senales_tecnicas` envuelve el combinador de señales del modo heredado del bot de
Alpaca (`construir_combinador`, con el generador técnico RSI + MACD + Bollinger + momentum) como una
`EstrategiaSenales`: entrada = señal > `umbral`, salida = señal < -`umbral`, y la base aporta posiciones,
stops y efectivo. Así el modo `--config estrategia_ejemplo.json` queda cubierto por el registro (backtest,
barridos, marcos y paper interno) y puede darse por obsoleto.

Diferencias con el modo heredado (`decidir_orden`):
- El tamaño pasa de proporcional a |señal| a peso fijo `1/max_posiciones` (o por volatilidad si se pide).
- No hay short: una señal por debajo de `-umbral` solo cierra la posición.
- La señal es un ranking transversal por fecha: depende de los tickers presentes en cada fecha (con el
  universo `sp500_pit` el ranking se calcula sobre todos los cargados; no mira al futuro).
- Se usa solo el generador técnico (los de factores, ML y régimen necesitan datos externos).

Se relaciona con: ypyquant.estrategias.senales (base), ypyquant.execution.nautilus.constructor_senales
(`construir_combinador`), ypyquant.signals (generadores), .registro y pipelines/paper_trading (modo heredado).
"""

from __future__ import annotations

from pydantic import Field

from ypyquant.estrategias.base import MARGEN_SESIONES
from ypyquant.estrategias.registro import registrar_estrategia
from ypyquant.estrategias.senales import EstrategiaSenales, ParametrosSenales

# Sesiones que necesita el generador técnico para tener todos sus indicadores (MACD lento + señal, Bollinger...)
SESIONES_TECNICAS = 60


@registrar_estrategia
class SenalesTecnicas(EstrategiaSenales):
    """Señal técnica combinada (RSI, MACD, Bollinger, momentum) convertida en reglas de entrada y salida."""

    nombre = "senales_tecnicas"
    descripcion = ("Puente con las señales técnicas heredadas: compra con señal > umbral y vende con señal < -umbral; "
                   "tamaño fijo por posición.")

    class Parametros(ParametrosSenales):
        """Parámetros de la base de señales más el umbral y la ventana del RSI."""

        umbral: float = Field(0.3, gt=0, lt=1, description="Entra con señal > umbral y sale con señal < -umbral")
        ventana_rsi: int = Field(14, ge=2, le=100, description="Sesiones del RSI del generador técnico")

    def indicadores(self, cierres, datos):
        """Señal combinada en [-1, 1] del generador técnico de `construir_combinador` (por defecto, solo técnico)."""
        from ypyquant.execution.nautilus.constructor_senales import (
            construir_combinador,  # import tardío: arrastra `ypyquant.execution`
        )

        combinador = construir_combinador({})
        return {"senal": combinador.generate(cierres, rsi_window=self.parametros.ventana_rsi)}

    def senal_entrada(self, cierres, indicadores):
        """Señal técnica por encima del umbral."""
        return indicadores["senal"] > self.parametros.umbral

    def senal_salida(self, cierres, indicadores):
        """Señal técnica por debajo de -umbral."""
        return indicadores["senal"] < -self.parametros.umbral

    def puntuacion(self, cierres, indicadores):
        """Entre candidatos, la señal más fuerte."""
        return indicadores["senal"]

    def sesiones_necesarias(self) -> int:
        """Historia mínima del generador técnico más el margen estándar."""
        return max(SESIONES_TECNICAS, self.parametros.ventana_rsi * 3) + MARGEN_SESIONES
