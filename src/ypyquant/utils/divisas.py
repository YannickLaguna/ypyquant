"""
Divisas: contrato del servicio de tipos de cambio y conversión de curvas de equity.

Responsabilidad:
- `ServicioCambio` (Protocol): lo que ypyquant necesita de un servicio de tipos de cambio,
  `convert_amount(importe, divisa_origen, divisa_destino, fecha)`. ypyquant no descarga
  tipos de cambio: la aplicación (p. ej. GrowSave `ExchangeRateService`) inyecta el suyo.
- `convertir_equity_a_divisa`: convierte una curva de equity fecha a fecha. Es la única
  conversión de curvas del backtest (cartera y benchmark usan los mismos tipos y fechas).
- `simbolo_divisa` / `SIMBOLOS_DIVISA`: símbolo para logs e informes (copia de la tabla de
  GrowSave `app/utils/currency.CURRENCY_SYMBOLS`).

Se relaciona con:
- ypyquant.portfolio.backtest_engine (BacktestEngine y PortfolioBacktestService).
"""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ServicioCambio(Protocol):
    """Servicio de tipos de cambio inyectable (duck typing)."""

    def convert_amount(self, amount: float, from_currency: str, to_currency: str, target_date=None) -> float:
        """Convierte `amount` de `from_currency` a `to_currency` con el tipo de `target_date`."""
        ...


SIMBOLOS_DIVISA: dict[str, str] = {
    'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'CNY': '¥', 'CHF': 'CHF',
    'CAD': 'CA$', 'AUD': 'A$', 'NZD': 'NZ$', 'MXN': 'MX$', 'ARS': 'AR$', 'BRL': 'R$',
    'CLP': 'CL$', 'COP': 'CO$', 'PEN': 'S/', 'UYU': 'UY$', 'INR': '₹', 'KRW': '₩',
    'RUB': '₽', 'TRY': '₺', 'ZAR': 'R', 'SEK': 'kr',
}


def simbolo_divisa(codigo: str) -> str:
    """Devuelve el símbolo de una divisa ISO 4217 (o el propio código si no está en la tabla)."""
    return SIMBOLOS_DIVISA.get(codigo, codigo)


def convertir_equity_a_divisa(
    servicio_cambio: ServicioCambio,
    equity: pd.Series,
    divisa_origen: str,
    divisa_destino: str,
) -> pd.Series:
    """
    Convierte una curva de equity de una divisa a otra con el tipo de cambio de cada fecha.

    Args:
        servicio_cambio: objeto con `convert_amount(importe, origen, destino, fecha)`.
        equity: serie de valores indexada por fecha.
        divisa_origen / divisa_destino: códigos ISO 4217.

    Returns:
        Serie convertida con el mismo índice (copia si las divisas coinciden).
    """
    if divisa_origen == divisa_destino:
        return equity.copy()
    valores = []
    for indice, valor in equity.items():
        fecha = indice.date() if hasattr(indice, 'date') else indice
        valores.append(servicio_cambio.convert_amount(valor, divisa_origen, divisa_destino, fecha))
    return pd.Series(valores, index=equity.index)
