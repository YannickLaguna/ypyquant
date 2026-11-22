"""
Adaptadores de la fuente de precios y su elección por entorno.

Responsabilidad:
- Reunir los adaptadores de `FuentePrecios`: `FuenteYFinanceCache` (producción), `FuenteSintetica` (tests y UI
  sin datos) y `fuente_por_defecto()`, que elige uno según el entorno (`YPYQUANT_PRECIOS=sintetico` da el
  sintético; en otro caso, yfinance con caché Parquet). Un adaptador de Norgate se añadiría aquí como
  `norgate.py` y una rama en `fuente_por_defecto` (`YPYQUANT_PRECIOS=norgate`); ver el plan en
  `docs/planes/2026-11-22-fuente-precios-universos.md`.

Se relaciona con:
- ypyquant.data.fuente_precios (interfaz), app/api/main.py (construye la fuente al arrancar) y
  pipelines/paper_trading/precios.py (el bot usa la misma).
"""

from __future__ import annotations

import os

from ypyquant.data.fuente_precios import FuentePrecios
from ypyquant.data.fuentes.sintetica import FuenteSintetica
from ypyquant.data.fuentes.yfinance_cache import FuenteYFinanceCache

__all__ = ["FuenteSintetica", "FuenteYFinanceCache", "fuente_por_defecto"]


def fuente_por_defecto() -> FuentePrecios:
    """`YPYQUANT_PRECIOS=sintetico` da el sintético; en otro caso, yfinance con caché en Parquet."""
    if os.environ.get("YPYQUANT_PRECIOS", "").lower() == "sintetico":
        return FuenteSintetica()
    return FuenteYFinanceCache()
