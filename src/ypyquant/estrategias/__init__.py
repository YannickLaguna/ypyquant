"""
Estrategias, experimentos y almacén de resultados de ypyquant.

Responsabilidad: punto de entrada público del subpaquete. Reúne el contrato `Estrategia`, el
registro con descubrimiento automático de estrategias de usuario, los experimentos
reproducibles (`ConfiguracionExperimento` + `ejecutar_experimento`), el almacén Parquet/DuckDB y los
barridos con su validación estadística (`barridos`, `validacion_barridos`, `reservado`).

Se relaciona con: portfolio.backtest_engine (motor), optimization (estimadores y presets),
app/api (API FastAPI) y estrategias_usuario/ (estrategias propias descubiertas al arrancar).

`AlmacenExperimentos` y los cargadores de precios requieren el extra `experimentos`
(pydantic, pyarrow, duckdb, yfinance); se importan de sus módulos para no exigirlos aquí.
"""

from ypyquant.estrategias.base import Estrategia, ParametrosBase, matriz_cierres, matriz_volumen, recortar_hasta
from ypyquant.estrategias.registro import (
    codigo_fuente,
    descubrir_estrategias_usuario,
    listar_estrategias,
    obtener_estrategia,
    registrar_estrategia,
)

__all__ = [
    "Estrategia",
    "ParametrosBase",
    "codigo_fuente",
    "descubrir_estrategias_usuario",
    "listar_estrategias",
    "matriz_cierres",
    "matriz_volumen",
    "obtener_estrategia",
    "recortar_hasta",
    "registrar_estrategia",
]
