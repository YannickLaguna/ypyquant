"""
Universo point-in-time `sp500_pit`: en cada fecha de rebalanceo solo son elegibles los miembros del
S&P 500 EN ESA FECHA con historia suficiente hasta t-1 (sin sesgo de supervivencia).

Motivo: `universos/sp500.txt` son los miembros ACTUALES del índice; backtestear con ellos premia a las
empresas que acabaron entrando o permaneciendo. Aquí se reutiliza el criterio del motor del TFM
(`optimization/estimacion_robusta/universo.py`) sin reimplementarlo: `elegibles_en_fecha` y
`construir_mascara_membresia`.

Responsabilidad:
- `UniversoPuntoEnElTiempo.crear_filtro(datos)` -> `filtro(fecha, datos_recortados) -> datos_filtrados`:
  precalcula (una vez) el panel mensual de cierres y la máscara de membresía y devuelve la función que la
  estrategia usa por fecha (`Estrategia.como_funcion(filtro_universo=...)`). Solo se usan precios de los
  meses completos anteriores a la fecha (t-T-1 .. t-1), de modo que ningún dato con fecha >= t influye
  (se comprueba con `verificar_sin_lookahead`).
- Las membresías (fuente `universos/sp500_membresias.json`, extracción, ventanas, cobertura y formato de
  símbolos) NO viven aquí sino en `ypyquant.data.universos`, el único módulo que las conoce; este recibe
  el diccionario de membresías y solo decide la elegibilidad por fecha.

Cómo se trata cada caso:
- Entra al índice en t+1: NO es elegible en t. Sale o deslista después de t: SÍ lo es en t.
- Historia: se exigen `meses_historia` retornos mensuales completos (por defecto 12; es la regla del TFM);
  las salidas a bolsa recientes no entran hasta cumplirlos.
- Los miembros históricos sin precios en la fuente (deslistados que yfinance ya no sirve) no pueden
  elegirse: es un sesgo de supervivencia RESIDUAL que `ejecutar_con_datos` anota en los avisos con la
  cobertura por fecha. Medida local (TFM.db): ver el informe de la fase 0 y `cobertura_membresias`.

Límites de la fuente: antes de 1999 las membresías son incompletas (~140 miembros en 1996); desde 2000
rondan los ~480-520 por fecha. Una empresa que cambió de ticker figura con el ticker vigente; si
yfinance tiene el histórico bajo ese ticker, el nombre antiguo no se resuelve.

Se relaciona con: ypyquant.data.universos (membresías y símbolos), ypyquant.optimization.estimacion_robusta.universo
(elegibilidad y máscara), ypyquant.estrategias.base (`como_funcion(filtro_universo=...)`), .simulacion (lo
construye con las membresías), .causalidad (prueba de ausencia de look-ahead).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pandas as pd

from ypyquant.data.universos import INDICE, Membresias
from ypyquant.estrategias.base import matriz_cierres
from ypyquant.optimization.estimacion_robusta.universo import construir_mascara_membresia, elegibles_en_fecha

MESES_HISTORIA_POR_DEFECTO = 12

# `filtro(fecha, datos_recortados_a_fecha) -> {ticker: OHLCV}` solo con los elegibles en `fecha`
FiltroUniverso = Callable[[datetime, dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]


# ----------------------------------------------------------------------------
# Universo point-in-time
# ----------------------------------------------------------------------------


class UniversoPuntoEnElTiempo:
    """
    Criterio de elegibilidad point-in-time sobre unas membresías dadas.

    Args:
        membresias: `{ticker: [{"index": "SP500", "date_added", "date_removed"}]}` (ver `cargar_membresias_sp500`).
        meses_historia: retornos mensuales completos que se exigen hasta t-1 (por defecto 12).
    """

    def __init__(self, membresias: Membresias, meses_historia: int = MESES_HISTORIA_POR_DEFECTO):
        if meses_historia < 1:
            raise ValueError("meses_historia debe ser >= 1")
        self.membresias = {t: [{"index": INDICE, **dict(m)} if "index" not in m else dict(m) for m in intervalos]
                           for t, intervalos in membresias.items()}
        self.meses_historia = meses_historia

    def crear_filtro(self, datos: dict[str, pd.DataFrame]) -> FiltroUniverso:
        """
        Filtro `(fecha, datos_recortados) -> datos_de_los_elegibles` calculado sobre `datos` (la unión del universo).

        Precalcula el panel mensual de cierres (último cierre ajustado de cada mes) y la máscara de
        membresía sobre el calendario de `datos`; solo se consultan meses anteriores a `fecha`, así que
        pasar el histórico completo no filtra información del futuro. El resultado de cada fecha se cachea.
        """
        cierres = matriz_cierres(datos)
        if cierres.empty:
            return lambda fecha, recortados: {}
        mensual = cierres.groupby(cierres.index.to_period("M")).last()
        mensual.index = mensual.index.to_timestamp(how="end").normalize()
        mascara = construir_mascara_membresia(self.membresias, cierres.index, list(cierres.columns), indice=INDICE)
        cache: dict[pd.Timestamp, pd.Index] = {}

        def filtro(fecha: datetime, recortados: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
            clave = pd.Timestamp(fecha)
            if clave not in cache:
                cache[clave] = elegibles_en_fecha(mensual, mascara, clave, self.meses_historia, 1.0)
            return {t: recortados[t] for t in cache[clave] if t in recortados}

        return filtro
