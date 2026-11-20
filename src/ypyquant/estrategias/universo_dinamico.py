"""
Universo point-in-time `sp500_pit`: en cada fecha de rebalanceo solo son elegibles los miembros del
S&P 500 EN ESA FECHA con historia suficiente hasta t-1 (sin sesgo de supervivencia).

Motivo: `universos/sp500.txt` son los miembros ACTUALES del índice; backtestear con ellos premia a las
empresas que acabaron entrando o permaneciendo. Aquí se reutiliza el criterio del motor del TFM
(`optimization/estimacion_robusta/universo.py`) sin reimplementarlo: `elegibles_en_fecha` y
`construir_mascara_membresia`.

Responsabilidad:
- Fuente de membresías: `universos/sp500_membresias.json` (ticker -> intervalos `date_added/date_removed`),
  generado desde la columna `tickers.index_memberships` de `TFM.db` (la misma que lee la prueba de paridad
  del TFM) con `scripts/exportar_membresias_sp500.py`. `extraer_membresias_desde_sqlite` hace la
  extracción (solo índice SP500, tickers con el formato de yfinance: `BRK.B` -> `BRK-B`);
  `cargar_membresias_sp500` lee el JSON; `miembros_en_ventana` lista los miembros en un rango de fechas.
- `UniversoPuntoEnElTiempo.crear_filtro(datos)` -> `filtro(fecha, datos_recortados) -> datos_filtrados`:
  precalcula (una vez) el panel mensual de cierres y la máscara de membresía y devuelve la función que la
  estrategia usa por fecha (`Estrategia.como_funcion(filtro_universo=...)`). Solo se usan precios de los
  meses completos anteriores a la fecha (t-T-1 .. t-1), de modo que ningún dato con fecha >= t influye
  (se comprueba con `verificar_sin_lookahead`).

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

Se relaciona con: ypyquant.optimization.estimacion_robusta.universo (elegibilidad y máscara),
ypyquant.estrategias.base (`como_funcion(filtro_universo=...)`), .experimentos (`resolver_universo`,
`ejecutar_con_datos`), .causalidad (prueba de ausencia de look-ahead), universos/sp500_membresias.json
y scripts/exportar_membresias_sp500.py.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ypyquant.estrategias.base import matriz_cierres
from ypyquant.optimization.estimacion_robusta.universo import construir_mascara_membresia, elegibles_en_fecha

RAIZ_REPO = Path(__file__).resolve().parents[3]
RUTA_MEMBRESIAS = RAIZ_REPO / "universos" / "sp500_membresias.json"
NOMBRE_UNIVERSO_PIT = "sp500_pit"
INDICE = "SP500"
MESES_HISTORIA_POR_DEFECTO = 12

# `filtro(fecha, datos_recortados_a_fecha) -> {ticker: OHLCV}` solo con los elegibles en `fecha`
FiltroUniverso = Callable[[datetime, dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]
Membresias = Mapping[str, Sequence[Mapping[str, Any]]]


def es_universo_pit(universo: Any) -> bool:
    """True si `universo` es el nombre del universo dinámico (`sp500_pit`)."""
    return isinstance(universo, str) and universo.partition(":")[0] == NOMBRE_UNIVERSO_PIT


# ----------------------------------------------------------------------------
# Fuente de membresías
# ----------------------------------------------------------------------------


def _fecha_corta(valor: str | None) -> str | None:
    """`'2010-02-16T00:00:00'` -> `'2010-02-16'` (None se conserva)."""
    return str(valor)[:10] if valor else None


def extraer_membresias_desde_sqlite(ruta_db: str | Path) -> dict[str, list[dict[str, str | None]]]:
    """
    Intervalos de pertenencia al S&P 500 de cada ticker leídos de `tickers.index_memberships` (SQLite, solo lectura).

    Devuelve `{ticker: [{"date_added", "date_removed"}]}` ordenado por ticker; solo entradas del índice
    SP500 y fechas `YYYY-MM-DD`. El ticker se normaliza al formato de yfinance (`.` -> `-`).
    """
    conn = sqlite3.connect(f"file:{Path(ruta_db).as_posix()}?mode=ro", uri=True)
    try:
        filas = conn.execute("SELECT symbol, index_memberships FROM tickers WHERE index_memberships IS NOT NULL").fetchall()
    finally:
        conn.close()
    membresias: dict[str, list[dict[str, str | None]]] = {}
    for simbolo, crudo in filas:
        intervalos = [
            {"date_added": _fecha_corta(m.get("date_added")), "date_removed": _fecha_corta(m.get("date_removed"))}
            for m in (json.loads(crudo) if isinstance(crudo, str) else crudo or []) if m.get("index") == INDICE
        ]
        if intervalos:
            membresias[simbolo.replace(".", "-")] = intervalos
    return dict(sorted(membresias.items()))


def cargar_membresias_sp500(ruta: str | Path | None = None) -> dict[str, list[dict[str, str | None]]]:
    """
    Membresías del S&P 500 desde `universos/sp500_membresias.json`, en el formato que espera
    `construir_mascara_membresia` (cada intervalo lleva `index="SP500"`).
    """
    ruta = Path(ruta) if ruta else RUTA_MEMBRESIAS
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {ruta}. Genéralo con `uv run python scripts/exportar_membresias_sp500.py` "
            "(necesita TFM.db; ver la cabecera de ypyquant/estrategias/universo_dinamico.py).")
    crudo = json.loads(ruta.read_text(encoding="utf-8"))["miembros"]
    return {t: [{"index": INDICE, **m} for m in intervalos] for t, intervalos in crudo.items()}


def _intervalo(m: Mapping[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    inicio = pd.Timestamp(m["date_added"]) if m.get("date_added") else pd.Timestamp.min
    fin = pd.Timestamp(m["date_removed"]) if m.get("date_removed") else pd.Timestamp.max
    return inicio, fin


def miembros_en_ventana(membresias: Membresias, inicio: str | pd.Timestamp, fin: str | pd.Timestamp) -> list[str]:
    """Tickers (ordenados) que fueron miembros en algún momento de `[inicio, fin]` (intervalos inclusivos)."""
    inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
    elegidos = []
    for ticker, intervalos in membresias.items():
        if any(i <= fin and f >= inicio for i, f in map(_intervalo, intervalos)):
            elegidos.append(ticker)
    return sorted(elegidos)


def cobertura_membresias(membresias: Membresias, tickers_con_precio: set[str] | Sequence[str],
                         fechas: Sequence[str | pd.Timestamp]) -> list[dict[str, Any]]:
    """
    Por fecha: cuántos miembros del índice había (`miembros`) y cuántos de ellos tienen precios en la fuente
    (`con_precio`); `sin_precio` son los que el universo no puede elegir (sesgo de supervivencia residual).
    """
    con_precio = set(tickers_con_precio)
    filas = []
    for fecha in fechas:
        miembros = miembros_en_ventana(membresias, fecha, fecha)
        n_con = sum(1 for t in miembros if t in con_precio)
        filas.append({"fecha": str(pd.Timestamp(fecha).date()), "miembros": len(miembros), "con_precio": n_con,
                      "sin_precio": len(miembros) - n_con})
    return filas


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
