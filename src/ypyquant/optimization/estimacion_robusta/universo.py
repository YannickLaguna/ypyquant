"""
Construcción del universo de activos por celda (N, T) y fecha, sin look-ahead.

Responsabilidad:
- `generar_prioridad_universo`: prioridad aleatoria u_i ~ U(0,1) por ticker y réplica,
  derivada del NOMBRE del ticker (SHA-256), estable entre sesiones y ante cambios del
  conjunto de tickers.
- `seleccionar_universo_anidado`: los N elegibles de menor prioridad (anidamiento exacto
  U_20 ⊂ U_40 ⊂ U_80 ⊂ U_150).
- `elegibles_en_fecha`: NUEVO respecto al TFM. Elegibilidad point-in-time en la fecha de
  rebalanceo t: cobertura completa de precios en los T meses previos y pertenencia al
  índice en t; solo usa precios con fecha <= t-1.
- `construir_mascara_membresia`: convierte los intervalos `date_added/date_removed` de
  cada ticker (formato de la columna `index_memberships` del TFM) en una máscara booleana.

Sesgos del TFM que `elegibles_en_fecha` corrige (y que el motor de rejilla.py permite
reproducir con una función de elegibles inyectada, para paridad):
1. Filtro de calidad/contaminación sobre TODO el histórico (incluido el periodo OOS
   futuro): el TFM excluía tickers detectados como corruptos mirando datos posteriores
   a t. Aquí no hay ningún filtro que mire más allá de t-1; la limpieza de datos debe
   hacerse antes y de forma point-in-time (o aplicarse como máscara fechada).
2. Exigir precio en t: el TFM descartaba los activos sin retorno en t (los que deslistan
   en ese mes), lo que elimina del universo OOS justo las bajas (sesgo de supervivencia
   que infla el retorno OOS). Aquí un activo que deslista después es elegible en t.
3. Umbral de cobertura del 95 % + `dropna()` posterior que acortaba la ventana de forma
   distinta según el universo de la celda: aquí se exige cobertura completa (T retornos
   observados), de modo que T es exactamente T para todos los activos.
4. Pertenencia al índice evaluada en t-1: aquí se evalúa en t (la composición vigente en
   la fecha en que se forma la cartera; los cambios se anuncian antes de ser efectivos).

Se relaciona con: rejilla.py (motor walk-forward que lo llama), w_mercado.py (usa los
mismos activos elegidos) y tests/test_estimacion_robusta.py.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

# ============================================================================
# Prioridad aleatoria y anidamiento
# ============================================================================


def seed_para(b: int, semilla_base: int = 42) -> int:
    """Semilla determinista de la réplica b."""
    return semilla_base + b


def _seed_ticker(ticker: str, b: int, semilla_base: int) -> int:
    """
    Semilla de 32 bits por (ticker, b, semilla_base): SHA-256 de la clave concatenada.
    Se evita `hash()` de Python porque está aleatorizado por proceso.
    """
    clave = f"{ticker}|{b}|{semilla_base}".encode()
    return int(hashlib.sha256(clave).hexdigest(), 16) % (2**32)


def generar_prioridad_universo(
    tickers, b: int, semilla_base: int = 42, metodo: Literal["hash", "posicional"] = "hash"
) -> pd.Series:
    """
    Prioridad u_i ~ U(0,1) por ticker, fija durante toda la réplica `b` (no se regenera
    por T ni por fecha). Cada ticker usa su propio generador semillado por su NOMBRE, no
    por su posición: añadir, quitar o reordenar otros tickers no mueve la prioridad de
    ninguno. Es aleatoria (no un ranking por cobertura) para que el efecto de N no se
    confunda con un gradiente de calidad de datos.

    `metodo="posicional"` reproduce el esquema ANTERIOR del TFM (un único vector
    `default_rng(semilla_base+b).random(len(tickers))` asignado por posición a `tickers`).
    Es frágil (cambia en cascada si cambia el conjunto u orden de tickers) pero es el que
    generó los parquet publicados de `rejilla_v2` (corrida del 2026-08-31 01:13, previa al
    arreglo por hash): solo sirve para paridad con esos resultados.
    """
    if metodo == "posicional":
        return pd.Series(np.random.default_rng(seed_para(b, semilla_base)).random(len(tickers)), index=list(tickers))
    if metodo != "hash":
        raise ValueError("metodo debe ser 'hash' o 'posicional'")
    return pd.Series({t: np.random.default_rng(_seed_ticker(t, b, semilla_base)).random() for t in tickers})


def seleccionar_universo_anidado(elegibles, prioridad: pd.Series, n: int) -> list:
    """
    Universo de la celda (N, T) en una fecha: los `n` elegibles de menor prioridad.
    Con prioridad fija por réplica el anidamiento entre N distintos es exacto (el mismo
    orden truncado en distintos puntos). Lista vacía si hay menos de `n` elegibles.
    """
    disponibles = prioridad.reindex(elegibles).dropna()
    if len(disponibles) < n:
        return []
    return disponibles.sort_values(kind="stable").index[:n].tolist()


# ============================================================================
# Membresía point-in-time
# ============================================================================


def construir_mascara_membresia(
    membresias_por_ticker: Mapping[str, str | Sequence[Mapping] | None],
    fechas: pd.DatetimeIndex,
    tickers: Sequence[str],
    indice: str = "SP500",
) -> pd.DataFrame:
    """
    Máscara booleana (fechas x tickers): True si el ticker era constituyente de `indice`.

    `membresias_por_ticker[ticker]` es una lista de dicts `{"index", "date_added",
    "date_removed"}` (o su JSON en texto, como la columna `index_memberships` del TFM).
    Intervalo inclusivo; `date_added` ausente = desde siempre, `date_removed` ausente =
    hasta hoy. Tickers sin entradas del índice quedan en False.
    """
    mascara = pd.DataFrame(False, index=fechas, columns=list(tickers))
    for ticker in mascara.columns:
        entradas = membresias_por_ticker.get(ticker)
        if not entradas:
            continue
        if isinstance(entradas, str):
            try:
                entradas = json.loads(entradas)
            except (TypeError, ValueError):
                continue
        activo = np.zeros(len(fechas), dtype=bool)
        for m in entradas:
            if m.get("index") != indice:
                continue
            ini = pd.Timestamp(m["date_added"]) if m.get("date_added") else pd.Timestamp.min
            fin = pd.Timestamp(m["date_removed"]) if m.get("date_removed") else pd.Timestamp.max
            activo |= (fechas >= ini) & (fechas <= fin)
        mascara[ticker] = activo
    return mascara


def _miembros_en_fecha(
    membresias: pd.DataFrame | Mapping | None, fecha: pd.Timestamp, columnas: pd.Index
) -> pd.Index:
    """Tickers (de `columnas`) que pertenecen al índice en `fecha`, usando solo información <= fecha."""
    if membresias is None:
        return columnas
    if isinstance(membresias, pd.DataFrame):
        historico = membresias.loc[:fecha]
        if historico.empty:
            return pd.Index([])
        fila = historico.iloc[-1].reindex(columnas).fillna(False).astype(bool)
        return fila.index[fila.values]

    miembros = []
    for ticker in columnas:
        entradas = membresias.get(ticker)
        if isinstance(entradas, str):
            entradas = json.loads(entradas)
        for m in entradas or []:
            if m.get("index", "SP500") != "SP500":
                continue
            ini = pd.Timestamp(m["date_added"]) if m.get("date_added") else pd.Timestamp.min
            fin = pd.Timestamp(m["date_removed"]) if m.get("date_removed") else pd.Timestamp.max
            if ini <= fecha <= fin:
                miembros.append(ticker)
                break
    return pd.Index(miembros)


# ============================================================================
# Elegibilidad point-in-time
# ============================================================================


def elegibles_en_fecha(
    precios_mensuales: pd.DataFrame,
    membresias: pd.DataFrame | Mapping | None,
    fecha: pd.Timestamp,
    T: int,
    cobertura_min: float = 1.0,
) -> pd.Index:
    """
    Tickers elegibles para formar la cartera en la fecha de rebalanceo `fecha` (= t).

    Un ticker es elegible si:
    1. Pertenece al índice en `fecha` (`membresias`: máscara bool fechas x tickers, o un
       mapping ticker -> intervalos como en `construir_mascara_membresia`); con
       `membresias=None` no se aplica filtro de índice.
    2. Tiene precio válido (> 0) en los T+1 meses t-T-1 .. t-1, es decir, los T retornos
       mensuales de la ventana de estimación (cobertura completa con `cobertura_min=1`;
       un valor menor admite huecos y obliga al llamador a gestionar los NaN).

    NO usa ningún dato con fecha >= t: no exige precio en t, ni mira si el activo sigue
    cotizando después. Por eso un activo que deslista en t+5 SÍ es elegible en t, y uno
    que entra al índice en t+1 NO lo es en t. Ver en la cabecera del módulo los sesgos
    del TFM que esto corrige.

    Parámetros
    ----------
    precios_mensuales : DataFrame (meses x tickers) de precios ajustados; el índice se
        interpreta por mes natural (admite primer o último día del mes).
    fecha : fecha de rebalanceo t (cualquier día del mes t).
    T : meses de la ventana de estimación.

    Retorna `pd.Index` de tickers (en el orden de las columnas de `precios_mensuales`).
    """
    periodo_t = pd.Period(fecha, freq="M")
    rango = pd.period_range(periodo_t - T - 1, periodo_t - 1, freq="M")

    periodos = precios_mensuales.index.to_period("M")
    ventana = precios_mensuales.loc[(periodos >= rango[0]) & (periodos <= rango[-1])]
    ventana = ventana.set_axis(ventana.index.to_period("M")).reindex(rango)
    ventana = ventana.where(ventana > 0)

    retornos = ventana.pct_change(fill_method=None).iloc[1:]
    cobertura = retornos.notna().sum() / T
    con_cobertura = cobertura.index[cobertura.values >= cobertura_min]

    miembros = _miembros_en_fecha(membresias, pd.Timestamp(fecha), precios_mensuales.columns)
    validos = set(con_cobertura) & set(miembros)
    return pd.Index([t for t in precios_mensuales.columns if t in validos])
