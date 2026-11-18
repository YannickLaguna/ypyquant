"""
test_paridad_tfm.py — Paridad numérica del motor `estimacion_robusta.rejilla` con los resultados del TFM.

Responsabilidad: reproducir una celda pequeña del experimento original (réplica b=0, N=20, T=12,
primeros meses OOS) leyendo los MISMOS datos (TFM.db en solo lectura) y comparar pesos y retornos
con `pesos_oos.parquet` / `resultados_oos.parquet` de `data/reportes/rejilla_v2/`. Reporta el delta maximo.

Marcado `datos`: se salta por defecto (necesita los ficheros locales del TFM, solo lectura).
Ejecutar: `uv run pytest -m datos -k paridad -s`. Ruta de los datos: variable de entorno
`TFM_DIR` (por defecto C:/Users/yanni/Documents/Codigo/Proyectos/tfm).

Cómo se reproduce exactamente el TFM (y qué difiere del motor por defecto):
- retornos de estimación SIMPLES en exceso (`tipo_retorno_estimacion="simple"`), como hizo el código del TFM;
- elegibilidad legada inyectada: cobertura >= 95 % de la ventana, pertenencia al S&P 500 en t-1 y
  retorno disponible en t (`exigir_retorno_oos=True`), sin los filtros point-in-time nuevos;
- tickers de la BD limitados a los del `manifest_draws.parquet` (el TFM excluyó los contaminados con un
  detector sobre todo el histórico que vive en su repositorio);
- prioridad aleatoria posicional (los parquet son anteriores al arreglo por hash del 2026-08-31);
- RNG global legado para Michaud (`rng_global_legado=True`) y el mismo orden de bucles (N completo).

Se relaciona con: src/ypyquant/optimization/estimacion_robusta/rejilla.py y universo.py.
"""

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ypyquant.optimization.estimacion_robusta import (
    ConfiguracionRejilla,
    DatosRejilla,
    construir_mascara_membresia,
    ejecutar_rejilla,
)

pytestmark = pytest.mark.datos

RAIZ_TFM = Path(os.environ.get("TFM_DIR", "C:/Users/yanni/Documents/Codigo/Proyectos/tfm"))
DB = RAIZ_TFM / "data" / "TFM.db"
DIR_REPORTES = RAIZ_TFM / "data" / "reportes" / "rejilla_v2"

MESES_COMPARADOS = 6
# Tolerancias: OSQP (eps 1e-3 por defecto, warm_start) no es bit a bit reproducible entre procesos con distinto historial.
TOLERANCIA_PESOS = 1e-3
TOLERANCIA_RETORNOS = 1e-4


def _panel_mensual(conn: sqlite3.Connection, columna: str, tickers: list[str]) -> pd.DataFrame:
    """Último valor de `columna` de cada mes por ticker (igual que `_cargar_panel` del orquestador del TFM)."""
    marcadores = ",".join("?" * len(tickers))
    df = pd.read_sql_query(
        f"SELECT date, ticker, {columna} FROM activos WHERE {columna} IS NOT NULL "
        f"AND ticker IN ({marcadores}) ORDER BY date",
        conn, params=tickers, parse_dates=["date"],
    )
    df["mes"] = df["date"].dt.to_period("M")
    panel = df.sort_values("date").groupby(["ticker", "mes"])[columna].last().unstack("ticker")
    panel.index = panel.index.to_timestamp()
    return panel


@pytest.fixture(scope="module")
def datos_tfm():
    if not (DB.exists() and (DIR_REPORTES / "resultados_oos.parquet").exists()):
        pytest.skip(f"No hay datos del TFM en {RAIZ_TFM}")
    manifest = pd.read_parquet(DIR_REPORTES / "manifest_draws.parquet", filters=[("b", "==", 0)])
    tickers = sorted(manifest["ticker"])
    conn = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        precios = _panel_mensual(conn, "adjusted_close", tickers).where(lambda x: x > 0)
        volumen = _panel_mensual(conn, "volume", tickers)
        marcadores = ",".join("?" * len(tickers))
        filas = conn.execute(
            f"SELECT symbol, index_memberships FROM tickers WHERE symbol IN ({marcadores})", tickers
        ).fetchall()
    finally:
        conn.close()
    precios = precios.reindex(columns=tickers)
    volumen = volumen.reindex(columns=tickers)
    retornos = precios.pct_change(fill_method=None).dropna(how="all")
    membresias = construir_mascara_membresia(
        {s: json.loads(m) if m else [] for s, m in filas}, retornos.index, tickers
    )
    rf = pd.read_parquet(DIR_REPORTES / "rf_mensual.parquet")["rf"]
    return {"precios": precios, "volumen": volumen, "retornos": retornos, "membresias": membresias, "rf": rf}


def _elegibles_tfm(datos, cobertura_min: float = 0.95):
    """Criterio de elegibilidad del orquestador del TFM: cobertura >= 95 % y miembro del índice en t-1."""
    retornos, membresias = datos["retornos"], datos["membresias"]

    def funcion(fecha_oos: pd.Timestamp, t_meses: int) -> pd.Index:
        ventana = retornos.loc[fecha_oos - pd.DateOffset(months=t_meses): fecha_oos - pd.DateOffset(months=1)]
        cobertura = ventana.notna().sum() / len(ventana)
        con_cobertura = cobertura[cobertura >= cobertura_min].index
        fecha_fin = fecha_oos - pd.DateOffset(months=1)
        miembros = membresias.columns[membresias.loc[fecha_fin]] if fecha_fin in membresias.index else pd.Index([])
        return con_cobertura.intersection(miembros)

    return funcion


@pytest.fixture(scope="module")
def resultado_motor(datos_tfm):
    """Corre el motor sobre T=12, los 4 tamaños N (para consumir el RNG global igual que el TFM) y 6 meses."""
    fecha_ini = pd.Timestamp("2010-01-01")
    fecha_fin = fecha_ini + pd.DateOffset(months=MESES_COMPARADOS - 1)
    config = ConfiguracionRejilla(
        valores_t=(12,), n_replicas=1, fecha_oos_ini=str(fecha_ini.date()), fecha_oos_fin=str(fecha_fin.date()),
        tipo_retorno_estimacion="simple", exigir_retorno_oos=True, rng_global_legado=True,
        metodo_prioridad="posicional", max_replicas_con_pesos=1,
    )
    datos = DatosRejilla(
        precios_mensuales=datos_tfm["precios"], volumen_mensual=datos_tfm["volumen"], rf_mensual=datos_tfm["rf"],
    )
    return ejecutar_rejilla(datos, config, funcion_elegibles=_elegibles_tfm(datos_tfm))


def _original(nombre: str, fechas) -> pd.DataFrame:
    filtros = [("b", "==", 0), ("T", "==", 12), ("N", "==", 20)]
    df = pd.read_parquet(DIR_REPORTES / nombre, filters=filtros)
    return df[df["fecha"].isin(fechas)]


def test_prioridad_coincide_con_manifest_del_tfm(datos_tfm):
    """La prioridad posicional (esquema antiguo) reproduce exactamente `manifest_draws.parquet` (b=0)."""
    from ypyquant.optimization.estimacion_robusta import generar_prioridad_universo

    manifest = pd.read_parquet(DIR_REPORTES / "manifest_draws.parquet", filters=[("b", "==", 0)]).set_index("ticker")
    propia = generar_prioridad_universo(sorted(manifest.index), b=0, semilla_base=42, metodo="posicional")
    assert np.abs(propia.reindex(manifest.index) - manifest["prioridad"]).max() < 1e-15
    # El esquema por hash (el vigente) NO coincide: los parquet son anteriores a ese arreglo.
    por_hash = generar_prioridad_universo(sorted(manifest.index), b=0, semilla_base=42)
    assert np.abs(por_hash.reindex(manifest.index) - manifest["prioridad"]).max() > 0.1


def test_paridad_retornos_y_turnover_celda_n20_t12(resultado_motor):
    """r_bruto, turnover, r_neto y diagnósticos de N=20, T=12, b=0 coinciden con resultados_oos.parquet."""
    mio = resultado_motor.resultados_oos.query("b == 0 and N == 20 and T == 12")
    orig = _original("resultados_oos.parquet", mio["fecha"].unique())
    claves = ["estrategia", "fecha"]
    unido = mio.merge(orig, on=claves, suffixes=("_py", "_tfm"))
    assert len(unido) == len(orig) == len(mio) > 0

    informe = {}
    for col in ("r_bruto", "turnover", "tc", "r_neto", "hhi", "w_max", "dist_w_mkt", "dist_w_prev"):
        delta = (unido[f"{col}_py"] - unido[f"{col}_tfm"]).abs()
        informe[col] = float(np.nanmax(delta.to_numpy()))
    print(f"\n[paridad] delta maximo por columna (N=20, T=12, b=0, {MESES_COMPARADOS} meses):")
    for col, delta in informe.items():
        print(f"   {col:12s} {delta:.3e}")
    deterministas = unido[unido["estrategia"] != "mpt_michaud"]
    for col in ("r_bruto", "turnover", "r_neto"):
        delta = (deterministas[f"{col}_py"] - deterministas[f"{col}_tfm"]).abs().max()
        assert delta < TOLERANCIA_RETORNOS, f"{col}: delta={delta:.3e}"
    assert (unido["n_activos_pos_py"] == unido["n_activos_pos_tfm"]).all()
    assert (unido["degenerate_py"] == unido["degenerate_tfm"]).all()
    assert (unido["solver_status_py"] == unido["solver_status_tfm"]).all()


def test_paridad_michaud_con_rng_global_legado(resultado_motor):
    """Con el RNG global legado, Michaud reproduce los retornos del TFM (mismo flujo de números aleatorios)."""
    mio = resultado_motor.resultados_oos.query("b == 0 and N == 20 and T == 12 and estrategia == 'mpt_michaud'")
    orig = _original("resultados_oos.parquet", mio["fecha"].unique())
    orig = orig[orig["estrategia"] == "mpt_michaud"]
    unido = mio.merge(orig, on="fecha", suffixes=("_py", "_tfm"))
    delta = (unido["r_bruto_py"] - unido["r_bruto_tfm"]).abs().max()
    print(f"\n[paridad] Michaud delta maximo r_bruto = {delta:.3e}")
    assert delta < TOLERANCIA_RETORNOS


def test_paridad_pesos_celda_n20_t12(resultado_motor):
    """Los pesos por ticker de las estrategias deterministas coinciden con pesos_oos.parquet."""
    mio = resultado_motor.pesos_oos.query("b == 0 and N == 20 and T == 12")
    orig = _original("pesos_oos.parquet", mio["fecha"].unique())
    unido = mio.merge(orig, on=["estrategia", "fecha", "ticker"], how="outer", suffixes=("_py", "_tfm")).fillna(0.0)
    unido["delta"] = (unido["w_py"] - unido["w_tfm"]).abs()
    por_estrategia = unido.groupby("estrategia")["delta"].max()
    print("\n[paridad] delta maximo de pesos por estrategia:\n" + por_estrategia.to_string())
    deterministas = por_estrategia.drop("mpt_michaud", errors="ignore")
    assert (deterministas < TOLERANCIA_PESOS).all(), deterministas.to_dict()
