"""
Universos: el único módulo que sabe qué valores componen un universo, en qué fechas y con qué símbolo.

Responsabilidad:
- Resolver un universo a una lista de símbolos: `resolver_universo` (lista explícita, nombre de archivo de
  `universos/*.txt` con límite opcional `'sp500:30'`, o el dinámico `'sp500_pit'`) y `listar_universos`.
- Membresías point-in-time del S&P 500: fuente `universos/sp500_membresias.json` (símbolo -> intervalos
  `date_added/date_removed`), generada desde `TFM.db` con `scripts/exportar_membresias_sp500.py`.
  `extraer_membresias_desde_sqlite` la extrae, `cargar_membresias_sp500` la lee, `miembros_en_ventana` lista los
  miembros de un rango y `cobertura_membresias` / `aviso_cobertura_pit` miden cuántos tienen precios en la fuente
  (sesgo de supervivencia residual). El criterio de ELEGIBILIDAD por fecha (historia mínima, máscara) vive en
  `estrategias/universo_dinamico.py`, que consume estas membresías. Una fuente con membresías propias (Norgate:
  `index_constituent_timeseries`) las serviría con este mismo formato `{símbolo: [{index, date_added, date_removed}]}`.
- Universo en vivo: `miembros_vigentes_en(membresias, fecha)` son los miembros de una fecha (con la de hoy, la membresía
  vigente) y `avisos_membresias` avisa si las membresías tienen más de 30 días de antigüedad respecto a la fecha o si el
  número de vigentes se aleja de ~500 (bajas sin registrar). Lo usan el paper interno y el bot de Alpaca.
- Traducción de símbolos: el símbolo CANÓNICO del proyecto es el de yfinance (`BRK-B`, con guion). Aquí se
  normaliza todo lo que entra desde fuera (`a_simbolo_canonico`: `BRK.B` -> `BRK-B`); las fuentes de precios
  reciben símbolos canónicos y traducen al de su proveedor si hace falta. Ningún otro módulo conoce el formato.
- Universo de un índice para el servicio de selección de activos (`universo_de_indice`,
  `resolver_seleccion`): el proveedor de membresías históricas se inyecta, como siempre.

Se relaciona con:
- ypyquant.estrategias.experimentos (re-exporta `resolver_universo`, `listar_universos`, `es_universo_pit` y usa
  este módulo al cargar datos), .simulacion (membresías y aviso de cobertura), .universo_dinamico (elegibilidad),
  .configuraciones y .marcos (resuelven universos por nombre).
- ypyquant.portfolio.asset_selection (`resolver_seleccion` en lugar de su antiguo `_resolver_universo`).
- scripts/exportar_membresias_sp500.py, universos/*.txt y universos/sp500_membresias.json.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

RAIZ_REPO = Path(__file__).resolve().parents[3]
CARPETA_UNIVERSOS = RAIZ_REPO / "universos"
RUTA_MEMBRESIAS = CARPETA_UNIVERSOS / "sp500_membresias.json"
NOMBRE_UNIVERSO_PIT = "sp500_pit"
INDICE = "SP500"
DIAS_MAXIMOS_ANTIGUEDAD = 30  # a partir de aquí se avisa de que las membresías están desactualizadas
MIEMBROS_ESPERADOS = 500  # el índice real ronda los 503 símbolos (500 empresas, algunas con dos clases de acciones)
TOLERANCIA_MIEMBROS = 15

Membresias = Mapping[str, Sequence[Mapping[str, Any]]]
# (nombre_indice, fecha_inicio, fecha_fin, solo_supervivientes) -> lista de símbolos
ProveedorMembresias = Callable[[str, datetime, datetime, bool], list[str]]


# ----------------------------------------------------------------------------
# Símbolos
# ----------------------------------------------------------------------------


def a_simbolo_canonico(simbolo: str) -> str:
    """Símbolo canónico del proyecto (formato yfinance, mayúsculas): `'brk.b'` -> `'BRK-B'`."""
    return simbolo.strip().upper().replace(".", "-")


def es_universo_pit(universo: Any) -> bool:
    """True si `universo` es el nombre del universo dinámico (`sp500_pit`)."""
    return isinstance(universo, str) and universo.partition(":")[0] == NOMBRE_UNIVERSO_PIT


# ----------------------------------------------------------------------------
# Membresías point-in-time
# ----------------------------------------------------------------------------


def _fecha_corta(valor: str | None) -> str | None:
    """`'2010-02-16T00:00:00'` -> `'2010-02-16'` (None se conserva)."""
    return str(valor)[:10] if valor else None


def extraer_membresias_desde_sqlite(ruta_db: str | Path) -> dict[str, list[dict[str, str | None]]]:
    """
    Intervalos de pertenencia al S&P 500 de cada símbolo leídos de `tickers.index_memberships` (SQLite, solo lectura).

    Devuelve `{símbolo: [{"date_added", "date_removed"}]}` ordenado por símbolo; solo entradas del índice
    SP500 y fechas `YYYY-MM-DD`. El símbolo se normaliza al canónico (`a_simbolo_canonico`).
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
            membresias[a_simbolo_canonico(simbolo)] = intervalos
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
            "(necesita TFM.db; ver la cabecera de ypyquant/data/universos.py).")
    crudo = json.loads(ruta.read_text(encoding="utf-8"))["miembros"]
    return {t: [{"index": INDICE, **m} for m in intervalos] for t, intervalos in crudo.items()}


def _intervalo(m: Mapping[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    inicio = pd.Timestamp(m["date_added"]) if m.get("date_added") else pd.Timestamp.min
    fin = pd.Timestamp(m["date_removed"]) if m.get("date_removed") else pd.Timestamp.max
    return inicio, fin


def miembros_en_ventana(membresias: Membresias, inicio: str | pd.Timestamp, fin: str | pd.Timestamp) -> list[str]:
    """Símbolos (ordenados) que fueron miembros en algún momento de `[inicio, fin]` (intervalos inclusivos)."""
    inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
    elegidos = []
    for simbolo, intervalos in membresias.items():
        if any(i <= fin and f >= inicio for i, f in map(_intervalo, intervalos)):
            elegidos.append(simbolo)
    return sorted(elegidos)


def miembros_vigentes_en(membresias: Membresias, fecha: str | pd.Timestamp) -> list[str]:
    """
    Símbolos (ordenados) que son miembros del índice EN `fecha` (point-in-time con esa fecha; con la de hoy es la
    membresía vigente). Intervalos inclusivos, igual que la máscara del backtest (`construir_mascara_membresia`).
    """
    return miembros_en_ventana(membresias, fecha, fecha)


def ultima_fecha_membresias(membresias: Membresias) -> pd.Timestamp | None:
    """La fecha más reciente que registran las membresías (altas o bajas); None si no hay ninguna."""
    fechas = [pd.Timestamp(m[campo]) for intervalos in membresias.values() for m in intervalos
              for campo in ("date_added", "date_removed") if m.get(campo)]
    return max(fechas) if fechas else None


def avisos_membresias(membresias: Membresias, fecha: str | pd.Timestamp,
                      esperados: int | None = MIEMBROS_ESPERADOS) -> list[str]:
    """
    Avisos de calidad de las membresías para operar/simular en `fecha` (lista vacía si todo cuadra):
    - antigüedad: la última fecha registrada es más de `DIAS_MAXIMOS_ANTIGUEDAD` días anterior a `fecha`, así que
      las altas y bajas recientes del índice pueden faltar; indica cómo refrescarlas;
    - número de miembros: los vigentes en `fecha` se alejan de `esperados` (+/- `TOLERANCIA_MIEMBROS`), señal de
      bajas sin registrar o de datos incompletos. `esperados=None` desactiva este aviso.
    """
    fecha = pd.Timestamp(fecha)
    avisos = []
    ultima = ultima_fecha_membresias(membresias)
    if ultima is not None and (fecha - ultima).days > DIAS_MAXIMOS_ANTIGUEDAD:
        avisos.append(
            f"Membresías del S&P 500 con antigüedad de {(fecha - ultima).days} días (último dato {ultima:%Y-%m-%d}, "
            f"fecha {fecha:%Y-%m-%d}): pueden faltar altas y bajas recientes. Refréscalas actualizando TFM.db y "
            "ejecutando `uv run python scripts/exportar_membresias_sp500.py`.")
    if esperados is not None:
        n = len(miembros_vigentes_en(membresias, fecha))
        if abs(n - esperados) > TOLERANCIA_MIEMBROS:
            avisos.append(
                f"El S&P 500 tiene {n} miembros vigentes el {fecha:%Y-%m-%d} en las membresías (se esperan ~{esperados}, "
                "503 con las clases de acciones duplicadas): probablemente hay bajas sin registrar y el universo en vivo "
                "incluye valores que ya no están en el índice. Revisa/refresca las membresías.")
    return avisos


def cobertura_membresias(membresias: Membresias, simbolos_con_precio: set[str] | Sequence[str],
                         fechas: Sequence[str | pd.Timestamp]) -> list[dict[str, Any]]:
    """
    Por fecha: cuántos miembros del índice había (`miembros`) y cuántos de ellos tienen precios en la fuente
    (`con_precio`); `sin_precio` son los que el universo no puede elegir (sesgo de supervivencia residual).
    """
    con_precio = set(simbolos_con_precio)
    filas = []
    for fecha in fechas:
        miembros = miembros_en_ventana(membresias, fecha, fecha)
        n_con = sum(1 for t in miembros if t in con_precio)
        filas.append({"fecha": str(pd.Timestamp(fecha).date()), "miembros": len(miembros), "con_precio": n_con,
                      "sin_precio": len(miembros) - n_con})
    return filas


def aviso_cobertura_pit(fecha_inicio: str, fecha_fin: str, simbolos_con_precio: set[str]) -> str:
    """
    Aviso de sesgo residual del universo `sp500_pit`: miembros históricos del periodo sin precios en la fuente.

    Los deslistados que la fuente ya no sirve no pueden elegirse aunque fueran miembros; el sesgo de
    supervivencia se reduce pero no desaparece. Se resume la cobertura al inicio, a mitad y al final.
    """
    membresias = cargar_membresias_sp500()
    historicos = miembros_en_ventana(membresias, fecha_inicio, fecha_fin)
    sin_precio = [t for t in historicos if t not in simbolos_con_precio]
    inicio, fin = pd.Timestamp(fecha_inicio), pd.Timestamp(fecha_fin)
    fechas = [inicio, inicio + (fin - inicio) / 2, fin]
    cobertura = cobertura_membresias(membresias, simbolos_con_precio, fechas)
    detalle = ", ".join(f"{c['fecha']}: {c['con_precio']}/{c['miembros']}" for c in cobertura)
    return (f"Universo {NOMBRE_UNIVERSO_PIT}: {len(sin_precio)} de {len(historicos)} miembros históricos del S&P 500 "
            f"entre {fecha_inicio} y {fecha_fin} no tienen precios en la fuente (típicamente deslistados) "
            f"y no pueden elegirse: el sesgo de supervivencia se reduce pero no desaparece. "
            f"Miembros con precio / miembros del índice: {detalle}.")


# ----------------------------------------------------------------------------
# Resolución de universos
# ----------------------------------------------------------------------------


def resolver_universo(universo: list[str] | str, carpeta: Path | None = None,
                      inicio: str | None = None, fin: str | None = None) -> list[str]:
    """
    Lista de símbolos desde una lista explícita o un nombre de `universos/` (`'sp500'`, `'sp500:30'`).

    Las líneas vacías y las que empiezan por `#` se ignoran. Lanza `ValueError` si el universo no existe.

    `'sp500_pit'` es el universo DINÁMICO point-in-time: devuelve todos los símbolos que fueron miembros
    del S&P 500 en algún momento de `[inicio, fin]` (sin ventana, todos los históricos). Cuáles son
    elegibles en cada fecha lo decide `UniversoPuntoEnElTiempo` durante la simulación. No admite `:N`.
    """
    if isinstance(universo, list):
        return universo
    if es_universo_pit(universo):
        if ":" in universo:
            raise ValueError(f"El universo '{NOMBRE_UNIVERSO_PIT}' no admite límite de tickers (':N')")
        return miembros_en_ventana(cargar_membresias_sp500(), inicio or "1900-01-01", fin or "2999-12-31")
    nombre, _, limite = universo.partition(":")
    ruta = (carpeta or CARPETA_UNIVERSOS) / f"{nombre}.txt"
    if not ruta.is_file():
        raise ValueError(f"Universo '{nombre}' no encontrado en {ruta.parent}")
    simbolos = [
        a_simbolo_canonico(linea) for linea in ruta.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.lstrip().startswith("#")
    ]
    return simbolos[: int(limite)] if limite else simbolos


def listar_universos(carpeta: Path | None = None) -> list[dict[str, Any]]:
    """
    Universos disponibles: `[{nombre, n_tickers, dinamico}]`.

    `dinamico=True` marca `sp500_pit` (membresía del índice en cada fecha; `n_tickers` = miembros históricos);
    se añade al final si existe `universos/sp500_membresias.json`.
    """
    carpeta = carpeta or CARPETA_UNIVERSOS
    universos = [{"nombre": ruta.stem, "n_tickers": len(resolver_universo(ruta.stem, carpeta)), "dinamico": False}
                 for ruta in sorted(carpeta.glob("*.txt"))]
    if RUTA_MEMBRESIAS.is_file():
        universos.append({"nombre": NOMBRE_UNIVERSO_PIT, "n_tickers": len(resolver_universo(NOMBRE_UNIVERSO_PIT)),
                          "dinamico": True})
    return universos


def universo_de_indice(indice: str, fecha_inicio: str, fecha_fin: str, evitar_sesgo_supervivencia: bool,
                       proveedor: ProveedorMembresias | None) -> list[str]:
    """
    Símbolos que pertenecieron al `indice` en el periodo, vía el proveedor de membresías inyectado.

    Con `evitar_sesgo_supervivencia=True` se incluyen también los que salieron del índice. Lanza
    `RuntimeError` si no hay proveedor.
    """
    if proveedor is None:
        raise RuntimeError("get_historical_universe necesita un proveedor_universo")
    return proveedor(indice, datetime.strptime(fecha_inicio, "%Y-%m-%d"), datetime.strptime(fecha_fin, "%Y-%m-%d"),
                     not evitar_sesgo_supervivencia)


def resolver_seleccion(simbolos: list[str] | None, fecha_inicio: str | None, fecha_fin: str | None,
                       indice: str | None, evitar_sesgo_supervivencia: bool,
                       proveedor: ProveedorMembresias | None) -> list[str]:
    """
    Símbolos a evaluar en una selección de activos: los explícitos o, si hay `indice`, su universo histórico.

    Lanza `ValueError` si hay índice sin fechas o si no queda ningún símbolo.
    """
    if indice:
        if not fecha_inicio or not fecha_fin:
            raise ValueError("start_date y end_date son requeridos cuando se usa index_name")
        simbolos = universo_de_indice(indice, fecha_inicio, fecha_fin, evitar_sesgo_supervivencia, proveedor)
        logger.info("Usando universo histórico de %s: %d tickers", indice, len(simbolos))
    if not simbolos:
        raise ValueError("Debe proporcionar 'tickers' o 'index_name'")
    return simbolos
