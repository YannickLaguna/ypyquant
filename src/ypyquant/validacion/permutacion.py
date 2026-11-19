"""
Permutación de precios para tests de Monte Carlo de sistemas de trading (método de Masters).

Responsabilidad:
- `permutar_precios(datos, semilla, inicio, longitud_bloque)`: genera una versión permutada de
  los precios (dict ticker -> OHLCV) que conserva la distribución marginal de los retornos y la
  correlación entre activos, pero destruye la estructura temporal (la «señal»). Se corre la
  estrategia sobre N versiones permutadas y se compara su métrica con la real mediante
  `estadisticos.p_valor_permutacion`.

Método (Masters, «Permutation and Randomization Tests for Trading System Development»; implementación
de referencia pública: neurotrader888/mcpt `bar_permute.py`):
- En log-precios, por barra: intrabarra relativos a la apertura (alto−apertura, bajo−apertura,
  cierre−apertura) y gap (apertura − cierre previo). Se baraja con UNA permutación los
  intrabarra y con OTRA los gaps, la MISMA para todos los tickers (conserva la correlación
  entre activos). Se reconstruye encadenando gaps e intrabarra desde el último dato real.
- Volume sigue la permutación de los intrabarra (la del retorno si solo hay cierre).

Decisiones documentadas:
- Columnas ajustadas: se permuta en el espacio AJUSTADO (open/high/low se escalan por
  f = Adj Close/Close de cada fila; el cierre ajustado es el precio base). Tras permutar, las
  columnas crudas se obtienen dividiendo por el mismo f de cada fila (el factor queda anclado a
  la fecha, no se baraja). Así el multiconjunto de log-retornos del cierre AJUSTADO (el que usa el
  proyecto, véase `extraer_serie_precios`) se conserva exactamente y todas las columnas siguen
  coherentes.
- Fechas distintas entre tickers: se alinea al índice común (intersección). Solo se permuta en
  el tramo común donde todos tienen OHLC completo, desde max(inicio, primera fila completa);
  filas anteriores, y filas que no estén en el índice común, quedan reales. Un NaN dentro del
  tramo a permutar lanza ValueError (no se inventan datos).
- `inicio`: la primera barra con fecha >= inicio queda real (ancla) y se permuta desde la
  siguiente. Se conserva el primer precio y el último cierre (la suma de log-retornos no cambia
  con ninguna permutación, tampoco con bloques).
- `longitud_bloque > 1`: se barajan bloques contiguos de barras. El método barra a barra (1)
  destruye el clustering de volatilidad y la autocorrelación, lo que hace el test OPTIMISTA para
  estrategias que aprovechan regímenes de volatilidad (la hipótesis nula es demasiado «fácil de
  batir»); con bloques se conserva parte de esa estructura intra-bloque y el test es más exigente.

Se relaciona con:
- ypyquant.data.matriz_precios (formato OHLCV del proyecto, nombres de columnas de precio).
- ypyquant.validacion.estadisticos (`p_valor_permutacion`).
- ypyquant.estrategias.experimentos (a futuro: ejecutar el experimento sobre precios permutados).
"""

from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd

# Nombres admitidos (en minúscula) por rol; el primero que exista en el DataFrame se usa.
_ALIAS = {
    "apertura": ("open",),
    "alto": ("high",),
    "bajo": ("low",),
    "cierre": ("close",),
    "ajustado": ("adj close", "adjusted_close"),
    "volumen": ("volume",),
}


def _mapa_columnas(df: pd.DataFrame) -> dict[str, str]:
    """Asigna a cada rol (apertura, alto, ...) el nombre real de columna, si existe."""
    por_minuscula = {str(c).lower(): c for c in df.columns}
    mapa: dict[str, str] = {}
    for rol, alias in _ALIAS.items():
        for nombre in alias:
            if nombre in por_minuscula:
                mapa[rol] = por_minuscula[nombre]
                break
    return mapa


def _orden_por_bloques(n: int, longitud_bloque: int, rng: np.random.Generator) -> np.ndarray:
    """Permutación de 0..n-1 que baraja bloques contiguos (con bloque 1, una permutación normal)."""
    if longitud_bloque <= 1:
        return rng.permutation(n)
    n_bloques = -(-n // longitud_bloque)
    orden = rng.permutation(n_bloques)
    return np.concatenate([np.arange(b * longitud_bloque, min((b + 1) * longitud_bloque, n)) for b in orden])


def _columnas_precio(mapa: dict[str, str]) -> list[str]:
    return [mapa[r] for r in ("apertura", "alto", "bajo", "cierre", "ajustado") if r in mapa]


def permutar_precios(
    datos: dict[str, pd.DataFrame],
    semilla: int,
    inicio: str | pd.Timestamp | None = None,
    longitud_bloque: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Devuelve una copia permutada de `datos` (ticker -> OHLCV) según el método de Masters.

    Args:
        datos: precios por ticker. Columnas reconocidas (sin distinguir mayúsculas): Open, High,
            Low, Close, Adj Close / adjusted_close, Volume. Con Open/High/Low y algún cierre se
            permuta con OHLC; si solo hay cierre (o ajustado) se barajan los log-retornos de cierre.
        semilla: semilla del generador (mismo resultado para la misma semilla).
        inicio: fecha desde la que se permuta (lo anterior queda real); None = desde la primera
            fila completa del índice común.
        longitud_bloque: 1 = barra a barra; >1 = bloques contiguos (conserva parte del clustering
            de volatilidad; ver cabecera del módulo).

    Returns:
        dict con los mismos tickers, índice, columnas y tipos que la entrada. No modifica la entrada.

    Raises:
        ValueError: datos vacíos, ticker sin columna de cierre, NaN o precios no positivos en el
            tramo a permutar, o índices sin fechas comunes.
    """
    if not datos:
        raise ValueError("datos vacío")
    if longitud_bloque < 1:
        raise ValueError("longitud_bloque debe ser >= 1")
    mapas = {t: _mapa_columnas(df) for t, df in datos.items()}
    for t, mapa in mapas.items():
        if "cierre" not in mapa and "ajustado" not in mapa:
            raise ValueError(f"{t}: sin columna de cierre")

    comun = reduce(lambda a, b: a.intersection(b), (df.index for df in datos.values())).sort_values()
    if len(comun) < 3:
        raise ValueError("Los tickers no tienen suficientes fechas comunes")

    # Máscara de filas completas (todas las columnas de precio válidas) en el índice común
    completa = np.ones(len(comun), dtype=bool)
    for t, df in datos.items():
        bloque = df.loc[comun, _columnas_precio(mapas[t])].astype(float)
        completa &= (bloque.notna() & (bloque > 0)).all(axis=1).to_numpy()
    if not completa.any():
        raise ValueError("No hay ninguna fila común con precios completos")
    posicion_inicio = int(np.argmax(completa))
    if inicio is not None:
        marca = pd.Timestamp(inicio)
        if comun.tz is not None and marca.tz is None:
            marca = marca.tz_localize(comun.tz)
        posicion_inicio = max(posicion_inicio, int(comun.searchsorted(marca, side="left")))

    salida = {t: df.copy() for t, df in datos.items()}
    m = len(comun) - posicion_inicio - 1  # barras a permutar (la ancla queda real)
    if m < 2:
        return salida
    if not completa[posicion_inicio:].all():
        raise ValueError("Hay NaN o precios no positivos dentro del tramo a permutar")

    rng = np.random.default_rng(semilla)
    orden_intrabarra = _orden_por_bloques(m, longitud_bloque, rng)
    orden_gap = _orden_por_bloques(m, longitud_bloque, rng)
    fechas_tramo = comun[posicion_inicio:]

    for t, df in datos.items():
        mapa = mapas[t]
        tramo = df.loc[fechas_tramo]
        cierre_crudo = tramo[mapa["cierre"]].to_numpy(float) if "cierre" in mapa else None
        cierre_ajustado = tramo[mapa["ajustado"]].to_numpy(float) if "ajustado" in mapa else None
        # Factor de ajuste por fila (1 si no hay doble columna de cierre)
        if cierre_crudo is not None and cierre_ajustado is not None:
            factor = cierre_ajustado / cierre_crudo
        else:
            factor = np.ones(len(tramo))
        log_cierre = np.log(cierre_ajustado if cierre_ajustado is not None else cierre_crudo)
        tiene_ohlc = all(r in mapa for r in ("apertura", "alto", "bajo"))
        nuevos: dict[str, np.ndarray] = {}

        nuevo_cierre = np.empty(len(tramo))
        nuevo_cierre[0] = log_cierre[0]
        if tiene_ohlc:
            log_ap = np.log(tramo[mapa["apertura"]].to_numpy(float) * factor)
            log_alto = np.log(tramo[mapa["alto"]].to_numpy(float) * factor)
            log_bajo = np.log(tramo[mapa["bajo"]].to_numpy(float) * factor)
            gap = (log_ap[1:] - log_cierre[:-1])[orden_gap]
            rel_alto = (log_alto[1:] - log_ap[1:])[orden_intrabarra]
            rel_bajo = (log_bajo[1:] - log_ap[1:])[orden_intrabarra]
            rel_cierre = (log_cierre[1:] - log_ap[1:])[orden_intrabarra]
            nuevo_cierre[1:] = log_cierre[0] + np.cumsum(gap + rel_cierre)
            nueva_apertura = nuevo_cierre[:-1] + gap
            nuevos[mapa["apertura"]] = np.exp(nueva_apertura) / factor[1:]
            nuevos[mapa["alto"]] = np.exp(nueva_apertura + rel_alto) / factor[1:]
            nuevos[mapa["bajo"]] = np.exp(nueva_apertura + rel_bajo) / factor[1:]
            orden_volumen = orden_intrabarra
        else:
            retornos = np.diff(log_cierre)[orden_gap]
            nuevo_cierre[1:] = log_cierre[0] + np.cumsum(retornos)
            orden_volumen = orden_gap

        cierre_nuevo_espacio = np.exp(nuevo_cierre[1:])
        if "ajustado" in mapa:
            nuevos[mapa["ajustado"]] = cierre_nuevo_espacio
        if "cierre" in mapa:
            nuevos[mapa["cierre"]] = cierre_nuevo_espacio / factor[1:]
        if "volumen" in mapa:
            nuevos[mapa["volumen"]] = tramo[mapa["volumen"]].to_numpy()[1:][orden_volumen]

        # Escribir en las filas reales del DataFrame original (el tramo puede ser subconjunto del índice)
        filas = df.index.get_indexer(fechas_tramo[1:])
        out = salida[t]
        for columna, valores in nuevos.items():
            tipo = df[columna].dtype
            serie = out[columna].to_numpy(dtype=float if tipo.kind in "iu" else tipo, copy=True)
            serie[filas] = valores
            out[columna] = serie.astype(tipo, copy=False)
    return salida
