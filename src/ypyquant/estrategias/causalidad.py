"""
Verificación de causalidad (ausencia de look-ahead) de una estrategia.

Responsabilidad:
- `envenenar_futuro(datos, fecha, semilla, modo)`: copia de los datos en la que todo lo POSTERIOR a
  `fecha` se sustituye por ruido (`modo="ruido"`: precios aleatorios positivos y volúmenes aleatorios)
  o por NaN (`modo="nan"`). El calendario (índice) no cambia; lo anterior o igual a `fecha` queda intacto.
- `verificar_sin_lookahead(clase, parametros, datos, fechas_muestra, semilla=0) -> list[str]`: para cada
  fecha de la muestra calcula los pesos con los datos reales y con el futuro envenenado (instancia
  nueva y mismas llamadas previas, para que el estado de la estrategia evolucione igual). Si los
  pesos difieren en alguna fecha, devuelve un mensaje por fecha:
  `look-ahead en <fecha>: <ticker> pesa X con futuro real y Y con futuro envenenado`.
  Lista vacía = no se detectó dependencia del futuro.

Por qué hace falta: `Estrategia.como_funcion` recorta los datos a `<= fecha` antes de `generar_pesos`,
pero `preparar(datos)` recibe TODO el histórico, y una estrategia puede guardar ahí indicadores que
miran hacia delante (`shift(-1)`, `rolling(center=True)`, normalizaciones con el máximo global...).
Un diseño así da un backtest espléndido y un paper decepcionante.

Límites: es un contraste empírico sobre las fechas de la muestra, no una prueba formal. Una estrategia
cuya dependencia del futuro no cambia sus pesos en esas fechas pasa el test; elige varias fechas
repartidas por todo el periodo. El coste es cuadrático en el número de fechas (cada fecha reproduce
las anteriores), así que conviene usar 5-20 fechas.

Se relaciona con: ypyquant.estrategias.base (`Estrategia.como_funcion`, que aplica el recorte y llama a
`preparar`), ypyquant.estrategias.universo_dinamico (el universo point-in-time se prueba con ella),
tests/test_causalidad.py y las estrategias de usuario (úsala en sus tests).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ypyquant.estrategias.base import Estrategia, FiltroUniverso

TOLERANCIA_PESOS = 1e-9
FabricaFiltroUniverso = Callable[[dict[str, pd.DataFrame]], FiltroUniverso]  # datos -> filtro de universo
MODOS_VENENO = ("ruido", "nan")
DESVIACION_RUIDO = 0.3  # desviación log del multiplicador aleatorio de precios


def envenenar_futuro(
    datos: dict[str, pd.DataFrame], fecha: pd.Timestamp, semilla: int = 0,
    modo: Literal["ruido", "nan"] = "ruido",
) -> dict[str, pd.DataFrame]:
    """
    Copia de `datos` con todo lo posterior a `fecha` sustituido por ruido o NaN (no modifica la entrada).

    `modo="ruido"`: las columnas de precio de cada fecha futura valen el último precio conocido por un
    multiplicador lognormal aleatorio (independiente por fecha y ticker) y las de volumen son enteros
    aleatorios; así cualquier cálculo que use el futuro cambia de valor. `modo="nan"`: todo NaN.
    """
    if modo not in MODOS_VENENO:
        raise ValueError(f"modo debe ser uno de {MODOS_VENENO}")
    fecha = pd.Timestamp(fecha)
    generador = np.random.default_rng(semilla)
    envenenados: dict[str, pd.DataFrame] = {}
    for ticker in sorted(datos):
        df = datos[ticker].copy()
        futuro = df.index > fecha
        if futuro.any():
            if modo == "nan":
                df.loc[futuro, :] = np.nan
            else:
                n_futuro = int(futuro.sum())
                multiplicador = np.exp(generador.normal(0.0, DESVIACION_RUIDO, n_futuro))
                for columna in df.columns:
                    if "vol" in str(columna).lower():
                        df.loc[futuro, columna] = generador.integers(1_000_000, 50_000_000, n_futuro).astype(float)
                        continue
                    previos = df.loc[~futuro, columna].dropna()
                    base = float(previos.iloc[-1]) if len(previos) else float(df[columna].dropna().iloc[0])
                    df.loc[futuro, columna] = base * multiplicador
        envenenados[ticker] = df
    return envenenados


def _pesos_en_fechas(clase: type[Estrategia], parametros: BaseModel | dict[str, Any] | None,
                     datos: dict[str, pd.DataFrame], fechas: Sequence[pd.Timestamp],
                     fabrica_filtro_universo: FabricaFiltroUniverso | None = None) -> list[dict[str, float]]:
    """Pesos de una instancia nueva en cada fecha (en orden, como en una simulación: el estado persiste)."""
    filtro = fabrica_filtro_universo(datos) if fabrica_filtro_universo else None
    funcion = clase(parametros).como_funcion(filtro_universo=filtro)
    return [funcion(fecha, datos) for fecha in fechas]


def _diferencia(real: dict[str, float], envenenado: dict[str, float]) -> tuple[str, float, float] | None:
    """El ticker con mayor diferencia de peso entre ambos (None si coinciden dentro de la tolerancia)."""
    mayor: tuple[str, float, float] | None = None
    for ticker in sorted(set(real) | set(envenenado)):
        a, b = real.get(ticker, 0.0), envenenado.get(ticker, 0.0)
        if abs(a - b) > TOLERANCIA_PESOS and (mayor is None or abs(a - b) > abs(mayor[1] - mayor[2])):
            mayor = (ticker, a, b)
    return mayor


def verificar_sin_lookahead(
    clase: type[Estrategia],
    parametros: BaseModel | dict[str, Any] | None,
    datos: dict[str, pd.DataFrame],
    fechas_muestra: Sequence[pd.Timestamp | str],
    semilla: int = 0,
    fabrica_filtro_universo: FabricaFiltroUniverso | None = None,
) -> list[str]:
    """
    Contrasta que los pesos en `fechas_muestra` no dependen de datos posteriores a cada fecha.

    Algoritmo: (1) pesos en cada fecha con los datos completos (una instancia, fechas en orden);
    (2) para cada fecha t, datos con todo lo posterior a t envenenado (ruido y NaN), instancia nueva
    que reproduce las fechas de la muestra hasta t y los mismos pesos en t; (3) si difieren, o si la
    estrategia falla solo con el futuro envenenado, se anota un mensaje.

    Devuelve la lista de mensajes (vacía = limpia). Los datos de entrada no se modifican. Si la
    estrategia falla con los datos reales la excepción se propaga (no es un problema de causalidad).

    `fabrica_filtro_universo` (opcional): `datos -> FiltroUniverso`, p. ej. `UniversoPuntoEnElTiempo.crear_filtro`.
    Se vuelve a invocar con los datos envenenados en cada contraste, de modo que también se comprueba
    que la elegibilidad del universo no mira al futuro.
    """
    fechas = sorted(pd.Timestamp(f) for f in fechas_muestra)
    reales = _pesos_en_fechas(clase, parametros, datos, fechas, fabrica_filtro_universo)
    errores: list[str] = []
    for posicion, fecha in enumerate(fechas):
        for modo in MODOS_VENENO:
            veneno = envenenar_futuro(datos, fecha, semilla, modo)
            try:
                envenenado = _pesos_en_fechas(clase, parametros, veneno, fechas[: posicion + 1], fabrica_filtro_universo)[-1]
            except Exception as error:  # noqa: BLE001 - una excepción solo con el futuro envenenado también delata look-ahead
                errores.append(f"look-ahead en {fecha:%Y-%m-%d}: la estrategia falla con el futuro envenenado "
                               f"({modo}): {type(error).__name__}: {error}")
                break
            diferencia = _diferencia(reales[posicion], envenenado)
            if diferencia is not None:
                ticker, peso_real, peso_envenenado = diferencia
                errores.append(f"look-ahead en {fecha:%Y-%m-%d}: {ticker} pesa {peso_real:.4g} con futuro real "
                               f"y {peso_envenenado:.4g} con futuro envenenado")
                break
    return errores
