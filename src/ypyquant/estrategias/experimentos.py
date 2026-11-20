"""
Experimentos reproducibles: configuración, ejecución con el motor de backtest y barridos.

Responsabilidad:
- `ConfiguracionExperimento` (Pydantic): estrategia + parámetros + universo + fechas + capital +
  costes + frecuencia + benchmark + semilla. `huella()` es un hash estable (misma configuración,
  mismo identificador); `normalizada()` rellena los parámetros por defecto de la estrategia.
- `cargar_datos_experimento(config, cargador)` -> `DatosExperimento` (precios con historia previa,
  benchmark, avisos) y `ejecutar_con_datos(config, datos)` -> `ResultadoExperimento`: carga y
  simulación separadas, para correr muchos backtests con una sola carga (barridos, permutación).
  `ejecutar_experimento` es su composición. La simulación usa `BacktestEngine` y calcula métricas
  con `PerformanceAnalyzer` (Sharpe, Sortino, Calmar, máx. drawdown, volatilidad, turnover, alfa/beta
  vs benchmark, Ulcer, Martin, `costes_sobre_ganancia_bruta`...). No persiste nada (lo hace
  `AlmacenExperimentos`).
- Procedencia (`ResultadoExperimento.procedencia`, ver `procedencia.py`): commit, versiones, hash de
  los precios y fecha; no entra en la huella.
- Universo `sp500_pit` (point-in-time): ver `universo_dinamico.py`. `ejecutar_con_datos` entrega a la estrategia
  solo los elegibles de cada fecha, el motor vende lo que sale del universo y avisa de los deslistados
  liquidados a precio caducado y de los miembros históricos sin precios (sesgo residual).
- `expandir_rejilla(config, {param: [valores]})`: N configuraciones de un barrido de parámetros (estricta);
  `expandir_rejilla_con_omitidas` omite y registra las combinaciones inválidas (p. ej. `rapida >= lenta`).
- Punto de extensión de validación: `registrar_validador(nombre)` registra funciones
  `(ResultadoExperimento) -> dict`; las que pida `config.validaciones` se ejecutan y su salida va a
  `metricas['validacion'][nombre]`. Vienen registrados `rachas` (prueba de rachas sobre los signos
  de los retornos diarios) y `psr` (Sharpe probabilístico frente a 0). La validación de un
  conjunto de configuraciones (DSR, PBO, permutación) vive en `validacion_barridos.py`.

Se relaciona con:
- ypyquant.estrategias.registro (obtiene la clase de estrategia) y .base (adaptador al motor).
- ypyquant.estrategias.precios (cargadores), .almacen (persistencia Parquet + DuckDB),
  .procedencia (huella de reproducibilidad), .barridos y .validacion_barridos (usan
  `cargar_datos_experimento` y `ejecutar_con_datos`).
- ypyquant.validacion (Ulcer, Martin, rachas y PSR para métricas y validadores).
- ypyquant.portfolio.backtest_engine (BacktestEngine, BacktestConfig) y .performance_metrics.
- universos/*.txt (listas de tickers por nombre) y .universo_dinamico (`sp500_pit`, membresías históricas).
- app/api (lanza y consulta experimentos).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ypyquant.data.matriz_precios import CargadorPrecios, extraer_serie_precios
from ypyquant.estrategias.base import recortar_hasta
from ypyquant.estrategias.procedencia import obtener_procedencia
from ypyquant.estrategias.registro import obtener_estrategia
from ypyquant.estrategias.universo_dinamico import (
    NOMBRE_UNIVERSO_PIT,
    RUTA_MEMBRESIAS,
    UniversoPuntoEnElTiempo,
    cargar_membresias_sp500,
    cobertura_membresias,
    es_universo_pit,
    miembros_en_ventana,
)
from ypyquant.portfolio.backtest_engine import BacktestConfig, BacktestEngine
from ypyquant.portfolio.performance_metrics import PerformanceAnalyzer
from ypyquant.validacion import indice_ulcer, prueba_rachas, ratio_martin, sharpe_probabilistico

logger = logging.getLogger(__name__)

RAIZ_REPO = Path(__file__).resolve().parents[3]
CARPETA_UNIVERSOS = RAIZ_REPO / "universos"
DIAS_ANIO = 252
DIAS_PRECIO_CADUCADO = 5  # días sin cotizar antes de la venta a partir de los cuales se avisa de la liquidación
MAX_COMBINACIONES_REJILLA = 500

Frecuencia = Literal["daily", "weekly", "monthly", "quarterly"]


class ConfiguracionExperimento(BaseModel):
    """
    Todo lo necesario para reproducir un backtest.

    Campos:
        estrategia: nombre registrado. parametros: valores de `Estrategia.Parametros` (los que falten usan el defecto).
        universo: lista de tickers o nombre de un archivo de `universos/` (`'sp500'`, o `'sp500:30'` = los 30 primeros);
            `'sp500_pit'` = miembros del S&P 500 en cada fecha de rebalanceo (point-in-time, sin sesgo de supervivencia).
        fecha_inicio / fecha_fin: 'YYYY-MM-DD'. capital_inicial: en USD.
        comision / slippage: fracciones por operación (0.001 = 10 pb).
        frecuencia_rebalanceo: None = la sugerida por la estrategia.
        benchmark: ticker de referencia (alfa, beta y curva comparada). semilla: para todo el azar.
        dias_historia_previa: días naturales anteriores a `fecha_inicio` que se cargan para estimar.
        validaciones: nombres de validadores registrados a ejecutar al terminar (extensión futura).
        etiqueta: nota libre; NO entra en la huella.
    """

    model_config = ConfigDict(extra="forbid")

    estrategia: str
    parametros: dict[str, Any] = Field(default_factory=dict)
    universo: list[str] | str
    fecha_inicio: str
    fecha_fin: str
    capital_inicial: float = Field(100_000.0, gt=0)
    comision: float = Field(0.001, ge=0, le=0.1)
    slippage: float = Field(0.001, ge=0, le=0.1)
    frecuencia_rebalanceo: Frecuencia | None = None
    benchmark: str = "SPY"
    semilla: int = 42
    dias_historia_previa: int = Field(1100, ge=0, le=7300)
    validaciones: list[str] = Field(default_factory=list)
    etiqueta: str = ""

    @field_validator("fecha_inicio", "fecha_fin")
    @classmethod
    def _fecha_valida(cls, valor: str) -> str:
        return pd.Timestamp(valor).strftime("%Y-%m-%d")

    @model_validator(mode="after")
    def _fechas_ordenadas(self) -> ConfiguracionExperimento:
        if self.fecha_inicio >= self.fecha_fin:
            raise ValueError("fecha_inicio debe ser anterior a fecha_fin")
        if isinstance(self.universo, list):
            if not self.universo:
                raise ValueError("el universo no puede estar vacío")
            self.universo = list(dict.fromkeys(t.strip().upper() for t in self.universo if t.strip()))
        return self

    def normalizada(self) -> ConfiguracionExperimento:
        """Copia con los parámetros validados y completados con los defectos, y la frecuencia resuelta."""
        clase = obtener_estrategia(self.estrategia)
        parametros = clase.Parametros(**self.parametros).model_dump()
        return self.model_copy(update={
            "parametros": parametros,
            "frecuencia_rebalanceo": self.frecuencia_rebalanceo or clase.frecuencia_rebalanceo,
        })

    def _hash_sin(self, excluir: set[str]) -> str:
        """SHA-256 del JSON canónico de la configuración sin los campos `excluir` (universo ordenado)."""
        datos = self.model_dump(exclude=excluir)
        _omitir_parametros_nuevos_en_defecto(datos)
        if isinstance(datos["universo"], list):
            datos["universo"] = sorted(datos["universo"])
        canonico = json.dumps(datos, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
        return hashlib.sha256(canonico.encode("utf-8")).hexdigest()

    def huella(self) -> str:
        """Hash SHA-256 estable de la configuración (sin la etiqueta; el orden del universo no importa)."""
        return self._hash_sin({"etiqueta"})

    def huella_receta(self) -> str:
        """
        Hash de la «receta»: estrategia, parámetros, universo, frecuencia, costes, benchmark, semilla e historia
        previa, SIN fechas, capital, validaciones ni etiqueta. Dos backtests de la misma receta en periodos o con
        capitales distintos comparten esta huella (la usan las configuraciones guardadas).
        """
        return self._hash_sin({"etiqueta", "fecha_inicio", "fecha_fin", "capital_inicial", "validaciones"})

    def identificador(self) -> str:
        """Identificador legible y estable: `<estrategia>-<10 primeros caracteres de la huella>`."""
        return f"{self.estrategia}-{self.huella()[:10]}"


def _omitir_parametros_nuevos_en_defecto(datos: dict[str, Any]) -> None:
    """
    Quita de `datos['parametros']` los parámetros que la estrategia declara como posteriores a los experimentos
    guardados (`parametros_fuera_de_huella_si_defecto`) mientras valgan su defecto. Así `normalizada()` puede
    rellenar los parámetros nuevos sin cambiar la huella de ningún experimento anterior.
    """
    try:
        clase = obtener_estrategia(datos["estrategia"])
    except KeyError:  # estrategia no registrada (p. ej. importada del TFM): la huella usa los parámetros tal cual
        return
    for nombre in clase.parametros_fuera_de_huella_si_defecto:
        campo = clase.Parametros.model_fields.get(nombre)
        if campo is not None and nombre in datos["parametros"] and datos["parametros"][nombre] == campo.default:
            del datos["parametros"][nombre]


@dataclass
class ResultadoExperimento:
    """
    Salida de `ejecutar_experimento` / `ejecutar_con_datos`: todo lo que persiste el almacén.

    equity: DataFrame con `fecha, equity, benchmark, drawdown, retorno` (benchmark reescalado al capital inicial).
    pesos: formato largo `fecha, ticker, peso` (pesos objetivo en cada rebalanceo).
    operaciones: `fecha, ticker, lado, cantidad, precio, comision, pnl`.
    procedencia: commit, versiones, hash de precios y fecha de ejecución (vacío con `completo=False`).
    """

    config: ConfiguracionExperimento
    metricas: dict[str, Any]
    equity: pd.DataFrame
    pesos: pd.DataFrame
    operaciones: pd.DataFrame
    avisos: list[str] = field(default_factory=list)
    procedencia: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Validación por experimento: punto de extensión (rachas y PSR vienen de serie)
# ----------------------------------------------------------------------------

Validador = Callable[[ResultadoExperimento], dict[str, Any]]
VALIDADORES: dict[str, Validador] = {}


def registrar_validador(nombre: str) -> Callable[[Validador], Validador]:
    """Decorador: registra un validador `(ResultadoExperimento) -> dict` para `config.validaciones`."""

    def decorador(funcion: Validador) -> Validador:
        VALIDADORES[nombre] = funcion
        return funcion

    return decorador


def _retornos_diarios(resultado: ResultadoExperimento) -> np.ndarray:
    """Retornos diarios del experimento sin NaN (insumo de los validadores)."""
    return resultado.equity["retorno"].dropna().to_numpy(dtype=float)


@registrar_validador("rachas")
def _validador_rachas(resultado: ResultadoExperimento) -> dict[str, Any]:
    """Prueba de rachas de Wald-Wolfowitz sobre los signos de los retornos diarios (Z < 0: signos agrupados)."""
    return prueba_rachas(np.sign(_retornos_diarios(resultado)))


@registrar_validador("psr")
def _validador_psr(resultado: ResultadoExperimento) -> dict[str, Any]:
    """Sharpe probabilístico: P(Sharpe verdadero > 0) corrigiendo por asimetría, curtosis y longitud."""
    retornos = _retornos_diarios(resultado)
    return {"psr": sharpe_probabilistico(retornos, 0.0), "n_observaciones": len(retornos)}


# ----------------------------------------------------------------------------
# Universos y precios
# ----------------------------------------------------------------------------


def resolver_universo(universo: list[str] | str, carpeta: Path | None = None,
                      inicio: str | None = None, fin: str | None = None) -> list[str]:
    """
    Lista de tickers desde una lista explícita o un nombre de `universos/` (`'sp500'`, `'sp500:30'`).

    Las líneas vacías y las que empiezan por `#` se ignoran. Lanza `ValueError` si el universo no existe.

    `'sp500_pit'` es el universo DINÁMICO point-in-time: devuelve todos los tickers que fueron miembros
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
    tickers = [
        linea.strip().upper() for linea in ruta.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.lstrip().startswith("#")
    ]
    return tickers[: int(limite)] if limite else tickers


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


def _cargar_precios(tickers: list[str], inicio: str, fin: str, cargador: CargadorPrecios) -> dict[str, pd.DataFrame]:
    """Carga los OHLCV en paralelo; los tickers sin datos se omiten (se avisa en el log)."""

    def cargar(ticker: str):
        try:
            return ticker, cargador(ticker, inicio, fin)
        except Exception as error:  # noqa: BLE001 - un ticker roto no aborta el experimento
            logger.warning("No se pudo cargar %s: %s", ticker, error)
            return ticker, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        resultados = list(pool.map(cargar, tickers))
    datos = {}
    for ticker, df in resultados:
        if df is None or df.empty:
            logger.warning("Sin datos para %s; se omite", ticker)
        else:
            datos[ticker] = df
    return datos


def _aviso_cobertura_pit(config: ConfiguracionExperimento, tickers_con_precio: set[str]) -> str:
    """
    Aviso de sesgo residual del universo `sp500_pit`: miembros históricos del periodo sin precios en la fuente.

    Los deslistados que yfinance ya no sirve no pueden elegirse aunque fueran miembros; el sesgo de
    supervivencia se reduce pero no desaparece. Se resume la cobertura al inicio, a mitad y al final.
    """
    membresias = cargar_membresias_sp500()
    historicos = miembros_en_ventana(membresias, config.fecha_inicio, config.fecha_fin)
    sin_precio = [t for t in historicos if t not in tickers_con_precio]
    inicio, fin = pd.Timestamp(config.fecha_inicio), pd.Timestamp(config.fecha_fin)
    fechas = [inicio, inicio + (fin - inicio) / 2, fin]
    cobertura = cobertura_membresias(membresias, tickers_con_precio, fechas)
    detalle = ", ".join(f"{c['fecha']}: {c['con_precio']}/{c['miembros']}" for c in cobertura)
    return (f"Universo {NOMBRE_UNIVERSO_PIT}: {len(sin_precio)} de {len(historicos)} miembros históricos del S&P 500 "
            f"entre {config.fecha_inicio} y {config.fecha_fin} no tienen precios en la fuente (típicamente deslistados) "
            f"y no pueden elegirse: el sesgo de supervivencia se reduce pero no desaparece. "
            f"Miembros con precio / miembros del índice: {detalle}.")


def _avisos_liquidaciones_caducadas(trades: list[Any], precios: dict[str, pd.DataFrame]) -> list[str]:
    """
    Avisos de posiciones vendidas con un precio caducado (el ticker dejó de cotizar antes de la venta).

    El motor valora una posición deslistada a su último precio conocido y la liquida a ese precio en el
    siguiente rebalanceo; si el deslistamiento fue una quiebra el resultado real habría sido peor.
    """
    avisos = []
    vistos: set[str] = set()
    for trade in trades:
        if trade.side != "sell" or trade.ticker in vistos or trade.ticker not in precios:
            continue
        ultimo = precios[trade.ticker].index[-1]
        if pd.Timestamp(trade.entry_date) - ultimo > pd.Timedelta(days=DIAS_PRECIO_CADUCADO):
            vistos.add(trade.ticker)
            avisos.append(f"{trade.ticker} se liquidó a su último precio conocido ({ultimo:%Y-%m-%d}) el "
                          f"{pd.Timestamp(trade.entry_date):%Y-%m-%d}: dejó de cotizar (deslistado) con la posición abierta.")
    return avisos


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------


def _limpiar(valor: Any) -> Any:
    """Convierte tipos NumPy a nativos y NaN/inf a None (JSON válido)."""
    if isinstance(valor, dict):
        return {k: _limpiar(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_limpiar(v) for v in valor]
    if isinstance(valor, (np.floating, float)):
        return None if not math.isfinite(float(valor)) else float(valor)
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    return valor


def _turnover(rotaciones: list[tuple[Any, float]], anios: float) -> tuple[float, float]:
    """
    (turnover medio por rebalanceo, turnover anual) con la rotación real del motor: lo operado
    desde los pesos a la deriva hasta el objetivo (`BacktestResult.rotaciones`). La 1.ª
    (compra inicial) se excluye. Antes se comparaban pesos objetivo consecutivos, y una
    estrategia de pesos fijos (p. ej. pesos iguales) salía con turnover 0 aunque operase.
    """
    cambios = [rotacion for _, rotacion in rotaciones[1:]]
    if not cambios:
        return 0.0, 0.0
    return float(np.mean(cambios)), float(np.sum(cambios) / anios) if anios > 0 else 0.0


def _calcular_metricas(config, resultado_bt, equity: pd.DataFrame, historial, benchmark_ret) -> dict[str, Any]:
    """
    Métricas del experimento: retorno/riesgo (PerformanceAnalyzer), Ulcer y Martin, costes, turnover y alfa/beta.

    `costes_sobre_ganancia_bruta` = comisiones_totales / (capital_final - capital_inicial + comisiones_totales);
    None si la ganancia bruta es <= 0. Un valor > 0,5 indica que los costes se comen la mitad del edge.
    """
    retornos = equity.set_index("fecha")["retorno"].dropna()
    base = PerformanceAnalyzer.calculate_metrics(retornos, risk_free_rate=0.0, periods_per_year=DIAS_ANIO)
    anios = max((pd.Timestamp(config.fecha_fin) - pd.Timestamp(config.fecha_inicio)).days / 365.25, 1e-9)
    turnover_medio, turnover_anual = _turnover(resultado_bt.rotaciones, anios)
    metricas: dict[str, Any] = {
        "retorno_total": base.total_return,
        "cagr": base.annualized_return,
        "volatilidad": base.volatility,
        "sharpe": base.sharpe_ratio,
        "sortino": base.sortino_ratio,
        "calmar": base.calmar_ratio,
        "max_drawdown": float(equity["drawdown"].min()),
        "var_95": base.var_95,
        "cvar_95": base.cvar_95,
        "capital_final": float(equity["equity"].iloc[-1]),
        "turnover_medio": turnover_medio,
        "turnover_anual": turnover_anual,
        "n_rebalanceos": len(historial),
        "n_operaciones": int(resultado_bt.total_trades),
        "comisiones_totales": float(resultado_bt.total_fees),
        "tasa_acierto": float(resultado_bt.win_rate),
        "dias": len(equity),
        "ulcer": indice_ulcer(equity["equity"].to_numpy(dtype=float)),
        "martin": ratio_martin(retornos.to_numpy(dtype=float), periodos_anio=DIAS_ANIO),
    }
    # Costes sobre ganancia bruta = comisiones / (ganancia neta + comisiones). Por encima de 0,5 los costes
    # se comen la mitad del edge; sin ganancia bruta positiva no está definido (None).
    ganancia_bruta = metricas["capital_final"] - config.capital_inicial + metricas["comisiones_totales"]
    metricas["costes_sobre_ganancia_bruta"] = (
        metricas["comisiones_totales"] / ganancia_bruta if ganancia_bruta > 0 else float("nan"))
    if benchmark_ret is not None and len(benchmark_ret) > 1:
        comparacion = PerformanceAnalyzer.compare_to_benchmark(retornos, benchmark_ret, risk_free_rate=0.0)
        if "error" not in comparacion:
            metricas.update({
                "alfa": comparacion["alpha"], "beta": comparacion["beta"],
                "tracking_error": comparacion["tracking_error"],
                "information_ratio": comparacion["information_ratio"],
                "benchmark_retorno_total": comparacion["benchmark"].total_return,
                "benchmark_cagr": comparacion["benchmark"].annualized_return,
                "benchmark_sharpe": comparacion["benchmark"].sharpe_ratio,
            })
    metricas["validacion"] = {}  # los validadores registrados (CPCV/DSR/PBO...) escriben aquí
    return _limpiar(metricas)


# ----------------------------------------------------------------------------
# Ejecución
# ----------------------------------------------------------------------------


@dataclass
class DatosExperimento:
    """
    Precios ya cargados para uno o varios experimentos (separa la E/S de la simulación).

    precios: `{ticker: OHLCV}` del universo CON la historia previa (puede ser la unión de varios
        universos: `ejecutar_con_datos` selecciona los tickers de cada configuración).
    benchmark: OHLCV del `ticker_benchmark` (None si no hubo datos). Los tests de permutación
        permutan `precios` pero dejan el benchmark intacto.
    avisos: avisos de la carga (tickers sin datos...).
    """

    precios: dict[str, pd.DataFrame]
    benchmark: pd.DataFrame | None
    ticker_benchmark: str
    avisos: list[str] = field(default_factory=list)


def cargar_datos_experimento(config: ConfiguracionExperimento, cargador_precios: CargadorPrecios) -> DatosExperimento:
    """
    Carga los precios del universo (con `dias_historia_previa` antes de `fecha_inicio`) y el benchmark.

    Lanza `RuntimeError` si no hay precios de ningún ticker. No simula nada: el resultado se pasa
    a `ejecutar_con_datos` (varias veces si se quiere, p. ej. un barrido o una permutación).
    """
    config = config.normalizada()
    tickers = resolver_universo(config.universo, inicio=config.fecha_inicio, fin=config.fecha_fin)
    inicio_carga = (pd.Timestamp(config.fecha_inicio) - pd.Timedelta(days=config.dias_historia_previa)).strftime("%Y-%m-%d")
    precios = _cargar_precios(tickers, inicio_carga, config.fecha_fin, cargador_precios)
    if not precios:
        raise RuntimeError("No hay precios para ningún ticker del universo (¿sin red y sin caché?)")
    if es_universo_pit(config.universo):
        avisos = [_aviso_cobertura_pit(config, set(precios))]  # cientos de deslistados: un resumen, no un aviso por ticker
    else:
        avisos = [f"Sin datos para {t}" for t in tickers if t not in precios]
    datos_bench = _cargar_precios([config.benchmark], inicio_carga, config.fecha_fin, cargador_precios)
    benchmark = datos_bench.get(config.benchmark)
    if benchmark is None:
        avisos.append(f"Sin datos del benchmark {config.benchmark}")
    return DatosExperimento(precios, benchmark, config.benchmark, avisos)


def _motivos_de_operaciones(operaciones: pd.DataFrame, eventos: list[dict[str, Any]]) -> list[str]:
    """
    Motivo de cada operación de un backtest de una `EstrategiaSenales`, cruzando `(fecha, ticker, lado)` con sus eventos
    (`senal`, `stop_perdida`, `stop_trailing`, `take_profit`, `tiempo`, `fuera_universo`, `regla_propia`). Una operación
    sin evento (compra o venta parcial que solo recoloca los pesos por la deriva o los costes) es `ajuste`.
    """
    por_clave = {(e["fecha"], e["ticker"], "compra" if e["tipo"] == "entrada" else "venta"): e["motivo"] for e in eventos}
    return [por_clave.get((f"{fecha:%Y-%m-%d}", ticker, lado), "ajuste")
            for fecha, ticker, lado in zip(operaciones["fecha"], operaciones["ticker"], operaciones["lado"], strict=True)]


def ejecutar_con_datos(config: ConfiguracionExperimento, datos: DatosExperimento,
                       completo: bool = True) -> ResultadoExperimento:
    """
    Corre el backtest de `config` sobre precios ya cargados y devuelve el resultado (sin persistir).

    Pasos: normaliza la configuración, fija las semillas, selecciona de `datos.precios` los
    tickers del universo y recorta todo a `<= fecha_fin` (así un barrido cargado hasta una fecha
    posterior no deja ver el futuro a `Estrategia.preparar`), simula con `BacktestEngine` y
    calcula métricas. Lanza `RuntimeError` si la estrategia no produjo ningún rebalanceo válido.

    Args:
        completo: si False (tests de permutación: miles de backtests) se omiten la procedencia
            (hash de precios, git) y los validadores de `config.validaciones`.
    """
    config = config.normalizada()
    clase = obtener_estrategia(config.estrategia)
    estrategia = clase(config.parametros)
    estrategia.semilla = config.semilla
    random.seed(config.semilla)
    np.random.seed(config.semilla)

    tickers = resolver_universo(config.universo, inicio=config.fecha_inicio, fin=config.fecha_fin)
    fin = pd.Timestamp(config.fecha_fin)
    datos_universo = {t: datos.precios[t] for t in tickers if t in datos.precios}
    if not datos_universo:
        raise RuntimeError("No hay precios para ningún ticker del universo (¿sin red y sin caché?)")
    precios = recortar_hasta(datos_universo, fin)
    pit = es_universo_pit(config.universo)
    if pit:
        avisos = [_aviso_cobertura_pit(config, set(precios))]
        # La estrategia solo ve, en cada fecha, los miembros del índice en esa fecha con historia hasta t-1
        filtro_universo = UniversoPuntoEnElTiempo(cargar_membresias_sp500()).crear_filtro(precios)
    else:
        avisos = [f"Sin datos para {t}" for t in tickers if t not in precios]
        filtro_universo = None

    motor = BacktestEngine(BacktestConfig(
        start_date=config.fecha_inicio, end_date=config.fecha_fin, initial_capital=config.capital_inicial,
        rebalance_frequency=config.frecuencia_rebalanceo, commission_rate=config.comision,
        slippage_pct=config.slippage, max_positions=max(len(precios), 1), benchmark_ticker=config.benchmark,
    ))
    resultado_bt = motor.run_backtest(estrategia.como_funcion(filtro_universo=filtro_universo), precios)
    if not estrategia.historial_pesos:
        detalle = estrategia.errores[0] if estrategia.errores else "sin pesos"
        raise RuntimeError(f"La estrategia no generó ningún rebalanceo válido ({detalle})")
    avisos.extend(estrategia.avisos)
    if estrategia.errores:
        avisos.append(f"{len(estrategia.errores)} rebalanceos fallaron; primero: {estrategia.errores[0]}")
    if pit:
        avisos.extend(_avisos_liquidaciones_caducadas(motor.trades, precios))

    # Curva de equity + benchmark reescalado al capital inicial
    curva = resultado_bt.equity_curve
    equity = pd.DataFrame({"fecha": curva.index, "equity": curva["equity"].values,
                           "drawdown": curva["drawdown_pct"].values, "retorno": curva["returns"].values})
    benchmark_ret = None
    benchmark_usado = None
    if datos.ticker_benchmark != config.benchmark:
        avisos.append(f"Sin datos del benchmark {config.benchmark} (los cargados son de {datos.ticker_benchmark})")
    elif datos.benchmark is None:
        avisos.append(f"Sin datos del benchmark {config.benchmark}")
    else:
        benchmark_usado = recortar_hasta({config.benchmark: datos.benchmark}, fin).get(config.benchmark)
    serie_bench = extraer_serie_precios(benchmark_usado, config.benchmark, avisar=False)
    if serie_bench is not None:
        alineada = serie_bench.reindex(curva.index).ffill()
        if alineada.notna().any():
            alineada = alineada.bfill()
            equity["benchmark"] = (alineada / alineada.iloc[0] * config.capital_inicial).values
            benchmark_ret = alineada.pct_change().dropna()
    if "benchmark" not in equity:
        equity["benchmark"] = np.nan

    pesos = pd.DataFrame(
        [(pd.Timestamp(f), t, w) for f, pesos_f in estrategia.historial_pesos for t, w in pesos_f.items()],
        columns=["fecha", "ticker", "peso"],
    )
    operaciones = pd.DataFrame([{
        "fecha": pd.Timestamp(t.entry_date), "ticker": t.ticker, "lado": "compra" if t.side == "buy" else "venta",
        "cantidad": t.quantity, "precio": t.entry_price if t.side == "buy" else t.exit_price,
        "comision": t.commission, "pnl": t.pnl,
    } for t in motor.trades], columns=["fecha", "ticker", "lado", "cantidad", "precio", "comision", "pnl"])
    if getattr(estrategia, "eventos", None):  # estrategias de señales: cada operación lleva su motivo
        operaciones["motivo"] = _motivos_de_operaciones(operaciones, estrategia.eventos)

    metricas = _calcular_metricas(config, resultado_bt, equity, estrategia.historial_pesos, benchmark_ret)
    resultado = ResultadoExperimento(config, metricas, equity, pesos, operaciones, avisos)
    if completo:
        resultado.procedencia = obtener_procedencia(precios, benchmark_usado)
        for nombre in config.validaciones:
            if nombre not in VALIDADORES:
                raise ValueError(f"Validador '{nombre}' no registrado. Disponibles: {sorted(VALIDADORES)}")
            resultado.metricas["validacion"][nombre] = _limpiar(VALIDADORES[nombre](resultado))
    return resultado


def ejecutar_experimento(config: ConfiguracionExperimento, cargador_precios: CargadorPrecios) -> ResultadoExperimento:
    """
    Corre el backtest de `config` y devuelve el resultado completo (sin persistir).

    Es la composición de `cargar_datos_experimento` y `ejecutar_con_datos`; los avisos de la
    carga (tickers sin datos, benchmark) se suman a los de la simulación. Lanza `RuntimeError`
    si no hay precios o la estrategia no produjo ningún rebalanceo válido.
    """
    config = config.normalizada()
    datos = cargar_datos_experimento(config, cargador_precios)
    resultado = ejecutar_con_datos(config, datos)
    resultado.avisos = list(dict.fromkeys(datos.avisos + resultado.avisos))
    return resultado


# ----------------------------------------------------------------------------
# Barridos
# ----------------------------------------------------------------------------


def _expandir(config: ConfiguracionExperimento, rejilla: dict[str, list[Any]],
              tolerante: bool) -> tuple[list[ConfiguracionExperimento], list[dict[str, Any]]]:
    """
    Núcleo común de `expandir_rejilla` y `expandir_rejilla_con_omitidas`.

    Con `tolerante=False` la primera combinación que no valida lanza `ValidationError` (comportamiento
    histórico). Con `tolerante=True` se omite y se anota `{valores, motivo}` en la segunda lista.
    """
    if not rejilla:
        return [config.normalizada()], []
    claves = list(rejilla)
    total = math.prod(len(valores) for valores in rejilla.values())
    if total > MAX_COMBINACIONES_REJILLA:
        raise ValueError(f"La rejilla tiene {total} combinaciones (máximo {MAX_COMBINACIONES_REJILLA})")
    configuraciones: dict[str, ConfiguracionExperimento] = {}
    omitidas: list[dict[str, Any]] = []
    for combinacion in itertools.product(*(rejilla[k] for k in claves)):
        campos = dict(zip(claves, combinacion, strict=True))
        propios = {k: v for k, v in campos.items() if k in ConfiguracionExperimento.model_fields}
        parametros = {**config.parametros, **{k: v for k, v in campos.items() if k not in propios}}
        etiqueta = ", ".join(f"{k}={v}" for k, v in campos.items())
        try:
            nueva = ConfiguracionExperimento(**{**config.model_dump(), **propios, "parametros": parametros,
                                                "etiqueta": etiqueta}).normalizada()
        except ValidationError as error:
            if not tolerante:
                raise
            motivo = "; ".join(str(e["msg"]).removeprefix("Value error, ") for e in error.errors(include_url=False))
            omitidas.append({"valores": campos, "motivo": motivo})
            continue
        configuraciones.setdefault(nueva.huella(), nueva)
    return list(configuraciones.values()), omitidas


def expandir_rejilla(config: ConfiguracionExperimento, rejilla: dict[str, list[Any]]) -> list[ConfiguracionExperimento]:
    """
    Producto cartesiano de valores sobre una configuración base.

    Cada clave es un parámetro de la estrategia o un campo de la configuración (`comision`,
    `frecuencia_rebalanceo`, `fecha_inicio`...). Devuelve configuraciones normalizadas sin
    duplicados, con la `etiqueta` `clave=valor, ...`. Máximo `MAX_COMBINACIONES_REJILLA`.
    Estricta: una combinación inválida lanza `ValidationError`; para tolerarlas ver
    `expandir_rejilla_con_omitidas`.
    """
    return _expandir(config, rejilla, tolerante=False)[0]


def expandir_rejilla_con_omitidas(
    config: ConfiguracionExperimento, rejilla: dict[str, list[Any]],
) -> tuple[list[ConfiguracionExperimento], list[dict[str, Any]]]:
    """
    Como `expandir_rejilla`, pero las combinaciones que no validan (p. ej. `rapida >= lenta`) se omiten.

    Devuelve `(configuraciones válidas, omitidas)`; cada omitida es `{valores: {eje: valor}, motivo: str}`.
    El llamador decide qué hacer si no queda ninguna válida (la API responde 422).
    """
    return _expandir(config, rejilla, tolerante=True)
