"""
Experimentos reproducibles: configuración, ejecución con el motor de backtest y barridos.

Responsabilidad:
- `ConfiguracionExperimento` (Pydantic): estrategia + parámetros + universo + fechas + capital +
  costes + frecuencia + benchmark + semilla. `huella()` es un hash estable (misma configuración,
  mismo identificador); `normalizada()` rellena los parámetros por defecto de la estrategia.
- `ejecutar_experimento(config, cargador_precios)`: carga precios, corre `BacktestEngine`,
  calcula métricas con `PerformanceAnalyzer` (Sharpe, Sortino, Calmar, máx. drawdown, volatilidad,
  turnover, alfa/beta vs benchmark...) y devuelve un `ResultadoExperimento` (no persiste nada;
  eso lo hace `AlmacenExperimentos`).
- `expandir_rejilla(config, {param: [valores]})`: N configuraciones de un barrido de parámetros.
- Punto de extensión de validación: `registrar_validador(nombre)` registra funciones
  `(ResultadoExperimento) -> dict` (p. ej. CPCV, DSR, PBO en el futuro); las que pida
  `config.validaciones` se ejecutan y su salida va a `metricas['validacion'][nombre]`.

Se relaciona con:
- ypyquant.estrategias.registro (obtiene la clase de estrategia) y .base (adaptador al motor).
- ypyquant.estrategias.precios (cargadores), .almacen (persistencia Parquet + DuckDB).
- ypyquant.portfolio.backtest_engine (BacktestEngine, BacktestConfig) y .performance_metrics.
- universos/*.txt (listas de tickers por nombre).
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ypyquant.data.matriz_precios import CargadorPrecios, extraer_serie_precios
from ypyquant.estrategias.registro import obtener_estrategia
from ypyquant.portfolio.backtest_engine import BacktestConfig, BacktestEngine
from ypyquant.portfolio.performance_metrics import PerformanceAnalyzer

logger = logging.getLogger(__name__)

RAIZ_REPO = Path(__file__).resolve().parents[3]
CARPETA_UNIVERSOS = RAIZ_REPO / "universos"
DIAS_ANIO = 252
MAX_COMBINACIONES_REJILLA = 500

Frecuencia = Literal["daily", "weekly", "monthly", "quarterly"]


class ConfiguracionExperimento(BaseModel):
    """
    Todo lo necesario para reproducir un backtest.

    Campos:
        estrategia: nombre registrado. parametros: valores de `Estrategia.Parametros` (los que falten usan el defecto).
        universo: lista de tickers o nombre de un archivo de `universos/` (`'sp500'`, o `'sp500:30'` = los 30 primeros).
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

    def huella(self) -> str:
        """Hash SHA-256 estable de la configuración (sin la etiqueta; el orden del universo no importa)."""
        datos = self.model_dump(exclude={"etiqueta"})
        if isinstance(datos["universo"], list):
            datos["universo"] = sorted(datos["universo"])
        canonico = json.dumps(datos, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
        return hashlib.sha256(canonico.encode("utf-8")).hexdigest()

    def identificador(self) -> str:
        """Identificador legible y estable: `<estrategia>-<10 primeros caracteres de la huella>`."""
        return f"{self.estrategia}-{self.huella()[:10]}"


@dataclass
class ResultadoExperimento:
    """
    Salida de `ejecutar_experimento`: todo lo que persiste el almacén.

    equity: DataFrame con `fecha, equity, benchmark, drawdown, retorno` (benchmark reescalado al capital inicial).
    pesos: formato largo `fecha, ticker, peso` (pesos objetivo en cada rebalanceo).
    operaciones: `fecha, ticker, lado, cantidad, precio, comision, pnl`.
    """

    config: ConfiguracionExperimento
    metricas: dict[str, Any]
    equity: pd.DataFrame
    pesos: pd.DataFrame
    operaciones: pd.DataFrame
    avisos: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Validación: punto de extensión (CPCV / DSR / PBO se enchufarán aquí)
# ----------------------------------------------------------------------------

Validador = Callable[[ResultadoExperimento], dict[str, Any]]
VALIDADORES: dict[str, Validador] = {}


def registrar_validador(nombre: str) -> Callable[[Validador], Validador]:
    """Decorador: registra un validador `(ResultadoExperimento) -> dict` para `config.validaciones`."""

    def decorador(funcion: Validador) -> Validador:
        VALIDADORES[nombre] = funcion
        return funcion

    return decorador


# ----------------------------------------------------------------------------
# Universos y precios
# ----------------------------------------------------------------------------


def resolver_universo(universo: list[str] | str, carpeta: Path | None = None) -> list[str]:
    """
    Lista de tickers desde una lista explícita o un nombre de `universos/` (`'sp500'`, `'sp500:30'`).

    Las líneas vacías y las que empiezan por `#` se ignoran. Lanza `ValueError` si el universo no existe.
    """
    if isinstance(universo, list):
        return universo
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
    """Universos disponibles: `[{nombre, n_tickers}]`."""
    carpeta = carpeta or CARPETA_UNIVERSOS
    return [{"nombre": ruta.stem, "n_tickers": len(resolver_universo(ruta.stem, carpeta))}
            for ruta in sorted(carpeta.glob("*.txt"))]


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


def _turnover(historial: list[tuple[Any, dict[str, float]]], anios: float) -> tuple[float, float]:
    """(turnover medio por rebalanceo, turnover anual) a partir de los pesos objetivo; el 1.º (compra inicial) se excluye."""
    cambios = []
    for (_, previos), (_, actuales) in itertools.pairwise(historial):
        tickers = set(previos) | set(actuales)
        cambios.append(0.5 * sum(abs(actuales.get(t, 0.0) - previos.get(t, 0.0)) for t in tickers))
    if not cambios:
        return 0.0, 0.0
    return float(np.mean(cambios)), float(np.sum(cambios) / anios) if anios > 0 else 0.0


def _calcular_metricas(config, resultado_bt, equity: pd.DataFrame, historial, benchmark_ret) -> dict[str, Any]:
    """Métricas del experimento: retorno/riesgo (PerformanceAnalyzer), costes, turnover y alfa/beta vs benchmark."""
    retornos = equity.set_index("fecha")["retorno"].dropna()
    base = PerformanceAnalyzer.calculate_metrics(retornos, risk_free_rate=0.0, periods_per_year=DIAS_ANIO)
    anios = max((pd.Timestamp(config.fecha_fin) - pd.Timestamp(config.fecha_inicio)).days / 365.25, 1e-9)
    turnover_medio, turnover_anual = _turnover(historial, anios)
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
    }
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


def ejecutar_experimento(config: ConfiguracionExperimento, cargador_precios: CargadorPrecios) -> ResultadoExperimento:
    """
    Corre el backtest de `config` y devuelve el resultado completo (sin persistir).

    Pasos: normaliza la configuración, fija las semillas, resuelve el universo, carga precios
    (más una ventana previa para estimar y el benchmark), simula con `BacktestEngine` y calcula
    métricas. Lanza `RuntimeError` si la estrategia no produjo ningún rebalanceo válido.
    """
    config = config.normalizada()
    clase = obtener_estrategia(config.estrategia)
    estrategia = clase(config.parametros)
    estrategia.semilla = config.semilla
    random.seed(config.semilla)
    np.random.seed(config.semilla)

    tickers = resolver_universo(config.universo)
    inicio_carga = (pd.Timestamp(config.fecha_inicio) - pd.Timedelta(days=config.dias_historia_previa)).strftime("%Y-%m-%d")
    datos = _cargar_precios(tickers, inicio_carga, config.fecha_fin, cargador_precios)
    if not datos:
        raise RuntimeError("No hay precios para ningún ticker del universo (¿sin red y sin caché?)")
    avisos = [f"Sin datos para {t}" for t in tickers if t not in datos]

    motor = BacktestEngine(BacktestConfig(
        start_date=config.fecha_inicio, end_date=config.fecha_fin, initial_capital=config.capital_inicial,
        rebalance_frequency=config.frecuencia_rebalanceo, commission_rate=config.comision,
        slippage_pct=config.slippage, max_positions=max(len(datos), 1), benchmark_ticker=config.benchmark,
    ))
    resultado_bt = motor.run_backtest(estrategia.como_funcion(), datos)
    if not estrategia.historial_pesos:
        detalle = estrategia.errores[0] if estrategia.errores else "sin pesos"
        raise RuntimeError(f"La estrategia no generó ningún rebalanceo válido ({detalle})")
    if estrategia.errores:
        avisos.append(f"{len(estrategia.errores)} rebalanceos fallaron; primero: {estrategia.errores[0]}")

    # Curva de equity + benchmark reescalado al capital inicial
    curva = resultado_bt.equity_curve
    equity = pd.DataFrame({"fecha": curva.index, "equity": curva["equity"].values,
                           "drawdown": curva["drawdown_pct"].values, "retorno": curva["returns"].values})
    benchmark_ret = None
    datos_bench = _cargar_precios([config.benchmark], inicio_carga, config.fecha_fin, cargador_precios)
    serie_bench = extraer_serie_precios(datos_bench.get(config.benchmark), config.benchmark, avisar=False)
    if serie_bench is not None:
        alineada = serie_bench.reindex(curva.index).ffill()
        if alineada.notna().any():
            alineada = alineada.bfill()
            equity["benchmark"] = (alineada / alineada.iloc[0] * config.capital_inicial).values
            benchmark_ret = alineada.pct_change().dropna()
    else:
        avisos.append(f"Sin datos del benchmark {config.benchmark}")
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

    metricas = _calcular_metricas(config, resultado_bt, equity, estrategia.historial_pesos, benchmark_ret)
    resultado = ResultadoExperimento(config, metricas, equity, pesos, operaciones, avisos)
    for nombre in config.validaciones:
        if nombre not in VALIDADORES:
            raise ValueError(f"Validador '{nombre}' no registrado. Disponibles: {sorted(VALIDADORES)}")
        resultado.metricas["validacion"][nombre] = _limpiar(VALIDADORES[nombre](resultado))
    return resultado


# ----------------------------------------------------------------------------
# Barridos
# ----------------------------------------------------------------------------


def expandir_rejilla(config: ConfiguracionExperimento, rejilla: dict[str, list[Any]]) -> list[ConfiguracionExperimento]:
    """
    Producto cartesiano de valores sobre una configuración base.

    Cada clave es un parámetro de la estrategia o un campo de la configuración (`comision`,
    `frecuencia_rebalanceo`, `fecha_inicio`...). Devuelve configuraciones normalizadas sin
    duplicados, con la `etiqueta` `clave=valor, ...`. Máximo `MAX_COMBINACIONES_REJILLA`.
    """
    if not rejilla:
        return [config.normalizada()]
    claves = list(rejilla)
    total = math.prod(len(valores) for valores in rejilla.values())
    if total > MAX_COMBINACIONES_REJILLA:
        raise ValueError(f"La rejilla tiene {total} combinaciones (máximo {MAX_COMBINACIONES_REJILLA})")
    configuraciones: dict[str, ConfiguracionExperimento] = {}
    for combinacion in itertools.product(*(rejilla[k] for k in claves)):
        campos = dict(zip(claves, combinacion, strict=True))
        propios = {k: v for k, v in campos.items() if k in ConfiguracionExperimento.model_fields}
        parametros = {**config.parametros, **{k: v for k, v in campos.items() if k not in propios}}
        etiqueta = ", ".join(f"{k}={v}" for k, v in campos.items())
        nueva = ConfiguracionExperimento(**{**config.model_dump(), **propios, "parametros": parametros,
                                            "etiqueta": etiqueta}).normalizada()
        configuraciones.setdefault(nueva.huella(), nueva)
    return list(configuraciones.values())
