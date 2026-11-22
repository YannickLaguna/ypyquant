"""
Experimentos reproducibles: configuración, ejecución con el motor de backtest y barridos.

Responsabilidad:
- `ConfiguracionExperimento` (Pydantic): estrategia + parámetros + universo + fechas + capital +
  costes + frecuencia + benchmark + semilla. `huella()` es un hash estable (misma configuración,
  mismo identificador); `normalizada()` rellena los parámetros por defecto de la estrategia.
- `cargar_datos_experimento(config, cargador)` -> `DatosExperimento` (precios con historia previa,
  benchmark, avisos) y `ejecutar_con_datos(config, datos)` -> `ResultadoExperimento`: carga y
  simulación separadas, para correr muchos backtests con una sola carga (barridos, permutación).
  `ejecutar_experimento` es su composición. `ejecutar_con_datos` es a su vez la composición de
  `simulacion.simular` (hechos: equity, pesos, operaciones, rotaciones; motor `BacktestEngine`) y
  `evaluacion.evaluar` (métricas con `PerformanceAnalyzer`: Sharpe, Sortino, Calmar, máx. drawdown,
  volatilidad, turnover, alfa/beta vs benchmark, Ulcer, Martin, `costes_sobre_ganancia_bruta`...).
  No persiste nada (lo hace `AlmacenExperimentos`). Este módulo conserva la configuración, los tipos
  (`DatosExperimento`, `ResultadoExperimento`), el registro de validadores y la carga; `simulacion` y
  `evaluacion` dependen de él (por eso `ejecutar_con_datos` los importa de forma diferida).
- Procedencia (`ResultadoExperimento.procedencia`, ver `procedencia.py`; la calcula `simular`): commit, versiones, hash de
  los precios y fecha; no entra en la huella.
- Universo `sp500_pit` (point-in-time): ver `universo_dinamico.py`. `simular` entrega a la estrategia
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
- ypyquant.estrategias.simulacion (`simular`) y .evaluacion (`evaluar`, `reevaluar`): el núcleo de la ejecución.
- ypyquant.data.fuente_precios y ypyquant.data.fuentes (fuente de precios), .almacen (persistencia Parquet + DuckDB),
  .procedencia (huella de reproducibilidad), .barridos y .validacion_barridos (usan
  `cargar_datos_experimento` y `ejecutar_con_datos`).
- ypyquant.validacion (Ulcer, Martin, rachas y PSR para métricas y validadores).
- ypyquant.portfolio.backtest_engine (BacktestEngine, BacktestConfig) y .performance_metrics.
- ypyquant.data.universos (resolución de universos y membresías; aquí se re-exportan `resolver_universo`, `listar_universos`
  y `es_universo_pit`) y .universo_dinamico (elegibilidad point-in-time de `sp500_pit`).
- ypyquant.data.fuente_precios (`FuentePrecios`: `cargar_datos_experimento` la acepta o envuelve un callable suelto).
- app/api (lanza y consulta experimentos).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ypyquant.data.fuente_precios import Deslistado, FuentePrecios, como_fuente
from ypyquant.data.matriz_precios import CargadorPrecios
from ypyquant.data.universos import (  # noqa: F401 - re-exportados: configuraciones, marcos, la API y los tests los importan de aquí
    aviso_cobertura_pit,
    es_universo_pit,
    listar_universos,
    resolver_universo,
)
from ypyquant.estrategias.registro import obtener_estrategia
from ypyquant.validacion import prueba_rachas, sharpe_probabilistico

logger = logging.getLogger(__name__)

DIAS_ANIO = 252
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
    rotaciones: `fecha, rotacion` de cada rebalanceo (para recalcular el turnover sin repetir el backtest); None
        en los importados del TFM.
    """

    config: ConfiguracionExperimento
    metricas: dict[str, Any]
    equity: pd.DataFrame
    pesos: pd.DataFrame
    operaciones: pd.DataFrame
    avisos: list[str] = field(default_factory=list)
    procedencia: dict[str, Any] = field(default_factory=dict)
    rotaciones: pd.DataFrame | None = None


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


def _cargar_precios(simbolos: list[str], inicio: str, fin: str, fuente: FuentePrecios,
                    ) -> tuple[dict[str, pd.DataFrame], dict[str, Deslistado]]:
    """
    Carga los OHLCV ajustados en paralelo y los deslistados que declara la fuente.

    Los símbolos sin datos se omiten (se avisa en el log). Los deslistados se consultan DESPUÉS de cargar la
    serie, porque un adaptador que los infiere (yfinance) lo hace con lo que acaba de leer.
    """

    def cargar(simbolo: str):
        try:
            return simbolo, fuente.ohlcv(simbolo, inicio, fin), fuente.deslistado(simbolo)
        except Exception as error:  # noqa: BLE001 - un ticker roto no aborta el experimento
            logger.warning("No se pudo cargar %s: %s", simbolo, error)
            return simbolo, None, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        resultados = list(pool.map(cargar, simbolos))
    datos, deslistados = {}, {}
    for simbolo, df, deslistado in resultados:
        if df is None or df.empty:
            logger.warning("Sin datos para %s; se omite", simbolo)
            continue
        datos[simbolo] = df
        if deslistado is not None:
            deslistados[simbolo] = deslistado
    return datos, deslistados


# ----------------------------------------------------------------------------
# Utilidades de serialización
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
    deslistados: `{ticker: Deslistado}` que declara la fuente de precios (explícitos si la fuente los
        sabe, inferidos si no). None = datos construidos sin fuente: `simular` los infiere de los precios.
    """

    precios: dict[str, pd.DataFrame]
    benchmark: pd.DataFrame | None
    ticker_benchmark: str
    avisos: list[str] = field(default_factory=list)
    deslistados: dict[str, Deslistado] | None = None


def cargar_datos_experimento(config: ConfiguracionExperimento,
                             cargador_precios: FuentePrecios | CargadorPrecios) -> DatosExperimento:
    """
    Carga los precios del universo (con `dias_historia_previa` antes de `fecha_inicio`) y el benchmark.

    `cargador_precios` es una `FuentePrecios` o, por compatibilidad, un callable `(ticker, inicio, fin)`
    (se envuelve con `como_fuente`). Recoge además los deslistados que declara la fuente.

    Lanza `RuntimeError` si no hay precios de ningún ticker. No simula nada: el resultado se pasa
    a `ejecutar_con_datos` (varias veces si se quiere, p. ej. un barrido o una permutación).
    """
    config = config.normalizada()
    tickers = resolver_universo(config.universo, inicio=config.fecha_inicio, fin=config.fecha_fin)
    inicio_carga = (pd.Timestamp(config.fecha_inicio) - pd.Timedelta(days=config.dias_historia_previa)).strftime("%Y-%m-%d")
    fuente = como_fuente(cargador_precios)
    precios, deslistados = _cargar_precios(tickers, inicio_carga, config.fecha_fin, fuente)
    if not precios:
        raise RuntimeError("No hay precios para ningún ticker del universo (¿sin red y sin caché?)")
    if es_universo_pit(config.universo):
        avisos = [aviso_cobertura_pit(config.fecha_inicio, config.fecha_fin, set(precios))]  # cientos de deslistados: un resumen, no un aviso por ticker
    else:
        avisos = [f"Sin datos para {t}" for t in tickers if t not in precios]
    datos_bench, _ = _cargar_precios([config.benchmark], inicio_carga, config.fecha_fin, fuente)
    benchmark = datos_bench.get(config.benchmark)
    if benchmark is None:
        avisos.append(f"Sin datos del benchmark {config.benchmark}")
    return DatosExperimento(precios, benchmark, config.benchmark, avisos, deslistados)


def ejecutar_con_datos(config: ConfiguracionExperimento, datos: DatosExperimento,
                       completo: bool = True) -> ResultadoExperimento:
    """
    Corre el backtest de `config` sobre precios ya cargados y devuelve el resultado (sin persistir).

    Es la composición `simular` -> `evaluar`: `simular` ejecuta la estrategia con el motor y devuelve los
    hechos (equity, pesos, operaciones, rotaciones); `evaluar` calcula métricas, benchmark y validaciones.
    Cada ejecución usa su propio generador aleatorio (semilla de la configuración), sin estado global.
    Lanza `RuntimeError` si no hay precios o la estrategia no produjo ningún rebalanceo válido.

    Args:
        completo: si False (tests de permutación: miles de backtests) se omiten la procedencia
            (hash de precios, git) y los validadores de `config.validaciones`.
    """
    # Import diferido: `simulacion` y `evaluacion` dependen de los tipos de este módulo (ciclo de importación).
    from ypyquant.estrategias.evaluacion import evaluar
    from ypyquant.estrategias.simulacion import simular

    config = config.normalizada()
    simulacion = simular(config, datos, completo=completo)
    evaluacion = evaluar(simulacion, validar=completo)
    return ResultadoExperimento(config, evaluacion.metricas, evaluacion.equity, simulacion.pesos,
                                simulacion.operaciones, simulacion.avisos, evaluacion.procedencia,
                                simulacion.rotaciones)


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
