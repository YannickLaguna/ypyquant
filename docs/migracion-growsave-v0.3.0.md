# Migración de GrowSave a ypyquant v0.3.0

> Responsabilidad: guía para que GrowSave borre sus copias de la lógica cuantitativa
> (backtest, selección de activos, métricas, optimización, ML, presets, rebalanceo) e
> importe de `ypyquant@v0.3.0` con cambios mínimos.
> Se relaciona con: `src/ypyquant/portfolio/*`, `src/ypyquant/optimization/*`,
> `src/ypyquant/data/matriz_precios.py`, `src/ypyquant/utils/*`, y en GrowSave con
> `app/services/{portafolio,optimizacion,ml}`, `app/utils/{serialization,currency}.py`.

Referencia comparada: GrowSave `integracion/2026-11-17` (`bfb6e43e`) frente a ypyquant `main` (`baf9f78`).

## Principio

ypyquant no importa nada de `app.*`, FastAPI ni SQLAlchemy. Todo lo que en GrowSave venía de la base
de datos se **inyecta**:

| Dependencia en GrowSave | Inyección en ypyquant |
|---|---|
| `app.DB.database.get_ticker_data(ticker, inicio, fin)` | `cargador_precios: CargadorPrecios` (`ypyquant.data.matriz_precios`) |
| `ExchangeRateService().convert_amount(...)` | `servicio_cambio` (Protocol `ServicioCambio`, `ypyquant.utils.divisas`) o `fabrica_servicio_cambio` |
| `TickerService.get_index_constituents_in_period(db, ...)` | `proveedor_universo(indice, inicio_dt, fin_dt, solo_supervivientes)` |
| `get_latest_fundamentals(ticker)` | `proveedor_fundamentales(ticker)` |
| `get_available_tickers()` | `proveedor_tickers_disponibles()` |
| Persistencia `Testeo` | gancho `PortfolioBacktestService._guardar_resultado(...)` |
| `ModelDataProvider(db).get_required_data(...)` | gancho `PortfolioOptimizationService._obtener_datos_modelo(...)` |
| Caché de frontera (stubs) | ganchos `_get_cached_frontier` / `_store_frontier_results` |

## Tabla de imports

| Módulo/símbolo en GrowSave | Pasa a importar | Qué queda en GrowSave |
|---|---|---|
| `app.services.portafolio.backtest_engine`: `BacktestConfig`, `BacktestEngine`, `BacktestResult`, `BacktestTrade`, `format_backtest_results` | `ypyquant.portfolio.backtest_engine` (mismos nombres y campos) | Subclase fina de `BacktestEngine` que inyecta `get_ticker_data` y `ExchangeRateService` (ver plantilla) |
| `...backtest_engine.PortfolioBacktestService(db)` | `ypyquant.portfolio.backtest_engine.PortfolioBacktestService` | Subclase con `db`, `_guardar_resultado` (ORM `Testeo`) y `get_latest_backtest_for_portfolio` |
| `app.services.portafolio.asset_selection.AssetSelectionService(db)` | `ypyquant.portfolio.asset_selection.AssetSelectionService` | Subclase que inyecta `get_ticker_data`, `TickerService`, `get_latest_fundamentals`, `get_available_tickers` |
| `app.services.portafolio.performance_metrics.PerformanceAnalyzer` (y `PerformanceMetrics`) | `ypyquant.portfolio.performance_metrics` | Nada (borrar el fichero) |
| `app.services.portafolio.rebalance_service` | `ypyquant.portfolio.rebalance_service` (idéntico, verificado por AST) | Nada |
| `app.services.optimizacion.optimizacion_ev.PortfolioOptimizationService(db)` | `ypyquant.optimization.optimizador_portafolio.PortfolioOptimizationService` | Subclase con `db`, `get_risk_free_rate`, `get_portfolio_metrics` (ORM) y `_obtener_datos_modelo` (ModelDataProvider) |
| `optimizacion_ev`: lógica pura (Markowitz, frontera, PMPT, metas, CML) | funciones `optimizar_*`, `calcular_*`, `metricas_cartera`, `validar_retorno_objetivo` del mismo módulo | — |
| `app.services.optimizacion.portfolio_models` | `ypyquant.optimization.portfolio_models` | Nada |
| `app.services.optimizacion.data_preprocessing` | `ypyquant.optimization.data_preprocessing` | Nada |
| `app.services.optimizacion.presets.*` | `ypyquant.optimization.presets` (idénticos) | `portfolio_presets.py` puede reexportar |
| `app.services.ml.*` | `ypyquant.ml.*` (idénticos) | Nada |
| `app.utils.serialization.convert_numpy_to_native` | `ypyquant.utils.serializacion.convert_numpy_to_native` (alias de `convertir_numpy_a_nativo`, compatible con NumPy 2) | Puede dejar un reexport |
| `app.utils.currency.CURRENCY_SYMBOLS`, `get_currency_symbol` | `ypyquant.utils.divisas.SIMBOLOS_DIVISA`, `simbolo_divisa` | `get_currency_from_country` (lo usa `authentication.py`) |
| `tests/test_backtest_motor.py` | portado a ypyquant `tests/test_backtest_motor.py` | Puede quedarse tal cual: con la subclase de enlace tardío (abajo) sus `patch.object(motor, 'get_ticker_data')` siguen funcionando |

Fuera de esta migración (dependen de BD/matplotlib o están obsoletos): `portafolio/backtest.py`,
`portafolio/analysis.py`, `portafolio/backtesting_service.py` (DEPRECATED) y `quant_bridge.py`
(su `run_backtest` usa una API que no existe en ninguna versión: `BacktestConfig(commission=...)`,
`engine.run(prices, signals)`; ya está roto).

## Plantillas de subclase (GrowSave)

```python
# app/services/portafolio/backtest_engine.py
"""Adaptador GrowSave del motor de backtest de ypyquant: inyecta BD y tipos de cambio y persiste Testeo."""
from ypyquant.portfolio import backtest_engine as _yq
from ypyquant.portfolio.backtest_engine import (  # noqa: F401  reexport
    BacktestConfig, BacktestResult, BacktestTrade, format_backtest_results, VENTANA_ESTIMACION_DIAS,
)
from app.DB.database import get_ticker_data, Testeo
from app.services.exchange_rate_service import ExchangeRateService


# Enlace tardío vía globals(): los tests hacen patch.object(este_modulo, 'get_ticker_data')
class BacktestEngine(_yq.BacktestEngine):
    def __init__(self, config):
        super().__init__(config, cargador_precios=lambda *a: globals()['get_ticker_data'](*a),
                         servicio_cambio=ExchangeRateService() if config.user_currency != 'USD' else None)


class PortfolioBacktestService(_yq.PortfolioBacktestService):
    def __init__(self, db=None):
        super().__init__(cargador_precios=lambda *a: globals()['get_ticker_data'](*a),
                         fabrica_servicio_cambio=ExchangeRateService)
        self.db = db

    def _guardar_resultado(self, cliente_id, portfolio_id, nombre, tickers, result, portfolio_result):
        if self.db is None:
            return None
        ...  # mismo bloque Testeo(...) que hoy; devolver testeo.id

    def get_latest_backtest_for_portfolio(self, portfolio_id):
        ...  # sin cambios
```

```python
# app/services/portafolio/asset_selection.py
"""Adaptador GrowSave de AssetSelectionService: inyecta precios, universo histórico y fundamentales."""
from ypyquant.portfolio.asset_selection import AssetSelectionService as _Base
from app.DB.database import get_ticker_data, get_available_tickers
from app.DB.ticker_fetcher import TickerService


class AssetSelectionService(_Base):
    def __init__(self, db):
        from app.DB.fundamentals_fetcher import get_latest_fundamentals
        super().__init__(
            cargador_precios=lambda *a: get_ticker_data(*a),
            proveedor_universo=lambda indice, ini, fin, solo: TickerService.get_index_constituents_in_period(
                db=db, index_name=indice, start_date=ini, end_date=fin, include_survivors_only=solo),
            proveedor_fundamentales=get_latest_fundamentals,
            proveedor_tickers_disponibles=get_available_tickers,
        )
        self.db = db
```

```python
# app/services/optimizacion/optimizacion_ev.py
"""Adaptador GrowSave del optimizador de ypyquant: BD, tasa libre de riesgo y datos externos de modelos."""
from ypyquant.optimization.optimizador_portafolio import PortfolioOptimizationService as _Base
from app.DB.database import get_ticker_data


class PortfolioOptimizationService(_Base):
    def __init__(self, db):
        super().__init__(cargador_precios=lambda *a: get_ticker_data(*a))
        self.db = db

    get_risk_free_rate = staticmethod(...)  # se queda igual (consulta ^TNX en BD)

    def get_portfolio_metrics(self, portfolio_id, start_date, end_date):
        # carga Portafolio/Posicion por ORM y delega:
        return self.calcular_metricas_cartera(tickers, pesos, start_date, end_date)

    def _obtener_datos_modelo(self, model, start_date, end_date):
        from app.services.model_data_provider import ModelDataProvider
        return ModelDataProvider(db_session=self.db).get_required_data(model, start_date, end_date)
```

`BacktestConfig` acepta los nombres en inglés de GrowSave (`start_date`, `end_date`, `initial_capital`,
`rebalance_frequency`, `monthly_contribution`, `user_currency`...) y los alias en español de ypyquant 0.2.0
(`fecha_inicio`, `capital_inicial`...). `BacktestResult` expone los campos de GrowSave
(`total_return_pct`, `drawdown_series`, `total_contributions`, `currency`...) y propiedades en español.

## Paridad numérica comprobada

Script de paridad (originales de GrowSave cargados con stubs de `app.*`, datos sintéticos de 5 activos,
2022-01-03 → 2024-12-31):

| Comparación | Resultado |
|---|---|
| Curva de equity, 3 escenarios (pesos iguales trimestral USD; rotación mensual con cierres; DCA 500 €/mes en EUR) | máx. \|Δ\| ≤ 7,3e-12; `total_return_pct`, Sharpe, nº de trades idénticos |
| Benchmark (USD sin DCA; EUR con DCA) | Δ = 0 |
| `compare_to_benchmark`: beta, alpha, tracking error, information ratio, captures | idénticos a 12 decimales |
| Screening (momentum, Sharpe, volatilidad, filtros) y diversificación (greedy, clustering, distance, mixed) | listas idénticas |
| Optimización (max_sharpe, min_vol, equal_weight, risk_parity, frontera, PMPT ×3, vol objetivo, metas, optimize_advanced) | Δpeso = 0 |

## Diferencias deliberadas (riesgos al migrar)

| # | Dónde | Antes (GrowSave) | Ahora (0.3.0) | Impacto |
|---|---|---|---|---|
| 1 | `BacktestResult.total_fees` | `commission_rate × final_capital` | suma real de comisiones (convertida a la divisa del usuario) | Cambia el número mostrado; ejemplo rotación mensual: 15,85 → 110,11 |
| 2 | PnL de las ventas | el precio de entrada se leía tras borrar la posición: cierre total = PnL −comisión | se captura antes | `win_rate`, `avg_win/loss`, `profit_factor` cambian si hay cierres (rotación: win_rate 0,520 → 0,660; profit_factor 11,32 → 4,74). Equity intacta |
| 3 | `compare_to_benchmark['alpha']` en ypyquant | intercepto de regresión × 252 (0.2.0) | alpha de Jensen CAPM (la de GrowSave); la antigua está en `alpha_regresion` | Solo afecta a usuarios de ypyquant 0.2.0 que leyeran `alpha` (`walk_forward` no la usa) |
| 4 | `diversify_distance` (función pura de ypyquant) | `default_rng(42)` (0.2.0) | `RandomState(42)`, igual que `np.random.seed(42)` de GrowSave y sin estado global | Resultado distinto respecto a ypyquant 0.2.0; idéntico a producción |
| 5 | `screen_custom` | `ohlcv.get(..., ohlcv['Close'])` → KeyError con columnas en minúscula: siempre 0 resultados | funciona | El endpoint empezará a devolver activos |
| 6 | `create_custom_model` | `ReturnModelType.HISTORICAL_MEAN` no existía → AttributeError siempre | se añaden `HISTORICAL_MEAN` y `HISTORICAL_MEDIAN` al enum | El endpoint de modelo personalizado empezará a funcionar |
| 7 | Etiquetado de pesos en optimización | `zip(tickers, pesos)` | columnas de la matriz tras el preprocesado (`strict=True`) | Si se descartaba un ticker sin datos, antes los pesos quedaban mal etiquetados |
| 8 | Servicios sin cargador | — | `RuntimeError` explícito | Solo si se instancian sin inyectar |

Se mantienen sin cambiar, para no mover cifras de producción (+94,5 % a 3 años con pesos iguales):

- `total_return_pct` de la cartera con DCA cuenta las aportaciones como ganancia; el del benchmark mide contra el total invertido, así que `excess_return` sale sesgado al alza con aportaciones.
- El rebalanceo `MS` solo ocurre si el día 1 del periodo es hábil.
- La primera aportación mensual se salta si `start_date` no cae en día 1.
