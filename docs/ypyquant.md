# ypyquant

Librería Python de lógica cuantitativa extraída del proyecto TFM GrowSave. Instalable en modo editable para reutilizar en notebooks, scripts y otros servicios sin dependencias de FastAPI ni bases de datos.

## Instalación

Desde GitHub:

```bash
pip install git+https://github.com/YannickLaguna/ypyquant.git@main
```

En modo editable (desarrollo local):

```bash
pip install -e /ruta/local/ypyquant
```

## Estructura

```
ypyquant/
├── optimization/       # Modelos de portafolio, walk-forward, preprocessing,
│   │                   # optimizador_portafolio (MPT/PMPT/frontera/objetivos; servicio)
│   └── presets/        # Configuraciones predefinidas (Markowitz, BL, PMPT...)
├── factors/            # CAPM, Fama-French (4/5/6 factores), regresión robusta
├── ml/                 # Clustering, regímenes de mercado, PCA, LSTM, RL
├── portfolio/          # Backtest (motor + benchmark), selección de activos, métricas, rebalanceo
├── data/               # Descargadores, indicadores y matriz_precios (cargador inyectado)
└── utils/              # Serialización NumPy→JSON y divisas (ServicioCambio)
```

Desde v0.3.0 ypyquant es la única fuente de la lógica cuantitativa de GrowSave (backtest,
selección de activos, métricas y optimización). Los servicios (`BacktestEngine`,
`PortfolioBacktestService`, `AssetSelectionService`, `PortfolioOptimizationService`) no
acceden a ninguna BD: reciben un `cargador_precios(ticker, inicio, fin) -> OHLCV` y, si la
divisa no es USD, un servicio de tipos de cambio. Migración de GrowSave:
`docs/migracion-growsave-v0.3.0.md`.

> `ypyquant.quantlib` está deprecated y existe solo como shim de retrocompatibilidad.
> Importar directamente desde `ypyquant.optimization`, `ypyquant.portfolio` o `ypyquant.data`.

## Uso rápido

```python
from ypyquant.optimization.portfolio_models import MarkowitzOptimizer, LedoitWolfShrinkageModel
from ypyquant.factors.factor_models import FamaFrench4FactorModel
from ypyquant.portfolio.performance_metrics import calcular_metricas
from ypyquant.data.french_data_fetcher import FrenchDataFetcher

# Backtest con datos precargados o con un cargador propio
from ypyquant import BacktestConfig, BacktestEngine
config = BacktestConfig(start_date="2022-01-03", end_date="2024-12-31",
                        initial_capital=10_000, rebalance_frequency="quarterly")
resultado = BacktestEngine(config, cargador_precios=mi_cargador).run_backtest(
    lambda fecha, datos: {"AAPL": 0.5, "MSFT": 0.5}, ["AAPL", "MSFT"])
```

## Dependencias opcionales

Para modelos de deep learning (LSTM/CNN):

```bash
pip install "ypyquant[ml] @ git+https://github.com/YannickLaguna/ypyquant.git@main"
```
