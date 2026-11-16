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
├── optimization/       # Modelos de portafolio, walk-forward, preprocessing
│   └── presets/        # Configuraciones predefinidas (Markowitz, BL, PMPT...)
├── factors/            # CAPM, Fama-French (4/5/6 factores), regresión robusta
├── ml/                 # Clustering, regímenes de mercado, PCA, LSTM, RL
├── portfolio/          # Métricas de performance, backtesting, rebalanceo
└── data/               # Descargadores: Fama-French factors, indicadores técnicos
```

> `ypyquant.quantlib` está deprecated y existe solo como shim de retrocompatibilidad.
> Importar directamente desde `ypyquant.optimization`, `ypyquant.portfolio` o `ypyquant.data`.

## Uso rápido

```python
from ypyquant.optimization.portfolio_models import MarkowitzOptimizer, LedoitWolfShrinkageModel
from ypyquant.factors.factor_models import FamaFrench4FactorModel
from ypyquant.portfolio.performance_metrics import calcular_metricas
from ypyquant.data.french_data_fetcher import FrenchDataFetcher
```

## Dependencias opcionales

Para modelos de deep learning (LSTM/CNN):

```bash
pip install "ypyquant[ml] @ git+https://github.com/YannickLaguna/ypyquant.git@main"
```
