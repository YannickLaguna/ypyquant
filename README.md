# quant

Librería Python de lógica cuantitativa extraída del proyecto TFM GrowSave. Instalable en modo editable para reutilizar en notebooks, scripts y otros servicios sin dependencias de FastAPI ni bases de datos.

## Instalación

```bash
pip install -e /ruta/a/tfm/quant
```

## Estructura

```
quant/
├── optimization/       # Modelos de portafolio, walk-forward, preprocessing
│   └── presets/        # Configuraciones predefinidas (Markowitz, BL, PMPT...)
├── factors/            # CAPM, Fama-French (4/5/6 factores), regresión robusta
├── quantlib/           # Curvas de tipos, pricing de bonos, Monte Carlo, Black-Litterman
├── ml/                 # Clustering, regímenes de mercado, PCA, LSTM, RL
├── portfolio/          # Métricas de performance, backtesting, rebalanceo
└── data/               # Descargadores: Fama-French factors, indicadores técnicos
```

## Uso rápido

```python
from quant.optimization.portfolio_models import MarkowitzOptimizer, LedoitWolfShrinkageModel
from quant.factors.factor_models import FamaFrench4FactorModel
from quant.portfolio.performance_metrics import calcular_metricas
from quant.data.french_data_fetcher import FrenchDataFetcher
```

## Dependencias opcionales

Para modelos de deep learning (LSTM/CNN):

```bash
pip install -e "quant[ml]"
```
