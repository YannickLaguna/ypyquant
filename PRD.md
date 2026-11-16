# PRD — Sistema de Trading

> **Propósito y prioridades: la fuente de verdad es la nota del proyecto en Obsidian**
> ([abrir en Obsidian](obsidian://open?vault=digiyann&file=1%20-%20%F0%9F%9B%A0%EF%B8%8F%20Organizaci%C3%B3n%2F14%20-%20%E2%9C%94%EF%B8%8FProyectos%2FProfesionales%2F2026-01-07%20Sistema%20de%20Trading) · `digiyann/1 - 🛠️ Organización/14 - ✔️Proyectos/Profesionales/2026-01-07 Sistema de Trading.md`).
> Este repo documenta el *cómo* (arquitectura, módulos, comandos). Si algo de aquí contradice la nota, manda la nota.
> Auditoría y plan vigente (2026-11-16): bitácora `5 - Código/AI/44 - Bitácoras/2026-11-16 Sistema trading - auditoría y plan.md`.
>
> **Objetivo (resumen de la nota):** entorno propio, sin lock-in, que va de la investigación a la ejecución: datos → generación de estrategias → backtest → contraste de hipótesis → paper trading → real tras aprobación.
>
> ⚠️ Las fases y la arquitectura de abajo son de 2026-03 y están desfasadas: Supabase ya no existe, el VPS es Contabo (no Hetzner) y GrowSave deja de ser la interfaz del trading. Se reescribirán según la bitácora del 2026-11-16.

---

## 1. Visión general

Sistema cuantitativo end-to-end dividido en dos verticales:

1. **Internal stack** — pipeline de datos → research → ejecución automatizada
2. ~~**GrowSave** — cara pública~~ → desde 2026-11, GrowSave es un producto independiente que solo consume `ypyquant`

El sistema corre en un servidor SIP (Hetzner), conteneirizado con Docker Compose, con acceso al dashboard interno via VPN.

---

## 2. Fases de desarrollo

### Fase 1 — Research local ✅ COMPLETA (2026-03-15)
**Meta:** Fetcher conectado a Supabase + módulos base de ypyquant con interfaces limpias

- [x] Definir interfaz limpia de `ypyquant` (signals/, portfolio/, risk/, data/)
- [x] `data/fetcher.py` — ingesta desde yfinance, FRED, FF Data, ECB → Supabase (triple fallback + gap detection)
- [x] Tests unitarios base para módulos core (`tests/test_fase1.py` → 19/19)
- [x] Documentar convenciones internas (tipos, retornos, naming)
- [x] 9 tablas Supabase: activos, tickers, factor_data, tipos_cambio, portafolios, posiciones, operaciones, signals, backtests

**Stack:** `yfinance → Supabase → ypyquant`

---

### Fase 2 — Entorno de investigación completo ✅ COMPLETA (2026-03-15)
**Meta:** Pipeline de research end-to-end sobre Fase 1

**Decisiones arquitectónicas:**
- Zipline DESCARTADO (no instala Python 3.11+ / NumPy 2.x)
- QuantConnect DESCARTADO (lock-in, decisión PRD §5)
- VectorBT SÍ — stage 3: scan rápido de señales (503 tickers en segundos)
- backtest_engine.py SÍ — stage 4: validación realista con fills/costes
- GUI Fase 2: solo `mlflow ui`; Plotly Dash espera a Fase 3

**Pipeline completo:** DATA → SIGNALS → VBT SCAN → BACKTEST → WALK-FORWARD → PERSIST

- [x] **M1** `ypyquant/signals/` — BaseSignalGenerator, TechnicalSignalGenerator, MLSignalGenerator, RegimeSignalGenerator, FactorSignalGenerator, SignalCombiner
- [x] **M2** MLflow tracking en walk_forward.py — MLflowTracker (try/import guard, no-op sin mlflow)
- [x] **M3** DB persistence — ExportadorResultados.exportar_a_db() → Supabase tabla backtests
- [x] **M4** VectorBT signal scanner — scripts/vbt_signal_scan.py
- [x] **M5** Orquestador pipeline — scripts/run_fase2.py + scripts/config_fase2.yaml
- [x] **M6** Package updates — __init__.py, pyproject.toml, requirements.txt
- [x] **Tests** `tests/test_fase2.py` → 40/40 OK

**Stack añadido:** VectorBT (opcional) · MLflow (operativo) · ypyquant.signals

---

### Fase 3+4 — Paper trading + Interface completa EN CURSO
**Meta:** Conectar el research de ypyquant con ejecución real en paper trading, usando TFM (GrowSave) como interfaz unificada.

**Principio arquitectónico central:**
> **ypyquant = toda la inteligencia de trading** | **TFM = capa de servicio fina (UI + API)**

TFM no contiene lógica de negocio. Solo importa de ypyquant y expone endpoints + UI.
`tfm/app/services/trading/` y `tfm/quant/` → **ELIMINADOS** (migrados a ypyquant.execution/).

**Proyectos involucrados:**
- `ypyquant` (Sistema_trading) — fuente única de verdad (signals, optimization, execution)
- `TFM` (`C:\Users\Yann\Documents\Codigo\Proyectos\tfm`) — FastAPI + React UI (thin layer)
- `tfm/nautilus_service/` — execution engine (Nautilus para backtesting, alpaca-py para live paper)

**Nota arquitectónica:** Nautilus Trader 1.221.0 no tiene adaptador Alpaca (eliminado). Live paper trading usa alpaca-py SDK directo. Nautilus solo para backtesting via BacktestEngine.

**Milestones:**

- [x] **M-DOC** Actualización documentación (PRD, README, project_fase34_status.md, MEMORY)
- [x] **M0** `ypyquant/execution/` completo — broker_interface, paper_broker, protection_manager, wallet_manager, trade_analytics, bot_executor (StorageInterface)
- [x] **M1** ypyquant como librería compartida en TFM — `pip install -e` + quant_bridge.py (TFMStorage)
- [x] **M2** nautilus_service/ — YpyquantStrategy, config, db_writer, run.py (live+backtest), 13/13 tests
- [x] **M3** TFM FastAPI endpoints — trading.py + strategies.py routers, schemas, nautilus_manager, registrado en main.py
- [x] **M4** React frontend sección Trading (TradingOverview, Positions, Orders, BotControl, PnL, StrategyBuilder) ✅ COMPLETA (2026-03-16)
- [ ] **M5** Docker Compose + deploy Hetzner

**Nuevas tablas TFM DB (M2):**
- `paper_account` — cash, equity, posiciones, timestamp
- `bot_log` — logs en tiempo real
- `strategy_config` — configuración de estrategias (signal_config + risk_config JSONB)

**Nuevos módulos ypyquant (M0):**
```
ypyquant/execution/
├── broker_interface.py   ← migrado de TFM
├── paper_broker.py       ← migrado de TFM
├── protection_manager.py ← migrado de TFM
├── wallet_manager.py     ← migrado de TFM
├── trade_analytics.py    ← migrado de TFM
├── bot_executor.py       ← migrado de TFM (adaptado)
└── nautilus/             ← nuevo (M2)
    ├── ypyquant_strategy.py
    ├── config.py
    └── db_writer.py
```

**Flujo de datos:**
1. Nautilus (`ypyquant.execution.nautilus`) genera señales → coloca órdenes en Alpaca paper
2. `db_writer.py` escribe fills/posiciones en PostgreSQL de TFM
3. TFM FastAPI llama a `ypyquant.execution.bot_executor` → lee PostgreSQL → endpoints
4. React frontend consume endpoints → visualiza estado en tiempo real

**Pre-requisitos:**
- Cuenta Alpaca paper (gratis): https://app.alpaca.markets
- `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` en `.env`
- `pip install nautilus_trader` (nautilus_trader>=1.190)

**Ver:** `project_fase34_status.md` para checklist detallado y comandos de verificación.

---

### Fase 5 — Producción real
**Meta:** Live trading con Interactive Brokers

- [ ] Interactive Brokers cuenta real (ib_insync)
- [ ] Mismo `YpyquantStrategy` de Fase 3+4 — solo cambiar adapter en config (no reescribir)
- [ ] TimescaleDB consolidado
- [ ] Proceso de sign-off antes de cada nueva estrategia en producción

---

## 3. Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│  TFM / GrowSave  —  CAPA DE SERVICIO FINA                       │
│                                                                 │
│  React/TypeScript (Vite)      FastAPI (thin API)                │
│  ├── [existing] UI            ├── routers/ (solo HTTP + schema) │
│  └── [NEW] Trading UI         │   └── llaman a ypyquant         │
│      /trading, /strategies    └── schemas/ (Pydantic)           │
│      /bot, /backtesting       NO lógica de negocio aquí         │
└──────────────────────────────────────┬──────────────────────────┘
                                       │ import ypyquant
                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  ypyquant  —  FUENTE ÚNICA DE VERDAD                            │
│                                                                 │
│  signals/          ← ya existe: TechnicalSignalGenerator, etc.  │
│  optimization/     ← ya existe: walk_forward, portfolio_models  │
│  portfolio/        ← ya existe: backtest_engine, metrics        │
│  ml/               ← ya existe: market_regime, ml_predictor     │
│  factors/          ← ya existe: fama-french                     │
│  data/             ← ya existe: fetchers, indicators            │
│  [NEW] execution/                                               │
│  ├── broker_interface.py   ← migrado de TFM                     │
│  ├── paper_broker.py       ← migrado de TFM                     │
│  ├── protection_manager.py ← migrado de TFM                     │
│  ├── wallet_manager.py     ← migrado de TFM                     │
│  ├── trade_analytics.py    ← migrado de TFM                     │
│  ├── bot_executor.py       ← migrado de TFM (orquestador)       │
│  └── nautilus/             ← nuevo (Fase 3+4)                   │
│      ├── ypyquant_strategy.py  (Nautilus Strategy wrapper)      │
│      ├── config.py             (Alpaca / IB config)             │
│      └── db_writer.py          (fills → DB TFM)                 │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
             ┌─────────────────────┴──────────────────┐
             ▼                                        ▼
  ┌─────────────────────┐              ┌──────────────────────────┐
  │  PostgreSQL (TFM)   │◄─────────────│  Nautilus process        │
  │  operacion          │  db_writer   │  (run.py entry point)    │
  │  posicion           │              │  Alpaca paper adapter    │
  │  testeo             │              │  → Alpaca paper API      │
  │  strategy_config    │              │                          │
  │  paper_account      │              │  Fase 5: IB adapter swap │
  │  bot_log            │              └──────────────────────────┘
  └─────────────────────┘
```

### Separación de capas

| | ypyquant (Sistema_trading) | TFM (GrowSave) |
|---|---|---|
| **Rol** | Fuente única de verdad | Capa de servicio fina |
| **Contiene** | Toda la lógica de trading | Solo HTTP + UI |
| **DB** | Supabase (research) | PostgreSQL TFM (ejecución) |
| **Imports** | Standalone | `import ypyquant` |

---

## 4. ypyquant — estructura actual

```
ypyquant/
├── data/
│   ├── french_data_fetcher.py    # Fama-French factors
│   ├── indicators.py             # Indicadores técnicos
│   ├── risk_free_rates.py        # Tipos libres de riesgo
│   └── yield_curves.py           # Curvas de tipos
├── factors/
│   └── factor_models.py          # CAPM, FF 4/5/6, robust regression
├── ml/
│   ├── asset_clustering.py
│   ├── lstm_predictor.py
│   ├── market_regime.py
│   ├── ml_predictor.py
│   ├── pca_risk.py
│   └── rl_optimizer.py
├── optimization/
│   ├── black_litterman.py
│   ├── portfolio_models.py       # Markowitz, Ledoit-Wolf...
│   ├── walk_forward.py
│   ├── mlflow_tracker.py         # MLflow tracking (Fase 2)
│   ├── exportador_resultados.py  # DB persistence (Fase 2)
│   └── presets/
│       ├── factor_models.py
│       ├── ml_avanzados.py
│       ├── pmpt.py
│       └── tradicionales.py
├── portfolio/
│   ├── backtest_engine.py
│   ├── performance_metrics.py
│   ├── monte_carlo.py
│   └── rebalance_service.py
├── signals/                      # Fase 2 ✅
│   ├── base.py
│   ├── technical.py
│   ├── ml_signal.py
│   ├── regime_signal.py
│   ├── factor_signal.py
│   └── combiner.py
└── execution/                    # Fase 3+4 — PENDIENTE M0
    ├── broker_interface.py       # Migrar de TFM
    ├── paper_broker.py           # Migrar de TFM
    ├── protection_manager.py     # Migrar de TFM
    ├── wallet_manager.py         # Migrar de TFM
    ├── trade_analytics.py        # Migrar de TFM
    ├── bot_executor.py           # Migrar de TFM (adaptado)
    └── nautilus/                 # Nuevo (M2)
        ├── ypyquant_strategy.py
        ├── config.py
        └── db_writer.py
```

**Estado por módulo:**
- `signals/` — ✅ Fase 2 (6 archivos, 40/40 tests)
- `optimization/` — ✅ Fase 2 (walk_forward, mlflow_tracker, exportador_resultados)
- `portfolio/` — ✅ Fase 1/2 (backtest_engine, metrics)
- `ml/` — ✅ Fase 1/2
- `factors/` — ✅ Fase 1/2
- `execution/` — pendiente M0 (migración desde TFM)

**Módulos pendientes:**
- `risk/` — gestión de riesgo dedicada (actualmente dispersa en portfolio/)
- Job automatizado ingesta diaria → Fase 3+4 infraestructura

---

## 5. Decisiones de diseño

- **No QuantConnect** — stack propio para control total, activos europeos, y valor en CV quant
- **Supabase en Fase 1** → **TimescaleDB en producción** — migración natural
- **Alpaca paper** obligatorio antes de IB real — mínimo 3-6 meses
- **Dash para dashboard interno** — Python puro, sin API intermedia, reutiliza ypyquant directamente
- **GrowSave solo ve lo que FastAPI expone** — aislamiento total entre datos internos y públicos

---

## 6. Stack de dependencias

| Capa | Librería | Estado |
|---|---|---|
| Datos | yfinance, FRED, ECB | Pendiente fetcher |
| Cuant core | pandas, numpy, scipy, statsmodels | Instalado |
| ML | scikit-learn, TensorFlow | Instalado |
| Optimización | PyPortfolioOpt, QuantLib | Instalado |
| Backtesting | VectorBT | Opcional instalado (pip install "ypyquant/[fast]") |
| Experimentos | MLflow | Opcional instalado (pip install "ypyquant/[tracking]") |
| Ejecución | Nautilus Trader | Fase 4 |
| Brokers | Alpaca, ib_insync | Fase 4-5 |
| Infra | FastAPI, Redis, Celery, Docker | Fase 3 |
| DB | Supabase (dev), TimescaleDB (prod) | Supabase en dev |
| Monitoreo | Grafana, Prometheus | Fase 3 |
| Frontend | Plotly Dash, Next.js | Fase 3 |
