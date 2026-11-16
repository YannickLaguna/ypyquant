# Fase 3+4 — Paper Trading + Interface — Status

> Principio central: **ypyquant = toda la inteligencia de trading** | **TFM = capa de servicio fina**

## Decisiones arquitectónicas

| Decisión | Elección | Razón |
|---|---|---|
| Framework ejecución | **Nautilus Trader** (desde el inicio) | Migración limpia a IB en Fase 5 sin reescribir estrategia |
| Broker paper | **Alpaca paper** (US equities) | Gratis, misma API que producción, bien documentado |
| Librería compartida | **ypyquant** (`pip install -e`) | Single source of truth para TFM y Nautilus |
| Interface | **TFM** (React + FastAPI) | Unifica backtesting + paper trading + bot + estrategias |
| DB | **PostgreSQL TFM** (compartida con Nautilus) | db_writer.py escribe fills directo en TFM |
| Deploy | **Hetzner** con Docker Compose | Mismo servidor que GrowSave, Caddy como proxy |
| Zipline | DESCARTADO | Incompatible Python 3.11+ / NumPy 2.x |

## Pre-requisitos

- [x] Cuenta Alpaca paper: https://app.alpaca.markets — cuenta PA356AG7IUCV, $100k paper cash
- [x] `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` → en `.env` de TFM
- [x] `alpaca-py>=0.38` instalado en venv TFM (via requirements.txt)
- [x] `nautilus_trader>=1.190` en requirements.txt (pendiente confirmar instalación completa)
- [x] Python 3.11.9 — venv TFM recreado con py -3.11

## Checklist de milestones

### M-DOC — Documentación ✅ COMPLETA (2026-03-15)
- [x] PRD.md actualizado (Fases 3+4 + arquitectura TFM/ypyquant)
- [x] README.md (Sistema_trading) reescrito
- [x] README.md (TFM) — sección ypyquant añadida
- [x] project_fase34_status.md creado (este archivo)
- [x] MEMORY.md actualizado

---

### M0 — Migrar execution TFM → ypyquant.execution/ ✅ COMPLETA
**Resultado:** ypyquant/execution/ ya existía con todos los módulos. tfm/services/trading/ y tfm/quant/ **ELIMINADOS** (no solo deprecated — no tenían consumers externos).

- [x] `ypyquant/execution/__init__.py`
- [x] `ypyquant/execution/broker_interface.py`
- [x] `ypyquant/execution/paper_broker.py` (fixed: datetime.utcnow → datetime.now(timezone.utc))
- [x] `ypyquant/execution/protection_manager.py`
- [x] `ypyquant/execution/wallet_manager.py`
- [x] `ypyquant/execution/trade_analytics.py` (fixed: datetime.utcnow → datetime.now(timezone.utc))
- [x] `ypyquant/execution/bot_executor.py` (StorageInterface abstracta ya implementada)
- [x] Exportado en `ypyquant/ypyquant/__init__.py`
- [x] `tests/test_execution.py` → 36/36
- [x] `tests/test_fase2.py` → 40/40 siguen pasando
- [x] `tfm/app/services/trading/` → **ELIMINADO**
- [x] `tfm/quant/` → **ELIMINADO**

---

### M1 — ypyquant como librería compartida en TFM ✅ COMPLETA
**Resultado:** ypyquant instalado con `pip install -e` en venv TFM. quant_bridge.py es el único punto de contacto.

- [x] `pip install -e ../Sistema_trading/ypyquant` en venv TFM (Python 3.11.9)
- [x] `tfm/requirements.txt`: añadido ypyquant editable
- [x] `tfm/app/services/quant_bridge.py` implementado:
  - `load_prices(tickers, start, end, db) -> pd.DataFrame`
  - `run_signals(prices_df, strategy_config) -> pd.DataFrame`
  - `run_backtest(strategy_config, tickers, start, end, db) -> dict`
  - `save_backtest(result, cliente_id, strategy_name, db) -> int`
  - `TFMStorage(StorageInterface)` — adaptador ORM TFM ↔ ypyquant
- [x] fixed: datetime.utcnow → datetime.now(timezone.utc) en quant_bridge.py

---

### M2 — Nautilus Trader + Alpaca setup ✅ COMPLETA
**Nota arquitectónica:** Nautilus Trader 1.221.0 **no tiene adaptador Alpaca** (fue eliminado en versiones recientes). Arquitectura adaptada: Nautilus solo para backtesting, alpaca-py SDK directo para live paper trading.

**Archivos creados:**
- [x] `tfm/nautilus_service/__init__.py`
- [x] `tfm/nautilus_service/config.py` — AlpacaConfig, StrategyRunConfig, NautilusServiceConfig
- [x] `tfm/nautilus_service/strategies/__init__.py`
- [x] `tfm/nautilus_service/strategies/ypyquant_strategy.py` — YpyquantStrategy(Strategy) + _build_combiner()
- [x] `tfm/nautilus_service/adapters/db_writer.py` — DBWriter → escribe fills en tabla operacion/posicion
- [x] `tfm/nautilus_service/run.py` — entry point: `--backtest` o live (poll loop alpaca-py)
- [x] `tfm/nautilus_service/tests/test_strategy.py` → **13/13 tests pasando**

**Tablas nuevas en TFM DB:**
- [ ] `paper_account` — **PENDIENTE** (Alembic migration)
- [ ] `bot_log` — **PENDIENTE** (Alembic migration)
- [x] `strategy_config` — modelo ORM creado en database.py, migration **PENDIENTE**

**Test de verificación M2:**
```bash
cd tfm && python -X utf8 nautilus_service/tests/test_strategy.py
# → 13/13 OK
```

---

### M3 — TFM FastAPI endpoints ✅ COMPLETA (2026-03-15)
**Resultado:** API completa implementada y registrada en main.py.

**Archivos creados:**
- [x] `app/schemas/__init__.py`
- [x] `app/schemas/trading.py` — PositionOut, OrderOut, PnLSummary, AccountStatus, BotStatus, BotLogEntry
- [x] `app/schemas/strategies.py` — GeneratorConfig, StrategyConfigIn/Out, BacktestRequest, BacktestResult, SignalRow
- [x] `app/services/nautilus_manager.py` — start/stop/status/logs proceso Nautilus
- [x] `app/routers/trading.py` — todos los endpoints `/api/trading/*`
- [x] `app/routers/strategies.py` — CRUD + backtest + señales `/api/strategies/*`
- [x] Registrado en `app/routers/__init__.py` y `app/main.py`

**Endpoints disponibles:**
```
GET  /api/trading/status          → BotStatus
GET  /api/trading/positions       → List[PositionOut]
GET  /api/trading/orders          → List[OrderOut] (paginado)
GET  /api/trading/pnl             → PnLSummary
GET  /api/trading/account         → AccountStatus (Alpaca paper)
POST /api/trading/bot/start       → lanza nautilus_service/run.py
POST /api/trading/bot/stop        → SIGTERM al proceso
GET  /api/trading/bot/logs        → List[BotLogEntry]

GET    /api/strategies            → List[StrategyConfigOut]
POST   /api/strategies            → StrategyConfigOut (201)
GET    /api/strategies/{id}       → StrategyConfigOut
PUT    /api/strategies/{id}       → StrategyConfigOut
DELETE /api/strategies/{id}       → 204
POST   /api/strategies/{id}/activate
POST   /api/strategies/{id}/backtest → BacktestResult
GET    /api/strategies/{id}/signals  → List[SignalRow]
GET    /api/strategies/{id}/backtest/{bt_id} → BacktestResult
```

**Pendiente M3:**
- [ ] APScheduler job: actualizar precios posiciones cada 60s (market hours) — no crítico para M4
- [ ] Alembic migrations para strategy_config, paper_account, bot_log

**Test de verificación M3:**
```bash
cd tfm && python -X utf8 scripts/run.py
curl http://localhost:8000/api/trading/status
curl http://localhost:8000/api/strategies
```

---

### M4 — React frontend sección Trading ✅ COMPLETA (2026-03-16)
**Objetivo:** UI completa para monitorizar y controlar el paper trading.

**Archivos creados en `tfm/app/frontend/`:**
- [x] `src/services/tradingApi.ts` — cliente API tipado para /api/trading/* y /api/strategies/*
- [x] `src/pages/Trading/index.tsx` — TradingOverview: KPIs + BotStatus + top 5 posiciones + tabs
- [x] `src/pages/Trading/Positions.tsx` — tabla completa de posiciones con P&L verde/rojo
- [x] `src/pages/Trading/Orders.tsx` — historial paginado + filtro compra/venta
- [x] `src/pages/Trading/BotControl.tsx` — start/stop bot + log viewer (polling cada 3s)
- [x] `src/pages/Trading/PnL.tsx` — equity curve (AreaChart) + drawdown (BarChart) + métricas
- [x] `src/pages/Strategies/index.tsx` — cards de estrategias con activate/delete
- [x] `src/pages/Strategies/StrategyBuilder.tsx` — wizard 3 pasos (señales/riesgo/activos)
- [x] `src/pages/Strategies/BacktestResults.tsx` — run backtest + métricas grid + equity curve

**Archivos modificados:**
- [x] `components/Sidebar.tsx` — añadidos ítems Trading (Activity) y Estrategias (Zap)
- [x] `src/App.tsx` — rutas /trading/*, /strategies/* registradas bajo ProtectedRoute

**Rutas activas:**
```
/trading             → TradingOverview
/trading/positions   → Positions
/trading/orders      → Orders
/trading/bot         → BotControl
/trading/pnl         → PnL
/strategies          → StrategiesList
/strategies/new      → StrategyBuilder
/strategies/:id/backtest        → BacktestResults (run nuevo)
/strategies/:id/backtest/:bt_id → BacktestResults (ver existente)
```

**Stack UI:** shadcn/ui (Card, Badge, Button, Table, Tabs, Select) + recharts + lucide-react

**Test de verificación M4:**
```bash
cd tfm/app/frontend && npm run dev
# Navegar a http://localhost:5173/trading
```

---

### M5 — Docker Compose + Hetzner
**Objetivo:** Stack completo containerizado.

- [ ] `docker-compose.yml` (postgres, redis, fastapi, nautilus, frontend, caddy)
- [ ] `Dockerfile` multi-stage para fastapi (incluye ypyquant)
- [ ] `Dockerfile.nautilus` (entrypoint: `python nautilus_service/run.py`)
- [ ] `.env.example` (ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, DATABASE_URL, SECRET_KEY)
- [ ] `scripts/deploy.sh` (pull → build → up → health check → rollback automático)
- [ ] Caddy config (HTTPS automático, /api/* → fastapi, / → frontend)

**Test de verificación M5:**
```bash
docker compose up -d && docker compose ps  # todos healthy
```

---

## Verificación end-to-end completa

```bash
# M0: ypyquant.execution importable
python -c "from ypyquant.execution import BrokerInterface, PaperBroker; print('M0 OK')"
python -X utf8 tests/test_execution.py

# M1: ypyquant importable desde TFM
cd tfm && python -c "from ypyquant.signals import TechnicalSignalGenerator; from ypyquant.execution import BotExecutor; print('M1 OK')"

# M2: Nautilus strategy en modo backtest (sin Alpaca)
cd tfm && python nautilus_service/tests/test_strategy.py

# M3: API endpoints
cd tfm && uvicorn app.main:app --reload
curl http://localhost:8000/api/trading/status

# M4: Frontend
cd tfm/app/frontend && npm run dev
# http://localhost:5173/trading

# M5: Docker
docker compose up -d && docker compose ps
```

## Notas para futuras sesiones

- **M5 es el siguiente paso**: Docker Compose + deploy Hetzner
- **Nautilus sin Alpaca adapter**: usar alpaca-py SDK directo para live; Nautilus solo backtesting
- **DB migrations pendientes**: strategy_config, paper_account, bot_log (Alembic) — necesario antes de usar /api/strategies en producción. Los endpoints tienen lazy import que retorna 503 si la tabla no existe (no crashea el servidor).
- **IB Fase 5**: mismo YpyquantStrategy — solo cambiar adapter en config. No reescribir.
- **Hetzner**: mismo servidor que GrowSave si el user confirma. Caddy como proxy.
- **TFM venv**: Python 3.11.9, recreado en esta sesión (`py -3.11 -m venv .venv`)
