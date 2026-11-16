# TODO — Sistema Trading

## Fase 1 — Research local (COMPLETADA ✅)

### Completado
- [x] `.env` + `.env.example` + `.gitignore`
- [x] `requirements.txt` (sqlalchemy, psycopg2-binary, python-dotenv, yfinance, requests, lxml)
- [x] `.venv` creado e instalado
- [x] `db/database.py` — engine + SessionLocal (sync, Supabase pooler)
- [x] `db/models.py` — 10 tablas ORM: tickers, activos, factor_data, tipos_cambio, portafolios, posiciones, operaciones, signals, backtests, **indicator_data**
- [x] `data/fetcher.py` — DataFetcher con triple fallback (yfinance → Twelve Data → Alpha Vantage) + gap detection + persistencia
- [x] `data/factors.py` — FactorDataManager FF5+MOM + double-write a indicator_data
- [x] `data/fred.py` — FREDFetcher (DFF, DGS10, DGS2, DEXUSEU, CPIAUCSL) → indicator_data
- [x] `data/loader.py` — load_prices, load_factors, load_fx, **load_indicators**
- [x] `data/fx.py` — FXFetcher tipos de cambio
- [x] `scripts/reset_tables.py` — reset controlado de tablas en Supabase
- [x] `scripts/fetch_sp500.py` — scrape Wikipedia S&P 500 → universos/sp500.txt
- [x] `tests/test_fase1.py` — 19/19 tests pasando
- [x] `tests/test_loader.py` — load_prices, load_factors, load_fx
- [x] `tests/test_factors.py` — FactorDataManager + double-write indicator_data
- [x] `tests/test_fred.py` — FREDFetcher
- [x] `tests/test_indicators.py` — load_indicators pivot
- [x] `tests/test_ypyquant_api.py` — interfaces ypyquant
- [x] Tablas creadas en Supabase con schema correcto
- [x] indicator_data — tabla unificada long-format (entity_id + metric + value)
- [x] yfinance 1.2.0 funcionando correctamente (errores previos eran festivos/fines de semana)

---

## Fase 2 — Entorno investigación (IMPLEMENTADA ✅ 2026-03-15)

### Completado
- [x] `ypyquant/signals/` — módulo completo (base, technical, ml, regime, factor, combiner)
- [x] MLflow tracking en walk_forward.py — MLflowTracker con no-op guard
- [x] DB persistence — ExportadorResultados.exportar_a_db() → tabla backtests
- [x] `scripts/vbt_signal_scan.py` — VectorBT scan (opcional, requiere vectorbt>=0.26)
- [x] `scripts/run_fase2.py` — orquestador pipeline stages 1-6
- [x] `scripts/config_fase2.yaml` — configuración YAML del pipeline
- [x] pyproject.toml: [tracking] mlflow>=2.10, [fast] vectorbt>=0.26
- [x] requirements.txt: pyyaml + comentarios opcionales Fase 2
- [x] PRD.md, TODO.md, project_fase2_status.md actualizados

### Pendiente Fase 2
- [ ] Instalar MLflow: `pip install "mlflow>=2.10"` o `pip install -e "ypyquant/[tracking]"`
- [ ] Instalar VectorBT: `pip install -e "ypyquant/[fast]"` (si Python/NumPy compatible)
- [ ] Ejecutar pipeline completo: `python scripts/run_fase2.py`
- [ ] `tests/test_fase2.py` — tests del módulo signals + mlflow_tracker + exportar_a_db
- [ ] Migrar `factor_data` → `indicator_data` (eliminar tabla legacy) — Fase 3

## Fase 3+4 — Paper trading + Interface completa (EN CURSO)

> Principio: **ypyquant = toda la inteligencia** | **TFM = thin service layer**

### M-DOC — Documentación (COMPLETADA ✅ 2026-03-15)
- [x] PRD.md actualizado con arquitectura Fases 3+4
- [x] README.md (Sistema_trading) reescrito
- [x] README.md (TFM) sección ypyquant añadida
- [x] project_fase34_status.md creado
- [x] MEMORY.md actualizado

### M0 — Migrar execution TFM → ypyquant.execution/ ✅ COMPLETO (2026-03-15)
- [x] Crear `ypyquant/execution/` con estructura completa
- [x] Migrar broker_interface.py (limpiado, sin dependencias TFM)
- [x] Migrar paper_broker.py (import actualizado a ypyquant.execution)
- [x] Migrar protection_manager.py
- [x] Migrar wallet_manager.py
- [x] Migrar trade_analytics.py (adaptado a StorageInterface, sin SQLAlchemy)
- [x] Adaptar bot_executor.py — usa StorageInterface + ypyquant.portfolio.rebalance_service
- [x] storage_interface.py — ABC abstracta para desacoplar de ORM
- [x] Exportar en ypyquant/__init__.py (v0.2.0)
- [x] Añadir `# DEPRECATED` en tfm/app/services/trading/*.py
- [x] tests/test_execution.py — 36/36 OK
- [x] tests/test_fase2.py → 40/40 siguen pasando

### M1 — ypyquant como librería compartida en TFM
- [ ] `pip install -e ../Sistema_trading/ypyquant` en venv TFM
- [ ] Añadir a tfm/requirements.txt: `-e ../Sistema_trading/ypyquant`
- [ ] Crear `tfm/app/services/quant_bridge.py` (load_prices, run_signals, run_backtest, save_backtest, get_bot_status)
- [ ] tests/test_quant_bridge.py con datos sintéticos

### M2 — Nautilus Trader + Alpaca setup
- [ ] `pip install nautilus_trader` (nautilus_trader>=1.190)
- [ ] Crear `tfm/nautilus_service/` con estructura completa
- [ ] Implementar YpyquantStrategy(Strategy) en Nautilus
- [ ] Implementar db_writer.py (fills/posiciones → PostgreSQL TFM)
- [ ] Crear tablas TFM: `paper_account`, `bot_log`, `strategy_config`
- [ ] Alembic migration para las 3 tablas nuevas
- [ ] config.py Nautilus (AlpacaDataClientConfig + AlpacaExecClientConfig)
- [ ] run.py entry point (--strategy-id, SIGTERM handling)
- [ ] tests/test_strategy.py con Nautilus BacktestEngine (sin Alpaca)

### M3 — TFM FastAPI endpoints
- [ ] app/routers/trading.py (status, positions, orders, pnl, account, bot/start, bot/stop, bot/logs)
- [ ] app/routers/strategies.py (CRUD + activate + backtest + signals)
- [ ] app/services/nautilus_manager.py (start/stop proceso Nautilus)
- [ ] app/schemas/trading.py + app/schemas/strategies.py
- [ ] APScheduler job: actualizar precios posiciones cada 60s
- [ ] Registrar nuevos routers en app/main.py

### M4 — React frontend sección Trading
- [ ] Añadir rutas en App.tsx (/trading, /strategies)
- [ ] Actualizar menú navegación Layout.tsx
- [ ] pages/Trading/TradingOverview.tsx (KPI cards + equity curve + bot status)
- [ ] pages/Trading/Positions.tsx (tabla con P&L)
- [ ] pages/Trading/Orders.tsx (historial paginado)
- [ ] pages/Trading/BotControl.tsx (start/stop + log viewer)
- [ ] pages/Trading/PnL.tsx (equity curve vs benchmark + drawdown)
- [ ] pages/Strategies/StrategiesList.tsx
- [ ] pages/Strategies/StrategyBuilder.tsx (wizard 3 pasos)
- [ ] pages/Strategies/BacktestResults.tsx
- [ ] components/trading/ (EquityCurve, DrawdownChart, MetricsGrid, PositionsTable, OrdersTable, BotStatusBadge)
- [ ] services/api.ts — todos los endpoints de trading + strategies

### M5 — Docker Compose + Hetzner
- [ ] docker-compose.yml (postgres, redis, fastapi, nautilus, frontend, caddy)
- [ ] Dockerfile multi-stage para fastapi (incluye ypyquant)
- [ ] Dockerfile.nautilus para nautilus service
- [ ] .env.example con todas las variables (ALPACA_*, DATABASE_URL, SECRET_KEY)
- [ ] scripts/deploy.sh (pull → build → up → health check → rollback)
- [ ] Caddy config (HTTPS automático, /api → fastapi, / → frontend)

## Fase 5 — Producción real
- [ ] Interactive Brokers cuenta real
- [ ] Mismo YpyquantStrategy — solo cambiar adapter en config (no reescribir)
- [ ] TimescaleDB consolidado
- [ ] Sign-off process antes de cada nueva estrategia
