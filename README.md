# Sistema Trading

> **Propósito y prioridades: la fuente de verdad es la nota del proyecto en Obsidian**
> ([abrir en Obsidian](obsidian://open?vault=digiyann&file=1%20-%20%F0%9F%9B%A0%EF%B8%8F%20Organizaci%C3%B3n%2F14%20-%20%E2%9C%94%EF%B8%8FProyectos%2FProfesionales%2F2026-01-07%20Sistema%20de%20Trading) · `digiyann/1 - 🛠️ Organización/14 - ✔️Proyectos/Profesionales/2026-01-07 Sistema de Trading.md`).
> Este repo documenta el *cómo* (arquitectura, módulos, comandos). Si algo de aquí contradice la nota, manda la nota.
> Auditoría y plan vigente (2026-11-16): bitácora `5 - Código/AI/44 - Bitácoras/2026-11-16 Sistema trading - auditoría y plan.md`.

Sistema cuantitativo end-to-end. Este repo contiene **ypyquant**, la librería Python compartida con toda la inteligencia de trading: datos, señales, optimización, ML y ejecución.

---

## Qué es este repo

- `ypyquant/` — librería Python instalable (`pip install -e ypyquant/`)
- `data/` — scripts de ingesta (fetcher.py, factors.py, fred.py, loader.py)
- `db/` — ORM SQLAlchemy para Supabase (database.py + models.py)
- `scripts/` — pipeline completo (run_fase2.py, vbt_signal_scan.py, etc.)
- `tests/` — pytest real: `python -m pytest` (offline, sin red; marcador `red` para los de red/BD)

GrowSave (repo `tfm`) **no forma parte del sistema**: es un producto de ahorro aparte que solo reutiliza ypyquant (optimización de carteras, Monte Carlo y perfiles de riesgo). El trading que todavía vive allí se migrará aquí (ver bitácora).

---

## Estado actual

| Fase | Estado | Descripción |
|---|---|---|
| Fase 1 | COMPLETA | fetcher.py → Supabase + interfaces ypyquant (19/19 tests) |
| Fase 2 | COMPLETA | signals/ + MLflow + VectorBT + walk-forward pipeline (40/40 tests) |
| Fase 3+4 | EN CURSO | Paper trading + Interface TFM (M-DOC completado, M0 pendiente) |
| Fase 5 | Pendiente | IB producción real |

---

## Arquitectura

```
ypyquant (este repo)  →  importado por  →  TFM (FastAPI + React)
                                                  ↓
                                         Nautilus Trader (ejecución)
                                                  ↓
                                         Alpaca paper API → IB (Fase 5)
```

Ver `PRD.md` §3 para la arquitectura completa y `project_fase34_status.md` para el plan detallado.

---

## Cómo correr el pipeline (Fase 2)

```bash
# Setup (uv; crea .venv con Python 3.11 e instala la librería editable + dev + entorno)
cd Sistema_trading
uv venv --python 3.11
uv pip install -e . --group dev --group entorno
uv pip install "mlflow>=2.10"   # opcional: activa los tests de MLflow (si no, se saltan)

# Tests (pytest real; por defecto sin red ni BD, ~10 s)
.venv/Scripts/python.exe -m pytest              # todo lo offline
.venv/Scripts/python.exe -m pytest -m red       # solo los que necesitan red o la BD Supabase
.venv/Scripts/python.exe -m pytest -m "red or not red"   # todo
.venv/Scripts/ruff.exe check tests              # lint de tests (debe quedar limpio)

# Los tests marcados `red` (marcador en pyproject.toml) descargan de internet
# (yfinance, FRED, Wikipedia, Kenneth French) o usan la BD Supabase, que ya no existe:
# sin BD se saltan con el motivo; fallan solo si la red/BD responde mal. Acceso a la BD: tests/_bd.py.

# Pipeline completo
python scripts/run_fase2.py

# MLflow UI
mlflow ui --backend-store-uri data/mlruns
```

---

## ypyquant — módulos

```
ypyquant/
├── signals/        # TechnicalSignalGenerator, MLSignalGenerator, SignalCombiner... ✅
├── optimization/   # WalkForwardEngine, MLflowTracker, ExportadorResultados ✅
├── portfolio/      # BacktestEngine, PerformanceMetrics ✅
├── ml/             # MarketRegimeClassifier, MLPredictor, LSTM, RL ✅
├── factors/        # FamaFrenchModel (FF5+MOM) ✅
├── data/           # Indicadores técnicos, datos FF, yield curves ✅
└── execution/      # BrokerInterface, PaperBroker, BotExecutor... ← M0 pendiente
```

---

## Interfaz (UI) y experimentos

Interfaz local (API FastAPI + React) para lanzar, comparar y monitorizar sistemas: **Estrategias**, **Experimentos**, **Comparar** y **Paper trading**. Las estrategias se escriben en Python en `estrategias_usuario/` (descubrimiento automático) sobre `ypyquant.estrategias`. Detalle, endpoints y cómo escribir una estrategia en 5 pasos: [`docs/ui.md`](docs/ui.md).

```bash
uv sync --extra ui --group dev --group entorno
uv run --extra ui uvicorn app.api.main:app --port 8010      # API en 127.0.0.1
cd app/web && npm install && npm run dev                     # http://127.0.0.1:5173 (proxy a la API)
```

## Paper trading

CLI en `pipelines/paper_trading/` (cuenta **paper** de Alpaca; no opera con dinero real). La librería está en `src/ypyquant/execution/nautilus/`. Las dependencias (`nautilus_trader`, `alpaca-py`, `python-dotenv`, `yfinance`) son el extra opcional `trading`, para que GrowSave no las arrastre.

```bash
uv sync --extra trading                       # instala el extra
cp .env.example .env                          # y rellena ALPACA_API_KEY / ALPACA_SECRET_KEY (claves paper)

uv run python -m pipelines.paper_trading --backtest          # backtest de las señales (sin claves de Alpaca)
uv run python -m pipelines.paper_trading --una-vez           # una iteración en vivo contra Alpaca paper
uv run python -m pipelines.paper_trading                     # bucle continuo (Ctrl+C para parar)
uv run python -m pipelines.paper_trading --config pipelines/paper_trading/estrategia_ejemplo.json
```

La estrategia (tickers, generadores de señales, umbral, riesgo) se define en un JSON (ver `estrategia_ejemplo.json`). Las órdenes son de mercado DAY y solo se envían con el mercado abierto. Los fills se guardan en `data/paper_trading/fills.jsonl`. Variables opcionales: `NAUTILUS_POLL_INTERVAL` (segundos entre iteraciones) y `NAUTILUS_LOOKBACK_DAYS`.

---

## Próximas sesiones (Fase 3+4)

Orden de ejecución:

```
M0 — Migrar tfm/app/services/trading/ → ypyquant/execution/
  ↓
M1 — pip install -e ypyquant en TFM + quant_bridge.py
  ↓
M2 — Nautilus Trader + Alpaca paper + tablas DB
  ↓
M3 — TFM FastAPI endpoints (trading.py, strategies.py)
  ↓
M4 — React frontend sección Trading
  ↓
M5 — Docker Compose + Hetzner deploy
```

Ver `TODO.md` para checklist detallado y `project_fase34_status.md` para pre-requisitos y verificación end-to-end.
