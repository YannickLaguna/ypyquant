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
- `tests/` — 40/40 tests pasando (Fase 1+2)

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
# Setup
cd Sistema_trading
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
pip install -e ypyquant/

# Tests
python -X utf8 tests/test_fase2.py   # 40/40

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
