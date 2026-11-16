# Fase 2 — Estado del proyecto (2026-03-15)

Análogo a `project_fase1_status.md`. Documenta qué se implementó, dónde vive
y cómo verificarlo.

---

## Resumen ejecutivo

Pipeline de research end-to-end implementado sobre Fase 1.
Zipline DESCARTADO (Python 3.11+/NumPy 2.x). MLflow y VectorBT opcionales.

**Pipeline:**
```
[1] DATA          data/loader.py → Supabase
[2] SIGNALS       ypyquant/signals/ → SignalFrame [-1, +1]
[3] VBT SCAN      scripts/vbt_signal_scan.py → ranking Sharpe bruto
[4] BACKTEST      ypyquant/portfolio/backtest_engine.py → fills realistas
[5] WALK-FORWARD  ypyquant/optimization/walk_forward.py → 21 exp OOS
[6] PERSIST       exportar_a_db() + tabla signals
```

---

## M1 — ypyquant/signals/ (COMPLETO)

| Archivo | Descripción |
|---|---|
| `signals/__init__.py` | API pública: re-exporta todo |
| `signals/base.py` | `BaseSignalGenerator` (ABC) + `SignalFrame` alias |
| `signals/technical.py` | `TechnicalSignalGenerator` — RSI, MACD, Bollinger %B, Momentum 12-1 |
| `signals/ml_signal.py` | `MLSignalGenerator` — wraps MLPredictor, predict_proba → [-1,1] |
| `signals/regime_signal.py` | `RegimeSignalGenerator` — wraps MarketRegimeClassifier, filtro Bull/Bear |
| `signals/factor_signal.py` | `FactorSignalGenerator` — wraps FamaFrenchModel, alpha rank → [-1,1] |
| `signals/combiner.py` | `SignalCombiner` — ensemble ponderado (weighted_mean / median / majority_vote) |

**Verificar:**
```bash
python -c "from ypyquant.signals import TechnicalSignalGenerator, SignalCombiner; print('OK')"
```

---

## M2 — MLflow tracking (COMPLETO)

| Archivo | Cambios |
|---|---|
| `optimization/mlflow_tracker.py` | Nuevo — `MLflowTracker` con no-op guard |
| `optimization/walk_forward.py` | Modificado — `mlflow_enabled`, `mlflow_experiment_name`, `mlflow_tracking_uri` en `__init__`; hooks `start_run`, `log_window`, `end_run` en `ejecutar_experimento` |

**Convención:**
- 21 experimentos = 1 MLflow experiment `"walk_forward_factorial"`
- 1 run = 1 `ConfigExperimento`
- URI local: `file:///[ROOT]/data/mlruns` (Fase 3 migra a servidor)

**Verificar sin MLflow instalado:**
```bash
python -c "
from ypyquant import WalkForwardEngine, ConfigWalkForward
e = WalkForwardEngine(ConfigWalkForward(), mlflow_enabled=False)
print('MLflow guard OK — sin mlflow instalado')
"
```

**Verificar con MLflow:**
```bash
pip install "mlflow>=2.10"
python -c "import mlflow; print(mlflow.__version__)"
mlflow ui --backend-store-uri data/mlruns
```

---

## M3 — DB persistence (COMPLETO)

| Archivo | Cambios |
|---|---|
| `optimization/exportador_resultados.py` | Añadido `exportar_a_db(session, nombre_exp, sobrescribir=False)` |

**Contrato:**
- Tabla `backtests` (ya existente en Supabase desde Fase 1)
- `parametros` JSONB ← `ConfigExperimento` fields
- `metricas` JSONB ← `resultado.metricas` + alpha_jensen/pvalue
- `resultados` JSONB ← equity_curve + pesos_historicos (compacto)
- `universo` JSONB ← tickers únicos del experimento
- Idempotente: lookup por nombre, `sobrescribir` flag

**Verificar:**
```bash
python -c "
from db.database import SessionLocal
from db.models import Backtest
with SessionLocal() as s:
    n = s.query(Backtest).filter_by(estado='completed').count()
    print(f'{n} backtests en Supabase')
"
```

---

## M4 — VectorBT signal scanner (COMPLETO)

| Archivo | Descripción |
|---|---|
| `scripts/vbt_signal_scan.py` | `run_vbt_scan()`, `scan_and_rank()`, CLI |

**Requiere:** `pip install "vectorbt>=0.26"` o `pip install -e "ypyquant/[fast]"`

**Uso:**
```bash
python scripts/vbt_signal_scan.py \
    --universe universos/sp500.txt \
    --start 2020-01-01 \
    --end   2025-01-01 \
    --top-n 10
```

---

## M5 — Pipeline orquestador (COMPLETO)

| Archivo | Descripción |
|---|---|
| `scripts/run_fase2.py` | Orquestador stages 1-6 |
| `scripts/config_fase2.yaml` | Configuración YAML del pipeline |

**Uso completo:**
```bash
python scripts/run_fase2.py
```

**Subset rápido (testing):**
```bash
python scripts/run_fase2.py --experiments 1,2,3 --no-mlflow --no-db
```

**Sin VectorBT:**
```bash
python scripts/run_fase2.py --skip-vbt
```

---

## M6 — Package updates (COMPLETO)

| Archivo | Cambios |
|---|---|
| `ypyquant/__init__.py` | Exporta módulo signals completo |
| `ypyquant/pyproject.toml` | `[tracking]` mlflow>=2.10, `[fast]` vectorbt>=0.26 |
| `requirements.txt` | pyyaml + comentarios opcionales |

**Instalar opcionales:**
```bash
pip install -e "ypyquant/[tracking]"       # MLflow
pip install -e "ypyquant/[fast]"           # VectorBT
pip install -e "ypyquant/[tracking,fast]"  # Ambos
```

---

## Verificación end-to-end

```bash
# M1: signals
python -c "from ypyquant.signals import TechnicalSignalGenerator, SignalCombiner; print('M1 OK')"

# M2: mlflow tracker (sin instalar mlflow)
python -c "
from ypyquant.optimization.mlflow_tracker import MLflowTracker
t = MLflowTracker(enabled=False)
print('M2 no-op guard OK')
"

# M3: exportar_a_db disponible
python -c "
from ypyquant.optimization.exportador_resultados import ExportadorResultados
assert hasattr(ExportadorResultados, 'exportar_a_db'), 'método no encontrado'
print('M3 OK')
"

# M4: vbt_signal_scan importable
python -c "
import sys; sys.path.insert(0, 'scripts')
from vbt_signal_scan import run_vbt_scan, scan_and_rank
print('M4 OK (VectorBT no instalado aún)')
" 2>/dev/null || echo "M4: necesita pip install vectorbt"

# Pipeline completo (subset)
python scripts/run_fase2.py --experiments 1 --no-mlflow --no-db --skip-vbt

# MLflow UI (una vez instalado)
# mlflow ui --backend-store-uri data/mlruns
```

---

## Dependencias opcionales

| Paquete | Estado | Comando |
|---|---|---|
| mlflow>=2.10 | **INSTALADO** — mlflow.db en `data/mlflow.db` | `pip install "mlflow>=2.10"` |
| vectorbt>=0.26 | Pendiente instalar | `pip install "vectorbt>=0.26"` |
| pyyaml | Instalado | ya en requirements.txt |

---

## Tests — resultado final

```
python -X utf8 tests/test_fase2.py  →  40/40 OK
```

Bugs corregidos en esta sesión (2026-03-15):
1. `signals/base.py::_cross_sectional_rank_signal` — fórmula `2*(rank-1)/(N-1)-1` garantiza media=0
2. `optimization/exportador_resultados.py::__init__` — acepta `list` además de `dict`; fix `resultados.items()` → `self.resultados.items()`

---

## Próximos pasos → Fase 3

Ver `PRD.md §Fase 3` para detalle completo.

Prioridades inmediatas:
1. Docker Compose en Hetzner (PostgreSQL, Redis, FastAPI, Celery, MLflow server)
2. Job de ingesta diaria automatizada (Celery beat)
3. Dashboard Plotly Dash — equity curves + señales
4. Migrar MLflow tracking URI de SQLite local → servidor HTTP
