# Interfaz interna (UI) de Sistema_trading

Herramienta local para **lanzar, comparar y monitorizar** sistemas de trading. Las estrategias se escriben en Python (como en LEAN, NautilusTrader, Freqtrade o Jesse); la UI no tiene constructor visual de reglas.

## Arquitectura

```
estrategias_usuario/*.py ──┐
src/ypyquant/estrategias/  │  núcleo (sin UI): Estrategia, registro, experimentos, almacén
  base.py        contrato Estrategia + adaptador al BacktestEngine
  registro.py    @registrar_estrategia, descubrimiento automático
  incluidas.py   pesos_iguales, momentum, bayes_stein, michaud, bl_prior, preset_ypyquant
  experimentos.py ConfiguracionExperimento (huella), ejecutar_experimento, expandir_rejilla, validadores
  almacen.py     Parquet + DuckDB en data/experimentos/<id>/
  precios.py     yfinance con caché Parquet (data/cache_precios) y precios sintéticos
app/api/   FastAPI fina (sin lógica de negocio), 127.0.0.1, sin autenticación
app/web/   React 18 + TypeScript + Vite + Tailwind + TanStack Query + lightweight-charts
pipelines/paper_trading/   bot (subproceso) que acepta estrategias del registro
```

Cada experimento vive en `data/experimentos/<estrategia>-<huella10>/`: `config.json`, `estado.json`, `metricas.json`, `equity.parquet`, `pesos.parquet`, `operaciones.parquet`. La misma configuración da el mismo id (se reutiliza el resultado; `?forzar=true` recalcula). `AlmacenExperimentos.ejecutar_sql(...)` expone vistas DuckDB `equity`, `pesos` y `operaciones` para análisis y validaciones futuras.

## Escribir una estrategia en 5 pasos

1. Crea `estrategias_usuario/mi_estrategia.py` (los archivos que empiezan por `_` se ignoran).
2. Hereda de `Estrategia`, decórala con `@registrar_estrategia` y pon `nombre` y `descripcion`.
3. Declara los parámetros en una subclase `Parametros(ParametrosBase)` con `Field(default, ge=, le=, description=)`; la UI genera el formulario desde su JSON Schema.
4. Implementa `generar_pesos(fecha, datos, estado) -> {ticker: peso}`. `datos` ya llega recortado a `<= fecha` (sin look-ahead); `estado` es un dict que persiste entre rebalanceos.
5. Reinicia la API o pulsa `POST /api/estrategias/recargar`: aparece en la UI.

```python
from pydantic import Field
from ypyquant.estrategias import Estrategia, ParametrosBase, matriz_cierres, registrar_estrategia

@registrar_estrategia
class InversaVolatilidad(Estrategia):
    nombre = "inversa_volatilidad"
    descripcion = "Pondera por la inversa de la volatilidad reciente."
    frecuencia_rebalanceo = "monthly"

    class Parametros(ParametrosBase):
        ventana_dias: int = Field(60, ge=10, le=500, description="Días para medir la volatilidad")

    def generar_pesos(self, fecha, datos, estado):
        retornos = matriz_cierres(datos).pct_change().iloc[-self.parametros.ventana_dias:]
        inversa = 1.0 / retornos.std().dropna()
        return (inversa / inversa.sum()).to_dict()
```

Ejemplo completo en `estrategias_usuario/ejemplo_inversa_volatilidad.py`. Ganchos opcionales: `preparar(datos)` (una vez, con todo el histórico: úsalo solo para cálculos causales), `semilla` (`self.semilla`, la fija el experimento) y `frecuencia_rebalanceo`. Los pesos se sanean (long-only, normalizados a 1).

## Arrancar

```bash
# Una vez (desde la raíz del repo)
uv sync --extra ui --group dev --group entorno
cd app/web && npm install && cd ../..

# API (solo 127.0.0.1)
uv run --extra ui uvicorn app.api.main:app --port 8010

# Frontend en desarrollo (proxy /api -> 127.0.0.1:8010)
cd app/web && npm run dev          # http://127.0.0.1:5173

# O compilado: la API sirve app/web/dist en http://127.0.0.1:8010
cd app/web && npm run build
```

Variables: `YPYQUANT_PRECIOS=sintetico` (precios sin red, para probar), `YPYQUANT_OFFLINE=1` (no descarga; usa la caché), `YPYQUANT_ESTRATEGIAS_USUARIO=<carpeta>`, `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (cuenta **paper**, en `.env`). Docker opcional: `docker compose up` (ver `docker-compose.yml`).

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/salud` | Estado |
| GET | `/api/estrategias`, `/api/estrategias/{nombre}` | Catálogo con JSON Schema de parámetros; detalle con código fuente |
| POST | `/api/estrategias/recargar` | Redescubre `estrategias_usuario/` |
| GET | `/api/universos` | Universos de `universos/` |
| POST | `/api/experimentos` | Lanza en segundo plano (202); 404 estrategia, 422 parámetros |
| POST | `/api/experimentos/barrido` | `{config, rejilla: {param: [valores]}}` |
| GET | `/api/experimentos?estrategia=&estado=&texto=` | Lista |
| GET / DELETE | `/api/experimentos/{id}` | Métricas, equity, drawdown, pesos, operaciones; borrar |
| GET | `/api/comparar?ids=a,b` | Equity base 100 + métricas (DuckDB) |
| GET | `/api/paper/{estado,cuenta,posiciones,ordenes,historial,fills}` | Alpaca paper; **503** sin claves, 502 si Alpaca falla |
| POST | `/api/paper/bot/{arrancar,parar}` | Subproceso `python -m pipelines.paper_trading --estrategia ...` |
| GET | `/api/paper/bot/{estado,log?desde=N}` | Estado y log (polling por línea) |

## Paper trading con estrategias del registro

```bash
uv run python -m pipelines.paper_trading --estrategia momentum --parametros '{"top_k": 3}' --tickers AAPL,MSFT,NVDA,AMZN
```

Sin `--estrategia` el bot sigue funcionando como antes (señales técnicas, `--config estrategia_ejemplo.json`). En modo registro cada iteración pide los pesos a la estrategia y envía órdenes de rebalanceo (`decidir_ordenes_rebalanceo`, ventas primero).

## Tests

`uv run pytest` (Python, sin red), `cd app/web && npx tsc --noEmit && npx vitest run && npm run build`.

## Limitaciones conocidas

- El motor solo rebalancea los días en que cae la fecha programada (`MS`, `QS`, `W-MON`): si el día 1 no es sesión, ese rebalanceo se pierde (un mensual de 3 años da ~27 en vez de 36). Pendiente de decidir si se corrige en `BacktestEngine`.
- Long-only y sin efectivo parcial (el motor normaliza los pesos a 1).
- La validación (CPCV, DSR, PBO) no está implementada: usa `registrar_validador` y `config.validaciones`.
- Sin autenticación: solo para `127.0.0.1`. El bot es hijo de la API; si la API muere en Windows puede seguir vivo.
