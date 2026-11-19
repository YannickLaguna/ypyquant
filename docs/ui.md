# Interfaz interna (UI) de Sistema_trading

Herramienta local para **lanzar, comparar y monitorizar** sistemas de trading. Las estrategias se escriben en Python (como en LEAN, NautilusTrader, Freqtrade o Jesse); la UI no tiene constructor visual de reglas.

## Arquitectura

```
estrategias_usuario/*.py ──┐
src/ypyquant/estrategias/  │  núcleo (sin UI): Estrategia, registro, experimentos, almacén
  base.py        contrato Estrategia + adaptador al BacktestEngine
  registro.py    @registrar_estrategia, descubrimiento automático
  incluidas.py   pesos_iguales, momentum, bayes_stein, michaud, bl_prior, robusta_elipsoide, robusta_caja, preset_ypyquant
  experimentos.py ConfiguracionExperimento (huella), cargar_datos_experimento + ejecutar_con_datos (= ejecutar_experimento), expandir_rejilla, validadores
  procedencia.py commit, versiones y hash de precios de cada experimento
  almacen.py     Parquet + DuckDB en data/experimentos/<id>/
  barridos.py    ConfiguracionBarrido, AlmacenBarridos (data/barridos/<id>/), matriz de retornos, superficie y meseta
  validacion_barridos.py  DSR, PBO, rachas, walk-forward y permutación del barrido
  reservado.py   periodo reservado con contador de usos (data/reservado.json)
  precios.py     yfinance con caché Parquet (data/cache_precios) y precios sintéticos
app/api/   FastAPI fina (sin lógica de negocio), 127.0.0.1, sin autenticación
app/web/   React 18 + TypeScript + Vite + Tailwind + TanStack Query + lightweight-charts
pipelines/paper_trading/   bot (subproceso) que acepta estrategias del registro
```

Cada experimento vive en `data/experimentos/<estrategia>-<huella10>/`: `config.json`, `estado.json`, `metricas.json`, `procedencia.json`, `equity.parquet`, `pesos.parquet`, `operaciones.parquet`. La misma configuración da el mismo id (se reutiliza el resultado; `?forzar=true` recalcula). `AlmacenExperimentos.ejecutar_sql(...)` expone vistas DuckDB `equity`, `pesos` y `operaciones` para análisis y validaciones futuras.

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
| POST | `/api/experimentos/barrido` | `{config, rejilla: {param: [valores]}}`; crea también el barrido y devuelve `barrido_id` |
| GET | `/api/validadores` | Validadores por experimento (`config.validaciones`): `rachas`, `psr` |
| POST / GET | `/api/barridos`, `/api/barridos/{id}` | Crear (202) / lista y detalle de barridos (ver contratos abajo) |
| DELETE | `/api/barridos/{id}?con_experimentos=` | Borra el barrido (y sus experimentos si `true`) |
| POST | `/api/barridos/{id}/{validar,walk_forward,reservado}` | Validación síncrona (segundos) |
| POST | `/api/barridos/{id}/permutacion[/cancelar]` | Test de permutación en segundo plano (202) y su cancelación |
| GET | `/api/experimentos?estrategia=&estado=&texto=` | Lista |
| GET / DELETE | `/api/experimentos/{id}` | Métricas, equity, drawdown, pesos, operaciones; borrar |
| GET | `/api/comparar?ids=a,b` | Equity base 100 + métricas (DuckDB) |
| GET | `/api/paper/{estado,cuenta,posiciones,ordenes,historial,fills}` | Alpaca paper; **503** sin claves, 502 si Alpaca falla |
| POST | `/api/paper/bot/{arrancar,parar}` | Subproceso `python -m pipelines.paper_trading --estrategia ...` |
| GET | `/api/paper/bot/{estado,log?desde=N}` | Estado y log (polling por línea) |

## Procedencia de un experimento

`procedencia.json` (y la clave `procedencia` en `GET /api/experimentos/{id}`) guarda con qué se obtuvo el resultado. **No entra en la huella**: misma configuración = mismo id aunque se ejecute otro día o con otro commit.

```json
{"commit": "da8e852d...", "arbol_sucio": false, "version_ypyquant": "0.5.0", "python": "3.11.9",
 "numpy": "2.4.6", "pandas": "2.3.3", "cvxpy": "1.9.3",
 "hash_precios": "9a8d0db5...(SHA-256)", "ejecutado": "2026-11-19T10:20:43"}
```

`commit` y `arbol_sucio` son `null` si la librería no está en un repo git (instalada en otra app). `arbol_sucio` solo cuenta cambios en archivos versionados y se calcula una vez por proceso. `hash_precios` es el SHA-256 de los precios usados (por ticker ordenado y el benchmark): dos ejecuciones con el mismo hash vieron exactamente los mismos datos. Los experimentos anteriores a v0.5.0 devuelven `procedencia: null`.

## Métricas añadidas (v0.5.0)

`ulcer` (índice Ulcer, %), `martin` (ratio de Martin = CAGR % / Ulcer %) y `costes_sobre_ganancia_bruta` = comisiones_totales / (capital_final − capital_inicial + comisiones_totales); `null` si la ganancia bruta es ≤ 0. **Un valor > 0,5 indica que los costes se comen la mitad del edge.**

## Barridos y validación

Un **barrido** es una rejilla de experimentos con identidad propia (`barrido-<10 hex>`, misma configuración = mismo id) en `data/barridos/<id>/`: `barrido.json` (configuración, combinaciones con su id de experimento y su posición en la rejilla, estado, progreso y tareas), `validacion.json`, `walk_forward.json` (+ `walk_forward_oos.parquet`), `permutacion_in_sample.json` / `permutacion_walk_forward.json` (+ `.parquet`) y `reservado.json`. Los experimentos siguen en `data/experimentos/`; un barrido solo guarda sus ids.

**Periodo reservado.** Con `fecha_fin_desarrollo` todas las combinaciones se ejecutan hasta esa fecha y el tramo (`fecha_fin_desarrollo`, `base.fecha_fin`] queda reservado: ninguna validación lo toca (la permutación ni siquiera carga esos precios). `POST .../reservado` ejecuta UNA combinación allí (con la misma historia previa) y **cuenta el uso** en `data/reservado.json`, con clave (universo resuelto ordenado, `fecha_fin_desarrollo`, `fecha_fin`): si ya se había mirado, la respuesta trae `aviso` explícito (el reservado deja de ser fuera de muestra por sesgo de selección). El contador no lo impide: lo hace visible.

**Matriz de retornos.** Fechas × combinaciones (columna = id de experimento; `etiqueta` en `combinaciones`). Si las combinaciones no cubren las mismas fechas (p. ej. `fecha_inicio` en la rejilla) se **recorta al tramo común** (`relleno: "recortar"`, por defecto) o se rellena con 0 (`"cero"`). Con la misma `fecha_inicio` no hay NaN.

**Superficie y meseta.** Cada fila trae la métrica objetivo y su **meseta**: media de la métrica objetivo de los vecinos (± 1 posición en un eje, el resto igual; la propia combinación no cuenta). Un pico rodeado de malos resultados puntúa bajo; una zona amplia de buenos, alto. Es la guía para elegir el centro de una meseta en vez del máximo aislado.

**Métricas objetivo admitidas** (mayor es mejor): `sharpe` (defecto), `sortino`, `calmar`, `martin`, `cagr`, `retorno_total`, `max_drawdown`, `information_ratio`, `alfa`, `tasa_acierto`.

**Convenciones de la validación.** `ventana_entreno_dias` y `paso_dias` son **días naturales** (se convierten a sesiones con la densidad real, ~0,69 sesiones por día natural). En la permutación `walk_forward` la selección usa siempre el Sharpe por periodo; en `in_sample` el estadístico es la mejor métrica objetivo entre combinaciones. Límite: 1000 permutaciones; aviso si combinaciones × permutaciones > 20 000 backtests. Con `procesos > 1` cada worker (spawn) carga las estrategias incluidas y redescubre `estrategias_usuario/`. Referencia de coste: un backtest de 10 años y 20 activos con rebalanceo mensual ronda 1–1,5 s; una permutación de un barrido de N combinaciones cuesta ≈ N backtests.

### Contratos JSON

Todos los errores son `{"detail": <texto o lista>}`: **404** barrido o estrategia inexistente, **409** experimentos sin terminar (o tarea ya en curso), **422** parámetros inválidos (mensaje legible). Números no finitos se serializan como `null`.

**`POST /api/barridos?forzar=false`** → 202
```json
// cuerpo (ConfiguracionBarrido)
{"base": {"estrategia": "momentum", "universo": ["AAPL","MSFT"], "fecha_inicio": "2015-01-01", "fecha_fin": "2024-12-31"},
 "rejilla": {"top_k": [1, 2, 3], "meses_retorno": [3, 6, 12]},
 "fecha_fin_desarrollo": "2022-12-31",   // opcional (null = sin periodo reservado)
 "metrica_objetivo": "sharpe", "etiqueta": ""}
// respuesta
{"id": "barrido-71237667dd", "ids": ["momentum-56cb4a4185", "..."], "total": 9, "reutilizado": false}
```
`base.fecha_fin` es el fin de TODO el periodo; con `fecha_fin_desarrollo` los experimentos se ejecutan hasta esa fecha.

**`GET /api/barridos`** → `{"barridos": [{"id", "etiqueta", "estrategia", "universo", "ejes": ["top_k", ...], "n_combinaciones", "metrica_objetivo", "fecha_fin_desarrollo", "estado", "progreso", "resultados": ["validacion", ...], "creado"}]}` (más recientes primero).

**`GET /api/barridos/{id}`**
```json
{"id": "barrido-71237667dd", "config": {"base": {...}, "rejilla": {...}, "fecha_fin_desarrollo": "...", "metrica_objetivo": "sharpe", "etiqueta": ""},
 "estado": "pendiente|en_curso|hecho|error",   // de los experimentos: error = terminó con algún fallo
 "progreso": {"total": 9, "hechos": 9, "errores": 0, "en_curso": 0, "pendientes": 0},
 "creado": "...", "actualizado": "...",
 "ejes": {"top_k": [1, 2, 3], "meses_retorno": [3, 6, 12]},
 "combinaciones": [{"id": "momentum-56cb4a4185", "etiqueta": "top_k=1, meses_retorno=3", "valores": {"top_k": 1, "meses_retorno": 3},
                    "posiciones": {"top_k": 0, "meses_retorno": 0}, "estado": "hecho"}],
 "superficie": [{"id": "momentum-56cb4a4185", "etiqueta": "...", "valores": {...}, "posiciones": {...},
                 "metricas": {"sharpe": -0.28, "cagr": -0.07, "ulcer": 19.3, "martin": -0.38, "...": 0},
                 "objetivo": -0.28, "meseta": -0.05, "n_vecinos": 2}],   // mapa de calor: x,y = posiciones de dos ejes, color = objetivo o meseta
 "tareas": {"permutacion_in_sample": {"estado": "pendiente|en_curso|hecho|error|cancelado", "progreso": {"completadas": 3, "total": 100},
            "p_valor": 0.04, "error": null, "parametros": {...}, "actualizado": "..."}},
 "validacion": null, "walk_forward": null, "permutacion_in_sample": null, "permutacion_walk_forward": null,   // resultados completos o null
 "reservado": [], "reservado_usos": [], "periodo_reservado": {"inicio": "2023-01-01", "fin": "2024-12-31"}}
```
Con un barrido en curso `superficie` contiene solo las combinaciones ya terminadas. Para un mapa de calor de 2 ejes, `posiciones` da la celda; con más ejes, fija los demás.

**`DELETE /api/barridos/{id}?con_experimentos=false`** → 204 (404 si no existe). Los experimentos se conservan salvo `con_experimentos=true` (cuidado: un experimento puede pertenecer a varios barridos).

**`POST /api/barridos/{id}/validar`** → 200 (síncrono). Cuerpo opcional `{"n_bloques": 16, "semilla": 0, "relleno": "recortar"}`.
```json
{"fecha": "...", "metrica_objetivo": "sharpe", "n_combinaciones": 9, "n_observaciones": 390, "fecha_inicio": "...", "fecha_fin": "...", "relleno": "recortar",
 "elegida": {"id": "...", "etiqueta": "...", "valores": {...}, "objetivo": 0.02, "meseta": -0.36, "n_vecinos": 2, "metricas": {...}},
 "dsr": {"dsr": 0.46, "sr": 0.0093, "sr_umbral": 0.0146, "sr_anualizado": 0.148, "n_pruebas": 9, "T": 390, "asimetria": 0.09, "curtosis": 3.46},
 "pbo": {"pbo": 0.74, "pendiente": -1.06, "prob_perdida_oos": 0.73, "n_combinaciones": 5000, "n_bloques": 16, "n_configuraciones": 9,
         "lambdas": [-1.39, "..."], "pares_is_oos": [[0.025, -0.053], "..."]},   // histograma de λ y dispersión IS vs OOS; pbo = null si T es muy pequeño
 "rachas": {"rachas": 211, "mu": 195.3, "sigma": 9.8, "z": 1.6, "p_valor": 0.11, "n_positivos": 183, "n_negativos": 207},
 "superficie": [ ...igual que en el detalle... ],
 "alternativa_meseta": {"id": "...", "etiqueta": "...", "valores": {...}, "meseta": -0.05, "objetivo": -0.28, "es_la_elegida": false},
 "avisos": ["PBO con 12 bloques en lugar de 16 (T=..., ...)"]}
```
Umbrales orientativos en `docs/validacion.md` (DSR > 0,95; PBO < 0,10–0,20; pendiente OOS~IS positiva). 409 si quedan experimentos sin terminar; 422 si hay menos de 2 combinaciones terminadas bien.

**`POST /api/barridos/{id}/walk_forward`** → 200. Cuerpo `{"ventana_entreno_dias": 756, "paso_dias": 90}` (días naturales).
```json
{"fecha": "...", "ventana_entreno_dias": 150, "paso_dias": 60, "ventana_filas": 107, "paso_filas": 43, "n_combinaciones": 9,
 "sharpe_periodo": -0.04, "metricas": {"sharpe_anualizado": -0.63, "cagr": -0.13, "max_drawdown": -0.24, "retorno_total": -0.14, "n_observaciones": 283},
 "elecciones": [{"fecha": "2020-06-01", "id": "...", "etiqueta": "top_k=2, meses_retorno=3", "metrica_entreno": 0.11}],
 "curva": {"fechas": ["2020-06-01", "..."], "equity": [1.0117, "..."]}}
```
Aproximación: ignora el coste de cambiar de combinación (optimista). 422 si la ventana no cabe en el periodo.

**`POST /api/barridos/{id}/permutacion`** → 202. Cuerpo (todo opcional salvo lo que exige el modo):
```json
{"n_permutaciones": 100, "longitud_bloque": 1, "modo": "in_sample|walk_forward",
 "ventana_entreno_dias": null, "paso_dias": null,   // obligatorios con modo walk_forward
 "semilla": 0, "procesos": 1}
// respuesta
{"tarea": "permutacion_in_sample", "estado": "pendiente", "n_backtests": 900, "aviso": null}   // aviso si n_backtests > 20000
```
Seguimiento: `GET /api/barridos/{id}` → `tareas.permutacion_<modo>` (`estado`, `progreso.completadas/total`) y, al terminar, el resultado en `permutacion_<modo>`:
```json
{"fecha": "...", "modo": "in_sample", "estadistico": "mejor_sharpe", "metrica_objetivo": "sharpe", "n_permutaciones": 100, "n_validas": 100,
 "longitud_bloque": 1, "semilla": 0, "procesos": 1, "ventana_entreno_dias": null, "paso_dias": null, "inicio_permutacion": null,
 "n_combinaciones": 9, "n_backtests": 900, "valor_real": 0.02, "p_valor": 0.75, "distribucion": [1.87, "..."],   // histograma + línea en valor_real
 "media_permutadas": 0.60, "p95_permutadas": 1.72, "duracion_s": 93.1, "avisos": []}
```
`p_valor` = (1 + #{permutadas ≥ real}) / (1 + n_validas). 409 si ya hay una igual en curso o experimentos sin terminar; 422 si `n_permutaciones` no está en 1..1000.
**`POST /api/barridos/{id}/permutacion/cancelar`** cuerpo `{"modo": "in_sample"}` → `{"tarea": "...", "cancelando": true}` (409 si no hay ninguna en curso). La tarea pasa a `cancelado` al acabar la permutación en curso. Las tareas que quedan `en_curso` al reiniciar el servidor pasan a `error`.

**`POST /api/barridos/{id}/reservado`** → 200 (síncrono, un backtest). Cuerpo `{"experimento": "<id o etiqueta de la combinación>"}`.
```json
{"fecha": "...", "experimento_reservado": "momentum-d34380e33b",   // guardado como experimento normal (equity, métricas, procedencia; usable en /api/comparar)
 "combinacion": {"id": "...", "etiqueta": "...", "valores": {...}, "posiciones": {...}},
 "periodo": {"inicio": "2023-01-01", "fin": "2024-12-31"}, "metricas": {...},
 "usos_previos": [{"fecha": "2026-11-19T10:20:46", "barrido": "barrido-...", "experimento": "momentum-..."}],
 "n_usos": 2, "aviso": "Este periodo reservado (...) ya se había usado 1 vez: deja de ser fuera de muestra (sesgo de selección). ...", "avisos": []}
```
`aviso` es `null` la primera vez. 422 si el barrido no tiene `fecha_fin_desarrollo` o la referencia no existe.

**`POST /api/experimentos/barrido`** (compatible) → `{"ids": [...], "total": 3, "barrido_id": "barrido-..."}`; `barrido_id` es `null` si no hay rejilla.

## Paper trading con estrategias del registro

```bash
uv run python -m pipelines.paper_trading --estrategia momentum --parametros '{"top_k": 3}' --tickers AAPL,MSFT,NVDA,AMZN
```

Sin `--estrategia` el bot sigue funcionando como antes (señales técnicas, `--config estrategia_ejemplo.json`). En modo registro cada iteración pide los pesos a la estrategia y envía órdenes de rebalanceo (`decidir_ordenes_rebalanceo`, ventas primero).

## Tests

`uv run pytest` (Python, sin red), `cd app/web && npx tsc --noEmit && npx vitest run && npm run build`.

## Limitaciones conocidas

- Long-only y sin efectivo parcial (el motor normaliza los pesos a 1).
- Las validaciones del barrido (DSR, PBO, permutación) cuentan solo las combinaciones de la rejilla: si probaste más cosas a mano, N real es mayor (ver `docs/validacion.md`). El PBO-CSCV y el walk-forward sobre la matriz ignoran el coste de cambiar de combinación.
- CPCV no está implementado (sí DSR, PBO-CSCV, rachas, permutación y walk-forward).
- Sin autenticación: solo para `127.0.0.1`. El bot es hijo de la API; si la API muere en Windows puede seguir vivo.
