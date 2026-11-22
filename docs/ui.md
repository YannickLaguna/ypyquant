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
  configuraciones.py  ConfiguracionGuardada (receta sin fechas, versionada), AlmacenConfiguraciones (data/configuraciones/<id>/v<n>.json)
  receta.py      Entorno, Receta y la huella canónica; servicio_experimentos.py: lanzar/reutilizar/barridos/comparativas sin FastAPI; indice_recetas.py
  marcos.py      MarcoComparacion, aplicar_marco, AlmacenMarcos (data/marcos/), comparativas y diferencias_ceteris_paribus
  validacion_barridos.py  DSR, PBO, rachas, walk-forward y permutación del barrido
  reservado.py   periodo reservado con contador de usos (data/reservado.json)
  precios.py     yfinance con caché Parquet (data/cache_precios) y precios sintéticos
app/api/   FastAPI fina (sin lógica de negocio), 127.0.0.1, sin autenticación
app/web/   React 18 + TypeScript + Vite + Tailwind + TanStack Query + lightweight-charts
pipelines/paper_trading/   bot (subproceso, uno por cuenta/perfil de Alpaca) que opera un experimento
src/ypyquant/perfiles_alpaca/   perfiles de claves, vínculo cuenta-experimento, almacén por perfil, redacción de secretos
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

### Estrategias de reglas (entrada/salida)

Si prefieres escribir reglas de compra y venta en vez de pesos, hereda de `EstrategiaSenales` (`ypyquant.estrategias.senales`): implementas `senal_entrada` y `senal_salida` (tablas booleanas causales) y la base aporta posiciones, stops (pérdida, trailing, take profit, tiempo), enfriamiento, tamaño y efectivo (`CASH`). Guía con ejemplo mínimo copiable y reglas para no engañarte: `docs/estrategias_senales.md`. Plantillas: `estrategias_usuario/cruce_medias.py` y `estrategias_usuario/canal_rsi_volatilidad.py`; puente heredado: `senales_tecnicas`. En el detalle del experimento, cada operación muestra su motivo.

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

Variables: `YPYQUANT_PRECIOS=sintetico` (precios sin red, para probar), `YPYQUANT_OFFLINE=1` (no descarga; usa la caché), `YPYQUANT_ESTRATEGIAS_USUARIO=<carpeta>`, `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (perfil `DEFECTO`) o `ALPACA_<PERFIL>_API_KEY` / `ALPACA_<PERFIL>_SECRET_KEY` (cuentas **paper**, en `.env`; ver «Alpaca multi-perfil»). Docker opcional: `docker compose up` (ver `docker-compose.yml`).

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
| GET | `/api/comparar?ids=a,b` | Equity base 100 + métricas (DuckDB) + `ceteris_paribus: {ok, diferencias}` |
| GET | `/api/comparar/series?ids=a,b&indicador=` | Series temporales por experimento: `drawdown`, `sharpe_movil`, `volatilidad_movil` (12 meses) y `turnover_rebalanceo`. Los tres primeros sobre el eje de fechas común de `/api/comparar` (con un TFM mensual el eje es mensual y la ventana son 12 observaciones; todo diario: 252); el turnover usa los rebalanceos propios (½Σ\|Δ peso objetivo\|). `sin_dato` lista los experimentos sin pesos o con serie corta (nunca ceros). 422 indicador desconocido |
| GET / POST | `/api/configuraciones?estrategia=&texto=&archivadas=` | Lista (última versión de cada una) / crea la v1 desde `desde_experimento`, `desde_barrido` (+ `criterio`, `id_experimento`) o `manual`; 201, 409 id duplicado, 422/404 |
| GET | `/api/configuraciones/{id}?version=` | Detalle: versión, historial, marco enlazado y backtests de la misma receta |
| POST | `/api/configuraciones/{id}/versiones` | `{cambios}` crea la versión n+1 (la anterior no cambia); 422 si el campo no es editable o no valida |
| POST | `/api/configuraciones/{id}/backtest` | `{fecha_inicio?, fecha_fin?, version?, capital_inicial?, forzar?}` lanza (o reutiliza) un experimento normal; sin fechas: las del marco o 5 años |
| POST | `/api/configuraciones/{id}/comparativa` | `{estrategias: [{estrategia, parametros}], fecha_inicio?, fecha_fin?, version?, forzar?}`: lanza la MISMA configuración de entorno de la receta contra N estrategias. Crea (o reutiliza: id `cfg-<hash>` de configuración + versión + fechas) un marco derivado y usa el flujo de `/api/marcos/{id}/comparativa` (valida todas, 202). Sin fechas: marco enlazado, experimento de origen o 5 años. El detalle devuelve `fechas_sugeridas` y `comparativas` |
| DELETE | `/api/configuraciones/{id}?definitivo=` | Archiva (o borra todas las versiones con `definitivo=true`) |
| GET / POST | `/api/marcos` | Lista / crea un marco (409 id duplicado, 422) |
| GET / PUT / DELETE | `/api/marcos/{id}` | Marco + sus comparativas / reemplaza (los experimentos ya lanzados no cambian) / borra |
| POST | `/api/marcos/{id}/comparativa` | `{estrategias: [{estrategia, parametros}], forzar?}`: aplica el marco, valida TODAS antes de lanzar (422), crea N experimentos normales y registra la comparativa (202) |
| GET | `/api/comparativas?id_marco=`, `/api/comparativas/{id}` | Registro de la comparativa con el estado actual de cada experimento |
| GET | `/api/paper/{estado,cuenta,posiciones,ordenes,historial,fills}?perfil=X` | Alpaca paper de UN perfil (por defecto `DEFECTO`); **503** sin claves (el mensaje nombra el perfil y las variables), 502 si Alpaca falla, 422 si el nombre del perfil es inválido |
| GET | `/api/paper/alpaca/perfiles?conexion=` | Cuentas detectadas: nombre, configurada sí/no, últimos 4 de la API key (nunca el secreto), variables que faltan, conexión y equity, bot y vínculo; límite blando (3) y aviso |
| GET | `/api/paper/alpaca/bots/{perfil}` / `.../log?desde=N` / `/api/paper/alpaca/{perfil}/instantaneas` | Estado del bot + vínculo / log redactado (polling por línea) / foto diaria de la cuenta |
| POST | `/api/paper/alpaca/bots/{perfil}/{arrancar,parar}` | `arrancar`: `{experimento_id \| configuracion_id, modo: continuo\|diario, intervalo_s}`; **404** experimento inexistente, **422** receta no operable (parámetros inválidos, apalancamiento > 1), **409** perfil ocupado o límite de 3 bots, **503** perfil sin claves |
| POST | `/api/paper/bot/{arrancar,parar}` | (bot ÚNICO heredado, perfil `DEFECTO`; la UI ya no lo usa) Subproceso `python -m pipelines.paper_trading --estrategia ...`; `arrancar` admite `dias_historico` (mínimo, 30-10000) y `frecuencia_rebalanceo` opcionales |
| GET | `/api/paper/bot/{estado,log?desde=N}` | Estado y log (polling por línea) |
| GET / POST | `/api/paper/interno/sesiones` | Sesiones de **paper interno** (distinto de `/api/paper`, que es Alpaca): lista con retorno, Sharpe, drawdown, costes, estado y `dias_pendientes` / `al_dia` / crea desde `{id_experimento o id_configuracion (+version), capital_inicial?, modo_ejecucion, comision?, slippage?, nombre?, fecha_alta?}` (201; 404 origen inexistente, 409 experimento sin terminar, 422 incoherente, estrategia sin código) |
| GET | `/api/paper/interno/sesiones/{id}` y `/equity`, `/operaciones`, `/pesos` | Detalle (receta congelada, métricas forward, posiciones, pesos objetivo y pendientes, avisos) y series |
| POST | `/api/paper/interno/sesiones/{id}/{pausar,reanudar,detener}` | Control; 409 si la transición no es válida (una detenida no se reanuda) |
| POST / GET | `/api/paper/interno/avanzar`, `/avanzar/estado` | Avanza todas las activas en un hilo del ejecutor (202; 409 si ya hay un avance); estado `{en_curso, inicio, fin, resultados, error}` para el sondeo |
| GET | `/api/paper/interno/sesiones/{id}/espejo` | Lanza (o reutiliza por huella) el backtest de la misma receta del alta a la última fecha y devuelve `real` y `espejo` (`estado` mientras se calcula) |

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

## Experimentos TFM (línea base importada)

Son los **resultados originales del TFM** (`rejilla_v2`) importados al almacén para compararlos con iteraciones posteriores. No se recalculan: se leen los Parquet del TFM (solo lectura) y se escriben experimentos con el mismo formato que un backtest (`tfm-<estrategia>-n<N>-t<T>-media`, estrategia `tfm_<estrategia>`, que no está en el registro: sin código y no se pueden relanzar). Llevan la insignia «TFM · importado» en Experimentos, Comparar y el detalle.

```
uv run python scripts/importar_tfm.py                    # (N,T) = (20,12), (80,36), (150,120) x 6 estrategias = 18
uv run python scripts/importar_tfm.py --celdas todas     # las 20 celdas (120 experimentos)
uv run python scripts/importar_tfm.py --celda 80,36 --estrategia mpt_bayes_stein
uv run python scripts/importar_tfm.py --replica 3        # una réplica (con pesos si b < 20) en vez de la media
```

Reimportar omite los existentes salvo `--forzar`. Opciones: `--origen` (carpeta `rejilla_v2`; por defecto `TFM_DIR`/data/reportes/rejilla_v2), `--almacen`, `--sin-benchmark`, `--cache-precios`.

- **Serie**: media de `r_neto` sobre las 100 réplicas en cada mes (la que da `sharpe_serie_promediada`), equity desde 100 000, con punto base en el cierre del mes anterior. Las fechas son el último día de negociación de cada mes (el TFM etiqueta el mes por su primer día) para que `GET /api/comparar`, que intersecta fechas, comparta cierres de mes con los backtests diarios. Se rebasa a 100 en la primera fecha común.
- **Métricas**: mismos nombres y funciones que los experimentos normales (rf = 0, 12 periodos/año), más `sharpe_media_desv`, `sharpe_exceso_rf` y las del TFM con prefijo `tfm_` (`tfm_sharpe_media_replicas`, `tfm_sharpe_serie_promediada`...). `sharpe` es CAGR/volatilidad (definición de la UI); el del TFM es √12·media/desv. de los retornos netos **sin restar la rf** (verificado: coincide con `sharpe_serie_promediada` a 1e-9).
- **Sesgos conocidos** (van en los `avisos` de cada experimento): universo con look-ahead/supervivencia (filtro de cobertura sobre la ventana OOS; ver `docs/migracion-v0.4.0.md`), costes de 10 pb ex post y motor distinto del de la interfaz. La serie promediada diversifica el ruido de selección del universo: su Sharpe supera al de una réplica (`tfm_sharpe_media_replicas`).
- **Procedencia**: origen, commit del repo `tfm`, ruta y SHA-256 de los Parquet fuente y fecha de importación. Sin benchmark SPY cargable no hay alfa/beta (se avisa).

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

## Interfaz de barridos y validación

- **Lanzar** (`/estrategias/<nombre>`, tarjeta «Lanzar barrido»): marca los parámetros a barrer, escribe sus valores (para numéricos, «Generar rango» con inicio/fin/paso), elige métrica objetivo, etiqueta y `fecha_fin_desarrollo` (lo posterior queda reservado). Muestra el nº de combinaciones (máx. 500) y al lanzar navega a `/barridos/<id>`. Los parámetros sin marcar usan el valor del formulario de experimento.
- **Lista** (`/barridos`): etiqueta, estrategia, ejes, combinaciones, métrica, estado/progreso y resultados disponibles.
- **Detalle** (`/barridos/<id>`, sondea cada 1,5 s mientras haya experimentos o tareas en curso): progreso; mapa de calor (métrica, valor/meseta, 1 o 2 ejes con valores fijos para el resto; ★ mejor, ◆ alternativa por meseta; clic abre el experimento); tabla ordenable (incluye `ulcer`, `martin` y `costes_sobre_ganancia_bruta`, en ámbar > 0,5); **Validar** (DSR, PBO con histograma de λ y dispersión IS/OOS, rachas); **Permutación** (progreso, cancelar, histograma y p-valor con semáforo); **Walk-forward**; **Periodo reservado** (solo con `fecha_fin_desarrollo`: contador de usos y confirmación si ya se usó).
- Semáforos orientativos (`src/lib/barridos.ts`): DSR > 0,95 verde; PBO < 0,1 verde, 0,1–0,2 ámbar, > 0,2 rojo; p-valor in-sample < 1 % verde, walk-forward < 5 % (periodo de ~1 año) o < 1 % (≥ 2 años).
- El detalle de experimento muestra la **procedencia** en un bloque plegable.
- Desarrollo en paralelo a otra instancia: `YPYQUANT_API_PUERTO` y `YPYQUANT_WEB_PUERTO` cambian el proxy y el puerto de Vite (por defecto 8010 y 5173).

## Configuraciones guardadas

Una **configuración** es una receta congelada y versionada (`data/configuraciones/<id>/v<n>.json` + `indice.json`): estrategia, parámetros completos, universo, frecuencia, costes, benchmark, capital, semilla e historia previa, **sin fechas** (las fechas son del backtest). `a_experimento(inicio, fin)` construye una `ConfiguracionExperimento` normal: misma huella que el experimento de origen si coinciden fechas y capital, así que se reutilizan los resultados ya calculados. `huella_estrategia` es la huella de la receta sin fechas ni capital (`ConfiguracionExperimento.huella_receta()`).

- **Origen**: `experimento`, `barrido` (criterio `mejor` | `meseta` | `elegida`) o `manual`; se guardan las métricas del experimento EN ESE MOMENTO para medir luego la degradación. Desde un barrido se avisa del riesgo de sobreajuste (winner's curse) y de que el periodo reservado no se ha evaluado.
- **Universo**: un nombre de archivo (`sp500:30`) se congela como lista de tickers (la etiqueta conserva el nombre); `sp500_pit` es dinámico por definición y se guarda como nombre.
- **Versiones inmutables**: editar crea la v(n+1) (`deriva_de`); solo `nombre`, `notas`, `etiquetas`, `parametros`, `universo`, `frecuencia_rebalanceo`, costes, `benchmark`, `capital_inicial`, `semilla`, `dias_historia_previa` e `id_marco` son editables (la estrategia no: otra estrategia es otra configuración).
- **UI** (`/configuraciones`, `/configuraciones/<id>?version=`): tabla, detalle con origen, historial de versiones, backtests de la receta, lanzar backtest y nueva versión (formulario autogenerado). «Guardar como configuración» está junto a ★ y ◆ en el detalle de barrido.
- **Fuera del menú (2026-11-21):** la página sigue accesible en `/configuraciones`, pero el menú ya no la muestra: para comparar estrategias se usan los marcos. Las configuraciones se reservan para congelar la receta que irá a paper/live (fases 3-5, capa «Ejecutor»). En el detalle de experimento, el botón es ahora **«Crear marco desde este experimento»**: abre `/marcos?desde=<id>` con el entorno del experimento (universo, fechas, rebalanceo, costes, benchmark, capital, semilla, historia previa y parámetros comunes) precargado (`lib/marcos.ts`, `marcoDesdeExperimento`).
- Las sesiones de paper interno se crean desde un experimento terminado (también admiten `id_configuracion`; ver «Paper interno»). Pendiente: vinculación con Alpaca (`/api/configuraciones/{id}/alpaca`).

## Marcos de comparación (ceteris paribus)

Un **marco** (`MarcoComparacion`, `data/marcos/<id>/marco.json`) fija una vez el entorno: universo (sugerido `sp500_pit`), fechas, frecuencia de rebalanceo, comisión, slippage, benchmark, capital, semilla, historia previa y `parametros_comunes` (`ventana_dias`, `peso_maximo`, `tasa_libre_riesgo`, `estimador_sigma`, `frecuencia_estimacion`...). Se lanzan varias estrategias sobre él para comparar solo el efecto de la estrategia.

- **`aplicar_marco(marco, estrategia, parametros_propios)`** (función pura) impone el entorno y aplica cada común SOLO si la `Parametros` de la estrategia lo declara; el informe dice qué se aplicó, qué no (momentum no tiene `ventana_dias`) y si el marco pisó un parámetro propio (el marco prevalece). Los parámetros propios (`top_k`, `penalizacion_turnover`...) se eligen aparte. Los `tfm_*` importados no se pueden lanzar (sin código), pero sí compararse.
- **Comparativa**: `POST /api/marcos/{id}/comparativa` crea N experimentos normales (mismo almacén y huella: los ya calculados se reutilizan) y guarda `data/marcos/_comparativas/cmp-<hash>.json` (marco, experimentos e informe por estrategia). Su id sale de los experimentos: relanzar lo mismo la reutiliza.
- **Estimadores comunes** (bayes_stein, michaud, bl_prior, robusta_elipsoide, robusta_caja): `estimador_sigma` (`ledoit_wolf`, defecto, o `muestral`; Michaud también) y `frecuencia_estimacion` (`diaria`, defecto, o `mensual`: cierres a fin de mes de la ventana, retornos mensuales y 12 periodos/año, como el TFM; el mes en curso se descarta salvo que la fecha sea su último día hábil; mínimo 12 retornos mensuales, por debajo 1/N, y aviso si hay menos de 24). Reutilizan `estimacion_robusta`. No miran al futuro (`verificar_sin_lookahead`). Son parámetros nuevos: **se omiten de la huella mientras valgan su defecto** (`Estrategia.parametros_fuera_de_huella_si_defecto`), así las huellas ya guardadas no cambian.
- **Comparar por indicador**: la página elige el indicador con `?indicador=` (por defecto `equity`). Escalares (Sharpe, Sortino, Calmar, CAGR, retorno total, volatilidad, máx. drawdown, alfa, beta, ratio de información, Ulcer, turnover anual, operaciones, comisiones): barras ordenadas de mejor a peor, «sin dato» al final si la métrica falta (importados del TFM: operaciones y comisiones). Temporales: líneas por experimento (catálogo en `app/web/src/lib/indicadores.ts`). Las series móviles se calculan en el servidor (`series_comparacion.py`) porque la periodicidad decide la ventana.
- **Aviso «no es ceteris paribus»**: `GET /api/comparar` devuelve `ceteris_paribus.diferencias` (función pura `diferencias_ceteris_paribus`): campos de entorno que difieren (universo, fechas, frecuencia, costes, benchmark, capital, semilla, historia) y comunes con valores distintos entre estrategias que los declaran. La página Comparar lo muestra también para comparaciones manuales. Con `?comparativa=<id>` muestra además los avisos del marco.
- **Lanzar una configuración contra otras estrategias**: `marco_desde_configuracion(config, inicio, fin)` (pura) copia el entorno de la receta (universo, frecuencia, costes, benchmark, capital, semilla, historia y los `COMUNES_CONOCIDOS` que traiga) a un `MarcoComparacion` con `id_configuracion` y `version_configuracion`. El enlace configuración a marco vive en el marco y NO se crea una versión nueva de la configuración (las versiones son inmutables y `id_marco` de la configuración fija las fechas por defecto de su backtest; un marco por rango de fechas no cabe en ese campo). La propia estrategia dentro del marco reproduce la misma huella que `a_experimento` y reutiliza su backtest. En la UI, el botón «Lanzar contra otras estrategias» del detalle de configuración reutiliza `LanzadorComparativa` con la propia estrategia marcada.
- **UI** (`/marcos`, `/marcos/<id>`): es una página propia y no una sección de Configuraciones porque tiene su flujo (definir el entorno, elegir estrategias, comparar) y agrupa N estrategias; una configuración es una sola receta y puede enlazar con un marco (`id_marco`, opcional). El detalle del marco permite editarlo, elegir estrategias con su formulario propio (sin los comunes que fija el marco), lanzar la comparativa y abrir Comparar con todos de golpe.

## Paper trading con estrategias del registro

```bash
uv run python -m pipelines.paper_trading --estrategia momentum --parametros '{"top_k": 3}' --tickers AAPL,MSFT,NVDA,AMZN
```

Sin `--estrategia` el bot sigue funcionando como antes (señales técnicas, `--config estrategia_ejemplo.json`). En modo registro cada iteración pide los pesos a la estrategia y envía órdenes de rebalanceo (`decidir_ordenes_rebalanceo`, ventas primero).

Reglas del bot en modo registro:

- **Historia**: se descarga la que necesita la estrategia, no un número fijo. Cada `Estrategia` declara `sesiones_necesarias()` según sus parámetros (momentum: `meses_retorno·21 + 1 + 10`; estimadores del TFM y `preset_ypyquant`: `ventana_dias + 1 + 10`), que se convierte a días naturales (×365/252 + 15 de margen). `NAUTILUS_LOOKBACK_DAYS` (252 por defecto), `--dias-historico` y el campo `dias_historico` de la API son el **mínimo**: si pides más se respeta; si pides menos se amplía a lo necesario. Si aun así faltan sesiones (p. ej. un ticker recién salido a bolsa) el bot escribe un `WARNING` («Historia insuficiente…») y la estrategia usa su caso de reserva (momentum reparte a pesos iguales; los estimadores usan una ventana más corta o pesos iguales): sigue operando, pero no es la estrategia pedida. Una estrategia propia puede sobrescribir `sesiones_necesarias()`; por defecto devuelve 0 (sin requisito).
- **Frecuencia**: solo rebalancea cuando toca según la frecuencia de la estrategia (`frecuencia_rebalanceo`: momentum y las del TFM son `monthly`), o la indicada con `--frecuencia` / `frecuencia_rebalanceo` en la API. Toca el primer día de mercado de un periodo de calendario nuevo (día, semana ISO, mes o trimestre) tras el último rebalanceo. La fecha se guarda en `data/paper_trading/estado_bot.json` (con la firma estrategia + parámetros + tickers: si cambias la configuración, el calendario se reinicia). Los días que no toca, el bot solo loguea. Si algún envío falla, no se marca como hecho y se reintenta en la siguiente iteración.
- **Órdenes abiertas**: antes de decidir consulta las órdenes abiertas y no envía ninguna sobre un ticker que ya tenga una; si no puede consultarlas, no envía nada.
- **Fills**: una orden de mercado recién enviada tiene `filled_qty` 0, así que el bot anota su id como pendiente (en `estado_bot.json`) y registra el fill real en `data/paper_trading/fills.jsonl` cuando la orden termina (al cerrar cada iteración, con unos reintentos, y al empezar la siguiente). Un mismo `id_orden` no se registra dos veces.

## Alpaca multi-perfil (fase 5)

Hasta 3 cuentas paper de Alpaca en paralelo, cada una operando UN experimento, para probar estrategias avanzadas a la vez.

**Claves (solo en `.env`, nunca en pantalla ni en la API).** Por cada cuenta, dos variables con el nombre del perfil en mayúsculas (`[A-Z0-9_]`):

```
ALPACA_TREND_API_KEY=...        ALPACA_TREND_SECRET_KEY=...
ALPACA_VALOR_API_KEY=...        ALPACA_VALOR_SECRET_KEY=...
ALPACA_ML_API_KEY=...           ALPACA_ML_SECRET_KEY=...
```

`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (sin nombre) son el perfil `DEFECTO` (compatibilidad). `ALPACA_PERFILES=TREND,VALOR,ML` (opcional) declara perfiles aunque aún no tengan claves: aparecen como «sin configurar» con las variables que faltan. La API reinicia para leer el `.env`. De una clave solo salen «configurada sí/no» y los últimos 4 caracteres de la API key; los logs del bot pasan por un filtro de redacción (y la API vuelve a redactar al servirlos). El límite blando de cuentas activas (3 bots a la vez) es `LIMITE_PERFILES_ACTIVOS`.

**Vínculo cuenta-experimento.** Al arrancar un bot (`--perfil X --experimento ID`, o el botón de la UI) se congela la `Receta` del experimento (estrategia, parámetros, entorno) en `data/paper_trading/<PERFIL>/vinculo.json`, con su huella, el universo ya resuelto a tickers, la frecuencia de rebalanceo (la de la receta o la sugerida por la estrategia), la fecha de alta y el equity inicial de la cuenta. Si el experimento se borra o se edita después, el bot sigue con lo congelado. Cambiar de experimento guarda el anterior en `vinculos_anteriores` y borra la memoria de la estrategia. `--configuracion ID` toma la receta de una configuración guardada (opcional; la UI trabaja con experimentos). El universo dinámico `sp500_pit` SÍ se puede operar: el vínculo guarda `universo_dinamico` (y los tickers de hoy como foto) y el bot resuelve la membresía vigente en cada rebalanceo (ver «Universo `sp500_pit` en vivo»).

**Qué hay en `data/paper_trading/<PERFIL>/`:** `vinculo.json`, `bot.json` (pid, marca del proceso, modo, intervalo; lo escribe la API), `bot.log`, `estado_bot.json` (último rebalanceo y órdenes pendientes con su precio de referencia), `estado_estrategia.json` (memoria de la estrategia entre reinicios: las posiciones de una `EstrategiaSenales`; solo la parte serializable), `fills.jsonl` y `instantaneas.parquet` (una fila por día: equity, efectivo y posiciones con cantidad y precio; Alpaca no devuelve posiciones pasadas y sin esto no se puede medir el descuadre de cartera). Los fills añaden `perfil`, `experimento_id` (o `configuracion_id`), `tipo` (`rebalanceo`) y `precio_referencia` (último cierre al decidir); las filas antiguas se leen igual.

**Modos del bot** (`--modo`): `diario` = una pasada y sale (para el programador de tareas; `--perfil X --modo diario` sin `--experimento` reanuda el vínculo guardado) y `continuo` = bucle cada `--intervalo` segundos, con un aviso: solo rebalancea cuando toca la frecuencia. **Cambio de comportamiento respecto al bot anterior:** antes, sin frecuencia, se rebalanceaba en cada iteración; ahora solo cuando toca, y el estado de la estrategia sobrevive a los reinicios. Con el mercado cerrado calcula los pesos pero no envía (como antes); `--enviar-fuera-de-horario` envía igualmente (ver la checklist).

**API y reinicios.** `GestorBots` lanza un subproceso por perfil; un perfil ocupado devuelve 409. Al arrancar la API **rehidrata**: lee los `bot.json`, comprueba que el pid siga vivo y sea el mismo proceso (pid + hora de creación, con `psutil` si está) y lo adopta (se puede parar por pid); los muertos se limpian. Los bots son hijos de la API pero siguen vivos si esta cae, por eso se rehidratan.

**UI** (`/paper`): una pestaña por cuenta (`?perfil=`) con su estado (sin configurar, sin conexión, bot en marcha), equity, posiciones, órdenes, fills, instantáneas y el control del bot (elegir experimento, modo e intervalo; arrancar/parar; log). Un perfil sin claves muestra qué variables poner en el `.env` (sin campo para pegarlas). En la ficha de un experimento terminado, **«Llevar a Alpaca»** abre `/paper?experimento=<id>` con el experimento preseleccionado.

### Checklist 5.7: orden de prueba manual (NO automática)

Verifica a mano, en el perfil de pruebas, la hipótesis del diseño: una orden de mercado `DAY` enviada con el mercado cerrado queda en cola y se ejecuta en la apertura siguiente (lo que casa con el modelo `apertura_siguiente`). No la ejecutes desde tests ni desde el bot sin vigilar.

1. Elige el perfil de pruebas (cuenta paper sin posiciones que importen) y comprueba en `/paper` que está «bot parado» y conectado.
2. Fuera del horario del mercado (noche o fin de semana), crea una orden de mercado `DAY` de 1 acción de un ticker líquido (p. ej. desde la consola de Alpaca paper o con `python -m pipelines.paper_trading --perfil X --experimento ID --modo diario --enviar-fuera-de-horario` sobre un experimento de un solo ticker).
3. Anota el estado inmediato (`new`/`accepted`/`pending_new`), la hora de creación y que aparece en «Órdenes recientes» y bloquea nuevas órdenes de ese ticker.
4. Tras la apertura siguiente, anota cuándo pasa a `filled`, el precio medio y compáralo con la apertura (`Open`) de la fuente de precios del proyecto: la diferencia es slippage más diferencia de fuente.
5. Confirma que el fill se registra en `data/paper_trading/<PERFIL>/fills.jsonl` con `precio_referencia` y `experimento_id` en la siguiente sincronización (siguiente pasada del bot).
6. Anota el resultado en el plan (sección «Estado 2026-11-22 (fase 5)»): si se confirma, `--enviar-fuera-de-horario` puede ser el defecto del modo `diario`; si no, el modo `diario` debe programarse con el mercado abierto.
## Paper interno

Página **Paper interno** (`/paper-interno`, entrada del menú; la de Alpaca sigue en `/paper`). Simula hacia delante, con precios
reales y sin broker, la receta congelada de un experimento terminado (núcleo: `docs/ypyquant.md`, «Paper interno»).

- **Crear una sesión**: en la ficha de un experimento terminado, botón **«Paper interno»**: nombre, modo de ejecución
  (apertura siguiente por defecto; «al cierre» es optimista), capital (100 000 por defecto), comisión y slippage opcionales
  (en %; vacío = los del experimento; 0 = costes tipo Alpaca) y fecha de alta (vacío = hoy). Navega al detalle de la sesión.
- **Lista** (`/paper-interno`): nombre, estrategia, modo, **periodo** (fecha de alta → última fecha procesada), **duración**
  («11 sesiones de mercado · 2 semanas», «3 meses y 5 días»: sesiones de trading procesadas y tiempo de calendario desde el
  alta), retorno total (%), Sharpe, drawdown máx. (%), costes (USD), estado, **pendientes** (días hábiles cerrados aún sin
  procesar) y **rebalanceo** (frecuencia de la receta, «rebalanceo mensual», y fecha del próximo, estimada sin festivos:
  `proximo_rebalanceo` en la API). Las cabeceras llevan su unidad. **Avanzar ahora** procesa los días cerrados pendientes (el botón se deshabilita y la página sondea mientras está
  en curso; al acabar muestra días procesados, avisos y errores por sesión). Selección múltiple: equity a base 100 en la
  primera fecha común, con el benchmark.
- **Detalle** (`/paper-interno/<id>`): bloque **«Tiempo en paper interno»** (periodo, duración, pendientes, frecuencia y
  próximo rebalanceo, con el último en la ayuda), KPIs forward, **curva real frente al backtest espejo** (la misma receta del alta a la
  última fecha, al cierre; en modo `cierre` deben coincidir), avisos (re-bases, liquidaciones, errores de estrategia), decisión
  pendiente (modo apertura siguiente), posiciones actuales, pesos objetivo, operaciones y pausar / reanudar / detener.
- **Al arrancar la API** se hace el catch-up en segundo plano si hay sesiones activas (`crear_app(catchup_paper_al_arrancar=False)`
  lo desactiva en los tests).
- **Comprobación automática mientras la API está en marcha**: un temporizador en segundo plano (`ypyquant/paper/planificador.py`,
  en el lifespan de la API) comprueba cada `YPYQUANT_PAPER_INTERVALO_MIN` minutos (30 por defecto; `0` lo desactiva; también
  `crear_app(intervalo_paper_min=...)`) si hay sesiones activas con días cerrados pendientes (día cerrado = tras las 16:30 de
  Nueva York) y las avanza con el mismo `ControlAvance` que el botón y el catch-up: si ya hay un avance en curso no lanza otro.
  Si no hay precios para el día pendiente (festivo o proveedor retrasado) deja de reintentar tras 4 intentos hasta que cambia la
  sesión cerrada objetivo. Se detiene limpio al cerrar la API. `GET /api/paper/interno/planificador` devuelve
  `{activo, intervalo_min, proxima_comprobacion, ultima: {hora, resultado, dias_procesados, error}}` y la página de lista muestra
  la próxima comprobación y qué hizo la última. Solo avanza mientras la API vive; los días perdidos se recuperan al volver
  (el simulador es idempotente y hace catch-up exacto). Es la vía recomendada: no hace falta tarea programada.
- **Programador de Windows (OPCIONAL, no necesario)**: `scripts/instalar_tarea_paper.ps1` registra la tarea «Sistema_trading paper interno» (diaria a
  las 23:30 hora local y al iniciar sesión, `StartWhenAvailable`: si el PC estaba apagado se ejecuta al encender y recupera todos
  los días pendientes), que lanza `uv run python -m ypyquant.paper avanzar` y añade la salida a `data/paper_interno/avance.log`.
  No registra nada sin que lo ejecutes tú: `-SoloMostrar` enseña lo que haría, `-Hora 23:45`, `-Despertar` (WakeToRun) y
  `-Desinstalar`. Ver el log: `Get-Content data\paper_interno\avance.log -Tail 50`. Solo sirve si quieres que el paper avance
  con la API apagada; con el temporizador y el catch-up al arrancar no es imprescindible.

### Universo `sp500_pit` en vivo

Regla estándar: en un backtest se usa la membresía point-in-time (los miembros en cada fecha; un universo fijo actual aplicado
hacia atrás introduce sesgo de supervivencia); en operativa en vivo el universo es la membresía VIGENTE en cada día (= point-in-time
con la fecha de hoy). Implementado en el paper interno y en el bot de Alpaca:

- **Paper interno**: la sesión conserva `sp500_pit` como nombre (ya no hay 422). En cada fecha procesada la estrategia ve solo a
  los miembros vigentes ese día con historia suficiente (el filtro del backtest, `UniversoPuntoEnElTiempo`); la membresía es la
  de la fecha de la barra (el paper avanza por fechas pasadas recientes). Los precios que se descargan son los de todo miembro
  entre el alta y la última sesión cerrada.
- **Bot de Alpaca**: en cada rebalanceo descarga los miembros de hoy más lo que hay en cartera (para poder venderlo) y la
  estrategia ve solo a los vigentes en la fecha del último precio. Necesita ~450 días de historia (el filtro exige 12 retornos
  mensuales completos), no los 252 por defecto.
- **Salidas del índice**: un valor que sale del S&P 500 no se vende ese día; la estrategia deja de verlo y se liquida en el
  siguiente rebalanceo. Hay aviso: evento `salida_indice` en el paper interno y WARNING en el log del bot.
- **Calidad de las membresías** (`universos/sp500_membresias.json`, exportado de TFM.db con
  `scripts/exportar_membresias_sp500.py`): aviso (una vez por sesión en el paper, en el log del bot y en el detalle de la sesión)
  si la última fecha registrada tiene más de 30 días respecto a la fecha procesada (indica cómo refrescarla: actualizar TFM.db y
  volver a ejecutar el script) y si el número de miembros vigentes se aleja de ~500 (±15).
- **Hallazgo (2026-11-22)**: el JSON da 525 miembros vigentes frente a ~503 símbolos reales. `universos/sp500.txt` (miembros a
  15-03-2026) tiene 503 y todos están vigentes en el JSON. Los 22 sobrantes: 17 altas posteriores a esa fecha (BE, CASY, COHR,
  ECHO, FDXF, FERG, FLEX, HONA, LITE, MRVL, OKTA, P, RDDT, TWLO, VEEV, VRT, VYLR) sin ninguna baja posterior registrada (la última
  baja del JSON es 2026-01-14; el índice mantiene su tamaño, así que faltan ~17 bajas de marzo-noviembre de 2026) y 5 filas con
  altas antiguas sin cierre que no están en el `sp500.txt` de marzo (BNY, DAY, ILMN, SKYD, VMRK: probables bajas o cambios de
  ticker sin cerrar). Las clases duplicadas (GOOG/GOOGL, FOX/FOXA, NWS/NWSA) NO explican la diferencia: ya están en los 503. No
  se pudo contrastar contra una fuente externa (no se descargó nada); hay que corregirlo en TFM.db y volver a exportar.

## Tests

`uv run pytest` (Python, sin red), `cd app/web && npx tsc --noEmit && npx vitest run && npm run build`.

## Limitaciones conocidas

- Long-only. El efectivo parcial se expresa con el pseudo-ticker `CASH` (lo usan las estrategias de señales; la UI lo muestra como «Efectivo»).
- Estrategias de señales: stops evaluados con el cierre diario (sin intradía ni gaps de apertura); las rejillas de barrido de la UI no admiten `null` (sin stop) en un parámetro opcional, solo la API.
- Las validaciones del barrido (DSR, PBO, permutación) cuentan solo las combinaciones de la rejilla: si probaste más cosas a mano, N real es mayor (ver `docs/validacion.md`). El PBO-CSCV y el walk-forward sobre la matriz ignoran el coste de cambiar de combinación.
- CPCV no está implementado (sí DSR, PBO-CSCV, rachas, permutación y walk-forward).
- Sin autenticación: solo para `127.0.0.1`. El bot es hijo de la API; si la API muere en Windows puede seguir vivo (la API lo rehidrata al volver a arrancar).
- Alpaca: acciones enteras (el simulador usa fracciones: el descuadre por enteros se mide en la fase 6).
- `sp500_pit` en vivo depende de que las membresías estén al día (aviso a los 30 días) y de que no falten bajas (ver «Universo `sp500_pit` en vivo»).
