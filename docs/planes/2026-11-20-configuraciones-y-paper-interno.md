# Plan: configuraciones guardadas, paper interno multi-estrategia y estrategias de señales

Repo: `C:\Users\yanni\Documents\Codigo\Proyectos\Sistema_trading` (paquete `src/ypyquant`, API `app/api`, web `app/web`).
Estado del plan: fase 0 y fase 1 (con el marco de comparación) implementadas; el resto, diseño para decidir. Basado en lectura del código (no ejecutado nada). Lo que afirmo sobre Alpaca que no está en el repo va marcado como "por verificar".

---

## 0. Resumen de diseño (1 página)

**Idea central.** Una *Configuración* es una receta congelada (estrategia + parámetros + universo + frecuencia + costes) que se puede lanzar en tres sitios con el mismo código de decisión: backtest, paper interno y Alpaca. Las tres piezas encajan así:

```
Experimento / mejor del barrido ──guardar──> CONFIGURACIÓN (inmutable, versionada)
                                                │
              ┌─────────────────────────────────┼───────────────────────────────┐
              ▼                                 ▼                               ▼
        Backtest (existe)            Sesión de PAPER INTERNO            Bot ALPACA (perfil X)
        lanzar_experimento           simulador forward local            1 cuenta = 1 configuración
                                     N sesiones a la vez                        │
                                           │                                    │
                                           └───── comparación interno vs Alpaca ┘
                                           └───── "backtest espejo" (esperado vs real)
```

**Decisiones clave (con recomendación):**

| # | Decisión | Recomendación | Alternativa descartada |
|---|---|---|---|
| 1 | Almacén de configuraciones | JSON por configuración en `data/configuraciones/<id>.json` (pocas, editables a mano, sin DuckDB) | Parquet/DuckDB: es para series, no para recetas |
| 2 | Edición | **Inmutables y versionadas**: editar = nueva versión. Una sesión de paper apunta a (id, versión) y guarda copia congelada | Editar en caliente: invalida el historial forward |
| 3 | Universo | Al guardar, **congelar la lista de tickers** (se conserva el nombre `sp500:30` solo como etiqueta) | Guardar solo el nombre: la lista de `universos/*.txt` cambia y rompe la reproducibilidad |
| 4 | Motor del paper interno | **Simulador incremental día a día** que reutiliza `BacktestEngine._execute_rebalance/_buy/_sell` (mismo modelo de comisión+slippage) y el calendario del motor; test de paridad: incremental == `run_backtest` | (a) Reejecutar el backtest completo cada día: no es forward real, no guarda estado de la estrategia y no permite comparar fills; (b) reescribir la ejecución: duplica el modelo de costes |
| 5 | Reanudación | Estado persistido por sesión (commit atómico por día) + *catch-up* de los días perdidos con barras diarias | Proceso residente siempre encendido: tu PC se apaga |
| 6 | Programación | Comando idempotente `python -m ypyquant.paper avanzar` + Programador de tareas de Windows con "ejecutar en cuanto sea posible si se perdió" + catch-up al arrancar la API + botón "Avanzar ahora" | APScheduler dentro de la API: solo corre si la API está abierta |
| 7 | Modelo de ejecución del paper | Por sesión: `apertura_siguiente` (decide con el cierre de D, ejecuta a la apertura de D+1; es lo que hace Alpaca con una orden de mercado DAY) o `cierre` (igual que el backtest actual, para la prueba de paridad) | Solo cierre: optimista (decide y ejecuta con el mismo precio) |
| 8 | Alpaca multi-cuenta | Perfiles `ALPACA_<PERFIL>_API_KEY/_SECRET_KEY`; un bot por perfil; 1 cuenta = 1 configuración (máx. 3) | Dividir una cuenta entre estrategias: Alpaca solo ve una cartera, no se puede atribuir |
| 9 | Señales | `EstrategiaSenales`: tabla booleana de entrada/salida (vectorizada y causal) + máquina de estados en la base (posiciones, stops, huecos, cash). Emite pesos con un pseudo-ticker `CASH` | Reglas por barra con bucles en cada estrategia: lento en barridos y cada usuario reimplementa los stops |
| 10 | Causalidad | Utilidad `verificar_sin_lookahead` que envenena el futuro y compara pesos; se aplica a cualquier estrategia | Confiar en que `recortar_hasta` basta: `preparar()` ve todo el histórico |

**Hallazgos del código que condicionan el diseño (verificados leyendo):**

1. El bot Alpaca actual llama a la estrategia **en cada iteración (cada 60 s)** sin respetar `frecuencia_rebalanceo` (`rebalancear_con_estrategia_registro` en `pipelines/paper_trading/alpaca_vivo.py`). Una estrategia mensual se re-ejecuta cada minuto y el estado vive solo en memoria. Hay que arreglarlo para que la comparación con el paper interno tenga sentido.
2. `BacktestEngine` **normaliza los pesos a 1 si se desvían más de un 1 %** (`_execute_rebalance`) y `sanear_pesos` también normaliza: hoy una estrategia no puede quedarse parcialmente en efectivo. Solución sin tocar el motor: un peso para un ticker sin precio (`CASH`) se salta en la compra (`if ticker not in current_prices: continue`) y deja ese tramo en caja. Hay que fijarlo con un test (fase 0).
3. El backtest decide con el cierre de D y ejecuta al **cierre de D** (precio `Close` del mismo día). Es optimista para todas las estrategias, sobre todo las diarias con stops. No se toca en las fases 1-6; en la fase 7 se propone una opción opt-in a nivel de motor.
4. `expandir_rejilla` lanza error si una combinación no valida (p. ej. `rapida >= lenta`): se rechaza el barrido entero. Hay que omitir y avisar (fase 0).
5. `Close` ya viene ajustado (yfinance `auto_adjust=True`): el historial se reescribe con cada dividendo/split. Un simulador forward debe re-basar las cantidades (tarea 3.5), si no pierde dividendos o registra falsas pérdidas.

**Decisiones de Yannick (2026-11-20):**
- Paper interno: ejecución a la `apertura_siguiente` por defecto (`cierre` solo para la paridad con backtest).
- Bot Alpaca diario (un job al día) en lugar del bucle de 60 s.
- Capital inicial de cada sesión interna: 100 000 fijo por defecto, configurable; cuando la sesión está vinculada a una cuenta Alpaca, la comparación (fase 6) normaliza ambas curvas a base 100.
- Nombre en la UI: «Configuraciones».
- Nueva fase 0 de higiene: universo point-in-time (`sp500_pit`) antes de todo lo demás (ver «FASE 0 — Salvaguardas», tareas 0.0a-0.0c).

**Orden y tamaño (S ≈ medio día, M ≈ 1-2 días, L ≈ 3-5 días de trabajo con tests):**

| Fase | Contenido | Tamaño | Depende de |
|---|---|---|---|
| 0 | Higiene y salvaguardas: universo point-in-time `sp500_pit`, tests de caracterización, `CASH`, rejilla tolerante, utilidad de causalidad | M | - |
| 1 | Configuraciones guardadas (modelo, almacén, API, UI) | M | 0 |
| 2 | `EstrategiaSenales` + ejemplo + estrategia puente de señales heredadas | M | 0 (paralelizable con 1) |
| 3 | Núcleo del paper interno (calendario, ejecutor, simulador, almacén, CLI) | L | 1 (modelo), 2 opcional |
| 4 | Paper interno en la API y la UI + programador de tareas | M | 3 |
| 5 | Alpaca multi-perfil + bot por configuración + instantáneas | M | 1, 3 |
| 6 | Comparación interno vs Alpaca y backtest espejo | M | 4, 5 |
| 7 | Opcional: ejecución `apertura_siguiente` en el motor, acciones enteras, intradía | M | 6 |

Tareas paralelizables (no comparten archivos): 1 y 2 tras la fase 0; dentro de la fase 3, el calendario y el almacén de sesiones.

---

## 1. Principios transversales

- **No romper GrowSave.** `BacktestEngine` y `BacktestConfig` los consume GrowSave (subclases). Reglas: ningún cambio de semántica por defecto; solo campos nuevos con valor por defecto idéntico al comportamiento actual; antes y después de tocar `portfolio/backtest_engine.py` pasan `tests/test_backtest_motor.py`, `tests/test_paridad_tfm.py` y `tests/test_ypyquant_api.py` con salida idéntica. El paper interno **no modifica** el motor en las fases 1-6: lo usa como ejecutor (acceso a métodos privados concentrado en un único módulo, `paper/ejecutor.py`, protegido por el test de paridad).
- **No cambiar huellas existentes.** Añadir campos a `ConfiguracionExperimento` cambia `huella()` de todos los experimentos guardados. Regla: un campo nuevo se excluye de la huella cuando vale su defecto (ver 7.1).
- **Español en código, tests y commits;** cabecera de cada archivo con responsabilidad y relaciones (regla del proyecto). Tests offline con `FuenteSintetica`.
- **Un único punto de verdad para el calendario de rebalanceo** (`ypyquant/execution/calendario.py`, ya implementado: ver `2026-11-21-ejecutor.md`), usado por el simulador interno y por el bot Alpaca. Reutiliza `BacktestEngine._get_rebalance_dates` y `_alinear_a_dias_de_trading` en vez de reimplementarlos.
- **Estado JSON-serializable.** Contrato nuevo de `Estrategia.estado`: solo tipos JSON (dict, list, str, float, int, bool, None). Lo verifica una utilidad y falla con mensaje claro. Las estrategias incluidas actuales (`_EstimadorRobusto` usa `estado['pesos_previos']`, arrays NumPy) se comprueban en la fase 3.
- **Secretos.** Las claves Alpaca nunca van en `config.json`, logs ni respuestas de API; la UI muestra solo el nombre del perfil y los 4 últimos caracteres.

---

## FASE 0 — Salvaguardas (S)

Archivos: `tests/test_backtest_motor.py` (ampliar), `tests/test_estrategias.py`, `tests/test_barridos.py`, `src/ypyquant/estrategias/experimentos.py`, `src/ypyquant/estrategias/causalidad.py` (nuevo), `tests/test_causalidad.py` (nuevo), `src/ypyquant/estrategias/universo_dinamico.py` (nuevo), `tests/test_universo_dinamico.py` (nuevo).

**Higiene: universo point-in-time** (motivo: los experimentos y barridos de la interfaz usan `universos/sp500.txt`, los miembros **actuales** del S&P 500, lo que mete sesgo de supervivencia; el motor del TFM ya lo resuelve con `optimization/estimacion_robusta/universo.py::elegibles_en_fecha`).
- [ ] **0.0a** Universo dinámico `sp500_pit` en `ConfiguracionExperimento.universo`: en cada fecha de rebalanceo, los elegibles son los miembros del índice en t (membresías históricas, las mismas que usa la rejilla del TFM) con historia suficiente hasta t-1. Reutilizar `elegibles_en_fecha` y `construir_mascara_membresia`; no reimplementar. Tests: un ticker que entra en t+1 no está en t; uno que deslista en t+5 sí está en t; ningún dato con fecha ≥ t influye (usar `verificar_sin_lookahead`, 0.3).
- [ ] **0.0b** La estrategia recibe en `datos` solo los elegibles de cada fecha (o una máscara), y el motor vende las posiciones que salen del universo. Comprobar el comportamiento con deslistados (precio que desaparece): liquidar al último precio conocido y registrarlo.
- [ ] **0.0c** Interfaz: `sp500_pit` como opción del selector de universo (por defecto en estrategias de sección cruzada); la lista fija sigue para universos a mano. La huella de los experimentos existentes no cambia (el universo es un valor más del campo).

- [ ] **0.1 Test de caracterización de `CASH`** (`tests/test_backtest_motor.py`): estrategia que devuelve `{"AAA": 0.5, "CASH": 0.5}`; afirmar que tras el primer rebalanceo `motor.cash ≈ 50 %` del capital, que no hay operaciones sobre `CASH` y que `sanear_pesos` conserva la clave. Si falla, se ajusta el diseño (alternativa: añadir `permitir_efectivo` a `BacktestConfig`, aditivo). Comando: `uv run pytest tests/test_backtest_motor.py -q`.
- [ ] **0.2 Rejilla tolerante.** Test en `tests/test_barridos.py`: rejilla `{rapida:[5,50], lenta:[20,40]}` con una estrategia cuyo `Parametros` exige `rapida < lenta`: `expandir_rejilla` devuelve 3 configuraciones y registra las omitidas. Implementar en `expandir_rejilla`: `try/except ValidationError` por combinación; devolver también la lista de omitidas (nueva función `expandir_rejilla_con_omitidas`; `expandir_rejilla` conserva su firma para no romper llamadas). Revisar que `ConfiguracionBarrido.combinaciones()` y `superficie` toleran huecos (los vecinos de la meseta se calculan por posición en `_vecinos`; añadir un test con hueco).
- [ ] **0.3 Utilidad de causalidad** `ypyquant/estrategias/causalidad.py`: `verificar_sin_lookahead(clase, parametros, datos, fechas_muestra, semilla=0) -> list[str]`. Algoritmo: (1) pesos en cada fecha con los datos completos vía `como_funcion`; (2) copia de los datos con todo lo posterior a la fecha **sustituido por ruido** (NaN y precios aleatorios), instancia nueva, mismos pesos; (3) si difieren en cualquier fecha, error `"look-ahead en <fecha>: <ticker> pesa X con futuro real y Y con futuro envenenado"`. Tests: una estrategia limpia pasa; una que usa `cierres.shift(-1)` en `preparar` falla; una que usa `rolling(center=True)` falla.
- [ ] **0.4** Ejecutar toda la batería (`uv run pytest -q`) y confirmar que no cambia ninguna huella: test que compara `ConfiguracionExperimento(...).huella()` con un valor fijo.
- [ ] **0.5 Commit** `fase 0: salvaguardas (CASH, rejilla tolerante, causalidad)`.

Riesgo: bajo. Es la red de seguridad de todo lo demás.

---

## FASE 1 — Configuraciones guardadas (M)

### 1.1 Decisiones
- **Qué es:** `ConfiguracionGuardada` (Pydantic) con:
  `id` (slug estable del nombre, p. ej. `momentum-top5-ejemplo`), `version` (int, empieza en 1), `nombre`, `notas`, `estrategia`, `parametros`, `universo` (lista congelada) + `universo_etiqueta` (opcional, p. ej. `sp500:30`), `frecuencia_rebalanceo`, `comision`, `slippage`, `benchmark`, `capital_inicial`, `semilla`, `origen` (ver abajo), `creada` (ISO), `etiquetas`, `huella_estrategia` (hash de estrategia+parámetros+universo+frecuencia+costes+benchmark, **sin fechas ni capital**), `archivada` (bool).
- **Sin fechas.** Las fechas pertenecen al backtest, no a la receta. Al lanzar backtest se piden `fecha_inicio/fecha_fin` (por defecto: últimos 5 años) y se construye una `ConfiguracionExperimento` normal → mismo almacén, misma huella, reutilización de resultados gratis. Método `a_experimento(fecha_inicio, fecha_fin)`.
- **Origen:** `{"tipo": "experimento"|"barrido"|"manual", "id_experimento", "id_barrido", "criterio": "mejor"|"meseta"|"elegida", "metricas": {sharpe, cagr, max_drawdown, meseta, ...}, "fechas": [inicio, fin]}`. Se guardan las métricas **en el momento de guardar** (para ver luego cuánto se degrada en paper). Si el barrido tiene periodo reservado sin evaluar, el origen lo anota y la UI avisa ("elegida sobre datos de desarrollo; sin validar en reservado").
- **Versionado:** `POST /api/configuraciones/{id}/versiones` crea la versión n+1 (copia editada). Se guardan todas en el mismo JSON (`versiones: [...]`) o en `<id>/v1.json, v2.json` (recomiendo carpeta: `data/configuraciones/<id>/v<n>.json` + `indice.json`).
- **Alternativas descartadas:** guardar el `ConfiguracionExperimento` tal cual (arrastra fechas y semántica de huella con fechas); una tabla en DuckDB (sobreingeniería); etiquetas libres en el experimento (no se pueden fijar ni versionar).
- **Nombre del ticker de efectivo y "origen barrido":** el servidor resuelve `desde_barrido`+`criterio` usando `superficie_registros` (ya existe en `barridos.py`; la web calcula ★/◆ en `lib/barridos.ts`, así que el cliente puede mandar directamente el `id_experimento` elegido y el criterio).

### 1.2 Archivos
Crear:
- `src/ypyquant/estrategias/configuraciones.py` — `ConfiguracionGuardada`, `OrigenConfiguracion`, `AlmacenConfiguraciones` (listar, leer(id, version=None), guardar, nueva_version, archivar, borrar), `desde_experimento(config_exp, metricas, origen)`.
- `app/api/rutas_configuraciones.py`.
- `tests/test_configuraciones.py`, `tests/test_api_configuraciones.py`.
- `app/web/src/paginas/Configuraciones.tsx`, `ConfiguracionDetalle.tsx`, `app/web/src/componentes/GuardarConfiguracion.tsx` (diálogo), `app/web/src/paginas/Configuraciones.test.tsx`.

Modificar:
- `app/api/contexto.py` (campo `almacen_configuraciones`, creado igual que `almacen_barridos`), `app/api/main.py` (router y raíz inyectable `raiz_configuraciones`).
- `src/ypyquant/estrategias/__init__.py` (no exportar nada que exija pydantic extra; igual que el almacén).
- `app/web/src/App.tsx` (rutas), `componentes/Layout.tsx` (menú), `api/hooks.ts`, `api/tipos.ts`, `paginas/ExperimentoDetalle.tsx` (botón "Guardar como configuración"), `paginas/BarridoDetalle.tsx` (botones junto a ★ y ◆: "Guardar esta"), `paginas/Estrategias.tsx` o `EstrategiaDetalle.tsx` (acceso a configuraciones de esa estrategia), `docs/ui.md` (sección y tabla de endpoints).

### 1.3 Endpoints
| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/configuraciones?estrategia=&texto=&archivadas=` | Lista (última versión de cada una) |
| POST | `/api/configuraciones` | Crea. Cuerpo: `{nombre, notas, desde_experimento: id}` o `{nombre, notas, desde_barrido: id, criterio: "mejor"\|"meseta", id_experimento?}` o manual (campos completos). 409 si el id ya existe |
| GET | `/api/configuraciones/{id}?version=` | Detalle con origen, huella, sesiones de paper y backtests asociados |
| POST | `/api/configuraciones/{id}/versiones` | Nueva versión con cambios (parámetros, universo...) |
| POST | `/api/configuraciones/{id}/backtest` | `{fecha_inicio, fecha_fin}` → `lanzar_experimento`; devuelve `id_experimento` (reutiliza si ya existe) |
| POST | `/api/configuraciones/{id}/paper` | Crea sesión de paper interno (fase 4) |
| POST | `/api/configuraciones/{id}/alpaca` | Vincula a un perfil Alpaca y arranca el bot (fase 5) |
| DELETE | `/api/configuraciones/{id}` | Archiva (no borra si tiene sesiones) |

### 1.4 UI
- Página **Configuraciones**: tabla (nombre, estrategia, parámetros resumidos, universo, métricas de origen, sesiones activas, fecha). Detalle: parámetros (reutiliza `FormularioSchema` en modo solo lectura), origen con enlace al experimento/barrido, historial de versiones, botones **Backtest**, **Paper interno**, **Alpaca**, **Nueva versión**.
- En `ExperimentoDetalle` y `BarridoDetalle` un diálogo `GuardarConfiguracion` (nombre sugerido `<estrategia> <parámetros clave>`, notas). Advertencia visible si viene de un barrido: "máximo de N combinaciones: puede estar sobreajustado; revisa DSR/PBO y la meseta".

### 1.5 Tareas (TDD)
- [x] **1.1** Test `test_configuraciones.py`: construir desde un `ConfiguracionExperimento` normalizada, comprobar que `huella_estrategia` no cambia con las fechas ni el capital y sí con un parámetro; que `a_experimento(inicio, fin)` produce la misma `huella()` que la configuración original cuando coinciden fechas y capital. Implementar el modelo.
- [x] **1.2** Test del almacén en `tmp_path`: guardar, leer, listar, nueva versión (v2 no modifica v1), archivar, id duplicado → error, escritura atómica (reutilizar `_escribir_json`/`_leer_json` de `almacen.py`). Implementar.
- [x] **1.3** Test: universo con nombre (`sp500:3`) se congela como lista de 3 tickers; `universo_etiqueta` conserva el nombre.
- [x] **1.4** Tests API (patrón de `tests/test_api.py` con `crear_app(cargador_precios=FuenteSintetica(), raiz_...=tmp)`): crear desde experimento terminado, desde barrido (`criterio=mejor`), 404 de experimento inexistente, 422 de parámetros inválidos, backtest desde configuración reutiliza el experimento si ya existe.
- [x] **1.5** Implementar rutas y registro en `main.py`/`contexto.py`.
- [x] **1.6** Frontend: tipos, hooks, páginas y diálogo; test Vitest del diálogo (valida nombre vacío; envía `desde_barrido`). `cd app/web && npm test -- --run`.
- [x] **1.7** Actualizar `docs/ui.md`; commit `fase 1: configuraciones guardadas`.

Riesgos: elegir el máximo de una rejilla (winner's curse); nombre/slug duplicados; universo `sp500` con sesgo de supervivencia (la lista actual solo contiene supervivientes: se avisa en la ficha, y el paper forward no lo sufre).


**Estado (fase 1 implementada en la rama `fase1/configuraciones`).** Hecho: modelo, almacén versionado, rutas, UI y el marco de comparación de la sección siguiente. Diferencias respecto al diseño de arriba, y por qué:
- `huella_estrategia` = `ConfiguracionExperimento.huella_receta()` (nuevo método): excluye fechas, capital, validaciones y etiqueta, e **incluye** semilla e historia previa (afectan al resultado).
- El universo `sp500_pit` no se congela como lista: es dinámico por definición (se guarda el nombre). Los nombres de archivo (`sp500:30`) sí se congelan.
- `a_experimento` conserva `dias_historia_previa`; sin él la huella de la reconstrucción no coincidiría con la del experimento de origen.
- Los endpoints `/paper` y `/alpaca` quedan para las fases 3-5. `DELETE` archiva y `?definitivo=true` borra (no hay sesiones aún que lo impidan).
- «Guardar» desde un barrido: el cliente puede mandar `id_experimento` (criterio `elegida`) y el servidor valida que pertenece al barrido; con `mejor`/`meseta` lo elige el servidor con `superficie_registros`.
- Los parámetros nuevos de una estrategia se excluyen de la huella mientras valgan su defecto mediante `Estrategia.parametros_fuera_de_huella_si_defecto` (aplica a `estimador_sigma` y `frecuencia_estimacion`; test `tests/test_estimadores_comunes.py`).

### 1.6 Marco de comparación (ceteris paribus) — requisito añadido

**Problema.** Para saber si una estrategia es mejor que otra hay que variar solo la estrategia. Hoy cada experimento lleva su propio universo, fechas, frecuencia, costes y estimadores, y es fácil comparar manzanas con peras.

**Diseño.**
- `MarcoComparacion` (`src/ypyquant/estrategias/marcos.py`, almacén `data/marcos/<id>/marco.json`, mismo JSON atómico que las configuraciones): `id`, `nombre`, `notas`, `universo` (+ `universo_etiqueta`; un nombre de archivo se congela como lista, `sp500_pit` se conserva), `fecha_inicio`, `fecha_fin`, `frecuencia_rebalanceo` (`None` = la de cada estrategia, con aviso), `comision`, `slippage`, `benchmark`, `capital_inicial`, `semilla`, `dias_historia_previa` y `parametros_comunes`.
- `aplicar_marco(marco, estrategia, parametros_propios) -> (ConfiguracionExperimento, InformeAplicacion)`: función pura. Los comunes se aplican **solo si la `Parametros` de la estrategia declara el campo**; el informe lista `aplicados`, `no_aplicados`, `sobrescritos` (el marco pisa un propio con el mismo nombre) y `avisos`. Estrategia inexistente o `tfm_*` → `ValueError` («no se puede lanzar»).
- Comparativa: `POST /api/marcos/{id}/comparativa` valida TODAS las estrategias antes de lanzar ninguna, crea N experimentos normales (misma huella: reutiliza los calculados) y guarda `data/marcos/_comparativas/cmp-<hash>.json` (marco copiado + experimentos + informe). El id sale de los experimentos, así que relanzar es idempotente.
- Estimadores comunes (`estimador_sigma`, `frecuencia_estimacion`) en `_EstimadorRobusto` y `_EstimadorIncertidumbre`; la variante mensual remuestrea los cierres de la ventana a fin de mes y usa 12 periodos/año. Reutiliza `estimar_sigma_ledoit_wolf`, `estimar_mu_bayes_stein`, etc.; `estimar_mu_michaud` gana un argumento opcional `estimador_sigma` (defecto = comportamiento anterior).
- `diferencias_ceteris_paribus(configs)`: función pura usada por `GET /api/comparar` (campo `ceteris_paribus`) y mostrada en Comparar para cualquier comparación, también manual.
- UI: página propia **Marcos** (`/marcos`, `/marcos/<id>`) en vez de una sección de Configuraciones: tiene su flujo y agrupa N estrategias. Una configuración puede enlazar con un marco (`id_marco`, opcional) y usa sus fechas por defecto al lanzar backtest.

**Decisiones.** (1) El marco prevalece sobre un parámetro propio que también es común: es lo que pide el ceteris paribus. (2) Editar un marco no toca los experimentos ya lanzados (son recetas congeladas; su huella incluye el entorno). (3) Frecuencia mensual de estimación: el mes en curso se descarta salvo que la fecha sea su último día hábil (calendario, sin mirar datos posteriores); mínimo 12 retornos mensuales (1/N por debajo) y aviso con menos de 24. (4) Michaud no usa `estimador_sigma` internamente en el TFM (cada réplica usa Ledoit-Wolf); se añadió el argumento opcional a `estimar_mu_michaud`.

**Tareas (TDD, hechas).**
- [x] **1.6a** Parámetros `estimador_sigma` y `frecuencia_estimacion`, fuera de la huella en su defecto, con test de huella fija por estrategia y de causalidad (`tests/test_estimadores_comunes.py`).
- [x] **1.6b** `MarcoComparacion`, `aplicar_marco`, `preparar_comparativa`, `AlmacenMarcos`, comparativas y `diferencias_ceteris_paribus` (`tests/test_marcos.py`).
- [x] **1.6c** Rutas `/api/marcos`, `/api/comparativas` y `ceteris_paribus` en `/api/comparar` (`tests/test_api_marcos.py`).
- [x] **1.6d** UI Marcos, lanzador de comparativas y aviso en Comparar (Vitest).

---

## FASE 2 — Estrategias de entrada/salida: `EstrategiaSenales` (M)

### 2.1 Diseño
Hoy hay dos mundos: `ypyquant.signals` + `construir_combinador` + `decidir_orden` (señal en [-1, 1], umbral, tamaño proporcional a |señal|, solo en el bot heredado) y el registro de `Estrategia` (pesos). La base nueva los une: **el usuario escribe reglas; la base las traduce a pesos**, de modo que backtest, barridos y paper usan exactamente la misma `generar_pesos`.

`EstrategiaSenales(Estrategia)` en `src/ypyquant/estrategias/senales.py`:

- **El usuario implementa** (solo estos tres, todos vectorizados y causales sobre `cierres: DataFrame fecha x ticker`):
  - `indicadores(cierres, datos) -> dict[str, DataFrame]` (opcional; cálculos reutilizables).
  - `senal_entrada(cierres, indicadores) -> DataFrame[bool]`.
  - `senal_salida(cierres, indicadores) -> DataFrame[bool]`.
  - Opcionales: `puntuacion(cierres, indicadores) -> DataFrame[float]` (para elegir candidatos cuando hay más señales que huecos) y `salida_personalizada(ticker, fecha, posicion, cierres_hasta) -> bool` para reglas que dependen de la posición.
- **La base aporta** (parámetros `ParametrosSenales`, que las subclases extienden): `max_posiciones`, `ponderacion` (`iguales` | `inversa_volatilidad`), `peso_maximo`, `stop_perdida`, `take_profit`, `stop_trailing`, `max_dias_posicion`, `enfriamiento_dias` (días sin reentrar tras un stop), `mantener_deriva` (True). Frecuencia sugerida `daily` (los stops exigen mirar cada día).
- **Máquina de estados** (en `estado`, JSON-serializable): `posiciones = {ticker: {fecha_entrada, precio_entrada, maximo, peso}}`, `enfriamiento = {ticker: fecha_fin}`, `fecha` (última fecha calculada), `pesos` (última salida). Cada llamada, en orden: (1) actualizar pesos por la deriva del precio desde la llamada anterior (así el motor no opera si no hay cambios: sin turnover espurio); (2) evaluar salidas: señal de salida, stop de pérdida, take profit, trailing, tiempo; (3) entradas sobre los huecos libres, ordenadas por `puntuacion` (empate: orden alfabético, determinista); (4) completar con `CASH` hasta 1.
- **Idempotente:** si se llama dos veces con la misma `fecha`, devuelve los mismos pesos sin mutar el estado (necesario para el bot Alpaca y el catch-up).
- **Rendimiento en barridos:** `preparar(datos)` calcula una vez `indicadores` y las dos tablas booleanas con todo el histórico (permitido porque las operaciones son causales; lo garantiza `verificar_sin_lookahead`, que se ejecuta en el test de cada estrategia). En paper, donde `datos` crece cada día, `generar_pesos` recalcula la caché si `fecha` supera lo precalculado.
- **Stops con cierres.** Se evalúan con el cierre diario. No se modelan stops intradía ni gaps de apertura (se documenta; opción `stop_intradia` queda para la fase 7).
- **Sin short.** El motor es long-only; las señales `-1` del módulo heredado solo se interpretan como salida.

Alternativas descartadas: (a) que cada estrategia implemente el bucle por barra: lento y duplica stops; (b) usar `vectorbt` (`fast` es extra opcional y su modelo de costes no es el del motor: los resultados no cuadrarían con el paper); (c) cambiar el motor para aceptar órdenes en vez de pesos: toca GrowSave.

**Puente con lo heredado:** `senales_tecnicas` (estrategia incluida) envuelve `construir_combinador(config)` de `execution/nautilus`: entrada = señal > `umbral`, salida = señal < -`umbral`. Diferencia documentada: el tamaño pasa de proporcional a |señal| a peso fijo `1/max_posiciones`. Con esto el modo `--config estrategia_ejemplo.json` queda cubierto por el registro y se puede marcar como obsoleto (no se borra en esta fase).

### 2.2 Código de referencia (boceto)

`src/ypyquant/estrategias/senales.py`:
```python
"""
Base para estrategias de entrada/salida (señales) dentro del registro.

Responsabilidad: convertir reglas de compra/venta (tablas booleanas causales) y stops en pesos objetivo
con una máquina de estados persistente, para que backtest, barridos y paper usen la misma lógica.
Se relaciona con: estrategias.base (Estrategia), estrategias.registro, paper (guarda `estado`).
"""
TICKER_EFECTIVO = "CASH"   # peso sin precio: el motor lo salta y esa parte queda en caja

class ParametrosSenales(ParametrosBase):
    max_posiciones: int = Field(10, ge=1, le=200, description="Máximo de posiciones simultáneas")
    ponderacion: Literal["iguales", "inversa_volatilidad"] = "iguales"
    peso_maximo: float = Field(0.25, gt=0, le=1, description="Tope de peso por activo al entrar")
    stop_perdida: float | None = Field(None, gt=0, lt=1, description="Sale si pierde esta fracción desde la entrada")
    take_profit: float | None = Field(None, gt=0, description="Sale si gana esta fracción desde la entrada")
    stop_trailing: float | None = Field(None, gt=0, lt=1, description="Sale si cae esta fracción desde el máximo")
    max_dias_posicion: int | None = Field(None, ge=1, description="Sale tras N sesiones")
    enfriamiento_dias: int = Field(5, ge=0, description="Sesiones sin reentrar tras un stop")

class EstrategiaSenales(Estrategia):
    frecuencia_rebalanceo = "daily"
    Parametros = ParametrosSenales

    def indicadores(self, cierres, datos) -> dict[str, pd.DataFrame]: return {}
    @abstractmethod
    def senal_entrada(self, cierres, indicadores) -> pd.DataFrame: ...
    @abstractmethod
    def senal_salida(self, cierres, indicadores) -> pd.DataFrame: ...
    def puntuacion(self, cierres, indicadores) -> pd.DataFrame | None: return None

    def preparar(self, datos): self._calcular(datos)
    def generar_pesos(self, fecha, datos, estado):
        if estado.get("fecha") == f"{fecha:%Y-%m-%d}":            # idempotente
            return dict(estado["pesos"])
        tablas = self._tablas_hasta(fecha, datos)                 # recalcula si la caché no llega a `fecha`
        posiciones = self._aplicar_deriva(estado, tablas.cierres, fecha)
        posiciones = self._aplicar_salidas(posiciones, estado, tablas, fecha)
        posiciones = self._aplicar_entradas(posiciones, estado, tablas, fecha)
        pesos = self._pesos_con_efectivo(posiciones)
        estado.update(fecha=f"{fecha:%Y-%m-%d}", pesos=pesos, posiciones=posiciones)
        return pesos
```

Ejemplo mínimo, `estrategias_usuario/cruce_medias.py` (se descubre solo y aparece en la UI con su formulario):
```python
"""Cruce de medias: largo mientras la media rápida esté por encima de la lenta. Ejemplo de EstrategiaSenales.
Se relaciona con: ypyquant.estrategias.senales (base y stops) y el registro (descubrimiento automático)."""
from pydantic import Field, model_validator
from ypyquant.estrategias import registrar_estrategia
from ypyquant.estrategias.senales import EstrategiaSenales, ParametrosSenales

@registrar_estrategia
class CruceMedias(EstrategiaSenales):
    nombre = "cruce_medias"
    descripcion = "Compra cuando la media rápida supera a la lenta; vende al revés o por stop (ejemplo de señales)."

    class Parametros(ParametrosSenales):
        rapida: int = Field(20, ge=2, le=200, description="Sesiones de la media rápida")
        lenta: int = Field(100, ge=5, le=400, description="Sesiones de la media lenta")

        @model_validator(mode="after")
        def _rapida_menor_que_lenta(self):
            if self.rapida >= self.lenta:
                raise ValueError("rapida debe ser menor que lenta")
            return self

    def indicadores(self, cierres, datos):
        p = self.parametros
        return {"rapida": cierres.rolling(p.rapida).mean(), "lenta": cierres.rolling(p.lenta).mean()}

    def senal_entrada(self, cierres, ind):
        return (ind["rapida"] > ind["lenta"]).fillna(False)

    def senal_salida(self, cierres, ind):
        return (ind["rapida"] < ind["lenta"]).fillna(False)

    def puntuacion(self, cierres, ind):                      # entre candidatos, el de mayor distancia relativa
        return ind["rapida"] / ind["lenta"] - 1.0
```
RSI sería idéntico cambiando `indicadores` (`rsi`), `senal_entrada = rsi < sobreventa` y `senal_salida = rsi > sobrecompra`. Barrido típico: `rapida=[10,20,30]`, `lenta=[50,100,150,200]`, `stop_perdida=[None,0.08,0.12]` (las combinaciones inválidas se omiten gracias a la tarea 0.2).

### 2.3 Archivos
Crear: `src/ypyquant/estrategias/senales.py`, `src/ypyquant/estrategias/senales_tecnicas.py` (puente; se registra en `incluidas.py`), `estrategias_usuario/cruce_medias.py`, `tests/test_senales.py`.
Modificar: `src/ypyquant/estrategias/__init__.py` (exportar `EstrategiaSenales`), `incluidas.py` (importar el puente), `docs/ui.md` (guía "estrategias de señales" con el cruce de medias), `pipelines/paper_trading/estrategia_ejemplo.json` y `docs` (nota de obsolescencia del modo heredado).

### 2.4 Tareas (TDD)
- [x] **2.1** Tests de la máquina de estados con precios sintéticos construidos a mano (serie que sube, cruza y baja): entrada el día esperado; con `max_posiciones=1` y dos candidatos entra el de mayor `puntuacion`; salida por señal; `stop_perdida` dispara con una caída del 10 % (`stop=0.08`); `stop_trailing` usa el máximo desde la entrada; `enfriamiento_dias` impide reentrar; `max_dias_posicion`.
- [x] **2.2** Tests de pesos: suman 1 contando `CASH`; sin señales todo es `CASH`; con `mantener_deriva` y sin cambios dos días seguidos devuelven pesos cercanos a los previos (turnover ≈ 0 en `BacktestEngine`, comprobado con `rotaciones`).
- [x] **2.3** Idempotencia y serialización: llamar dos veces con la misma fecha no cambia `estado`; `json.dumps(estado)` funciona; instancia nueva con el estado cargado de JSON continúa igual que la original (base de la reanudación del paper).
- [x] **2.4** Integración con el motor: `ejecutar_con_datos` con `cruce_medias` sobre `FuenteSintetica` devuelve métricas, pesos con filas `CASH`, y `costes_sobre_ganancia_bruta` definido. Una rejilla `rapida×lenta` corre en un barrido con celdas omitidas.
- [x] **2.5** `verificar_sin_lookahead` (fase 0) sobre `cruce_medias` y `senales_tecnicas` sin errores; test con una subclase tramposa (`shift(-1)` en `senal_entrada`) que sí falla.
- [x] **2.6** Puente `senales_tecnicas`: con el generador técnico por defecto devuelve pesos válidos; documentar diferencias de tamaño.
- [x] **2.7** Implementar, documentar y commit `fase 2: EstrategiaSenales y cruce_medias`.

**Implementada (fase 2).** Guía: `docs/estrategias_senales.md`; tests: `tests/test_senales.py` (máquina de estados) y `tests/test_senales_ejemplos.py` (integración). Desviaciones del boceto: (1) segunda plantilla `canal_rsi_volatilidad` (ruptura de canal + RSI + tamaño por volatilidad); (2) parámetros extra `ventana_volatilidad` y `peso_maximo` aplica también al tamaño por volatilidad; (3) las operaciones de estas estrategias llevan la columna `motivo` (`ejecutar_con_datos` cruza `EstrategiaSenales.eventos` con las operaciones del motor; el resto de estrategias no cambia) y `AlmacenExperimentos.ejecutar_sql` lee con `union_by_name`; (4) el detalle del experimento muestra el motivo y un resumen de entradas/salidas, y el formulario admite parámetros opcionales (`float | None`); (5) el enfriamiento cuenta días hábiles (`BDay`) y solo se activa tras `stop_perdida`/`stop_trailing`; (6) el modo heredado `--config estrategia_ejemplo.json` queda **obsoleto** (documentado en la guía; el JSON y el CLI no se tocan). Pendiente para la fase 4: tabla de «viajes» (entrada+salida emparejadas con P&L por posición) y marcas de señal sobre el gráfico de precios.

Riesgos: look-ahead al precalcular en `preparar` (mitigado con el verificador y documentando "solo operaciones causales"); el `CASH` aparece en `pesos.parquet` y en la UI de pesos (filtrar o mostrarlo como "Efectivo"); metrica `n_operaciones` y `tasa_acierto` del motor cuentan operaciones por rebalanceo, no "viajes completos" (ver nota en la fase 4: tabla de viajes por posición).

---

## FASE 3 — Núcleo del paper interno (L)

### 3.1 Diseño

**Paquete nuevo `src/ypyquant/paper/`** (separado de `estrategias/` para que `estrategias` no dependa del tiempo real):

| Módulo | Responsabilidad |
|---|---|
| `calendario.py` (**hecho** en `execution/calendario.py`: `fechas_de_rebalanceo`, `toca_rebalancear`) | `es_dia_de_rebalanceo(dia, dias_trading, inicio, frecuencia, ultimo_rebalanceo)` reutilizando los métodos estáticos del motor; `dias_pendientes(ultima_fecha, dias_con_datos, ahora_ny)` (solo sesiones cerradas) |
| `ejecutor.py` (**hecho**, otro diseño: `execution/ejecutor.py` puro, sin envolver el motor privado; ver `2026-11-21-ejecutor.md`) | `EjecutorCartera`: envuelve una instancia de `BacktestEngine` con estado restaurado (`cash`, `positions`) y llama a `_execute_rebalance`; devuelve operaciones y rotación. **Único** sitio que toca métodos privados del motor |
| `sesion.py` | `SesionPaper` (Pydantic: configuración congelada, capital, modo de ejecución, estado, perfil Alpaca opcional) y `EstadoCartera` (efectivo, posiciones, `estado_estrategia`, pesos pendientes, último rebalanceo) |
| `almacen.py` | `AlmacenPaper`: `data/paper_interno/<id_sesion>/` con `sesion.json`, `estado.json`, `equity.parquet`, `operaciones.parquet`, `pesos.parquet`, `eventos.jsonl`; bloqueo por archivo |
| `simulador.py` | `SimuladorPaper.avanzar(sesion, hasta=None)` y `avanzar_todas(...)` con precios cargados una vez para la unión de tickers |
| `__main__.py` | CLI: `avanzar [--sesion ID] [--hasta FECHA]`, `estado`, `crear --configuracion ID`, `pausar/reanudar/detener` |
| `comparacion.py` | (fase 6) métricas interno vs Alpaca |

**Bucle diario por sesión** (para cada sesión activa, cada día de trading cerrado pendiente `D`):
1. Precios: cierre ajustado de `D` y apertura de `D` (modo `apertura_siguiente` también necesita la apertura de `D`).
2. **Re-base por ajuste** (tarea 3.5): comparar el cierre ajustado de la última fecha procesada guardado con el recién descargado; si el factor `f` difiere más de `1e-6`, escalar `cantidad /= f` y `precio_medio *= f` (dividendos reinvertidos implícitamente; coherente con `Close` ajustado del backtest).
3. Ejecutar órdenes pendientes (modo `apertura_siguiente`): `EjecutorCartera.rebalancear(pesos_pendientes, apertura_D, D)` → operaciones con comisión y slippage idénticos al backtest.
4. Valorar a cierre de `D` y escribir una fila de equity (equity, efectivo, nº posiciones, benchmark reescalado).
5. Si `D` es día de rebalanceo (calendario del motor): `pesos = estrategia.como_funcion()(D, datos)`; en modo `cierre` se ejecutan ya al cierre de `D` (equivale a `run_backtest`); en `apertura_siguiente` quedan en `pesos_pendientes` para `D+1`. Se guarda `pesos.parquet` (objetivo) y el `estado` de la estrategia.
6. **Commit del día:** se escriben los Parquet/JSONL y, **al final**, `estado.json` con `ultima_fecha_procesada = D`. Si el proceso cae a mitad, al reanudar se truncan las filas posteriores a `ultima_fecha_procesada` y se repite el día (idempotente).

**Calendario y festivos.** Los días de trading salen de los datos (unión de fechas con barra, como el motor), no de un calendario de bolsa externo. Una sesión `D` solo se procesa si (a) `D` es anterior a hoy en Nueva York, o es hoy y son más de las 16:30 NY (`zoneinfo.ZoneInfo("America/New_York")`), y (b) el benchmark y al menos el 80 % de los tickers de la configuración tienen barra en `D` (evita procesar barras parciales de yfinance). Un ticker sin barra N días seguidos se liquida a su último precio con aviso en `eventos.jsonl` (deslistado/cambio de ticker).

**Reanudación y PC apagado.** `avanzar` calcula `dias_pendientes` desde `ultima_fecha_procesada` y procesa en orden todos los días con datos (catch-up). En modo `apertura_siguiente` el catch-up es exacto porque las barras históricas incluyen la apertura. Nada depende de que el proceso estuviera vivo.

**Bloqueo.** `sesion.lock` con `os.open(..., O_CREAT|O_EXCL)` + pid; si el pid ya no existe, se roba el bloqueo. Evita que el programador, la API y un `avanzar` manual pisen la misma sesión. Escrituras con los helpers `_escribir_json`/`_escribir_parquet` de `estrategias/almacen.py` (reintentos de Windows).

**Costes.** Mismos `comision` y `slippage` de la configuración (por defecto 0,001/0,001). Se pueden **sobrescribir por sesión** (`comision_paper`, `slippage_paper`) porque Alpaca no cobra comisión: así se puede simular "costes realistas" frente a "costes Alpaca" sin tocar la receta.

**Precios y fuentes.** Reutiliza `FuenteYFinanceCache` (antes `CargadorPreciosCache`) (yfinance con caché Parquet). Hay que forzar refresco del tramo reciente: hoy el cargador solo descarga cuando el rango pedido no está cubierto (`cubierto[1] >= fin_pedido`), así que cada día se pide `fecha_fin = hoy` y se vuelve a descargar el rango completo de cada ticker; para universos pequeños (<50 tickers) es aceptable; si el universo es grande se añade descarga incremental (fase 7). Segunda fuente opcional para validar: cliente de datos de Alpaca (ya se usa `StockHistoricalDataClient` en el bot). Intradía queda fuera: las estrategias del registro son de barra diaria y rebalanceo ≥ diario; yfinance da horario solo ~730 días y Alpaca IEX es gratuito pero parcial (por verificar). Se deja la interfaz `FuenteBarras` para enchufarlo en la fase 7.

**Alternativas descartadas:** guardar cada sesión en una sola base SQLite (más piezas, y el resto del sistema ya vive en Parquet/JSON); un proceso por sesión (N procesos residentes); calcular el equity a partir de Alpaca (no hay Alpaca para las N configuraciones).

### 3.2 Tareas (TDD, cada una con su commit)
- [x] **3.1 Calendario.** Test `tests/test_paper_calendario.py`: para frecuencias `daily/weekly/monthly/quarterly` y un calendario con festivos sintéticos, el conjunto de días donde `es_dia_de_rebalanceo` es True coincide con `BacktestEngine._alinear_a_dias_de_trading(_get_rebalance_dates(...), dias)` (propiedad con Hypothesis, skill `property-based-testing`). Caso "PC apagado": llamar con varios días pendientes en bloque da el mismo resultado que día a día.
- [x] **3.2 Ejecutor.** Test: `EjecutorCartera` con estado vacío + pesos + precios produce las mismas operaciones (cantidad, precio, comisión) que `BacktestEngine._execute_rebalance` del motor puro; con el estado guardado/restaurado a mitad produce el mismo resultado que sin interrupción.
- [x] **3.3 Paridad incremental (test clave).** `tests/test_paper_paridad.py`: con `FuenteSintetica`, estrategias `pesos_iguales`, `momentum` y `cruce_medias` (modo `cierre`), avanzar día a día durante 250 sesiones da una curva de equity y una lista de operaciones **iguales** (tolerancia 1e-9) a `ejecutar_con_datos` del mismo rango. Variante: partir el avance en dos procesos (guardar/cargar `estado.json` en medio) y obtener lo mismo. Este test protege contra cambios futuros del motor (GrowSave).
- [x] **3.4 Modo `apertura_siguiente`.** Test con barras de apertura/cierre fabricadas: decisión en `D`, fill en `apertura(D+1)` con slippage; el equity de `D` aún refleja la cartera anterior; la última decisión queda pendiente al terminar los datos.
- [x] **3.5 Re-base por ajuste.** Test: serie con un "dividendo" (el histórico ajustado se reescala 2 % entre dos descargas); el equity no tiene salto y las cantidades se re-basan; sin cambio de histórico no ocurre nada.
- [x] **3.6 Almacén y reanudación.** Test de crash: simular excepción tras escribir `equity.parquet` y antes de `estado.json`; al reanudar se trunca y se repite el día sin duplicar filas. Test de bloqueo: dos avances concurrentes (hilos) no corrompen; bloqueo huérfano con pid inexistente se recupera.
- [x] **3.7 Día no cerrado y datos parciales.** Test: con "ahora" a las 12:00 NY no se procesa el día en curso; barra del benchmark ausente → se pospone; ticker que desaparece 5 sesiones → liquidación con aviso.
- [x] **3.8 Varias sesiones, una descarga.** Test: `avanzar_todas` con 3 sesiones sobre universos solapados llama al cargador una vez por ticker (cargador espía) y cada sesión conserva su cartera independiente.
- [x] **3.9 Estado de estrategias incluidas.** Test: cada estrategia registrada (`listar_estrategias()`) soporta `json.dumps(estado)` tras un rebalanceo; las que usan arrays NumPy (`_EstimadorRobusto`: `pesos_previos`) se corrigen guardando listas (cambio en `incluidas.py`) o el almacén convierte con un codificador (`numpy -> list`, con test de ida y vuelta).
- [x] **3.10 CLI.** Test de `python -m ypyquant.paper avanzar` con `YPYQUANT_PRECIOS=sintetico` y carpeta temporal (variable `YPYQUANT_PAPER_RAIZ`); salida con resumen por sesión y código de salida ≠ 0 si alguna falló (para el programador).
- [x] **3.11** `docs/ypyquant.md`: sección "Paper interno"; commit `fase 3: paper interno (núcleo)`.

Tamaño estimado: 900-1300 líneas de código + 700-900 de tests.

Riesgos de la fase: acceso a métodos privados del motor (mitigado por `ejecutor.py` único y el test 3.3); precisión de yfinance (revisiones, huecos, límites de peticiones: reintentos y caché ya existen); sesgo de decidir con barras parciales (3.7); el modo `cierre` es optimista (se muestra en la ficha de la sesión como "ejecución optimista"); estrategias de optimización pesadas (max-Sharpe con 756 días) tardan: en `daily` pueden ser minutos por día y sesión, aceptable para el job nocturno pero hay que medirlo.

---

## FASE 4 — Paper interno en API, UI y programador (M)

### 4.1 Endpoints (`app/api/rutas_paper_interno.py`, prefijo `/api/paper/interno`)
| Método | Ruta | Descripción |
|---|---|---|
| GET | `/sesiones` | Lista con estado, equity actual, retorno, drawdown, última fecha, días pendientes |
| POST | `/sesiones` | `{configuracion_id, version?, capital_inicial?, modo_ejecucion, comision?, slippage?, fecha_alta?}`; el alta fija `ultima_fecha_procesada = día anterior` |
| GET | `/sesiones/{id}` | Detalle: configuración congelada, métricas forward (Sharpe, volatilidad, drawdown, turnover, costes pagados), avisos |
| GET | `/sesiones/{id}/equity` `/operaciones` `/pesos` `/viajes` | Series y tablas (viajes = posiciones cerradas con entrada, salida, motivo si es de señales) |
| POST | `/sesiones/{id}/{pausar,reanudar,detener}` | Control |
| POST | `/avanzar` | Avanza todas las activas en un hilo del `ejecutor` (estado `en_curso` consultable); idempotente |
| GET | `/avanzar/estado` | Último avance (hora, días procesados, errores) |
| GET | `/sesiones/{id}/espejo` | Lanza (o reutiliza) el backtest `fecha_alta → última fecha` de la misma configuración (modo `cierre`) y devuelve ambas curvas: "esperado" vs "real" |

El arranque de la API (`crear_app`) ejecuta el catch-up en segundo plano si hay sesiones activas (con parámetro `catchup_al_arrancar=True`, desactivable en tests) y marca como "interrumpido" un avance que quedó en curso, igual que `marcar_interrumpidos` del almacén de experimentos.

### 4.2 UI
- La página **Paper trading** actual se divide en pestañas: **Sesiones internas** (nueva, por defecto), **Alpaca** (lo actual) y **Comparación** (fase 6).
- Sesiones internas: tabla (configuración, modo, días, retorno, Sharpe, drawdown, costes, estado, "al día / n días pendientes"); botón **Avanzar ahora**; selección múltiple → gráfico de equity normalizada a 100 reutilizando `Graficos.tsx` y la lógica de `Comparar.tsx` (varias sesiones y su benchmark en un solo gráfico). Detalle de sesión: curva real vs backtest espejo, operaciones, pesos actuales, avisos.
- Botón **Paper interno** en `ConfiguracionDetalle`.

### 4.3 Programación (Windows)
`scripts/instalar_tarea_paper.ps1` (lo ejecuta el usuario; no se registra nada sin su OK) crea una tarea diaria con `schtasks`/`Register-ScheduledTask`: a las 23:30 hora local (cierre de EE. UU. = 22:00 en Madrid con calendarios sincronizados; margen para datos de yfinance), `StartWhenAvailable` (si el PC estaba apagado, se ejecuta al encender), `WakeToRun` opcional, `RunOnlyIfNetworkAvailable`, ejecutando `uv run python -m ypyquant.paper avanzar` con directorio de trabajo en el repo, log en `data/paper_interno/avance.log`. Dos disparadores: 23:30 diario y "al iniciar sesión" (catch-up).

### 4.4 Tareas
- [x] **4.1** Tests de API con `FuenteSintetica`, raíz temporal y reloj inyectable (`ahora` fijo): crear sesión desde configuración, avanzar, listar, equity coincide con la del simulador, pausar/reanudar, 409 al avanzar dos veces a la vez, 404/422.
- [x] **4.2** Implementar rutas, `Contexto.almacen_paper`, `reloj` inyectable, catch-up al arrancar.
- [x] **4.3** Backtest espejo: test de que reutiliza el experimento si existe (misma huella) y que las curvas coinciden en modo `cierre` (continuación del test de paridad 3.3 a nivel API).
- [x] **4.4** Frontend (tipos, hooks con polling mientras `avanzar` está en curso, pestañas, tabla, gráfico). Tests Vitest de la tabla (estados vacíos, "n días pendientes") y de la selección múltiple.
- [x] **4.5** Script PowerShell + sección en `docs/ui.md` y README (cómo instalar la tarea, qué hace si el PC está apagado, cómo ver el log).
- [x] **4.6** Commit `fase 4: paper interno en API/UI`.

---

## FASE 5 — Alpaca multi-perfil y bot por configuración (M)

### 5.1 Diseño
- **Perfiles de claves en `.env`:** `ALPACA_<PERFIL>_API_KEY` y `ALPACA_<PERFIL>_SECRET_KEY` (PERFIL en mayúsculas, `[A-Z0-9_]`). Compatibilidad: `ALPACA_API_KEY/ALPACA_SECRET_KEY` equivalen al perfil `DEFECTO`. `.env.example` actualizado. `GET /api/paper/alpaca/perfiles` lista perfiles detectados (nombre, últimos 4 caracteres de la clave, estado de conexión), nunca el secreto.
- **`ServicioAlpaca(perfil)`** y `Contexto.alpacas: dict[str, ServicioAlpaca]`; las rutas actuales (`/api/paper/cuenta`, etc.) reciben `?perfil=` (por defecto `DEFECTO`) para no romper la UI ni los tests existentes.
- **`GestorBots`** reemplaza a `BotControl` único: un subproceso por perfil (`data/paper_trading/<perfil>/bot.log`, `bot.json` con pid, configuración y hora). Al arrancar la API se **rehidrata** leyendo `bot.json` y comprobando que el pid sigue vivo y es el proceso esperado (hoy, si la API muere, el bot queda huérfano y la API lo ignora). Límite blando de 3 perfiles activos.
- **1 cuenta = 1 configuración.** El vínculo (`configuracion_id`, versión, perfil, fecha de alta, equity inicial de la cuenta) se guarda en `data/paper_trading/<perfil>/vinculo.json`. Una configuración puede estar a la vez en una sesión interna y en una cuenta Alpaca (eso es lo que se compara). El paper interno vinculado toma por defecto `capital_inicial = equity de la cuenta al vincular`.
- **Bot refactorizado** (`pipelines/paper_trading`): `--perfil X --configuracion ID [--version N]` (lee estrategia, parámetros, universo y frecuencia de la configuración; deja de recibirlos por línea de comandos). Cambios en `alpaca_vivo.py`:
  1. **Respeta la frecuencia:** solo llama a la estrategia y envía órdenes si `es_dia_de_rebalanceo` (mismo calendario que el simulador) y si no se hizo ya hoy (`ultimo_rebalanceo` persistido).
  2. **Estado persistido** de la estrategia en `data/paper_trading/<perfil>/estado_estrategia.json` (hoy se pierde al reiniciar; imprescindible para `EstrategiaSenales`).
  3. **Modo diario recomendado:** `--modo diario` ejecuta una pasada y sale (ya existe `--una-vez`), pensado para el mismo programador de tareas; el bucle de 60 s queda como `--modo continuo`. Hipótesis a verificar con una orden real de prueba: una orden de mercado `DAY` enviada con el mercado cerrado queda en cola y se ejecuta en la apertura siguiente, lo que casa con el modelo `apertura_siguiente`.
  4. **Fills enriquecidos:** `RegistroFills` añade `perfil`, `configuracion_id`, `precio_referencia` (apertura prevista = último cierre en el momento de la decisión) y `tipo` (rebalanceo). Los JSONL pasan a `data/paper_trading/<perfil>/fills.jsonl`.
  5. **Acciones enteras:** `decidir_ordenes_rebalanceo` trabaja con enteros (`int(...)`), lo que desvía la cartera frente al simulador fraccionario. Alpaca admite acciones fraccionarias en paper (por verificar con la versión de `alpaca-py` fijada); se deja una opción `fraccionarias` y, mientras tanto, el informe de comparación separa el "descuadre por enteros".
- **Instantánea diaria de la cuenta:** el job diario guarda `instantaneas.parquet` (fecha, equity, efectivo, posiciones con cantidad y precio) por perfil. Alpaca devuelve el historial de equity (`get_portfolio_history`), pero no las posiciones pasadas; sin esta tabla no se puede medir el descuadre de cartera.

### 5.2 Archivos
Crear: `src/ypyquant/paper/perfiles.py` (descubrir perfiles desde el entorno, sin leer secretos para mostrar), `src/ypyquant/paper/instantaneas.py`, `tests/test_alpaca_perfiles.py`, `tests/test_bot_configuracion.py`.
Modificar: `app/api/paper.py` (`ServicioAlpaca(perfil)`, `GestorBots`), `app/api/rutas_paper.py` (`perfil`, `/perfiles`, `/bots`), `app/api/contexto.py`, `app/api/main.py`, `pipelines/paper_trading/{__main__,config,alpaca_vivo}.py`, `src/ypyquant/execution/nautilus/registro_fills.py`, `.env.example`, `docs/ui.md`, `app/web/src/paginas/PaperTrading.tsx` (selector de perfil), `app/web/src/api/*`.

### 5.3 Tareas (TDD)
- [ ] **5.1** `perfiles.py`: con un entorno simulado (`monkeypatch.setenv`) detecta `DEFECTO`, `A`, `B`; ignora variables incompletas; no expone secretos.
- [ ] **5.2** `ServicioAlpaca(perfil)` con `fabrica_cliente` falsa: claves de cada perfil por separado; error claro y 503 con el nombre del perfil.
- [ ] **5.3** `GestorBots`: con `construir_comando` falso, dos perfiles arrancan a la vez, un perfil ocupado devuelve 409, y la rehidratación reconoce un pid vivo (proceso de prueba) y descarta uno muerto.
- [ ] **5.4** Bot: con un cliente Alpaca falso y reloj inyectable, un martes sin día de rebalanceo no envía órdenes; en día de rebalanceo envía una vez y no repite en la segunda iteración; tras "reiniciar" (nueva instancia leyendo `estado_estrategia.json`) una `cruce_medias` conserva sus posiciones.
- [ ] **5.5** Fills enriquecidos: formato, retrocompatibilidad (lectura de filas antiguas sin `perfil`).
- [ ] **5.6** Rutas con `?perfil=`; la UI conserva el comportamiento con el perfil por defecto (tests existentes de `tests/test_api.py` pasan sin cambios).
- [ ] **5.7** Orden de prueba manual documentada (checklist, no automática): una orden DAY fuera de horario en el perfil de pruebas y comprobar cuándo se ejecuta.
- [ ] **5.8** Commit `fase 5: Alpaca multi-perfil y bot por configuración`.

Riesgos: claves en logs (filtro de redacción en `GestorBots.log`); límite de 3 cuentas y rate limits; cambiar el comportamiento del bot actual (hoy rebalancea cada iteración): se anuncia en `docs/ui.md` y se conserva `--modo continuo` con una advertencia; órdenes huérfanas si el bot cae entre decidir y enviar (se reconcilia leyendo posiciones al arrancar y comparando con el objetivo).

---

## FASE 6 — Comparación interno vs Alpaca y backtest espejo (M)

### 6.1 Qué se compara (funciones puras en `src/ypyquant/paper/comparacion.py`)
Para una configuración con sesión interna **y** cuenta Alpaca vinculada, sobre el periodo común:
1. **Equity normalizada** (base 100 en el alta) de interno, Alpaca, backtest espejo y benchmark; diferencia acumulada interno−Alpaca en puntos y en %.
2. **Tracking error** anualizado de los retornos diarios (interno − Alpaca), correlación y beta de uno sobre otro (`PerformanceAnalyzer.compare_to_benchmark` ya calcula alfa/beta/TE/IR; se reutiliza con interno como "benchmark").
3. **Slippage real por operación:** para cada fill de Alpaca emparejado con la operación interna del mismo día y ticker: `lado·(precio_fill − precio_modelo)/precio_modelo` en puntos básicos, con `precio_modelo` = apertura (modo `apertura_siguiente`) y también contra el cierre del día; resumen (media, mediana, percentil 95, por lado, por ticker) y comparación con el `slippage` supuesto (0,1 % = 10 pb).
4. **Costes:** comisiones modeladas internas frente a 0 de Alpaca → "impacto de costes modelados" (equity interna con costes de Alpaca = comisión 0 y slippage real medido). Es la métrica que dice si 0,001+0,001 es pesimista.
5. **Descuadre de cartera:** por día, `Σ|peso_alpaca − peso_interno|/2` usando `instantaneas.parquet` y `pesos` internos a deriva; separado en "por enteros" (diferencia de redondeo) y "por decisión distinta" (señal distinta o ejecución perdida).
6. **Divergencias de decisión:** días en que una cartera rebalanceó y la otra no, órdenes rechazadas/no ejecutadas, ejecuciones parciales.
7. **Esperado vs real (backtest espejo):** mismo periodo que la sesión; diferencia de retorno y de Sharpe, para ver si la estrategia se degrada fuera de muestra (el uso estadístico principal del paper: test forward real). Se acompaña de una banda sencilla de "¿es compatible con el Sharpe de desarrollo?" con el error estándar del Sharpe con N días (solo orientativo: con unos meses de datos la potencia es muy baja; el informe lo dice).

### 6.2 API y UI
- `GET /api/paper/comparacion/{configuracion_id}` → series y tablas anteriores; 404 si falta alguna de las dos mitades (la UI explica qué falta).
- Pestaña **Comparación** en Paper trading: selector de configuración, gráfico de equity normalizada (interno / Alpaca / espejo / benchmark), tarjetas KPI (diferencia acumulada, tracking error, slippage medio vs supuesto, descuadre medio), tabla de fills emparejados, histograma de slippage, lista de divergencias.

### 6.3 Tareas (TDD)
- [ ] **6.1** `alinear_equity`: series con fechas distintas (festivos, día en que falló el avance) → intersección y rebase a 100; test con huecos.
- [ ] **6.2** `tracking_error`/`correlacion` con series sintéticas de TE conocido.
- [ ] **6.3** `emparejar_fills`: fills y operaciones internas con tickers/días que no emparejan, ventas parciales, dos fills el mismo día; slippage signo correcto (compra con fill mayor = coste positivo).
- [ ] **6.4** `descuadre_cartera` separando enteros de decisión (caso con fraccionarias vs enteros a mano).
- [ ] **6.5** Ruta de API con Alpaca falso y datos sintéticos; casos de mitad ausente.
- [ ] **6.6** Frontend: pestaña y gráficos; test de estado vacío ("falta vincular una cuenta").
- [ ] **6.7** Commit `fase 6: comparación interno vs Alpaca`.

Riesgos: emparejar operaciones cuando Alpaca no ejecuta por falta de fondos/lote entero (se reporta como divergencia, no se oculta); el precio de apertura de yfinance no es el de la subasta de apertura de Alpaca: parte del "slippage" será diferencia de fuente de datos (se documenta; con el cliente de datos de Alpaca como segunda fuente se puede acotar); poca muestra durante los primeros meses.

---

## FASE 7 — Opcional (M)

- **Modo de ejecución en el motor** (`ejecucion: "cierre" | "apertura_siguiente"` en `BacktestConfig` y `ConfiguracionExperimento`), de modo que backtest, barridos y paper compartan el mismo supuesto. Regla de huella: solo se incluye en `huella()` cuando no es `cierre` (no invalida experimentos existentes). Test: con `cierre` los resultados son idénticos byte a byte a los actuales; GrowSave no lo usa y no cambia.
- **Acciones enteras** (`acciones_enteras: bool = False` en `BacktestConfig`): `_buy/_sell` redondean cantidades; permite que interno y Alpaca casen. Misma regla de no regresión.
- **Stops intradía** en `EstrategiaSenales` (usar `High/Low`; modelar gaps con el peor de apertura y nivel del stop).
- **Descarga incremental** de precios y segunda fuente (Alpaca data) con comparación de calidad.
- **Alertas** (correo/Telegram) cuando una sesión no avanza N días, cuando el descuadre supera un umbral o cuando una sesión rebasa un drawdown fijado.

---

## 8. Mapa de archivos (resumen)

Crear:
`src/ypyquant/estrategias/{configuraciones,senales,senales_tecnicas,causalidad}.py`
`src/ypyquant/paper/{__init__,calendario,ejecutor,sesion,almacen,simulador,perfiles,instantaneas,comparacion,__main__}.py`
`app/api/{rutas_configuraciones,rutas_paper_interno}.py`
`estrategias_usuario/cruce_medias.py`
`scripts/instalar_tarea_paper.ps1`
`app/web/src/paginas/{Configuraciones,ConfiguracionDetalle}.tsx`, `componentes/GuardarConfiguracion.tsx`, pestañas de paper
`tests/{test_causalidad,test_configuraciones,test_api_configuraciones,test_senales,test_paper_calendario,test_paper_ejecutor,test_paper_paridad,test_paper_almacen,test_api_paper_interno,test_alpaca_perfiles,test_bot_configuracion,test_comparacion}.py`

Modificar:
`estrategias/experimentos.py` (rejilla tolerante), `estrategias/incluidas.py` (estado serializable, puente), `estrategias/__init__.py`, `app/api/{contexto,main,paper,rutas_paper}.py`, `pipelines/paper_trading/{__main__,config,alpaca_vivo}.py`, `execution/nautilus/registro_fills.py`, `.env.example`, `docs/ui.md`, `docs/ypyquant.md`, `README.md`, `app/web/src/{App,api/hooks,api/tipos,componentes/Layout}.tsx` y las páginas de detalle de experimento y barrido.
Sin tocar en fases 1-6: `portfolio/backtest_engine.py`.

## 9. Riesgos transversales y mitigaciones

| Riesgo | Dónde | Mitigación |
|---|---|---|
| **Look-ahead** en `preparar()` (ve todo el histórico) o en reglas con `shift(-n)`/`center=True` | Señales y estrategias de usuario | `verificar_sin_lookahead` en tests de cada estrategia; documentación; el paper forward no puede hacer trampa pero un diseño con look-ahead produciría un backtest espléndido y un paper decepcionante |
| **Decidir y ejecutar con el mismo cierre** | Backtest actual | Modo `apertura_siguiente` en paper; opción de motor en fase 7; etiqueta "ejecución optimista" en la UI |
| **Winner's curse** al guardar la mejor del barrido | Fase 1 | Guardar métricas de origen, avisar, sugerir meseta (◆), mostrar DSR/PBO y estado del reservado; el paper es la validación fuera de muestra |
| **Festivos / medios días / barras parciales** | Fase 3 | Calendario derivado de datos + umbral de cobertura + cierre de sesión en hora NY |
| **Supervivencia** (`universos/sp500.txt` actual) | Backtest y barridos | Aviso en la configuración; paper forward lo evita pero los deslistados se gestionan (liquidación con aviso) |
| **Ajustes por dividendos/splits** | Fase 3 | Re-base por ajuste (3.5); documentar diferencias con Alpaca (dividendo en efectivo) |
| **Fuente de datos única (yfinance)** | Todo | Caché, reintentos, umbral de cobertura, segunda fuente opcional (fase 7) |
| **Motor compartido con GrowSave** | Fases 3 y 7 | Cero cambios al motor hasta la fase 7; test de paridad 3.3; cambios solo aditivos con defecto igual al comportamiento actual; ejecutar `test_backtest_motor`, `test_paridad_tfm` antes de cada merge |
| **Estado no serializable** | Fases 2-5 | Contrato JSON + test sobre todas las estrategias registradas |
| **Concurrencia** (programador + API + CLI) | Fase 3-4 | Bloqueo por sesión, commit atómico por día, avance idempotente |
| **PC apagado/dormido** | Fase 4 | `StartWhenAvailable`, catch-up al arrancar la API, "días pendientes" visibles |
| **Claves en logs/respuestas** | Fase 5 | Redacción, solo últimos 4 caracteres, `.env` ya ignorado por git |
| **Bot actual rebalancea cada minuto** | Fase 5 | Corregir con el calendario compartido y estado persistido; cambio anunciado |
| **Poca muestra en paper** | Fase 6 | El informe lo advierte: unos meses no bastan para distinguir Sharpe real de ruido; sirve sobre todo para validar ejecución, costes y degradación gruesa |

## 10. Estimación global

- Código nuevo: ≈ 3 500-4 500 líneas Python (≈ 40 % tests por encima de ese rango) y ≈ 1 500 líneas TS/TSX.
- Tiempo de trabajo con tests: fases 0-2 ≈ 4-5 días, fase 3 ≈ 4-5 días, fase 4 ≈ 2-3 días, fase 5 ≈ 3 días, fase 6 ≈ 3 días; total ≈ 3-4 semanas de dedicación parcial. Hito útil mínimo (guardar configuraciones + señales + paper interno por CLI y programador): fases 0-3 + el script de la 4.3.
- Verificación al cerrar cada fase: `uv run pytest -q`, `uv run ruff check .`, `cd app/web && npm test -- --run && npm run build`, y una pasada manual con `YPYQUANT_PRECIOS=sintetico`.


---

## Estado 2026-11-22 (fases 3 y 4: paper interno, núcleo + API + web)

Hecho en la rama `fase3/paper-interno` (tareas 3.1-3.11 y 4.1-4.6 marcadas arriba). Lo implementado se aparta del plan original en lo siguiente.

**Arquitectura (usa lo que ya existía en lugar de lo que el plan proponía)**
- No hay `EjecutorCartera` que envuelva el motor. El simulador **planifica con el Ejecutor** (`planificar_rebalanceo` + `POLITICA_MOTOR`) y **ejecuta con el `BrokerSimulado`** aplicando la misma comisión y slippage que el motor. Para igualar `_buy`/`_sell` al céntimo traduce cada `Orden` a la cantidad que calculan (compra recortada a `efectivo / (1 + comisión)` y sin operar menos de 1; venta de `importe` al precio con slippage, limitada a la posición; lo que queda por debajo de 0,001 acciones se descarta). Ningún método privado del motor se toca.
- Calendario: `toca_rebalancear` / `fechas_de_rebalanceo` de `execution/calendario.py`; `paper/calendario.py` solo añade las sesiones cerradas y los días pendientes (la tarea 3.1 se redujo a esto). Los tests del calendario viven en `tests/test_paper_simulador.py`, no en un `test_paper_calendario.py` propio.
- Persistencia con `AlmacenDocumentos` (`AlmacenPaper`), sin helpers JSON propios; `RaizDatos.paper_interno`. `eventos` es `eventos.json` (lista), no `eventos.jsonl`.
- Métricas forward con `evaluar(simulacion_desde_tablas(...), validar=False)`, la misma función que el backtest.
- Paquete `ypyquant.paper` = `sesion`, `almacen`, `simulador`, `calendario`, `servicio` (`ServicioPaper`, `ControlAvance`) y `__main__`; no hay `comparacion.py` (fase 6).

**Cambio de producto: la sesión nace de un experimento terminado, no de una configuración**
- `POST /api/paper/interno/sesiones` con `{id_experimento, capital_inicial?, modo_ejecucion, comision?, slippage?, nombre?, fecha_alta?}`; admite además `id_configuracion` (+`version`) opcional. La receta se congela con el universo resuelto a lista de tickers (`sp500_pit` se rechaza con 422: una sesión forward necesita un universo fijo). Los costes por sesión sustituyen a los de la receta solo en la sesión.
- El botón «Paper interno» está en `ExperimentoDetalle` (no en `ConfiguracionDetalle`). Capital por defecto 100 000; el modo por defecto es `apertura_siguiente`.

**Otras desviaciones**
- Sesiones en el menú: página nueva `/paper-interno` y `/paper-interno/:id` (la de Alpaca sigue en `/paper`; el plan la dividía en pestañas).
- `FuentePrecios.refrescar_reciente(simbolos, hasta)` (no-op por defecto) y su versión en `FuenteYFinanceCache`: añade solo las sesiones nuevas; si el solape con lo cacheado difiere (> 1e-6: dividendo o split) vuelve a descargar ese símbolo entero. Los precios cacheados no cambian salvo en ese caso. El re-base de cantidades (3.5) compara el cierre guardado de cada posición con el recién descargado y es independiente de la fuente.
- Cobertura mínima del 80 % de los tickers del universo para dar por buena la barra de un día: con universos muy pequeños, un ticker que desaparece deja los días como «incompletos»; la liquidación tras 5 sesiones sin barra solo se alcanza en universos de >= 5-10 tickers.
- `RNG`: el estado del generador de la estrategia se persiste (`estado_rng`) para que reanudar equivalga a no parar (también con estrategias aleatorias como Michaud). Las claves de caché del estado de una estrategia no serializables (`modelo` de `preset_ypyquant`) no se persisten (`CLAVES_CACHE_ESTADO`); el test 3.9 recorre todas las estrategias registradas. No hizo falta tocar `incluidas.py`.
- Bloqueo (3.6): `sesion.lock` con pid; en Windows la comprobación de «proceso vivo» usa `OpenProcess` (con `os.kill(pid, 0)` Windows TERMINA el proceso). Un bloqueo de un hilo del propio proceso cuenta como activo (`SesionBloqueada`, la sesión se omite y la API responde 409 si es el avance completo).
- API (4.2): `Contexto.almacen_paper_interno`, `servicio_paper_interno`, `control_avance_paper` y `reloj_paper` inyectable; `crear_app(catchup_paper_al_arrancar=True, reloj_paper=None)`. No se «marca como interrumpido» un avance en curso al arrancar: el estado del avance vive en memoria (se pierde con el proceso) y el avance es idempotente. Endpoint `/espejo` implementado (GET que lanza o reutiliza por huella el backtest de la misma receta; la web lo sondea).
- El estado de la estrategia de señales (precios de entrada, máximos) no se re-basa por ajustes: solo las posiciones.
- `scripts/instalar_tarea_paper.ps1` creado y NO ejecutado (`-SoloMostrar` enseña lo que haría).

**Verificación**: `tests/test_paper_paridad.py` (paridad con `simular` a 1e-9: día a día, en bloque y partido en dos procesos, con `pesos_iguales`, `momentum` y `cruce_medias`), `test_paper_simulador.py`, `test_paper_cli.py`, `test_api_paper_interno.py`, `test_fuente_refresco.py` y los tests Vitest de la tabla, el avance, la selección, el panel de alta y el detalle.

**Pendiente**: vincular una sesión con una cuenta Alpaca (fase 5) y la comparación interno vs Alpaca (fase 6); intradía y acciones enteras en el paper interno (fase 7).
