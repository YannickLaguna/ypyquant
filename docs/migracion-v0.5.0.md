# Migración a ypyquant v0.5.0

> Responsabilidad: describir las novedades de v0.5.0 y cómo usarlas. Cada apartado documenta una
> funcionalidad nueva (las secciones se añaden por separado).
> Se relaciona con: `src/ypyquant/optimization/estimacion_robusta/elipsoide.py`, `portfolio_models.py`,
> `optimizador_portafolio.py`, `presets/robusta.py`, `estrategias/incluidas.py`, `tests/test_elipsoide.py`,
> `src/ypyquant/validacion/`, `estrategias/{barridos,validacion_barridos,reservado,procedencia}.py`,
> `app/api/rutas_barridos.py`, `tests/test_validacion.py` y `tests/test_barridos.py`.

v0.5.0 es compatible hacia atrás: no cambia ninguna firma existente. Lo único que cambia de comportamiento
visible: `metricas.json` trae tres métricas más, cada experimento nuevo guarda `procedencia.json` y
`POST /api/experimentos/barrido` devuelve además `barrido_id`.

## Optimización robusta con elipsoide

Media-varianza long-only robusta a la incertidumbre en µ: en vez de fiarse de la estimación muestral µ̂,
se optimiza en el peor caso de un conjunto de incertidumbre alrededor de ella. Referencias: Goldfarb e
Iyengar (2003), *Robust Portfolio Selection Problems*, Math. Oper. Res. 28(1):1-38; Ceria y Stubbs (2006),
J. Asset Management 7(2):109-127; Fabozzi, Huang y Zhou (2010), Ann. Oper. Res. 176.

**Elipsoide** `U = {µ : (µ-µ̂)'Θ⁻¹(µ-µ̂) <= κ²}`, con Θ la covarianza del estimador de µ (`Θ = Σ/T`):

    max_w  µ̂'w − κ‖Θ^{1/2}w‖₂ − (aversión/2)·w'Σw    s.a.  1'w = 1,  0 <= w <= peso_maximo
    [opcional]  ‖Lᵀw‖₂ <= σ*   (volatilidad objetivo, Σ = LLᵀ)

Es un SOCP (cvxpy; CLARABEL → ECOS → SCS). **Caja** `|µ_i-µ̂_i| <= δ_i` con `δ_i = z·σ_i/√T`: con pesos
long-only el peor caso es `(µ̂-δ)'w`.

### API (`ypyquant.optimization.estimacion_robusta`)

| Símbolo | Qué hace |
|---|---|
| `matriz_incertidumbre(sigma, n_observaciones, tipo="sigma"\|"diagonal", frecuencia=1)` | Θ = frecuencia·Σ/T (o su diagonal) |
| `kappa_confianza(confianza, n_activos)` | κ = √χ²_N(confianza) |
| `z_confianza(confianza)`, `delta_caja(sigma, n_observaciones, z, frecuencia=1)` | z bilateral normal y δ de la caja |
| `optimizar_robusto_elipsoide(mu, sigma, theta, kappa, aversion=None, volatilidad_objetivo=None, peso_maximo=1.0)` | `(pesos, estado, diagnostico)` |
| `optimizar_robusto_caja(mu, sigma, delta, aversion=None, volatilidad_objetivo=None, peso_maximo=1.0)` | ídem |

`diagnostico`: `rentabilidad_nominal` (µ̂'w), `rentabilidad_peor_caso` (µ̂'w − penalización), `volatilidad`,
`penalizacion` (κ‖Θ^{1/2}w‖ o δ'w) y `aversion`. Estados: `robusto_elipsoide` / `robusto_caja`,
`volatilidad_objetivo`, `volatilidad_objetivo_holgada`, `gmv_volatilidad_inalcanzable` (σ* bajo la mínima
varianza: se devuelve la GMV, igual que `optimizar_volatilidad_objetivo`) y `gmv_fallback`.

Casos límite (cubiertos en `tests/test_elipsoide.py`): κ=0 (o δ=0) es media-varianza y, con
`volatilidad_objetivo`, coincide con `optimizar_volatilidad_objetivo`; κ→∞ con Θ ∝ Σ tiende a la GMV.

### Elección de κ y escalas

- `kappa_confianza` da el radio que contiene a µ con la confianza dada bajo normalidad, pero con N grande
  κ ≈ √N (N=100, 90 % → κ ≈ 10,7) y la cartera se vuelve casi la de mínimo Θ-riesgo. Por eso se admite un
  κ directo, que en la práctica se calibra como parámetro de regularización (p. ej. 0,5-3).
- Con Σ y µ anualizados, la covarianza de la media anualizada es `Θ = frecuencia·Σ_anual/T` (252 en datos
  diarios); `matriz_incertidumbre` lo hace con el argumento `frecuencia` (con 1 es literalmente Σ/T).
  `T` es la longitud de la ventana de estimación.

### Integración

- **Strategy**: `RobustoElipsoideOptimizer(kappa=None, confianza=0.9, tipo_theta="sigma", aversion=2.5,
  volatilidad_objetivo=None, frequency=252, ...)` y `RobustoCajaOptimizer(z=None, confianza=0.9, ...)` en
  `portfolio_models.py`; `OptimizerType.ROBUSTO_ELIPSOIDE` / `ROBUSTO_CAJA` y las mismas ramas en
  `crear_modelo_personalizado`. Reciben µ y Σ del modelo de rentabilidad/riesgo; T sale de
  `retornos_historicos` (lo rellena `PortfolioOptimizationModel`) o de `n_observaciones`.
- **Preset**: `obtener_preset("robusta_elipsoide")` (µ muestral, Σ Ledoit-Wolf, confianza 90 %, λ = 2,5).
- **Estrategias de la UI**: `robusta_elipsoide` y `robusta_caja`, con parámetros `estimador_mu`
  (`muestral` | `bayes_stein` | `bl_prior`), `confianza`, `kappa` (elipsoide) o `z` (caja), `tipo_theta`,
  `aversion`, `volatilidad_objetivo` (0 = sin restricción), `peso_maximo` y `ventana_dias`.
  **Prioridad**: si `kappa` (o `z`) es > 0 prevalece sobre `confianza`; con ambos en 0 no hay robustez
  (media-varianza). Todos los parámetros se pueden barrer en una rejilla.

```python
from ypyquant.optimization.estimacion_robusta import matriz_incertidumbre, optimizar_robusto_elipsoide

theta = matriz_incertidumbre(sigma_anual, n_observaciones=756, tipo="sigma", frecuencia=252)
pesos, estado, diag = optimizar_robusto_elipsoide(mu, sigma_anual, theta, kappa=1.5, aversion=2.5, peso_maximo=0.3)
```

## Validación estadística (`ypyquant.validacion`)

Librería pura (numpy/pandas/scipy, sin motor ni API) para contestar «¿este backtest se distingue de la suerte?». Fórmulas, referencias y umbrales en `docs/validacion.md`.

| Símbolo | Qué hace |
|---|---|
| `sharpe`, `sharpe_probabilistico` (PSR), `sharpe_deflactado` (DSR), `sharpe_maximo_esperado` | Sharpe por periodo y su deflactación por el nº de pruebas |
| `pbo_cscv(matriz, n_bloques=16)` | Probabilidad de sobreajuste por CSCV (+ λ y pares IS/OOS para gráficos) |
| `prueba_rachas(signos)` | Rachas de Wald-Wolfowitz |
| `indice_ulcer`, `ratio_martin` | Riesgo de drawdown y retorno/Ulcer |
| `permutar_precios(datos, semilla, inicio, longitud_bloque)` | Precios permutados (método de Masters) |
| `p_valor_permutacion(real, permutados)` | (1 + #{perm ≥ real}) / (1 + n) |
| `seleccion_walk_forward(matriz, ventana, paso)` | Re-optimización periódica sobre una matriz de retornos |

**No uses estas medidas como objetivo de optimización** (ley de Goodhart): úsalas una vez sobre el conjunto de pruebas completo y declarado de antemano.

## Experimentos: carga y ejecución separadas, procedencia y métricas

- `cargar_datos_experimento(config, cargador) -> DatosExperimento` (precios con la historia previa, benchmark, avisos) y `ejecutar_con_datos(config, datos, completo=True) -> ResultadoExperimento`. `ejecutar_experimento` es su composición y da los mismos resultados que antes. `ejecutar_con_datos` selecciona los tickers de la configuración y recorta los precios a `<= fecha_fin` (así un barrido cargado hasta una fecha posterior no deja ver el futuro a `Estrategia.preparar`); `completo=False` omite procedencia y validadores (miles de backtests en permutación).
- **Procedencia** (`ResultadoExperimento.procedencia`, `procedencia.json`, `AlmacenExperimentos.leer(...)["procedencia"]`): `commit` y `arbol_sucio` (git; `None` fuera de un repo, sin fallar; cacheados por proceso y solo cuentan archivos versionados), `version_ypyquant`, `python`, `numpy`, `pandas`, `cvxpy`, `hash_precios` (SHA-256 de los precios por ticker ordenado y el benchmark) y `ejecutado`. No entra en la huella.
- **Métricas nuevas**: `ulcer`, `martin` y `costes_sobre_ganancia_bruta` = comisiones / (capital_final − capital_inicial + comisiones), `None` si la ganancia bruta es ≤ 0; > 0,5 = los costes se comen la mitad del edge.
- **Validadores por experimento** registrados de serie (`config.validaciones`): `rachas` (signos de los retornos diarios) y `psr` (Sharpe probabilístico frente a 0); resultado en `metricas.validacion.<nombre>`. `GET /api/validadores` los lista.

## Barridos

`ConfiguracionBarrido(base, rejilla, fecha_fin_desarrollo=None, metrica_objetivo="sharpe", etiqueta="")` da un barrido con identidad propia (`barrido-<10 hex>`) y `AlmacenBarridos` (`data/barridos/<id>/`; raíz configurable como `AlmacenExperimentos`) lo persiste con su estado y los resultados de validación.

```python
from ypyquant.estrategias.barridos import AlmacenBarridos, ConfiguracionBarrido, matriz_retornos, superficie
from ypyquant.estrategias.validacion_barridos import validar_barrido, walk_forward_barrido, prueba_permutacion_barrido
from ypyquant.estrategias.reservado import evaluar_reservado

almacen = AlmacenBarridos()                       # data/barridos/ (los experimentos, en data/experimentos/)
config = ConfiguracionBarrido(base=base, rejilla={"top_k": [3, 5, 8], "meses_retorno": [6, 9, 12]},
                              fecha_fin_desarrollo="2022-12-31")
id_barrido, _ = almacen.crear(config)             # registra; los experimentos los ejecutas tú (o la API)
barrido = almacen.cargar(id_barrido)
validacion = validar_barrido(barrido)             # DSR, PBO-CSCV, rachas, superficie con meseta
wf = walk_forward_barrido(barrido, ventana_entreno_dias=756, paso_dias=90)
perm = prueba_permutacion_barrido(barrido, cargador, n_permutaciones=100, procesos=4)
reservado = evaluar_reservado(barrido, "top_k=5, meses_retorno=9", cargador)
```

- **Matriz de retornos** (`matriz_retornos`): fechas × combinaciones (columna = id de experimento). Si las combinaciones no cubren las mismas fechas se **recorta al tramo común** por defecto (`relleno="cero"` conserva la unión con 0 los días sin posición).
- **Superficie** (`superficie`): parámetros → métricas + **meseta** (media de la métrica objetivo de los vecinos a ± 1 posición por eje; la propia combinación no cuenta). `alternativa_meseta` de la validación sugiere la combinación de mejor meseta.
- **Permutación del barrido**: cada permutación de precios ejecuta TODAS las combinaciones y se queda con el mejor estadístico (`in_sample`: mejor métrica objetivo; `walk_forward`: Sharpe de la serie OOS, permutando solo desde `fecha_inicio + ventana_entreno_dias`). Carga los precios una vez y reparte las permutaciones en procesos (`ProcessPoolExecutor`, spawn; cada worker redescubre las estrategias, también `estrategias_usuario/`). Máximo 1000 permutaciones; aviso si combinaciones × permutaciones > 20 000. `ventana_entreno_dias` y `paso_dias` son días naturales.
- **Periodo reservado**: el tramo (`fecha_fin_desarrollo`, `base.fecha_fin`] no lo toca ninguna validación. `evaluar_reservado` lo evalúa y cuenta el uso en `data/reservado.json` (clave: universo resuelto ordenado + fechas); si ya se había usado, `aviso` explícito: deja de ser fuera de muestra.
- **API**: `/api/barridos` (ver `docs/ui.md`, con los contratos JSON). `POST /api/experimentos/barrido` sigue funcionando y devuelve también `barrido_id`.

Cambios menores: `tests/test_fase2.py::test_mlflow_db_exists` se salta si no existe `data/mlflow.db`; `docs/ui.md` ya no lista la limitación del rebalanceo perdido en festivos (corregida en v0.4.0: pasa al siguiente día hábil).
