# Migración a ypyquant v0.4.0: estimación robusta del TFM

> Responsabilidad: describir la API nueva de v0.4.0 (subpaquete `ypyquant.optimization.estimacion_robusta`,
> clases Strategy, presets `tfm_*`), los cambios respecto a v0.3.0 y cómo usarlo desde GrowSave y desde el
> propio repo de investigación.
> Se relaciona con: `src/ypyquant/optimization/estimacion_robusta/*`, `portfolio_models.py`,
> `optimizador_portafolio.py`, `presets/{tfm,pmpt}.py`, `tests/test_estimacion_robusta.py`,
> `tests/test_paridad_tfm.py` y, en GrowSave, `app/services/optimizacion`.

v0.4.0 migra la lógica del TFM «Optimización robusta de portafolios» (rejilla v2: estimación robusta de µ con
Σ Ledoit-Wolf y max-Sharpe long-only). Es compatible hacia atrás: no se ha cambiado ninguna firma existente.
La única modificación de comportamiento es el preset `pmpt_cvar` (ver «Cambios»).

## API nueva

### Subpaquete `ypyquant.optimization.estimacion_robusta`

Funciones puras (sin BD ni estado salvo la caché de problemas QP). Todas aceptan `frecuencia`
(12 = mensual, la del TFM y el defecto del subpaquete; 252 = diaria).

| Módulo | Símbolos |
|---|---|
| `sigma` | `estimar_sigma_ledoit_wolf(retornos, frecuencia=12)` |
| `estimadores_mu` | `estimar_mu_muestral`, `estimar_mu_bayes_stein(retornos, sigma_lw, frecuencia=12, si_t_menor_n2="avisar")`, `estimar_mu_bl_prior(sigma_lw, w_mkt, delta=2.5)`, `calcular_omega_momentum`, `estimar_mu_bl_momentum(ventana_signal, sigma_lw, w_mkt, delta=2.5, tau=0.05, frecuencia=12) -> (µ, M)`, `calcular_diagnostico_bl`, `estimar_mu_michaud(retornos, n_replicas=50, rf=0.0, peso_maximo=1.0, devolver_diagnostico=False, metodo_bootstrap="iid", longitud_media_bloque=6.0, rng=None, frecuencia=12)`, `indices_bootstrap` |
| `max_sharpe_qp` | `optimizar_max_sharpe(mu, sigma, rf=0.0, peso_maximo=1.0, pesos_previos=None, penalizacion_turnover=0.0) -> (pesos, degenerate, solver_status)`, `optimizar_min_varianza` |
| `universo` | `generar_prioridad_universo(tickers, b, semilla_base=42, metodo="hash")`, `seleccionar_universo_anidado`, `elegibles_en_fecha(precios_mensuales, membresias, fecha, T, cobertura_min=1.0)`, `construir_mascara_membresia`, `seed_para` |
| `w_mercado` | `calcular_w_mkt_volumen`, `ProveedorCapitalizacion` (Protocol), `CapitalizacionEstatica`, `pesos_mercado(activos, fecha, proveedor=None, precios=None, volumen=None) -> (pesos, fuente)` |
| `rejilla` | `ConfiguracionRejilla`, `DatosRejilla`, `CargadorDatosRejilla` (Protocol), `ejecutar_rejilla(datos, config=None, funcion_elegibles=None, replicas=None) -> ResultadosRejilla`, `ejecutar_replica`, `preparar_panel`, `calcular_turnover`, `calcular_retorno_neto` |

Todo se reexporta desde `ypyquant.optimization.estimacion_robusta`.

### Clases Strategy (`ypyquant.optimization.portfolio_models`)

- `BayesSteinReturnsModel(frequency=252, si_t_menor_n2="avisar")`, `BLPriorReturnsModel(risk_aversion=2.5, frequency=252, market_caps=None)`,
  `BLMomentumReturnsModel(risk_aversion=2.5, tau=0.05, frequency=252, ventana_senal=12, omitir_ultimas=1, market_caps=None)`:
  subclases de `ExpectedReturnsModel`. En BL, la capitalización llega por `market_caps` (constructor o kwarg de `optimize`) o por
  `proveedor_capitalizacion`; si no hay, se usa el proxy por volumen (kwargs `precios` y `volumen`) o equiponderado, y el aviso
  queda en `resultado["optimizer_metadata"]["avisos"]`. `BLMomentumReturnsModel.covarianza_predictiva(sigma)` devuelve Σ+M.
- `MaxSharpeQPOptimizer(risk_free_rate=0.0, penalizacion_turnover=0.0)` y `MichaudOptimizer(n_replicas=50, metodo_bootstrap="iid",
  longitud_media_bloque=6.0, frequency=252, risk_free_rate=0.0, seed=None)`: subclases de `PortfolioOptimizer`. Michaud necesita el
  histórico (`retornos_historicos`); `PortfolioOptimizationModel.optimize` se lo pasa solo (lo hace con cualquier optimizador que
  declare `requiere_historico = True`, también `PMPTMinCVaROptimizer`).
- Enums nuevos: `ReturnModelType.{BAYES_STEIN, BL_PRIOR, BL_MOMENTUM}` y `OptimizerType.{MAX_SHARPE_QP, MICHAUD}`.
- `crear_modelo_personalizado` construye los tipos nuevos y ahora filtra la configuración del optimizador por tipo (antes solo
  dejaba pasar `risk_free_rate`; los parámetros desconocidos se descartan con un aviso en el log). `modelos_disponibles()` lista
  únicamente lo que `crear_modelo_personalizado` sabe construir (antes incluía tipos del enum sin implementación, como `fama_french_4`).

### Presets

`obtener_preset("tfm_bayes_stein" | "tfm_michaud" | "tfm_bl_prior")` (datos diarios, 252). `tfm_bl_prior` usa rf=0 en el optimizador
(Π ya es un exceso de retorno) y requiere `market_caps` para ser fiel; sin ellos avisa.

## Cambios respecto a v0.3.0

1. **`pmpt_cvar` cambia de comportamiento**: antes minimizaba la volatilidad sobre semicovarianza (no el CVaR); ahora usa
   `PMPTMinCVaROptimizer` (CVaR empírico al 95 %). La matriz de riesgo pasa a covarianza muestral, solo para las métricas.
2. `cvxpy>=1.4` es dependencia directa de la librería (antes solo estaba en el grupo `entorno`); GrowSave debe poder instalarlo.
3. `modelos_disponibles()` y el filtro de config de `crear_modelo_personalizado` (ver arriba).
4. Marker pytest `datos` (excluido por defecto) para los tests que leen los ficheros locales del TFM.
5. Versión `0.4.0` (`pyproject.toml`, `ypyquant.__version__`).

## Uso

### Desde GrowSave (datos diarios, sin cambios de código)

```python
from ypyquant.optimization.presets import obtener_preset
from ypyquant.optimization.optimizador_portafolio import crear_modelo_personalizado

modelo = obtener_preset("tfm_bl_prior")
resultado = modelo.optimize(retornos_df, market_caps={"AAPL": 3.0e12, "MSFT": 2.8e12})   # kwargs hasta calculate()

modelo = crear_modelo_personalizado(
    "michaud_bloques", "historical_mean", "ledoit_wolf", "michaud",
    optimizer_config={"n_replicas": 50, "metodo_bootstrap": "bloques", "longitud_media_bloque": 21, "seed": 7},
)
```

Para fijar la dependencia: `ypyquant @ git+https://github.com/YannickLaguna/ypyquant.git@v0.4.0` (la etiqueta la crea el responsable
tras revisar).

### Datos mensuales (como el TFM) con las funciones puras

```python
from ypyquant.optimization.estimacion_robusta import (
    estimar_sigma_ledoit_wolf, estimar_mu_bayes_stein, optimizar_max_sharpe,
)
sigma = estimar_sigma_ledoit_wolf(excesos_mensuales)               # frecuencia=12
mu = estimar_mu_bayes_stein(excesos_mensuales, sigma, si_t_menor_n2="avisar")
pesos, degenerate, estado = optimizar_max_sharpe(mu, sigma.values, peso_maximo=0.2)
```

Con las clases Strategy usa `frequency=12` (p. ej. `BayesSteinReturnsModel(frequency=12)` junto con `LedoitWolfShrinkageModel(frequency=12)`).

### Motor de rejilla

```python
from ypyquant.optimization.estimacion_robusta import ConfiguracionRejilla, DatosRejilla, ejecutar_rejilla

datos = DatosRejilla(precios_mensuales=precios, volumen_mensual=volumen, rf_mensual=rf, membresias=mascara_o_intervalos,
                     proveedor_capitalizacion=None)          # o cualquier objeto con .capitalizaciones(tickers, fecha)
resultado = ejecutar_rejilla(datos, ConfiguracionRejilla(n_replicas=100, coste_pb=10))
resultado.resultados_oos, resultado.pesos_oos, resultado.manifest_draws
```

`ejecutar_rejilla` acepta también un cargador (`.cargar() -> DatosRejilla`); no hay dependencia de SQLite. Es secuencial y no escribe
checkpoints. `resultados_oos` tiene las columnas del parquet del TFM más `frac_contingencia` (fracción de réplicas de Michaud en la
contingencia GMV).

## Qué corrige `elegibles_en_fecha` respecto al TFM

Elegible en `t` = pertenece al índice en `t` y tiene precio válido los T+1 meses `t-T-1..t-1` (cobertura completa); **no** mira nada
posterior a `t-1`. Corrige: (1) el filtro de «datos contaminados» calculado sobre todo el histórico (incluido el futuro OOS);
(2) exigir precio en `t` (excluía las bajas del propio mes, sesgo de supervivencia); (3) cobertura ≥95 % + `dropna` que acortaba la ventana
de forma distinta por universo; (4) pertenencia evaluada en `t-1` en vez de `t`. El motor lo usa por defecto. Para reproducir el
criterio del TFM, inyecta `funcion_elegibles` y `exigir_retorno_oos=True` (así lo hace `tests/test_paridad_tfm.py`).

## Paridad con el TFM

`uv run pytest -m datos -k paridad -s` reproduce la celda b=0, N=20, T=12 (6 meses, con N=20..150 para consumir el RNG igual) contra
`pesos_oos.parquet` y `resultados_oos.parquet`. Requisitos de la reproducción: retornos de estimación **simples** en exceso
(`tipo_retorno_estimacion="simple"`; el motor usa **log** por defecto), prioridad **posicional** (`metodo_prioridad="posicional"`: los
parquet son anteriores al arreglo por hash del 2026-08-31) y `rng_global_legado=True`.

## Cosas del original que conviene revisar (no se han «arreglado»)

- El código del TFM estima µ y Σ con retornos **simples** en exceso (`pct_change() - rf`), no log como indica el resumen metodológico;
  el motor usa log por defecto y `"simple"` para paridad.
- Los parquet publicados usan la prioridad posicional anterior al arreglo por hash (ver arriba).
- Bayes-Stein con T < N+2 (p. ej. N=150, T=12..120) es la norma en la rejilla; el motor pasa `si_t_menor_n2="ignorar"` para no cambiar resultados.
- El retorno OOS de un activo sin precio en `t` se imputa a 0 % (`fillna(0)`); con `exigir_retorno_oos=False` (defecto) el universo incluye
  bajas y ese 0 % es optimista (una baja real suele ser una pérdida).
- Michaud: una réplica que falla se descarta sin registro; la bandera `degenerate` de la celda es la regla de mayoría (>50 %).
- BL-prior se resuelve por el atajo `w = w_mkt` (exacto sin views); con la capitalización por proxy de volumen, `w_mkt` sobrepondera
  activos líquidos de baja capitalización.
- OSQP usa tolerancias por defecto y `warm_start`: los pesos no son reproducibles bit a bit entre procesos (diferencias ~1e-4 en
  la paridad); un aviso «Solution may be inaccurate» de cvxpy es posible con datos mal escalados.
