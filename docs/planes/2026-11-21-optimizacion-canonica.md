# Núcleo canónico de optimización (candidato 4 de la revisión de arquitectura)

> Responsabilidad: documentar el mapa de implementaciones duplicadas de los algoritmos de optimización, cuál es la
> canónica, qué delega ya en ella, qué queda pendiente y por qué.
> Se relaciona con: `src/ypyquant/optimization/{estimacion_robusta,portfolio_models,presets}`,
> `optimizador_portafolio.py`, `estrategias/incluidas.py`, `tests/test_optimizacion_caracterizacion.py`,
> `tests/test_optimizacion_compatibilidad.py`, `docs/ypyquant.md` y `docs/migracion-v0.4.0.md`.

Rama `arq/optimizacion-canonica` (desde `0258079`). Restricción dura: GrowSave (repo `tfm`) no se rompe.

## 1. Qué usa GrowSave (inventario, solo lectura)

Ramas inspeccionadas: `feature/landing-objetivos` (checkout), `feature/modelo-tfm`, `deploy/vps` y el resto. La rama
`modelo-tfm` ya subclasifica ypyquant; las demás conservan su propia clase pre-migración.

- `from ypyquant.optimization.optimizador_portafolio import PortfolioOptimizationService as _OptimizadorBase`: GrowSave
  la subclasifica (`__init__(db, tipo_retorno)` llama `super().__init__(cargador_precios=..., tipo_retorno=...)`),
  sobrescribe `optimize_advanced` (llama `super().optimize_advanced(tickers, start, end, model, constraints, **kw)`) y
  `_obtener_datos_modelo`, y usa `calcular_metricas_cartera`. Los routers llaman `_get_returns_matrix`, `optimize_advanced`,
  `compare_models`, `create_custom_model`, `get_available_models`, `get_preconfigured_model`, `list_available_presets`,
  `get_preset_details`, `calculate_efficient_frontier`, `optimize_portfolio`, `adjust_portfolio_to_volatility`.
- Funciones: `crear_modelo_personalizado`, `modelos_disponibles` (tests de GrowSave con ypyquant real).
- Presets: `from ypyquant.optimization.presets import obtener_preset, listar_presets, obtener_detalles_preset, PRESETS_REGISTRADOS`.
- Shim `app/services/optimizacion/portfolio_models.py`: reexporta 22 nombres de `ypyquant.optimization.portfolio_models`.
- Shims de `data_preprocessing`, `exportador_resultados`, `risk_profiles`, `walk_forward` (no tocados aquí).
- `ypyquant.optimization.black_litterman.BlackLittermanQuantLib, crear_views_desde_dict` (router de optimización avanzada).
- Los métodos privados `_optimize_*`, `_get_cached_frontier`, `_store_frontier_results`, `calculate_target_return_from_goals` y
  `validate_target_return` los usa la clase pre-migración de GrowSave (`self.`): se conservan.
- `scripts/run_experimento_rejilla.py` usa su propio `quant.portfolio.estrategias_rejilla`, no ypyquant.
- GrowSave fija `ypyquant@v0.2.0` en `requirements*.txt`; el contrato que cuenta es el de la rama `modelo-tfm`.

El contrato queda fijado en `tests/test_optimizacion_compatibilidad.py`.

## 2. Mapa de duplicados

Diferencia = máx. |Δ peso| (o Δ relativa) medida con `tests/test_optimizacion_caracterizacion.py` (5 activos, 500 días, 3 semillas,
tope de peso 1,0 y 0,3).

| Algoritmo | Implementaciones | Llamadores | Δ medida | Veredicto |
|---|---|---|---|---|
| max-Sharpe | `estimacion_robusta.optimizar_max_sharpe` (QP, canónica); `optimizador_portafolio.optimizar_max_sharpe_scipy` (SLSQP, sin rf); `optimizar_max_sharpe_pypfopt`; `MarkowitzOptimizer` (pypfopt + `clean_weights`); `MaxSharpeQPOptimizer` | estrategias, presets TFM; GrowSave `optimize_portfolio`, preset `markowitz` | QP-clase = 0 (envoltorio); scipy-QP hasta 1,4e-4; pypfopt-QP hasta 5e-6 | QP ya canónico; scipy y pypfopt NO delegan (cambiarían pesos de GrowSave) |
| mín. varianza | `optimizar_min_varianza` (QP); `optimizar_min_volatilidad` (SLSQP); `MarkowitzOptimizer('min_volatility')` | GrowSave `optimize_portfolio('min_vol')` | scipy-QP hasta 6,5e-4; Markowitz-QP hasta 2,3e-5 | Pendiente |
| volatilidad objetivo | `estimacion_robusta.optimizar_volatilidad_objetivo` (SOCP); `optimizador_portafolio.optimizar_volatilidad_objetivo` (SLSQP, con apalancamiento opcional y `ValueError`); `MarkowitzOptimizer('efficient_risk')` | estrategias; GrowSave `optimize_target_volatility` | Markowitz-canónica = 0; scipy-SOCP hasta 5,3e-4 | Markowitz ya delega; la scipy no (semántica distinta: apalancamiento, error en vez de GMV) |
| paridad de riesgo | `optimizar_risk_parity` y `RiskParityOptimizer` (mismo objetivo duplicado) | GrowSave preset `risk_parity` | 0 | Unificado: `estimacion_robusta/riesgo.py::optimizar_paridad_riesgo` |
| CVaR histórico | `calcular_cvar` y `PMPTMinCVaROptimizer._calcular_cvar` | `optimizar_pmpt`, preset `pmpt_cvar` | 0 | Unificado: `riesgo.py::var_cvar_historicos` |
| Ledoit-Wolf | `estimar_sigma_ledoit_wolf` y `LedoitWolfShrinkageModel` | todo | 0 (misma llamada pypfopt) | Sin cambio: las reservas ante fallo difieren (sklearn frente a muestral); queda documentado |
| Bayes-Stein | función canónica; `BayesSteinReturnsModel`; estrategia `bayes_stein` | presets `tfm_*`, estrategias | 0 | Ya por construcción (todas llaman a la función) |
| Michaud | función canónica; `MichaudOptimizer`; estrategia `michaud` | ídem | 0 (misma semilla) | Ya por construcción |
| BL-prior | función canónica; `BLPriorReturnsModel`; `BlackLittermanReturnsModel` (prior pypfopt); estrategia `bl_prior` | ídem | < 1e-17 | Ya por construcción / equivalentes |
| semicovarianza | `calcular_semivarianza` (ceros + cov) y `SemiCovarianceModel` (pypfopt) | `optimizar_pmpt`, presets PMPT | 38 % relativa | Son estimadores distintos, no duplicados; no se tocan |

Hallazgo: la premisa de la revisión («Bayes-Stein, Michaud y BL existen dos veces») era imprecisa. Los presets, las clases de
`portfolio_models` y las estrategias comparten ya una única implementación en `estimacion_robusta` y la paridad con el TFM se
mantiene por construcción; `tests/test_paridad_tfm.py` sigue siendo la red de seguridad frente al TFM, no frente a un duplicado.

## 3. Implementación canónica y qué delega

- Canónica: `estimacion_robusta` (con el nuevo `riesgo.py`, que no depende del resto de `optimization`, evitando ciclos).
- Delega ahora: `optimizar_risk_parity`, `RiskParityOptimizer`, `calcular_cvar`, `PMPTMinCVaROptimizer._calcular_cvar`. Resultados
  idénticos (tests de caracterización).
- `PortfolioOptimizationService` y `optimizador_portafolio`: cabecera «fachada de compatibilidad; no ampliar; usar `estimacion_robusta`».
- Borrado (prueba de borrado: sin llamadores en ypyquant ni en ninguna rama de GrowSave): `_anualizar_media`,
  `_anualizar_volatilidad`, `_anualizar_covarianza`, `DIAS_TRADING_ANUAL` y `ProveedorDatosModelo`.
- `portfolio_models.py` (1861 líneas) pasa a subpaquete `portfolio_models/` (`base`, `modelos_retorno`, `modelos_riesgo`,
  `optimizadores`, `modelo`); `__init__` reexporta los 31 nombres públicos.

## 4. Pendiente (decisión del usuario)

1. Delegar `optimizar_max_sharpe_scipy`, `optimizar_min_volatilidad` y `optimizar_volatilidad_objetivo` (scipy) en el QP/SOCP.
   Ganancia: óptimo exacto y una sola implementación. Coste: pesos de GrowSave cambian hasta 1,4e-4 (max-Sharpe), 6,5e-4 (mín.
   varianza) y 5,3e-4 (vol. objetivo), y la vol. objetivo cambia de semántica (sin `ValueError`, sin apalancamiento). Requiere avisar a GrowSave.
2. Sustituir `MarkowitzOptimizer` (pypfopt + `clean_weights`, hasta 5e-6) y `optimizar_max_sharpe_pypfopt` por el QP (cambia el redondeo de pesos).
3. Unificar el respaldo de Ledoit-Wolf (sklearn frente a covarianza muestral).
4. Fragilidad detectada: el QP usa `warm_start` con caché global (`_CACHE_QP`); el resultado de Michaud con semilla fija
   depende de qué se resolvió antes en el proceso (un test mío rompía `test_michaud_reproducible...` hasta aislar la caché).
   Conviene no compartir estado entre llamadas o limpiar la caché por llamada.
5. Los 8 `except Exception` (BLE001) de `optimizadores.py`/`modelos_retorno.py` son anteriores al movimiento y siguen sin tratar.
