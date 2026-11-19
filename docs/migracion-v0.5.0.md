# Migración a ypyquant v0.5.0

> Responsabilidad: describir las novedades de v0.5.0 y cómo usarlas. Cada apartado documenta una
> funcionalidad nueva (las secciones se añaden por separado).
> Se relaciona con: `src/ypyquant/optimization/estimacion_robusta/elipsoide.py`, `portfolio_models.py`,
> `optimizador_portafolio.py`, `presets/robusta.py`, `estrategias/incluidas.py` y `tests/test_elipsoide.py`.

v0.5.0 es compatible hacia atrás: no cambia ninguna firma existente.

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
