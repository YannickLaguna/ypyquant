# Validación estadística de estrategias (`ypyquant.validacion`)

Librería pura (numpy/pandas/scipy) que responde a una pregunta: **¿este resultado de backtest es distinguible de la suerte?** No toca el motor ni la API; recibe una *matriz de retornos diarios* (DataFrame, índice = fechas, columnas = configuraciones de un barrido) o precios `dict[str, DataFrame]` ticker → OHLCV.

Módulos: `estadisticos.py`, `permutacion.py`, `walk_forward.py` (todo reexportado en `ypyquant.validacion`).

## 1. Sharpe y su deflactación

- `sharpe(r)` = media/desv. típica muestral (ddof=1), **por periodo**. `anualizar=True` multiplica por √252.
- **Sharpe máximo esperado** con N pruebas y varianza V de los Sharpes (por periodo):
  `E[max SR] = √V · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))`, γ = 0,5772156649. N=1 → 0. Ej.: N=10, V=1 → ≈ 1,574.
- **PSR** (Sharpe probabilístico): `Φ((SR̂−SR*)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²))`. γ₃ asimetría, γ₄ curtosis *no* en exceso. Todo por periodo.
- **DSR** (Sharpe deflactado): PSR con `SR* = E[max SR]` calculado con la varianza de los Sharpes por periodo de las N pruebas del barrido. Con N=1 es PSR(0).
- Referencia: Bailey y López de Prado (2014), «The Deflated Sharpe Ratio», *Journal of Portfolio Management* 40(5).

## 2. PBO por CSCV

`pbo_cscv(matriz, n_bloques=16)`: se parten las T filas en S bloques contiguos; para cada combinación de S/2 bloques (IS) y su complemento (OOS, más de 5000 combinaciones → muestreo con semilla) se toma n\* = argmax métrica IS, ω̄ = rango relativo de n\* en OOS = rango/(N+1) y λ = ln(ω̄/(1−ω̄)). **PBO = P(λ ≤ 0)**: probabilidad de que la mejor configuración in-sample quede por debajo de la mediana fuera de muestra. Devuelve también los pares (IS, OOS) para el gráfico de degradación, la pendiente de la regresión OOS~IS (negativa = degradación) y la probabilidad de pérdida OOS.
Referencia: Bailey, Borwein, López de Prado y Zhu (2017), «The Probability of Backtest Overfitting», *Journal of Computational Finance* 20(4).

## 3. Test de permutación (método de Masters)

`permutar_precios(datos, semilla, inicio, longitud_bloque)` genera precios sintéticos que conservan los retornos y la correlación entre activos pero destruyen la estructura temporal. Se corre la estrategia (con su optimización, si la tiene) sobre N permutaciones y `p_valor_permutacion(valor_real, valores_permutados) = (1 + #{perm ≥ real}) / (1 + n)`. La corrección +1 cuenta el dato real como una permutación más (p nunca es 0).

- En log-precios, por barra: intrabarra (H−O, L−O, C−O) y gap (O − C previo). Una permutación para los intrabarra y otra para los gaps, **la misma para todos los tickers**.
- Con Adj Close/`adjusted_close` se permuta en el espacio ajustado y las columnas crudas se recuperan con el mismo factor de ajuste de cada fila. Volume sigue la permutación de los intrabarra.
- Tickers con fechas distintas: índice común; se permuta desde max(`inicio`, primera fila completa); lo anterior queda real (la primera barra permutable es el ancla y también queda real). Se conservan el primer precio y el último cierre.
- `longitud_bloque > 1` baraja bloques contiguos. **El método barra a barra destruye el clustering de volatilidad** y la autocorrelación, por lo que el test es optimista para estrategias que explotan regímenes de volatilidad; los bloques lo hacen más exigente.
- Referencia: Masters, *Permutation and Randomization Tests for Trading System Development*; explicación y código de neurotrader888 (`mcpt`).

## 4. Walk-forward

`seleccion_walk_forward(matriz, ventana_entreno, paso)`: en cada t (cada `paso` filas tras `ventana_entreno`) elige la columna con mejor métrica en las filas anteriores a t y usa sus retornos en [t, t+paso). Devuelve la serie OOS, las elecciones y la métrica OOS. **Aproximación:** ignora el coste de cambiar de configuración y el estado de la estrategia (cada columna es un backtest completo independiente); es optimista en costes.

## 5. Otros

- **Rachas de Wald-Wolfowitz** (`prueba_rachas`): R, μ = 2n₊n₋/n + 1, σ² = 2n₊n₋(2n₊n₋−n)/(n²(n−1)), Z = (R−μ)/σ, p bilateral. Z < 0: signos agrupados (persistencia).
- **Índice Ulcer** = √media(drawdown%²); **ratio de Martin** = retorno anualizado (%) / Ulcer (%).

## Cómo interpretar (umbrales orientativos)

| Medida | Señal favorable |
|---|---|
| p de permutación (in-sample) | < 1 % |
| PBO | < 0,10–0,20 |
| DSR | > 0,95 |
| Pendiente OOS~IS (PBO) | positiva |

Son orientativos, no un veredicto: dependen de T, del número real de pruebas (también las que no se guardaron) y de su independencia (N efectivo < N si las configuraciones están muy correlacionadas).

## Advertencia

**No conviertas estas medidas en el objetivo de optimización.** Cuando un estadístico de validación se maximiza o se usa para filtrar candidatas, deja de medir la validez y pasa a ser otro parámetro que se sobreajusta (Ley de Goodhart). Úsalas una vez, sobre el conjunto de pruebas completo y declarado de antemano, y si se repite el ciclo cuenta cada iteración en N.

## Uso con barridos

`ypyquant.estrategias.validacion_barridos` aplica estas funciones a la matriz de retornos de un barrido (N = nº de combinaciones de la rejilla) y a su permutación; ver `docs/ui.md` (API y contratos) y `docs/migracion-v0.5.0.md`. Recuerda que N cuenta solo lo guardado en la rejilla: si exploraste más combinaciones a mano, el DSR y el PBO son optimistas. El periodo reservado del barrido no entra en ninguna de estas validaciones; se evalúa aparte con contador de usos.
