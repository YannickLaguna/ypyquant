# Plan: partir `ejecutar_con_datos` en `simular` + `evaluar` (2026-11-22)

Candidato 7 de la revisión de arquitectura. Rama `arq/simular-evaluar`.

## Qué se hizo

`ejecutar_con_datos` (~100 líneas) mezclaba simulación, métricas, validaciones, procedencia y semillas globales.
Ahora son dos módulos profundos y la función antigua es su composición (misma firma, mismos resultados):

- `estrategias/simulacion.py`: `Simulacion` + `simular(config, datos, rng=None, completo=True)`. Solo hechos.
- `estrategias/evaluacion.py`: `Evaluacion` + `evaluar(simulacion, benchmark=None, validar=True)` y `reevaluar(...)`.
- `estrategias/experimentos.py`: conserva configuración, `DatosExperimento`, `ResultadoExperimento`, validadores y la
  composición `ejecutar_con_datos` / `ejecutar_experimento` (los usan la API, barridos, marcos y el importador TFM).
- Persistencia: `rotaciones.parquet` nuevo; `AlmacenExperimentos.reevaluar(id)` y `scripts/reevaluar_experimentos.py`.

## Interfaz

```
Simulacion(config, equity[fecha,equity,drawdown,retorno], pesos[fecha,ticker,peso],
           operaciones[fecha,ticker,lado,cantidad,precio,comision,pnl(+motivo)],
           rotaciones[fecha,rotacion], avisos, benchmark_precios, procedencia)
simular(config, datos, rng=None, completo=True) -> Simulacion          # adaptador: motor de backtest
simulacion_desde_tablas(config, equity, pesos, operaciones, rotaciones=None, ...) -> Simulacion  # adaptador: tablas
evaluar(simulacion, benchmark=None, validar=True) -> Evaluacion(metricas, equity+benchmark, procedencia)
reevaluar(config, equity, pesos, operaciones, rotaciones=None) -> Evaluacion
```

`evaluar` no sabe de dónde viene la simulación: esa es la costura. Todas las métricas salen de las tablas
(n_operaciones, comisiones, tasa de acierto y turnover incluidos), no del `BacktestResult`.

## Semillas

- Antes: `random.seed` y `np.random.seed` globales al inicio de cada ejecución (dos hilos se pisaban).
- Ahora: cada ejecución crea `np.random.default_rng(config.semilla)` (o recibe `rng`) y se lo inyecta a la
  estrategia (`Estrategia.rng`, `generador_azar()`). Michaud ya usaba `default_rng(semilla)` en `estado`; ahora
  usa `generador_azar()`: misma secuencia, por eso los resultados no cambian (test de caracterización exacto).
- Ningún otro consumidor del camino de un experimento usaba el estado global (`grep np.random|random.seed|default_rng`):
  permutación de precios, causalidad y cargador sintético ya usan generadores locales; `walk_forward` y
  `monte_carlo` (que sí siembran el global) no están en este camino.
- Convención nueva para estrategias de usuario: sortear solo con `self.generador_azar()`; `np.random.*` global ya no
  se siembra.

## Pendiente conocido (fuera de alcance: `optimization/`)

`optimization/estimacion_robusta/max_sharpe_qp.py` comparte entre hilos un problema cvxpy compilado con
`cp.Parameter` mutables (`_CACHE_QP`), así que dos Michaud simultáneos (o cualquier estrategia con QP max-Sharpe)
pueden dar resultados distintos a los de serie. Arreglo de una línea: incluir `threading.get_ident()` en la clave de
la caché. Hay un test `xfail(strict=True)` (`test_michaud_en_hilos_distintos_no_interfiere`) que pasará a verde con
ese arreglo y habrá que quitarle el marcador.

## Cómo encajará el paper interno (fases 3-5)

1. El paper interno registra lo que realmente ocurre (equity diaria, fills, pesos objetivo, rotaciones planificadas
   por el Ejecutor) y lo vuelca en tablas con las mismas columnas.
2. Un adaptador `simulacion_desde_tablas(...)` (o una función propia `simulacion_desde_paper(id)`) construye la
   `Simulacion`; el benchmark se pasa explícito (`evaluar(sim, benchmark=serie)`).
3. `evaluar` produce las mismas métricas y validaciones, comparables 1:1 con el backtest de la misma configuración;
   Alpaca sería un segundo adaptador. Las divergencias backtest vs. paper se miden comparando dos `Evaluacion`.
4. `reevaluar` permite recalcular métricas históricas si cambia una fórmula, sin volver a simular.

## Limitaciones de `reevaluar`

- Experimentos guardados antes de este cambio no tienen `rotaciones.parquet`: el turnover se estima desde
  operaciones (aprox., marcado en `metricas['reevaluacion']`); el resto de métricas coinciden con tolerancia 1e-9
  (alfa/beta usan el benchmark ya reescalado: diferencias de redondeo de ~1e-16).
- Las importaciones del TFM se omiten (sus métricas no salen de operaciones).
