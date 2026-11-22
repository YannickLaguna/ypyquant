# Apalancamiento y cortos (decisiones del 2026-11-22)

> Responsabilidad: documentar qué se implementó para el apalancamiento (opción de las estrategias con volatilidad
> objetivo), qué queda pendiente y el ESTUDIO de las posiciones cortas: qué modelos de la librería las admiten de forma
> natural, cómo modelarlas en el motor y en el Ejecutor, qué dice la documentación de Alpaca y una propuesta priorizada.
> Se relaciona con: `src/ypyquant/estrategias/{incluidas,base,simulacion}.py`, `src/ypyquant/portfolio/backtest_engine.py`,
> `src/ypyquant/execution/ejecutor.py`, `src/ypyquant/optimization/estimacion_robusta/`,
> `docs/planes/2026-11-21-optimizacion-canonica.md` y `tests/test_decisiones_optimizacion.py`.

Rama `arq/decisiones-optimizacion`. Estado de las fuentes: las páginas de Alpaca se leyeron el 2026-11-22; los DOI de la
bibliografía se citan de memoria y **no se han abierto en esta sesión** (marcados «sin verificar el enlace»).

## 1. Apalancamiento: qué se hizo

**Cómo se activa.** Las estrategias con `volatilidad_objetivo` (`bayes_stein`, `bl_prior`, `michaud`, `robusta_elipsoide`,
`robusta_caja`) tienen dos parámetros nuevos, que el formulario de lanzamiento de la web muestra solos (se genera del
esquema JSON de `Parametros`):

- `apalancamiento_maximo` (1 a 2, defecto 1 = sin apalancamiento). Validación: > 1 exige `volatilidad_objetivo` > 0.
- `diferencial_financiacion` (0 a 10 %, defecto 1 %): lo que se paga sobre la tasa libre por el préstamo.

No lo declaran `pesos_iguales`, `momentum`, `preset_ypyquant` ni `senales_tecnicas`: no tienen un objetivo de riesgo al
que escalar (los presets de `optimization/presets` tampoco usan volatilidad objetivo). Ambos parámetros van en
`parametros_fuera_de_huella_si_defecto`: con el defecto, huellas, ids y resultados guardados no cambian (test).

**Qué decide la estrategia.** La cartera long-only de siempre se escala por `min(L, σ*/σ)` solo cuando su σ queda por
debajo del objetivo (es decir, cuando sin apalancar no se llega a σ*). Las posiciones pasan a sumar `factor` y se añade
`CASH = 1 - factor` (peso NEGATIVO: el préstamo). La suma total sigue siendo 1. El tope `peso_maximo` se aplica antes de
apalancar (el peso final puede llegar a `peso_maximo x L`). Es una heurística razonable, no el óptimo con coste de
financiación (ver pendientes).

**Cómo lo modela el motor.** El peligro era que el Ejecutor normaliza pesos que no suman 1 (`POLITICA_MOTOR.tolerancia_suma
= 0,01`) y `sanear_pesos` descartaba lo no positivo: el apalancamiento se habría perdido en silencio. Ahora:

1. `sanear_pesos` conserva un `CASH` negativo y escala las posiciones a sumar `1 - CASH`; sin `CASH` negativo, igual que antes.
2. El Ejecutor no cambia: la suma total (posiciones + CASH negativo) es 1, no se normaliza, y `CASH` ya era «efectivo sin orden».
3. `BacktestEngine(apalancamiento_maximo, tasa_financiacion)` (argumentos nuevos, no campos de `BacktestConfig`, para no
   alterar el `asdict` que consume GrowSave): `_buy` puede gastar `efectivo + (L-1) x equity` (el efectivo queda negativo);
   cada día con efectivo negativo se cargan `-efectivo x tasa x días / 360` (convención de Alpaca; el fin de semana cuenta 3 días).
   La tasa es `tasa_libre_riesgo + diferencial_financiacion`. Con L = 1 el motor no entra en ninguna de estas ramas.
4. `simular` pasa L y la tasa al motor y añade a los avisos un resumen (intereses totales, exposición bruta máxima); la
   estrategia añade otro, agregado (en cuántos rebalanceos se apalancó, exposición media y máxima).

**Pendiente (decisiones para el usuario).**

- Márgenes de mantenimiento y margin call: hoy solo hay tope en la compra (`exposición <= L x equity` al rebalancear).
  Si el mercado cae entre rebalanceos la exposición puede superar L sin liquidación forzosa. Alpaca exige 50 % inicial y
  mantenimiento mayor (sección 3). Propuesta: añadir al motor una comprobación diaria `equity / valor_posiciones <
  mantenimiento` con liquidación parcial y aviso.
- Optimizar de verdad con financiación: `max µ'w - spread·(1'w - 1)+` con `0 <= 1'w <= L` y `‖Lᵀw‖ <= σ*` (SOCP convexo)
  en vez de escalar. Solo cambia la cartera apalancada.
- Paper trading y bot (`POLITICA_BOT`, otro agente): con `CASH` negativo la suma de pesos es 1 pero las posiciones suman > 1;
  el bot compraría más que su efectivo. Hasta que se decida, el bot debería **rechazar** estrategias con
  `apalancamiento_maximo > 1` (o exigir cuenta de margen en Alpaca). No se tocó su código.
- Web: la tabla de pesos muestra `CASH` (Efectivo) con peso negativo; podría rotularse «Préstamo (margen)».

## 2. Estudio de cortos: qué modelos los admiten

Cortos = pesos negativos. Interesan dos restricciones: exposición neta `Σw` y bruta `Σ|w|`.

| Modelo (módulo) | ¿Admite cortos? | Cómo / qué cambia |
|---|---|---|
| Mínima varianza (`optimizar_min_varianza`) | Sí, de forma natural | Sin `w >= 0` es un QP con solución cerrada `Σ⁻¹1 / 1'Σ⁻¹1`. Con cortos casi siempre hay pesos negativos. Conviene acotar `‖w‖₁ <= B` (bruta) o `w >= -c`. |
| Max-Sharpe QP de Cornuéjols-Tütüncü (`max_sharpe_qp`) | Sí con un cambio pequeño | La formulación actual (`min y'Σy`, `µ'y = 1`, `y >= 0`) impone `y >= 0` y `κ >= 0` para ser long-only. Quitando `y >= 0` sigue siendo el máximo Sharpe con cortos (cerrado: `w ∝ Σ⁻¹(µ-rf)`), pero hay que tratar el signo de `κ = 1'y`. Hay que acotar la bruta: sin cota los pesos explotan con µ ruidoso. El tope por activo se vuelve `|w_i| <= peso_maximo`. |
| Volatilidad objetivo SOCP (`optimizar_volatilidad_objetivo`) | Sí | Convexo sin `w >= 0`: `max µ'w, ‖Lᵀw‖ <= σ*, 1'w = 1` más cota de bruta. «σ* inalcanzable» deja de existir (sin `w >= 0` se puede bajar σ; el mínimo sigue siendo la GMV). |
| Black-Litterman (`black_litterman.py`, `bl_prior`) | Es un estimador de µ: independiente del signo | µ_BL se obtiene sin restricciones; el signo lo decide el optimizador de después. El prior de equilibrio usa pesos de mercado (long-only por construcción), pero los *views* (p. ej. «A superará a B») son relativos y son cortos de forma natural. |
| Bayes-Stein, Michaud, µ muestral | Estimadores de µ: independientes del signo | Bayes-Stein contrae hacia la GMV (en largo-corto la GMV tiene cortos). Michaud promedia pesos de réplicas: promediar carteras largo-corto es válido, pero con cortos las réplicas divergen más y el efecto estabilizador es menor. |
| Elipsoide / caja robusta (`elipsoide.py`) | Sí | La penalización `κ‖Θ^½ w‖` ya usa la norma, no depende del signo; la de caja usa `δ'|w|` y con cortos debe escribirse con `|w|` (hoy `δ'w` es válida porque `w >= 0`). |
| Paridad de riesgo (`riesgo.optimizar_paridad_riesgo`) | No en su formulación estándar | Contribuciones iguales al riesgo con `w > 0` (Maillard-Roncalli-Teïletche); con cortos las contribuciones pueden igualarse con pesos de signo distinto, pero la solución deja de ser única y de tener la interpretación habitual. Hoy `bounds=(0, peso_maximo)` en SLSQP. Se deja long-only. |
| CVaR histórico / PMPT (`optimizar_pmpt`, `var_cvar_historicos`) | Sí | CVaR de Rockafellar-Uryasev es LP/convexo para cualquier `w`. La semivarianza también es convexa en `w`. Hay que sustituir `bounds=(0, ·)` y acotar la bruta. |
| Momentum / señales (`momentum`, `senales_tecnicas`) | Sí, es la aplicación más natural | Largo-corto por ranking: comprar el decil superior y vender el inferior (dólar-neutral, `Σw = 0`, `Σ|w| = 2`) o variantes 130/30 (`Σw = 1`, `Σ|w| = 1,6`). Es el único caso donde los cortos aportan alfa que no se obtiene de otra forma (en un largo-solo se cae el lado perdedor del ranking). |

Resumen: lo que de verdad condiciona es el optimizador, no el estimador de µ; las tres piezas con `w >= 0` explícito son
el max-Sharpe QP, la vol. objetivo y la paridad de riesgo. Sin cota de exposición bruta, cualquier optimizador con cortos
produce pesos enormes y de signo inestable por el ruido de µ (Clarke-de Silva-Sapra sobre el coste de las restricciones;
sin verificar el enlace).

## 3. Cómo modelarlos

**Restricciones.** Parámetros nuevos (con defecto «sin cortos», fuera de la huella): `exposicion_bruta_maxima` (p. ej. 1,6)
y `exposicion_neta` (1 = largo-solo, 0 = dólar-neutral); `cortos_prohibidos` (lista de tickers no vendibles, p. ej. los
no «easy to borrow»); `peso_corto_maximo` por activo. 130/30 = bruta 1,6 y neta 1.

**Coste.** Préstamo de títulos: tasa anual por ticker (en el motor, por defecto una plana, p. ej. 0,3 % para ETB) sobre el
valor de mercado del corto; los hard-to-borrow cuestan más y pueden no estar disponibles. En Alpaca: comisión diaria sobre
el valor de mercado y la tasa HTB del valor, dividida entre 360, en lotes redondeados de 100 acciones (fuente 1). Además el
corto cobra el interés del producto de la venta (rebate) en el mundo real; se puede ignorar y dejar en «sin verificar».

**Margen.** Reg T: 50 % inicial; para un corto de valor V se exige colateral extra. Mantenimiento mayor; un corto no
tiene pérdida máxima acotada, así que el motor debe liquidar por mantenimiento (ver pendiente del apalancamiento).

**Efecto en el Ejecutor y el motor (hoy).**

- `planificar_rebalanceo` solo trata posiciones largas: `posiciones = {t: q ... if q > 0}`, ignora pesos `<= 0` y tiene
  `if lado == VENTA and actual <= 0: continue  # no se abren cortos`. Soportar cortos exige cantidades con signo en
  `EstadoCuenta.posiciones`, órdenes de `venta_corto` / `cobertura` (en Alpaca son ventas con posición negativa), umbrales
  y rotación con `abs()`, y los cierres cuando el peso cambia de signo (cruce por cero = dos operaciones).
- `BacktestEngine`: `positions[t]['quantity']` negativo, valoración `Σ q·p` (ya funciona con signo), el producto de la venta
  entra en efectivo, intereses de préstamo de títulos y de margen, y `_sell` hoy limita la cantidad a la posición.
- `sanear_pesos` filtra `w > 0`: tendría que aceptar negativos cuando la estrategia declare que usa cortos.
- Ticker sin precio y deslistados: un corto sobre un valor deslistado/quebrado gana (o queda sin cerrar); el modelo actual de
  «liquidar al último precio» no es válido para cortos.

**Alpaca (verificado el 2026-11-22).**

- Margen y cortos requieren cuenta de margen y al menos 2.000 USD de capital; sin ellos, solo 1x. Overnight hasta 2x (Reg T);
  intradía hasta 4x. Requisito inicial 50 % en valores marginables y 100 % en no marginables. [Fuente 1]
- Cortos: más de 5.000 valores *easy-to-borrow* sin comisión de locate; los *hard-to-borrow* exigen locate aprobado antes de
  enviar la orden (lotes de 100, no reutilizables, no reembolsables) y comisión diaria de préstamo. [Fuente 1]
- Interés de margen: 6,50 % anual (5,00 % usuarios «elite») sobre el saldo deudor al cierre, saldo x tasa / 360; el viernes
  cobra 3 días. La tabla vigente está en la tarifa de Alpaca; las cifras pueden haber cambiado. [Fuente 1]
- Paper trading: la tabla comparativa marca venta en corto y margen como disponibles en paper y live, **pero los «Borrow
  Fees» no están disponibles en paper («Coming soon»)**. Es decir, en paper un corto no pagará préstamo: el paper
  sobrestimará el resultado de largo-corto frente al real. [Fuente 2] Sin verificar: comportamiento de locates/HTB, margin
  calls e interés de margen en paper (la página no lo cubre).

## 4. Propuesta priorizada

1. **(Hecho)** Apalancamiento por escala con coste de financiación y exposición bruta en el motor.
2. **Mantenimiento y margin call diario en el motor** (pequeño, ~1 día): sin él, el apalancamiento subestima el riesgo de cola.
3. **Bloquear apalancamiento en el bot** hasta decidir (trivial; lo toca el agente de paper/Alpaca): ver sección 1.
4. **Largo-corto por señales (`momentum` 130/30 y dólar-neutral)** como primera pieza de cortos: es donde dan alfa, no hay
   optimizador de por medio (pesos por ranking), y evita el problema de la inestabilidad de µ. Requiere: cantidades con signo
   en Ejecutor y motor, `exposicion_bruta_maxima` / `exposicion_neta`, `cortos_prohibidos`, coste de préstamo plano.
5. **Min. varianza y vol. objetivo con cortos acotados** (`‖w‖₁ <= B`): cambios locales en `estimacion_robusta`, tras (4).
6. **Max-Sharpe QP con cortos**: quitar `y >= 0` con cota de bruta; probar contra el cerrado. Menor prioridad: con µ ruidoso
   el valor es dudoso.
7. **No hacer**: paridad de riesgo con cortos (sin interpretación estándar).

Decisión del usuario pendiente antes de (4): si el motivo de los cortos es alfa (largo-corto) o cobertura/mayor eficiencia
(130/30): cambia qué estrategias reciben los parámetros nuevos.

## Fuentes

1. Alpaca, «Margin and Short Selling»: <https://docs.alpaca.markets/docs/margin-and-short-selling> (leída 2026-11-22).
2. Alpaca, «Paper Trading»: <https://docs.alpaca.markets/docs/paper-trading> (leída 2026-11-22).
3. Cornuéjols y Tütüncü, *Optimization Methods in Finance* (Cambridge UP, 2006), cambio de variable del max-Sharpe: <https://doi.org/10.1017/CBO9780511753886> (sin verificar el enlace).
4. Maillard, Roncalli, Teïletche (2010), «On the properties of equally-weighted risk contributions portfolios», *J. Portfolio Management*: <https://doi.org/10.3905/jpm.2010.36.4.060> (sin verificar el enlace).
5. Rockafellar y Uryasev (2000), «Optimization of conditional value-at-risk», *J. Risk*: <https://doi.org/10.21314/JOR.2000.038> (sin verificar el enlace).
6. Black y Litterman (1992), «Global portfolio optimization», *Financial Analysts Journal*: <https://doi.org/10.2469/faj.v48.n5.28> (sin verificar el enlace).
7. Jorion (1986), «Bayes-Stein estimation for portfolio analysis», *JFQA*: <https://doi.org/10.2307/2331042> (sin verificar el enlace).
8. Clarke, de Silva y Thorley (2002), «Portfolio constraints and the fundamental law of active management», *FAJ*; y Jacobs y Levy sobre carteras 130/30 (sin enlace: sin verificar).
