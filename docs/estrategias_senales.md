# Cómo escribir una estrategia de señales

> Responsabilidad: guía para escribir una estrategia que emite señales de compra, venta, venta en corto y recompra, con stops,
> take profit, trailing y salida por tiempo, y entender qué hace el marco con ellas. Se relaciona con
> `src/ypyquant/estrategias/senales.py` (adaptador y parámetros), `senales_maquina.py` (máquina de estados),
> `senales_modelo.py` (datos y motivos), `senales_tamano.py` (dimensionamiento), `estrategias_usuario/` (plantillas) y
> `docs/planes/2026-11-22-apalancamiento-y-cortos.md` (margen y cortos).

Una estrategia de señales se define con **reglas de entrada y de salida por lado** (largo y corto) en vez de pesos objetivo. El
marco separa tres cosas, cada una con su responsabilidad y cambiable sin tocar las otras:

| Pieza | Quién la escribe | Qué decide |
|---|---|---|
| **Reglas** (`EstrategiaSenales`) | tú | *Cuándo hay señal*: tablas booleanas causales fecha x ticker, por lado, y su fuerza |
| **Gestión de la posición** (`GestionPosicion`) | parámetros | *Cuándo se sale*: stop de pérdida, take profit, trailing, tiempo máximo, enfriamiento, giro |
| **Dimensionamiento** (`Dimensionamiento`, `LimitesExposicion`) | parámetros | *Cuánto se entra*: peso fijo, por volatilidad, por fuerza o por riesgo; límites de exposición bruta y neta |

Las señales viajan como datos (`Senal(ticker, lado, tipo, fuerza)`) hacia una máquina de estados pura por ticker
(`MaquinaPosiciones`), que no sabe de pandas ni de indicadores. Backtest, barridos, marcos de comparación, paper interno y bot usan
exactamente la misma `generar_pesos`; el motor (`BacktestEngine`, compartido con GrowSave) no cambia: sigue recibiendo pesos.

## La interfaz en 12 líneas

Guarda esto como `estrategias_usuario/mi_cruce.py`; aparece solo en la UI (Estrategias) con su formulario:

```python
@registrar_estrategia
class MiCruce(EstrategiaSenales):
    nombre = "mi_cruce"
    descripcion = "Largo con la media rápida sobre la lenta; corto al revés."

    class Parametros(ParametrosSenales):          # hereda lados, stops, límites de exposición, tamaño...
        rapida: int = Field(20, ge=2, le=200)
        lenta: int = Field(100, ge=5, le=400)

    def indicadores(self, cierres, datos):        # opcional: cálculos compartidos (causales)
        return {"r": cierres.rolling(self.parametros.rapida).mean(), "l": cierres.rolling(self.parametros.lenta).mean()}

    def senal_entrada(self, cierres, i):       return (i["r"] > i["l"]).fillna(False)   # comprar
    def senal_salida(self, cierres, i):        return (i["r"] < i["l"]).fillna(False)   # vender el largo
    def senal_entrada_corta(self, cierres, i): return (i["r"] < i["l"]).fillna(False)   # vender en corto
    def senal_salida_corta(self, cierres, i):  return (i["r"] > i["l"]).fillna(False)   # recomprar el corto
```

Con los parámetros por defecto (`lados="largo"`) solo se usan `senal_entrada` y `senal_salida`: la estrategia es solo larga como
siempre. Al lanzar el experimento con `lados="largo_corto"` (o `"corto"`) se activan las reglas cortas. Los cuatro métodos son
opcionales: define solo los de los lados que uses.

## Ejemplo completo con cortos: reversión a la media por z-score

El archivo `estrategias_usuario/reversion_zscore_largo_corto.py` (plantilla lista para copiar) hace esto: compra lo muy por debajo
de su media móvil, vende en corto lo muy por encima, cierra ambos al volver cerca de la media y reparte el tamaño por la fuerza
de la señal. Es el esqueleto de cualquier estrategia largo-corto:

```python
class Parametros(ParametrosSenales):
    lados: Literal["largo", "corto", "largo_corto"] = "largo_corto"   # por defecto opera los dos lados
    ponderacion: Ponderacion = "fuerza"                                # más peso a la señal más extrema
    stop_perdida: float | None = Field(0.08, gt=0, lt=1)               # en un corto, salta si el precio SUBE un 8 %
    exposicion_neta_min: float = -0.5                                  # casi neutral al mercado: neta entre -0,5 y 0,5
    exposicion_neta_max: float = 0.5
    ventana: int = Field(20, ge=5, le=250)
    z_entrada: float = Field(2.0, gt=0, le=5)
    z_salida: float = Field(0.5, ge=0, le=3)

def indicadores(self, cierres, datos):                     # z-score hacia atrás: causal
    media, desviacion = cierres.rolling(self.parametros.ventana).mean(), cierres.rolling(self.parametros.ventana).std()
    return {"z": (cierres - media) / desviacion}

def senal_entrada(self, cierres, i):        return (i["z"] < -self.parametros.z_entrada).fillna(False)
def senal_salida(self, cierres, i):         return (i["z"] > -self.parametros.z_salida).fillna(False)
def senal_entrada_corta(self, cierres, i):  return (i["z"] > self.parametros.z_entrada).fillna(False)
def senal_salida_corta(self, cierres, i):   return (i["z"] < self.parametros.z_salida).fillna(False)
def puntuacion(self, cierres, i):           return -i["z"]      # fuerza del largo (mayor = antes y más peso)
def puntuacion_corta(self, cierres, i):     return i["z"]       # fuerza del corto
```

Otras plantillas: `cruce_medias.py` (cruce con `puntuacion` y stops) y `canal_rsi_volatilidad.py` (ruptura de canal, filtro RSI y
tamaño por volatilidad); ambas tienen el lado corto como espejo y `senales_tecnicas` (puente con las señales heredadas) también.

## Qué hace la máquina de estados

Por ticker: **plano → largo / corto**. En cada fecha, con datos hasta esa fecha, y en este orden:

1. **Deriva.** Los pesos cambian con el precio desde la llamada anterior (sin operar si no hay cambios; `mantener_deriva`) y se
   actualiza el extremo favorable de cada posición (el máximo en un largo, el **mínimo** en un corto).
2. **Salidas**, por prioridad: valor fuera del universo → `stop_loss` → `trailing` → `take_profit` → señal contraria → regla de
   salida del lado → tiempo máximo → `salida_personalizada`. Los stops son **simétricos**: en un corto el stop de pérdida está
   **encima** de la entrada (salta si el precio sube `stop_perdida`), el take profit **debajo** y el trailing sigue al mínimo.
3. **Entradas** sobre los huecos libres: primero largas, luego cortas; entre candidatos, mayor `fuerza` primero (sin puntuación,
   orden alfabético). El tamaño sale del dimensionamiento y se **escala** para no pasar de los límites de exposición.
4. **Efectivo.** `CASH` es el resto hasta 1: mayor que 1 con cortos (el producto de la venta) y **negativo** si hay préstamo de margen.
   Los pesos de los largos son positivos, los de los cortos **negativos**, y con `CASH` suman siempre 1.

Reglas finas (decisiones de diseño):
- Un ticker con entrada larga y corta el mismo día (señales contradictorias) no entra por ningún lado.
- Una entrada y una salida del mismo lado el mismo día tampoco entra.
- **Señal contraria:** con una posición abierta, que aparezca la señal de entrada del lado opuesto la cierra (`senal_contraria`).
  Sin `permitir_giro` queda plana esa sesión (entra al día siguiente si la señal persiste); con `permitir_giro=True` cierra **y
  abre la contraria el mismo día**.
- Tras un `stop_loss` o `trailing`, el ticker espera `enfriamiento_dias` días hábiles antes de reentrar (en cualquiera de los dos
  lados). Una salida por señal, tiempo o take profit no enfría.
- Un valor que sale del universo (p. ej. deja de ser del índice) se cierra con `salida_universo`.
- Una llamada de margen liquida proporcionalmente en el motor (no la decide la estrategia): sus operaciones se etiquetan
  `llamada_margen` y el siguiente rebalanceo reconstruye la cartera.

## Parámetros comunes

Heredados de `ParametrosSenales`; la UI los muestra como formulario junto a los tuyos.

| Grupo | Parámetros |
|---|---|
| Posiciones | `max_posiciones` (total), `max_posiciones_largas`, `max_posiciones_cortas` |
| Gestión | `stop_perdida`, `take_profit`, `stop_trailing`, `max_dias_posicion`, `enfriamiento_dias`, `permitir_giro`, `mantener_deriva` |
| Lados | `lados`: `largo` (defecto) \| `corto` \| `largo_corto` |
| Tamaño | `ponderacion`: `iguales` (1/`max_posiciones`) \| `inversa_volatilidad` \| `fuerza` \| `riesgo`; `peso_maximo`; `ventana_volatilidad`; `riesgo_por_operacion` |
| Exposición | `exposicion_bruta_max` (suma de \|pesos\|; 1,6 = 130/30; tope 2x), `exposicion_neta_max`, `exposicion_neta_min` |
| Costes del margen | `coste_prestamo_titulos` (cortos), `diferencial_financiacion` (préstamo, con neta > 1) |

- **`riesgo`**: el peso es `riesgo_por_operacion / stop_perdida` (si salta el stop pierdes ese porcentaje del equity); exige `stop_perdida`.
- **`fuerza`**: proporcional a la fuerza de la señal (`puntuacion` / `puntuacion_corta`).
- Los cortos exigen cuenta de margen. Las reglas (requisito inicial 50 %, mantenimiento 25 %/30 %, llamadas de margen, préstamo de
  títulos) son las de `ypyquant.execution.margen` y se aplican igual en motor, paper interno y bot (Alpaca). Los parámetros de
  esta tabla derivan las `ReglasMargen` (`reglas_desde_parametros`).
- **Compatibilidad:** todos los parámetros nuevos (lados, límites, giro, riesgo, costes) están **fuera de la huella** mientras valgan su
  defecto, así que un experimento solo largo guardado antes sigue teniendo el mismo id y el mismo resultado (test de caracterización
  `tests/test_caracterizacion_senales.py`).

## Reglas para no engañarte

1. **Solo operaciones causales** (`rolling`, `ewm`, `shift(1)`...). Nunca `shift(-1)`, `center=True` ni normalizar con el máximo global.
   La base precalcula las tablas con todo el histórico en `preparar`, así que un look-ahead daría un backtest espléndido y un paper decepcionante.
2. **Pasa `verificar_sin_lookahead`** en el test de cada estrategia, también en modo largo-corto (ver `tests/test_senales_ejemplos.py`):
   ```python
   from ypyquant.estrategias.causalidad import verificar_sin_lookahead
   assert verificar_sin_lookahead(MiCruce, {"rapida": 5, "lenta": 20, "lados": "largo_corto"}, datos, ["2021-08-02", "2021-12-01"]) == []
   ```
3. **Ejecución.** Las reglas y los stops se evalúan con el cierre de la fecha de rebalanceo y, en el backtest, la orden se ejecuta a ese
   mismo cierre (el paper sigue el modo de ejecución de la sesión). No hay stops intradía ni gaps de apertura: un stop real puede ejecutarse peor.
4. **Frecuencia.** Lo sugerido es `daily` (los stops exigen mirar cada día). Con una frecuencia menor las reglas solo se evalúan
   en las fechas de rebalanceo (el extremo del trailing sí cuenta los días intermedios). Un marco con frecuencia fija la impone a todas.
5. **Cortos: lo que NO se modela.** No hay rebate ni valores *hard-to-borrow* (el préstamo de títulos es una tasa plana), ni cierre
   forzoso de un corto por deslistado, ni liquidación selectiva (la llamada de margen reduce todo proporcionalmente).
6. **Efectivo.** `CASH` no tiene precio: el motor lo salta y aparece en `pesos.parquet`; la UI lo muestra como «Efectivo» o «Préstamo (margen)».
7. **Parámetros opcionales.** `stop_perdida: float | None = None` deja el campo vacío en el formulario. Una rejilla de barrido por la UI
   solo admite números; para probar «sin stop» en un barrido usa la API con `null` en la rejilla.
8. **Costes.** Un sistema de señales rota mucho más que uno mensual; mira `turnover_anual` y `costes_sobre_ganancia_bruta` antes de fiarte del retorno bruto.

## Qué ves en la UI

- **Estrategias:** las estrategias de señales aparecen con su formulario (los parámetros comunes más los tuyos).
- **Experimento → Operaciones:** cada operación lleva su **motivo** y un resumen de entradas y salidas por motivo.
- **Experimento → Salidas por motivo:** tabla con, por cada motivo de salida y lado de la posición, el nº de salidas, el PnL total y
  medio, la tasa de acierto y la duración media (días). Sirve para ver qué regla gana o pierde dinero (¿los stops salvan o muerden?,
  ¿el trailing recorta ganadoras?). El PnL es el de la operación de cierre, neto de su comisión pero no de la de apertura, y los
  ajustes parciales por deriva no cuentan. Está en las métricas (`metricas['motivos_salida']`) y en la vista SQL `operaciones`
  (columna `motivo`; NULL en estrategias sin señales).
- **Marcos:** se lanzan como cualquier estrategia; los parámetros comunes del marco se aplican solo si los declara su `Parametros`.

### Motivos de las operaciones

`entrada_larga`, `entrada_corta`, `stop_loss`, `take_profit`, `trailing`, `tiempo`, `senal_salida` (regla de salida del lado),
`senal_contraria`, `salida_universo`, `regla_propia`, `llamada_margen` (liquidación forzosa del motor) y `ajuste` (compra o venta
parcial que solo recoloca los pesos). Los experimentos guardados antes del marco largo-corto usan el vocabulario anterior
(`senal`, `stop_perdida`, `stop_trailing`, `fuera_universo`): la web y el desglose los leen igual (`a_motivo_actual`).

## Puente con las señales heredadas

`senales_tecnicas` (incluida) envuelve el combinador técnico (RSI, MACD, Bollinger, momentum) de `ypyquant.signals`: entrada larga = señal >
`umbral`, salida = señal < -`umbral`; con `lados="largo_corto"` la señal < -`umbral` abre un corto. Diferencia con el modo heredado del bot
(`pipelines/paper_trading --config estrategia_ejemplo.json`): el tamaño pasa de proporcional a |señal| a un peso fijo `1/max_posiciones`
(o `ponderacion="fuerza"`). **El modo `--config` queda obsoleto** (no se borra): usa `senales_tecnicas` desde el registro.

## Estado y reanudación

`estado` guarda solo tipos JSON: `posiciones` (`lado`, `fecha_entrada`, `precio_entrada`, `maximo` o `minimo`, `peso`; negativo en
los cortos), `enfriamiento`, `fecha` y `pesos`. Una instancia nueva con ese estado continúa igual (base de la reanudación del paper
y del bot; las posiciones guardadas sin `lado` se leen como largas), y llamar dos veces con la misma fecha devuelve los mismos pesos
sin mutar nada. La lista `eventos` (entradas y salidas con motivo y lado) vive en la instancia, no en el estado.

## Tests que protegen el marco

- `tests/test_caracterizacion_senales.py`: solo largo, resultado idéntico al marco anterior (equity, pesos, operaciones, estado, huella).
- `tests/test_senales_maquina.py`: casos a mano e invariantes con Hypothesis (nunca largo y corto a la vez, stop del lado correcto,
  exposición dentro de límites, estado JSON).
- `tests/test_senales.py`, `tests/test_senales_ejemplos.py`: adaptador de pandas, motor con margen, causalidad, plantillas.
- `tests/test_paper_paridad.py`: el paper reproduce el backtest de una estrategia de señales largo-corto con stops y giro.
- `tests/test_desglose_motivos.py`: el desglose por motivo de salida.
