# Cómo escribir una estrategia de señales

Una estrategia de señales se define con **reglas de compra y de venta** en vez de pesos objetivo. La base
`EstrategiaSenales` (`src/ypyquant/estrategias/senales.py`) traduce las reglas a pesos con una máquina de
estados (posiciones, stops, huecos, efectivo), de modo que backtest, barridos, marcos de comparación y paper
usan exactamente la misma `generar_pesos`. El motor (`BacktestEngine`, compartido con GrowSave) no cambia.

## Ejemplo mínimo copiable

Guarda esto como `estrategias_usuario/mi_cruce.py`; aparece solo en la UI (Estrategias) con su formulario:

```python
from pydantic import Field, model_validator
from ypyquant.estrategias import registrar_estrategia
from ypyquant.estrategias.senales import EstrategiaSenales, ParametrosSenales


@registrar_estrategia
class MiCruce(EstrategiaSenales):
    nombre = "mi_cruce"
    descripcion = "Compra con la media rápida sobre la lenta; vende al revés o por stop."

    class Parametros(ParametrosSenales):          # hereda max_posiciones, stops, enfriamiento...
        rapida: int = Field(20, ge=2, le=200, description="Sesiones de la media rápida")
        lenta: int = Field(100, ge=5, le=400, description="Sesiones de la media lenta")

        @model_validator(mode="after")
        def _orden(self):
            if self.rapida >= self.lenta:
                raise ValueError("rapida debe ser menor que lenta")
            return self

    def indicadores(self, cierres, datos):        # opcional: cálculos compartidos
        p = self.parametros
        return {"rapida": cierres.rolling(p.rapida).mean(), "lenta": cierres.rolling(p.lenta).mean()}

    def senal_entrada(self, cierres, indicadores):  # DataFrame bool, fecha x ticker
        return (indicadores["rapida"] > indicadores["lenta"]).fillna(False)

    def senal_salida(self, cierres, indicadores):
        return (indicadores["rapida"] < indicadores["lenta"]).fillna(False)
```

Plantillas completas: `estrategias_usuario/cruce_medias.py` (cruce con `puntuacion` y stops) y
`estrategias_usuario/canal_rsi_volatilidad.py` (ruptura de canal, filtro RSI y tamaño por volatilidad).

## Qué implementas y qué aporta la base

| Tú escribes (vectorizado y causal sobre `cierres`, fecha x ticker) | La base aporta |
|---|---|
| `senal_entrada(cierres, indicadores)` y `senal_salida(cierres, indicadores)`: tablas booleanas | Posiciones, huecos libres (`max_posiciones`) y efectivo (`CASH`) |
| `indicadores(cierres, datos)` (opcional): dict de DataFrames | Stops: `stop_perdida`, `stop_trailing`, `take_profit`, `max_dias_posicion` |
| `puntuacion(cierres, indicadores)` (opcional): elige candidatos si hay más señales que huecos (sin ella, orden alfabético) | `enfriamiento_dias` tras un stop de pérdida o trailing |
| `salida_personalizada(ticker, fecha, posicion, cierres_hasta)` (opcional) para reglas que dependen de la posición | Tamaño: `iguales` (1/`max_posiciones`) o `inversa_volatilidad`, con tope `peso_maximo` |
| `sesiones_necesarias()` (opcional): historia mínima para el paper | Deriva de pesos entre rebalanceos (`mantener_deriva`), idempotencia y estado JSON |

Orden de evaluación en cada fecha: deriva de los pesos con el precio, salidas (fuera del universo, stop de
pérdida, trailing, take profit, señal, tiempo, regla propia), entradas sobre los huecos libres y relleno con `CASH`.

## Reglas para no engañarte

1. **Solo operaciones causales** (`rolling`, `ewm`, `shift(1)`...). Nunca `shift(-1)`, `center=True` ni normalizar con el máximo global.
   La base precalcula las tablas con todo el histórico en `preparar`, así que un look-ahead daría un backtest espléndido y un paper decepcionante.
2. **Pasa `verificar_sin_lookahead`** en el test de cada estrategia (ver `tests/test_senales_ejemplos.py`):
   ```python
   from ypyquant.estrategias.causalidad import verificar_sin_lookahead
   assert verificar_sin_lookahead(MiCruce, {"rapida": 5, "lenta": 20}, datos, ["2021-08-02", "2021-12-01"]) == []
   ```
3. **Ejecución a cierre.** Las reglas y los stops se evalúan con el cierre de la fecha de rebalanceo y la orden se ejecuta a ese
   mismo cierre. No hay stops intradía ni gaps de apertura (los stops reales pueden ejecutarse peor).
4. **Frecuencia.** Lo sugerido es `daily` (los stops exigen mirar cada día). Con una frecuencia menor las reglas solo se evalúan
   en las fechas de rebalanceo (el máximo del trailing sí cuenta los días intermedios). Un marco de comparación con frecuencia
   fija la impone a todas las estrategias.
5. **Long-only.** No hay cortos: una señal de venta solo cierra la posición.
6. **Efectivo.** Lo no invertido se devuelve como el pseudo-ticker `CASH` (sin precio: el motor lo salta y queda en caja). Aparece en
   `pesos.parquet` y la UI lo muestra como «Efectivo».
7. **Parámetros opcionales.** `stop_perdida: float | None = None` deja el campo vacío en el formulario. Una rejilla de barrido por la UI
   solo admite números; para probar «sin stop» en un barrido usa la API con `null` en la rejilla.
8. **Costes.** Un sistema de señales rota mucho más que uno mensual; mira `turnover_anual` y `costes_sobre_ganancia_bruta` antes de fiarte del retorno bruto.

## Qué ves en la UI

- **Estrategias:** las estrategias de señales aparecen con su formulario (los parámetros comunes más los tuyos).
- **Experimento → Operaciones:** cada operación lleva su **motivo** (`senal`, `stop_perdida`, `stop_trailing`, `take_profit`,
  `tiempo`, `fuera_universo`, `regla_propia`, o `ajuste` si solo recoloca pesos por la deriva) y una línea resumen de entradas y salidas por motivo.
  También está en la vista SQL `operaciones` (columna `motivo`; NULL en estrategias sin señales).
- **Marcos:** se lanzan como cualquier estrategia; los parámetros comunes del marco se aplican solo si los declara su `Parametros`.

## Puente con las señales heredadas

`senales_tecnicas` (incluida) envuelve el combinador técnico (RSI, MACD, Bollinger, momentum) de `ypyquant.signals`: entrada = señal > `umbral`,
salida = señal < -`umbral`. Diferencia con el modo heredado del bot (`pipelines/paper_trading --config estrategia_ejemplo.json`): el tamaño pasa
de proporcional a |señal| a un peso fijo `1/max_posiciones`. **El modo `--config` queda obsoleto** (no se borra): usa `senales_tecnicas` desde el registro.

## Estado y reanudación

`estado` guarda solo tipos JSON: `posiciones` (`fecha_entrada`, `precio_entrada`, `maximo`, `peso`), `enfriamiento`, `fecha` y `pesos`. Una instancia nueva
con ese estado continúa igual (base de la reanudación del paper), y llamar dos veces con la misma fecha devuelve los mismos pesos sin mutar nada.
La lista `eventos` (entradas/salidas con motivo) vive en la instancia, no en el estado.
