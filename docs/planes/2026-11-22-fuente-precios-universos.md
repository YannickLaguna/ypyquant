# Plan: fuente de precios y universos (candidato 3 de la revisión de arquitectura)

Rama `arq/fuente-precios`. Objetivo: dos módulos profundos que oculten de dónde vienen los precios y qué
valores componen un universo, de modo que cambiar yfinance por Norgate (previsto en una semana) no toque la
interfaz ni a los llamadores.

## Problema de partida

- La costura de precios era un callable `CargadorPrecios` que devolvía `DataFrame | None`: sin ajustado/crudo,
  sin identificador del valor, sin deslistados.
- Cada llamador hablaba con yfinance por su cuenta: `estrategias/precios.py` (con caché Parquet) y el bot
  (`pipelines/paper_trading/precios.py`, otro `yf.download` sin caché): backtest y bot podían ver datos distintos.
- `universo_dinamico.py` conocía el formato de símbolos de yfinance (`BRK.B` -> `BRK-B`); `experimentos.py`
  resolvía universos y `asset_selection.py` tenía un `_resolver_universo` paralelo.
- Los deslistados se inferían a posteriori en `simulacion.py` (`_avisos_liquidaciones_caducadas`) comparando el
  último precio con la fecha de venta.

## Módulo 1: `ypyquant/data/fuente_precios.py` (interfaz) y `ypyquant/data/fuentes/` (adaptadores)

```python
class FuentePrecios(ABC):
    def ohlcv(self, simbolo, fecha_inicio, fecha_fin, ajuste="ajustado" | "crudo") -> DataFrame | None
    def identificador(self, simbolo) -> str          # id estable, distinto del símbolo de mercado
    def deslistado(self, simbolo) -> Deslistado | None  # fecha del último día, motivo?, inferido?
    def __call__(self, ticker, inicio, fin)          # = ohlcv ajustado: sigue siendo un CargadorPrecios
```

Profundidad: tres operaciones detrás de las que viven la caché Parquet, la descarga, la normalización de
columnas, el ajuste y la inferencia de deslistados. Como cada fuente es llamable, los consumidores del contrato
antiguo (`BacktestEngine`, `AssetSelectionService`, `PortfolioOptimizationService`, la importación del TFM) no
cambian. `cargar_datos_experimento` acepta una `FuentePrecios` o un callable suelto (`como_fuente` lo envuelve en
`FuenteDesdeCargador`, que infiere los deslistados de lo que devolvió).

Adaptadores:

| Adaptador | Uso | Ajustado | Crudo | Identificador | Deslistados |
|---|---|---|---|---|---|
| `FuenteYFinanceCache` | producción hoy | `auto_adjust=True` (igual que siempre) | `auto_adjust=False`, caché `<símbolo>.crudo.*` | `yf:<símbolo>` | inferidos de la caché (`inferido=True`, sin motivo) |
| `FuenteSintetica` | tests, UI sin datos | el de siempre (resultados intactos) | factor determinista por fecha | `sint:<símbolo>` | explícitos opcionales (`deslistados={'X': (fecha, motivo)}`) |

`fuente_por_defecto()` elige por entorno (`YPYQUANT_PRECIOS=sintetico`; si no, yfinance con caché). El bot usa la
misma función: `descargar_precios(tickers, dias, fuente=None)` pide cierres ajustados a la fuente (hasta ayer).

Test de contrato: `tests/test_fuente_precios_contrato.py` corre el mismo conjunto contra cada adaptador
(yfinance sin red: caché precargada con `descargar=False`). Para añadir un adaptador basta registrar una fábrica
en `FABRICAS`.

## Cómo entrará Norgate (NO implementado)

Adaptador `ypyquant/data/fuentes/norgate.py` con la clase `FuenteNorgate(FuentePrecios)`; la API Python es
`norgatedata` y requiere la app Norgate Data Updater (NDU) abierta en Windows. Correspondencias previstas (a
verificar con la documentación de `norgatedata` al instalarlo; no se han podido comprobar aquí):

- `ohlcv(..., "ajustado")` -> `norgatedata.price_timeseries(simbolo, stock_price_adjust=TOTALRETURN,
  padding_setting=NONE, start_date, end_date, timeseriesformat="pandas-dataframe")`; `"crudo"` -> el mismo con
  `stock_price_adjust=NONE`. Hay que renombrar columnas al formato estándar (`Open, High, Low, Close, Volume`) y
  construir `adjusted_close` desde la serie ajustada.
- `identificador(simbolo)` -> `str(norgatedata.assetid(simbolo))` (id numérico estable de Norgate).
- `deslistado(simbolo)` -> si el valor está deslistado (los símbolos deslistados llevan sufijo `-YYYYMM`),
  `Deslistado(fecha=norgatedata.last_quoted_date(simbolo), motivo=<cuando el dato lo permita>, inferido=False)`.
- Membresías: `norgatedata.index_constituent_timeseries(simbolo, "S&P 500")` se convierte al formato
  `{símbolo: [{index, date_added, date_removed}]}` que ya consume `ypyquant.data.universos` (sustituiría al JSON
  exportado de `TFM.db`; `cargar_membresias_sp500` sería la única función a cambiar).
- Símbolos: el canónico es el de yfinance (`BRK-B`); el adaptador traduce al de Norgate dentro de su propio
  módulo y `universos` solo conoce el canónico. Los deslistados con sufijo deben mapearse al símbolo canónico
  usando `identificador` (decisión pendiente cuando se tenga el producto).
- Registro: una rama en `fuente_por_defecto()` (`YPYQUANT_PRECIOS=norgate`). Nada más cambia: motor, API, bot y
  avisos consumen la interfaz; los deslistados explícitos sustituyen a la inferencia sin tocar `simular`.

## Módulo 2: `ypyquant/data/universos.py`

Único módulo que conoce membresías point-in-time y la traducción de símbolos:

- `resolver_universo`, `listar_universos`, `es_universo_pit` (movidos de `experimentos.py`, que los re-exporta
  para no romper `configuraciones`, `marcos`, `reservado`, la API ni los tests).
- Membresías: `extraer_membresias_desde_sqlite`, `cargar_membresias_sp500`, `miembros_en_ventana`,
  `cobertura_membresias`, `aviso_cobertura_pit` (movidos de `universo_dinamico.py` y de `experimentos.py`).
- Símbolos: `a_simbolo_canonico` (`BRK.B` -> `BRK-B`), usado al extraer membresías y al leer `universos/*.txt`.
- Selección de activos: `resolver_seleccion` / `universo_de_indice` sustituyen a `AssetSelectionService._resolver_universo`
  y al cuerpo de `get_historical_universe` (el proveedor de membresías sigue inyectándose).

`universo_dinamico.py` queda solo con la elegibilidad por fecha (`UniversoPuntoEnElTiempo`), que consume las
membresías. La única costura para sustituir membresías en tests es `ypyquant.data.universos.cargar_membresias_sp500`.

## Deslistados

`DatosExperimento.deslistados` (campo nuevo, fuera de `ConfiguracionExperimento`, por tanto fuera de la huella)
lo rellena `cargar_datos_experimento` consultando `fuente.deslistado` tras cargar cada serie. `simular` avisa
con la misma redacción de siempre (más `(motivo)` si la fuente lo conoce). Con yfinance el aviso es idéntico al
anterior; sin fuente (`deslistados is None`) se infiere de los precios como antes. Regla de inferencia
compartida: `inferir_deslistado` (último dato más de `DIAS_PRECIO_CADUCADO = 5` días antes del final cubierto).

## Qué se borró

- `src/ypyquant/estrategias/precios.py` (`CargadorPreciosCache` -> `FuenteYFinanceCache`, `CargadorSintetico` ->
  `FuenteSintetica`, `cargador_por_defecto` -> `fuente_por_defecto`).
- El `yf.download` propio del bot (`pipelines/paper_trading/precios.py`): ahora usa la fuente compartida.
- `resolver_universo`, `listar_universos`, `_aviso_cobertura_pit`, `CARPETA_UNIVERSOS` y `DIAS_PRECIO_CADUCADO` de
  `experimentos.py` (los dos primeros se re-exportan); `_resolver_universo` de `asset_selection.py`; toda la parte
  de membresías de `universo_dinamico.py`.
- Sin tocar: `data/yield_curves.py` descarga rendimientos del Tesoro con yfinance; no es una serie de precios de
  valores y queda fuera de esta fuente.

## Compatibilidad y resultados

La huella de un experimento no cambia (la fuente no entra en la configuración). Los precios ajustados del
sintético y de la caché de yfinance son los mismos bytes de siempre (la caché existente en `data/cache_precios`
sigue siendo válida; los crudos van a archivos nuevos `*.crudo.*`). Verificación: tests de huella fija y de
caracterización, suite completa y paridad TFM (ver el informe de la rama).
