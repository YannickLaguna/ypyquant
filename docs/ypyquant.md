# ypyquant

Librería Python de lógica cuantitativa extraída del proyecto TFM GrowSave. Instalable en modo editable para reutilizar en notebooks, scripts y otros servicios sin dependencias de FastAPI ni bases de datos.

## Instalación

Desde GitHub:

```bash
pip install git+https://github.com/YannickLaguna/ypyquant.git@main
```

En modo editable (desarrollo local):

```bash
pip install -e /ruta/local/ypyquant
```

## Estructura

```
ypyquant/
├── optimization/       # Núcleo canónico: estimacion_robusta/ (max-Sharpe QP, µ, Σ, riesgo.py);
│   │                   # portfolio_models/ (subpaquete: base, modelos_retorno, modelos_riesgo,
│   │                   #   optimizadores, modelo), walk-forward, preprocessing,
│   │                   # optimizador_portafolio (fachada de compatibilidad con GrowSave: no ampliar)
│   └── presets/        # Configuraciones predefinidas (Markowitz, BL, PMPT...)
├── factors/            # CAPM, Fama-French (4/5/6 factores), regresión robusta
├── ml/                 # Clustering, regímenes de mercado, PCA, LSTM, RL
├── portfolio/          # Backtest (motor + benchmark), selección de activos, métricas, rebalanceo
├── data/               # Descargadores, indicadores y matriz_precios (cargador inyectado)
├── execution/          # Ejecutor (pesos -> órdenes), brokers (simulado, Alpaca), calendario de rebalanceo
├── paper/              # Paper interno: sesiones forward de una receta congelada (ver «Paper interno»)
└── utils/              # Serialización NumPy→JSON y divisas (ServicioCambio)
```

Desde v0.3.0 ypyquant es la única fuente de la lógica cuantitativa de GrowSave (backtest,
selección de activos, métricas y optimización). Los servicios (`BacktestEngine`,
`PortfolioBacktestService`, `AssetSelectionService`, `PortfolioOptimizationService`) no
acceden a ninguna BD: reciben un `cargador_precios(ticker, inicio, fin) -> OHLCV` y, si la
divisa no es USD, un servicio de tipos de cambio. Migración de GrowSave:
`docs/migracion-growsave-v0.3.0.md`.

Desde v0.4.0 incluye `ypyquant.optimization.estimacion_robusta` (estimadores robustos de µ, max-Sharpe QP, universo
point-in-time y motor de rejilla del TFM). Guía: `docs/migracion-v0.4.0.md`.

Optimización canónica (2026-11-21): el código nuevo usa `estimacion_robusta` (implementación única de max-Sharpe QP,
mínima varianza, volatilidad objetivo, estimadores, CVaR y paridad de riesgo en `riesgo.py`), los modelos de
`portfolio_models` (ahora subpaquete; la ruta de importación no cambia) o las estrategias. `optimizador_portafolio`
y `PortfolioOptimizationService` quedan congelados como fachada para GrowSave. Mapa de duplicados y decisiones:
`docs/planes/2026-11-21-optimizacion-canonica.md`.

Desde 2026-11-22 la fachada DELEGA en el núcleo canónico (mismos nombres y firmas; cambian los pesos numéricos):
`optimizar_max_sharpe_scipy`, `optimizar_max_sharpe_pypfopt` (rf = 2 %), `MarkowitzOptimizer` y
`optimizar_min_volatilidad` usan el QP / mínima varianza de `estimacion_robusta`; `optimizar_volatilidad_objetivo`
usa el SOCP: ya no lanza `ValueError` cuando σ* es inalcanzable (devuelve la GMV con la clave `estado`), el
apalancamiento escala la tangente long-only (sin cortos, peso <= 2). `LedoitWolfShrinkageModel` y
`estimar_sigma_ledoit_wolf` comparten reserva (covarianza muestral) vía `ledoit_wolf_con_metodo`.

> `ypyquant.quantlib` está deprecated y existe solo como shim de retrocompatibilidad.
> Importar directamente desde `ypyquant.optimization`, `ypyquant.portfolio` o `ypyquant.data`.

## Margen y cortos

`ypyquant.execution.margen` (módulo puro) reúne las reglas que comparten el motor, el paper interno y el bot de Alpaca:

- `ReglasMargen`: `apalancamiento_maximo` (exposición bruta / equity, tope 2x = Reg T), requisito inicial 50 %, mantenimiento
  25 % (largos) y 30 % (cortos), `tasa_financiacion` y `coste_prestamo_titulos` (anuales, base 360 días como Alpaca) y
  `permite_cortos`. Por defecto (L = 1, sin cortos) está inactiva y nada cambia: huellas, resultados guardados y tests de
  paridad quedan intactos.
- Definiciones: equity = efectivo + largos - cortos; bruta = largos + |cortos|; mantenimiento = 0,25 x largos + 0,30 x |cortos|.
  Comprobación DIARIA tras valorar: si `equity < mantenimiento` hay llamada de margen y se liquida TODA la cartera en
  proporción (con comisión y slippage) hasta el requisito inicial (`factor_de_conservacion`); queda un aviso con la fecha
  (`Llamada de margen AAAA-MM-DD: ...`). Los pesos objetivo no cambian: el siguiente rebalanceo reconstruye la cartera.
- Ejecutor: `PoliticaEjecucion.permite_cortos` hace que un peso negativo sea una posición corta (venta sin posición larga; compra
  para cubrir; un cambio de signo son DOS órdenes, cierre y apertura). Con cortos no se normaliza la suma de pesos.
  `BrokerSimulado(reglas_margen=...)` rechaza lo que pase del límite o cruce por cero.
- Estrategias: `momentum` es largo-corto con `exposicion_larga` / `exposicion_corta` / `top_k_cortos` /
  `coste_prestamo_titulos` (130/30 = 1,3 y 0,3; dólar-neutral = 1 y 1). `Estrategia.admite_cortos()` decide si `sanear_pesos`
  conserva los negativos. Los parámetros nuevos van fuera de la huella en su defecto y el formulario de la web los genera.
- Evaluación: con cortos o préstamo las métricas incluyen `exposicion_bruta_media/maxima`, `exposicion_neta_media`,
  `exposicion_corta_media` y `llamadas_margen`; la ficha del experimento los muestra y rotula `CASH` negativo como
  «Préstamo (margen)» y los pesos negativos como «(corto)».
- Alpaca: `ALPACA_<PERFIL>_MARGEN=1` (`ALPACA_MARGEN` en `DEFECTO`) declara la cuenta de margen; las recetas apalancadas o con
  cortos solo se vinculan así (hasta 2x overnight) y el bot comprueba al arrancar la cuenta real (`multiplier`,
  `shorting_enabled`). Detalle y decisiones: `docs/planes/2026-11-22-apalancamiento-y-cortos.md`.

## Paper interno

`ypyquant.paper` simula **hacia delante** (forward, con precios reales y sin broker) la receta congelada de un experimento
terminado. Una sesión fija `fecha_alta`, `capital_inicial`, `modo_ejecucion` y, opcionalmente, comisión y slippage propios
(Alpaca no cobra comisión: así se compara «costes realistas» con «costes Alpaca» sin tocar la receta). No envuelve el
motor: **planifica con el Ejecutor** (`planificar_rebalanceo` + `POLITICA_MOTOR`) y **ejecuta con el `BrokerSimulado`**
aplicando la misma comisión y slippage que `BacktestEngine`; el calendario de rebalanceo es el único de
`execution/calendario.py` (`toca_rebalancear`).

```python
from ypyquant.data.fuentes import fuente_por_defecto
from ypyquant.paper import AlmacenPaper, ServicioPaper
from ypyquant.paper.servicio import receta_de_experimento

servicio = ServicioPaper(AlmacenPaper("data/paper_interno"), fuente_por_defecto())
sesion = servicio.crear_sesion(receta_de_experimento(almacen_experimentos, "momentum-abc123"),
                               modo_ejecucion="apertura_siguiente", capital_inicial=100_000)
servicio.avanzar()           # procesa los días cerrados pendientes de las sesiones activas (idempotente, con catch-up)
servicio.listar()            # una fila por sesión: retorno, Sharpe, drawdown, costes, estado, días pendientes
```

- **Modos**: `apertura_siguiente` (por defecto) decide con el cierre de D y ejecuta a la apertura de D+1; `cierre` decide y
  ejecuta con el cierre del mismo día, igual que el backtest (optimista; sirve para la prueba de paridad y se avisa como
  «ejecución optimista»).
- **Bucle diario** (`SimuladorPaper`): precios -> re-base por ajuste -> ejecución -> valoración (fila de equity) -> decisión
  si toca. El commit de cada día escribe las tablas y, AL FINAL, `estado.json` con `ultima_fecha_procesada`: una caída a mitad
  repite el día entero sin duplicar filas. Una sesión `D` solo se procesa si ya es definitiva (en Nueva York, tras las 16:30)
  y su barra es completa (benchmark y al menos el 80 % del universo; si no, se pospone el último día o se omite uno intermedio
  con aviso). Un ticker sin barra 5 sesiones seguidas se liquida a su último precio con aviso.
- **Re-base por ajuste**: `Close` viene ajustado, así que cada dividendo o split reescribe el histórico. Antes de avanzar se
  compara el último cierre guardado de cada posición con el recién descargado; si el factor `f` difiere más de 1e-6 se hace
  `cantidad /= f` y `precio_medio *= f` (el valor no salta). `FuentePrecios.refrescar_reciente` (no-op por defecto; en
  `FuenteYFinanceCache` añade solo las sesiones nuevas y vuelve a descargar un símbolo únicamente si el solape revela un ajuste).
- **Persistencia** (`data/paper_interno/<id>/`, vía `AlmacenDocumentos`): `sesion.json`, `estado.json`, `eventos.json` y
  `equity|operaciones|pesos|rotaciones.parquet` (mismo formato que un experimento, con `efectivo`, `n_posiciones` y `slippage`
  de más). Bloqueo entre procesos con `sesion.lock` (pid del dueño; un bloqueo huérfano se roba). Raíz: `RaizDatos.paper_interno`.
- **Métricas forward**: `evaluar(simulacion_desde_tablas(...), validar=False)`, la misma función que el backtest. Costes =
  comisiones + slippage.
- **Paridad** (`tests/test_paper_paridad.py`): en modo `cierre`, avanzar día a día, de golpe o partido en dos procesos da la
  misma equity, operaciones, pesos y rotaciones que `simular` del mismo rango (tolerancia 1e-9). Protege el motor que usa GrowSave.
- **CLI**: `uv run python -m ypyquant.paper avanzar [--sesion ID] [--hasta AAAA-MM-DD]`, `estado`, `crear --experimento ID`,
  `pausar|reanudar|detener ID`. Código de salida 1 si alguna sesión falló. `YPYQUANT_PAPER_RAIZ` cambia la raíz y
  `YPYQUANT_PRECIOS=sintetico` usa precios sin red. Con la API en marcha, un temporizador (`YPYQUANT_PAPER_INTERVALO_MIN`, 30 min por defecto) avanza solo las sesiones con días cerrados
  pendientes (`paper/planificador.py`). Tarea diaria de Windows OPCIONAL: `scripts/instalar_tarea_paper.ps1` (ver `docs/ui.md`).
- **Universo `sp500_pit`**: se admite; se resuelve en cada fecha procesada con los miembros vigentes (ver `docs/ui.md`,
  «Universo `sp500_pit` en vivo»). `proximo_rebalanceo` (calendario) estima la fecha del siguiente rebalanceo.
- **Limitaciones**: solo largo, cantidades fraccionarias (política del motor),

  `YPYQUANT_PRECIOS=sintetico` usa precios sin red. Tarea diaria de Windows: `scripts/instalar_tarea_paper.ps1` (ver `docs/ui.md`).
- **Margen y cortos**: el paper acepta recetas apalancadas (`apalancamiento_maximo` > 1) y largo-corto (`exposicion_corta` > 0)
  con las MISMAS reglas que el motor, de una sola implementación (`ypyquant.execution.margen`): intereses del préstamo y
  préstamo de títulos al empezar el día, capacidad de compra/venta en corto por el límite de exposición bruta,
  comprobación diaria del mantenimiento con liquidación forzosa y un evento `llamada_margen` (aparece en los avisos de la
  sesión). La paridad cubre 130/30, dólar-neutral, apalancamiento y una llamada de margen.
- **Limitaciones**: universo fijo (`sp500_pit` no se admite), cantidades fraccionarias (política del motor),
  barras diarias (sin intradía), y el estado de la estrategia (p. ej. precios de entrada de las de señales) no se re-basa por ajustes.

## Uso rápido

```python
from ypyquant.optimization.portfolio_models import MarkowitzOptimizer, LedoitWolfShrinkageModel
from ypyquant.factors.factor_models import FamaFrench4FactorModel
from ypyquant.portfolio.performance_metrics import calcular_metricas
from ypyquant.data.french_data_fetcher import FrenchDataFetcher

# Backtest con datos precargados o con un cargador propio
from ypyquant import BacktestConfig, BacktestEngine
config = BacktestConfig(start_date="2022-01-03", end_date="2024-12-31",
                        initial_capital=10_000, rebalance_frequency="quarterly")
resultado = BacktestEngine(config, cargador_precios=mi_cargador).run_backtest(
    lambda fecha, datos: {"AAPL": 0.5, "MSFT": 0.5}, ["AAPL", "MSFT"])
```

## Dependencias opcionales

Para modelos de deep learning (LSTM/CNN):

```bash
pip install "ypyquant[ml] @ git+https://github.com/YannickLaguna/ypyquant.git@main"
```
