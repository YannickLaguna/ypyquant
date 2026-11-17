"""
backtest.py — Backtest rápido de las señales con el motor de ypyquant (sin Alpaca).

Responsabilidad:
    Descargar ~2 años de precios, generar señales y ejecutarlas con
    `ypyquant.portfolio.backtest_engine` con pesos proporcionales a la señal positiva.
    OJO: pese al flag `--backtest` heredado de GrowSave, NO usa el BacktestEngine de
    Nautilus (necesitaría datos en formato Nautilus); valida las señales con el motor propio.

Se relaciona con:
    - precios.py: datos de entrada.
    - ypyquant.execution.nautilus.constructor_senales: señales.
    - ypyquant.portfolio.backtest_engine: motor (se importa al ejecutar, no al importar el módulo,
      porque ese paquete se está reescribiendo aparte; si cambia su API hay que ajustar aquí).
"""

import logging

from pipelines.paper_trading.config import ConfiguracionEstrategia
from pipelines.paper_trading.precios import descargar_precios
from ypyquant.execution.nautilus.constructor_senales import construir_combinador

logger = logging.getLogger(__name__)

METRICAS = ["sharpe_ratio", "sortino_ratio", "total_return_pct", "max_drawdown_pct", "win_rate"]


def pesos_desde_senales(senales, fecha) -> dict[str, float]:
    """Pesos de cartera en la fecha: señal positiva normalizada por la suma de |señales|."""
    if fecha not in senales.index:
        return {}
    fila = senales.loc[fecha]
    total = fila.abs().sum()
    if total == 0:
        return {}
    return {t: max(0.0, float(fila[t]) / total) for t in fila.index if float(fila[t]) > 0}


def ejecutar_backtest(estrategia: ConfiguracionEstrategia):
    """Ejecuta el backtest y registra las métricas. Devuelve el resultado del motor (o None si falla)."""
    from ypyquant.portfolio.backtest_engine import BacktestConfig, BacktestEngine

    logger.info("Backtest: %s | %s", estrategia.nombre, estrategia.tickers)
    precios = descargar_precios(estrategia.tickers, dias_historico=504)
    if precios.empty:
        logger.error("No hay datos para el backtest")
        return None

    try:
        senales = construir_combinador(estrategia.config_senales).generate(precios)
    except Exception as e:
        logger.error("Error generando señales para el backtest: %s", e)
        return None
    if senales is None or senales.empty:
        logger.error("No se generaron señales")
        return None

    riesgo = estrategia.config_riesgo
    configuracion = BacktestConfig(
        fecha_inicio=str(precios.index[0].date()),
        fecha_fin=str(precios.index[-1].date()),
        capital_inicial=riesgo.get("initial_capital", 100_000.0),
        frecuencia_rebalanceo="monthly",
        commission_rate=riesgo.get("commission", 0.001),
        slippage_pct=riesgo.get("slippage", 0.001),
    )
    datos = {t: precios[[t]].rename(columns={t: "close"}) for t in estrategia.tickers if t in precios.columns}
    resultado = BacktestEngine(config=configuracion).run_backtest(
        strategy_func=lambda fecha, _datos: pesos_desde_senales(senales, fecha),
        tickers=datos,
    )

    logger.info("RESULTADOS DEL BACKTEST")
    for atributo in METRICAS:
        valor = getattr(resultado, atributo, None)
        if valor is not None:
            logger.info("  %s: %.4f", atributo, valor)
    return resultado
