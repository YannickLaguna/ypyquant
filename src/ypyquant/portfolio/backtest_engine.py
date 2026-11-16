"""
Backtest Engine puro — Simulacion de portafolios sin dependencias de base de datos.
====================================================================================

Responsabilidades:
- Simular estrategias de rebalanceo historicas con costes de transaccion (comision + slippage)
- Calcular equity curve dia a dia con soporte para aportaciones mensuales (DCA)
- Calcular metricas de performance del backtest (Sharpe, Sortino, Drawdown, trades)
- Mantener registro de trades individuales con PnL

Interfaz:
- Recibe historicos pre-cargados como Dict[str, pd.DataFrame] (sin acceso a BD)
- La capa de aplicacion (app/) es responsable de cargar datos y persistir resultados

Se relaciona con:
- ypyquant.portfolio.performance_metrics: PerformanceAnalyzer para metricas avanzadas
- ypyquant.optimization.data_preprocessing: DiasTrading para constantes de anualizacion

Costes de transaccion modelados:
- Comision: porcentaje sobre valor de la operacion (ambos lados)
- Slippage: precio efectivo ajustado al ejecutar la orden (compra: +slippage, venta: -slippage)
- Total fees rastreados en BacktestResult.total_fees

TODO (costes avanzados):
- [ ] Market impact: slippage variable segun volumen de la orden vs ADV
- [ ] Spread bid-ask explicito como porcentaje del precio
- [ ] Costes de financiamiento para posiciones largas/cortas
- [ ] Impuestos sobre ganancias de capital (configurable por jurisdiccion)
- [ ] Tracking de turnover para optimizar frecuencia de rebalanceo
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Dias de trading anuales (consistente con data_preprocessing.py)
_DIAS_ANUAL = 252


@dataclass
class BacktestConfig:
    """
    Configuracion de un backtest.

    Parametros de costes:
        commission_rate: Porcentaje sobre valor operado, aplicado en compras y ventas.
        slippage_pct: Ajuste de precio por impacto de mercado (compra: precio * (1 + slippage),
                      venta: precio * (1 - slippage)).
    """
    fecha_inicio: str            # 'YYYY-MM-DD'
    fecha_fin: str               # 'YYYY-MM-DD'
    capital_inicial: float
    frecuencia_rebalanceo: str   # 'daily', 'weekly', 'monthly', 'quarterly'
    commission_rate: float = 0.001    # 0.1% por operacion
    slippage_pct: float = 0.001       # 0.1% de deslizamiento de precio
    max_positions: int = 10
    benchmark_ticker: str = '^GSPC'
    aportacion_mensual: float = 0.0   # DCA: aportacion al inicio de cada mes


@dataclass
class BacktestTrade:
    """Representa un trade individual ejecutado durante el backtest."""
    ticker: str
    fecha_entrada: datetime
    fecha_salida: Optional[datetime]
    precio_entrada: float
    precio_salida: Optional[float]
    cantidad: float
    lado: str            # 'buy' o 'sell'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    comision: float = 0.0


@dataclass
class BacktestResult:
    """
    Resultado completo de un backtest.

    Todos los valores monetarios estan en la misma unidad que capital_inicial.
    """
    # Capitales
    capital_inicial: float
    capital_final: float
    retorno_total: float          # Absoluto
    retorno_total_pct: float      # Porcentual

    # Metricas de riesgo
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float           # Absoluto
    max_drawdown_pct: float       # Porcentual
    calmar_ratio: float
    recovery_factor: float

    # Metricas de trades
    total_trades: int
    trades_ganadores: int
    trades_perdedores: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Series temporales
    equity_curve: pd.DataFrame    # columnas: equity, cash, returns, drawdown, drawdown_pct
    trades: List[BacktestTrade]

    # Costes
    total_fees: float
    total_aportaciones: float = 0.0


class BacktestEngine:
    """
    Motor puro de backtesting para estrategias de portafolio.

    Recibe historicos pre-cargados y simula la estrategia dia a dia.
    Aplica comisiones y slippage en cada operacion.

    Uso tipico:
        config = BacktestConfig(fecha_inicio='2010-01-01', ...)
        engine = BacktestEngine(config)
        resultado = engine.run_backtest(mi_estrategia, historical_data)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._reset_estado()

    def _reset_estado(self):
        """Reinicia el estado interno para permitir reusar el engine."""
        self.valor_portfolio = self.config.capital_inicial
        self.cash = self.config.capital_inicial
        self.posiciones: Dict[str, Dict] = {}   # {ticker: {quantity, avg_price}}
        self.equity_curve: List[Dict] = []
        self.trades: List[BacktestTrade] = []
        self.total_aportaciones = 0.0
        self.pesos_objetivo: Dict[str, float] = {}
        self.fecha_actual = None

    def run_backtest(
        self,
        strategy_func: Callable[[datetime, Dict[str, pd.DataFrame]], Dict[str, float]],
        historical_data: Dict[str, pd.DataFrame]
    ) -> BacktestResult:
        """
        Ejecuta el backtest completo.

        Args:
            strategy_func: Funcion que recibe (fecha, historical_data) y retorna
                           {ticker: peso} con pesos objetivo para el rebalanceo.
            historical_data: Dict {ticker: DataFrame con columna 'Close'} con datos
                             pre-cargados para el periodo de backtest.

        Returns:
            BacktestResult con metricas completas, equity curve y lista de trades.
        """
        self._reset_estado()

        logger.info(
            f"[BacktestEngine] Iniciando backtest {self.config.fecha_inicio} → "
            f"{self.config.fecha_fin} | {len(historical_data)} tickers"
        )

        fechas_rebalanceo = set(self._calcular_fechas_rebalanceo())
        fechas_aportacion = set()
        if self.config.aportacion_mensual > 0:
            fechas_aportacion = set(
                d.to_pydatetime()
                for d in pd.date_range(
                    start=self.config.fecha_inicio,
                    end=self.config.fecha_fin,
                    freq='MS'
                )
            )

        # Union de todas las fechas de trading disponibles
        todas_fechas = sorted({
            d
            for df in historical_data.values()
            for d in df.index.tolist()
            if pd.to_datetime(self.config.fecha_inicio) <= d <= pd.to_datetime(self.config.fecha_fin)
        })

        logger.info(f"[BacktestEngine] {len(todas_fechas)} dias de trading")

        asignacion_inicial_hecha = False
        n_rebalanceos = 0

        for fecha in todas_fechas:
            self.fecha_actual = fecha
            precios = self._obtener_precios(historical_data, fecha)

            if not precios:
                continue

            # Asignacion inicial en el primer dia con precios
            if not asignacion_inicial_hecha:
                try:
                    pesos = strategy_func(fecha, historical_data)
                    self.pesos_objetivo = pesos.copy()
                    self._ejecutar_rebalanceo(pesos, precios, fecha)
                    asignacion_inicial_hecha = True
                    n_rebalanceos += 1
                except Exception as exc:
                    logger.error(f"Error en asignacion inicial ({fecha}): {exc}")
                self._actualizar_valor(precios)
                self._registrar_equity(fecha)
                continue

            # DCA: aportacion mensual
            if self.config.aportacion_mensual > 0:
                if self._es_fecha_aportacion(fecha, fechas_aportacion, todas_fechas):
                    self._ejecutar_aportacion(precios, fecha)

            self._actualizar_valor(precios)
            self._registrar_equity(fecha)

            # Rebalanceo periodico
            if fecha in fechas_rebalanceo:
                try:
                    pesos = strategy_func(fecha, historical_data)
                    self.pesos_objetivo = pesos.copy()
                    self._ejecutar_rebalanceo(pesos, precios, fecha)
                    n_rebalanceos += 1
                except Exception as exc:
                    logger.error(f"Error en rebalanceo ({fecha}): {exc}")

        logger.info(f"[BacktestEngine] Simulacion completada: {n_rebalanceos} rebalanceos")
        return self._calcular_metricas()

    # ------------------------------------------------------------------
    # Metodos de simulacion (logica pura, sin IO)
    # ------------------------------------------------------------------

    def _calcular_fechas_rebalanceo(self) -> List[datetime]:
        """Genera la lista de fechas de rebalanceo segun la frecuencia configurada."""
        freq_map = {
            'daily': 'D',
            'weekly': 'W-MON',
            'monthly': 'MS',
            'quarterly': 'QS',
        }
        freq = freq_map.get(self.config.frecuencia_rebalanceo)
        if not freq:
            raise ValueError(f"Frecuencia no soportada: {self.config.frecuencia_rebalanceo}")
        return [
            d.to_pydatetime()
            for d in pd.date_range(self.config.fecha_inicio, self.config.fecha_fin, freq=freq)
        ]

    def _obtener_precios(
        self,
        historical_data: Dict[str, pd.DataFrame],
        fecha: datetime
    ) -> Dict[str, float]:
        """Extrae los precios de cierre para una fecha concreta."""
        precios = {}
        for ticker, df in historical_data.items():
            try:
                if fecha in df.index:
                    precios[ticker] = float(df.loc[fecha, 'Close'])
                else:
                    fechas_previas = df.index[df.index <= fecha]
                    if len(fechas_previas):
                        precios[ticker] = float(df.loc[fechas_previas[-1], 'Close'])
            except Exception:
                pass
        return precios

    def _actualizar_valor(self, precios: Dict[str, float]):
        """Recalcula el valor del portafolio con los precios actuales."""
        valor_posiciones = sum(
            pos['cantidad'] * precios[ticker]
            for ticker, pos in self.posiciones.items()
            if ticker in precios
        )
        self.valor_portfolio = self.cash + valor_posiciones

    def _registrar_equity(self, fecha: datetime):
        """Agrega un punto a la equity curve."""
        self.equity_curve.append({
            'date': fecha,
            'equity': self.valor_portfolio,
            'cash': self.cash,
        })

    def _es_fecha_aportacion(
        self,
        fecha: datetime,
        fechas_aportacion: set,
        todas_fechas: List[datetime]
    ) -> bool:
        """Determina si en esta fecha corresponde procesar una aportacion DCA."""
        if fecha in fechas_aportacion:
            return True
        idx = todas_fechas.index(fecha) if fecha in todas_fechas else -1
        if idx > 0:
            fecha_anterior = todas_fechas[idx - 1]
            if fecha.month != fecha_anterior.month or fecha.year != fecha_anterior.year:
                for fa in fechas_aportacion:
                    if fa.year == fecha.year and fa.month == fecha.month and fa <= fecha:
                        return True
        return False

    def _ejecutar_aportacion(self, precios: Dict[str, float], fecha: datetime):
        """
        Ejecuta una aportacion mensual distribuyendola hacia activos infrarepresentados
        (DCA con rebalanceo natural, sin necesidad de vender).
        """
        aportacion = self.config.aportacion_mensual
        self.cash += aportacion
        self.total_aportaciones += aportacion
        self.valor_portfolio += aportacion

        pesos_objetivo = self.pesos_objetivo or {
            t: 1.0 / len(precios) for t in precios
        }

        # Pesos actuales
        pesos_actuales = {
            ticker: (self.posiciones[ticker]['cantidad'] * precios[ticker] / self.valor_portfolio)
            if ticker in self.posiciones and ticker in precios else 0.0
            for ticker in pesos_objetivo
        }

        # Distribuir la aportacion hacia los activos infrarepresentados
        desviaciones = {
            t: max(0.0, pesos_objetivo[t] - pesos_actuales.get(t, 0.0))
            for t in pesos_objetivo
        }
        total_desv = sum(desviaciones.values())
        if total_desv == 0:
            desviaciones = {t: w for t, w in pesos_objetivo.items() if w > 0}
            total_desv = sum(desviaciones.values())

        pesos_aportacion = {t: d / total_desv for t, d in desviaciones.items() if total_desv > 0}

        for ticker, peso in pesos_aportacion.items():
            if ticker not in precios:
                continue
            importe = aportacion * peso
            self._comprar(ticker, importe, precios[ticker], fecha)

    def _ejecutar_rebalanceo(
        self,
        pesos_objetivo: Dict[str, float],
        precios: Dict[str, float],
        fecha: datetime
    ):
        """Rebalancea el portafolio hacia los pesos objetivo."""
        total = sum(pesos_objetivo.values())
        if abs(total - 1.0) > 0.01 and total > 0:
            pesos_objetivo = {t: w / total for t, w in pesos_objetivo.items()}

        # Cerrar posiciones que salen del universo
        for ticker in list(self.posiciones.keys()):
            if ticker not in pesos_objetivo or pesos_objetivo[ticker] == 0:
                self._cerrar_posicion(ticker, precios.get(ticker), fecha)

        # Ajustar hacia valores objetivo
        for ticker, peso in pesos_objetivo.items():
            if ticker not in precios:
                continue
            valor_objetivo = self.valor_portfolio * peso
            valor_actual = (
                self.posiciones[ticker]['cantidad'] * precios[ticker]
                if ticker in self.posiciones else 0.0
            )
            diferencia = valor_objetivo - valor_actual
            if abs(diferencia) < 10.0:
                continue
            if diferencia > 0:
                self._comprar(ticker, diferencia, precios[ticker], fecha)
            else:
                self._vender(ticker, abs(diferencia), precios[ticker], fecha)

    def _comprar(self, ticker: str, importe: float, precio: float, fecha: datetime):
        """Ejecuta una orden de compra con comision y slippage."""
        comision = importe * self.config.commission_rate
        slippage_abs = importe * self.config.slippage_pct
        coste_total = importe + comision + slippage_abs

        if coste_total > self.cash:
            return

        precio_efectivo = precio * (1 + self.config.slippage_pct)
        cantidad = (importe - comision - slippage_abs) / precio_efectivo

        if ticker in self.posiciones:
            old_qty = self.posiciones[ticker]['cantidad']
            old_avg = self.posiciones[ticker]['precio_medio']
            new_qty = old_qty + cantidad
            self.posiciones[ticker] = {
                'cantidad': new_qty,
                'precio_medio': (old_qty * old_avg + cantidad * precio_efectivo) / new_qty,
            }
        else:
            self.posiciones[ticker] = {'cantidad': cantidad, 'precio_medio': precio_efectivo}

        self.cash -= coste_total
        self.trades.append(BacktestTrade(
            ticker=ticker,
            fecha_entrada=fecha,
            fecha_salida=None,
            precio_entrada=precio_efectivo,
            precio_salida=None,
            cantidad=cantidad,
            lado='buy',
            comision=comision,
        ))

    def _vender(self, ticker: str, importe: float, precio: float, fecha: datetime):
        """Ejecuta una orden de venta con comision y slippage."""
        if ticker not in self.posiciones:
            return

        precio_efectivo = precio * (1 - self.config.slippage_pct)
        cantidad_vender = min(importe / precio_efectivo, self.posiciones[ticker]['cantidad'])

        ingresos_brutos = cantidad_vender * precio_efectivo
        comision = ingresos_brutos * self.config.commission_rate
        ingresos_netos = ingresos_brutos - comision

        precio_entrada = self.posiciones[ticker]['precio_medio']
        pnl = (precio_efectivo - precio_entrada) * cantidad_vender - comision

        self.posiciones[ticker]['cantidad'] -= cantidad_vender
        if self.posiciones[ticker]['cantidad'] < 0.001:
            del self.posiciones[ticker]

        self.cash += ingresos_netos
        self.trades.append(BacktestTrade(
            ticker=ticker,
            fecha_entrada=fecha,
            fecha_salida=fecha,
            precio_entrada=precio_entrada,
            precio_salida=precio_efectivo,
            cantidad=cantidad_vender,
            lado='sell',
            pnl=pnl,
            pnl_pct=(precio_efectivo - precio_entrada) / precio_entrada if precio_entrada > 0 else 0,
            comision=comision,
        ))

    def _cerrar_posicion(self, ticker: str, precio: Optional[float], fecha: datetime):
        """Cierra completamente una posicion."""
        if ticker not in self.posiciones or precio is None:
            return
        cantidad = self.posiciones[ticker]['cantidad']
        self._vender(ticker, cantidad * precio, precio, fecha)

    def _calcular_metricas(self) -> BacktestResult:
        """Calcula todas las metricas de performance a partir de la equity curve."""
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['cummax']
        equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['cummax']

        capital_inicial = self.config.capital_inicial
        capital_final = self.valor_portfolio
        retorno_total = capital_final - capital_inicial
        retorno_total_pct = retorno_total / capital_inicial if capital_inicial > 0 else 0.0

        retornos = equity_df['returns'].dropna()
        sharpe = (retornos.mean() / retornos.std()) * np.sqrt(_DIAS_ANUAL) if retornos.std() > 0 else 0.0

        retornos_negativos = retornos[retornos < 0]
        sortino = (
            (retornos.mean() / retornos_negativos.std()) * np.sqrt(_DIAS_ANUAL)
            if len(retornos_negativos) > 0 and retornos_negativos.std() > 0 else 0.0
        )

        max_drawdown = equity_df['drawdown'].min()
        max_drawdown_pct = equity_df['drawdown_pct'].min()

        anos = (
            pd.to_datetime(self.config.fecha_fin) - pd.to_datetime(self.config.fecha_inicio)
        ).days / 365.25
        retorno_anual = (capital_final / capital_inicial) ** (1 / anos) - 1 if anos > 0 else 0.0
        calmar = retorno_anual / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0.0
        recovery = retorno_total / abs(max_drawdown) if max_drawdown != 0 else 0.0

        trades_cerrados = [t for t in self.trades if t.pnl is not None]
        ganadores = [t for t in trades_cerrados if t.pnl > 0]
        perdedores = [t for t in trades_cerrados if t.pnl <= 0]

        win_rate = len(ganadores) / len(trades_cerrados) if trades_cerrados else 0.0
        avg_win = float(np.mean([t.pnl for t in ganadores])) if ganadores else 0.0
        avg_loss = float(np.mean([t.pnl for t in perdedores])) if perdedores else 0.0
        total_wins = sum(t.pnl for t in ganadores)
        total_losses = abs(sum(t.pnl for t in perdedores))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        total_fees = sum(t.comision for t in self.trades)

        return BacktestResult(
            capital_inicial=capital_inicial,
            capital_final=capital_final,
            retorno_total=retorno_total,
            retorno_total_pct=retorno_total_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            calmar_ratio=calmar,
            recovery_factor=recovery,
            total_trades=len(self.trades),
            trades_ganadores=len(ganadores),
            trades_perdedores=len(perdedores),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_df,
            trades=trades_cerrados,
            total_fees=total_fees,
            total_aportaciones=self.total_aportaciones,
        )


def format_backtest_result(resultado: BacktestResult) -> str:
    """Formatea un BacktestResult en texto legible."""
    return (
        f"{'='*50}\n"
        f"  BACKTEST RESULT\n"
        f"{'='*50}\n"
        f"Capital inicial:  {resultado.capital_inicial:>12,.2f}\n"
        f"Capital final:    {resultado.capital_final:>12,.2f}\n"
        f"Retorno total:    {resultado.retorno_total_pct:>+11.2%}\n"
        f"Sharpe ratio:     {resultado.sharpe_ratio:>12.2f}\n"
        f"Sortino ratio:    {resultado.sortino_ratio:>12.2f}\n"
        f"Max drawdown:     {resultado.max_drawdown_pct:>11.2%}\n"
        f"Calmar ratio:     {resultado.calmar_ratio:>12.2f}\n"
        f"Recovery factor:  {resultado.recovery_factor:>12.2f}\n"
        f"Total trades:     {resultado.total_trades:>12}\n"
        f"Win rate:         {resultado.win_rate:>11.1%}\n"
        f"Profit factor:    {resultado.profit_factor:>12.2f}\n"
        f"Total fees:       {resultado.total_fees:>12,.2f}\n"
        f"Aportaciones:     {resultado.total_aportaciones:>12,.2f}\n"
        f"{'='*50}"
    )
