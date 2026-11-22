"""
Backtest Engine — simulación de estrategias de portafolio (única fuente desde v0.3.0).
=====================================================================================

Responsabilidades:
- `BacktestEngine`: simula día a día una estrategia de rebalanceo con comisión y slippage,
  asignación inicial, rebalanceos periódicos (en el primer día de trading de cada periodo:
  si la fecha programada cae en fin de semana o festivo se pasa al siguiente día hábil),
  aportaciones mensuales (DCA con rebalanceo implícito), rotación real de cada rebalanceo
  (frente a los pesos a la deriva) y conversión de la equity a la divisa del usuario.
- `PortfolioBacktestService`: backtest + benchmark buy & hold (con el mismo DCA y la misma
  divisa), comparación, comparación de estrategias y métricas rolling.
- Dataclasses `BacktestConfig`, `BacktestTrade`, `BacktestResult` con los nombres de campo de
  GrowSave (inglés) y alias en español de la API 0.2.0 (fecha_inicio, capital_inicial,
  retorno_total_pct, ...), para que funcionen ambos consumidores.

Unificación (v0.3.0): la lógica es la de GrowSave validada en producción (arreglos del
2026-11-15/17: columna 'Close' desde 'close'/'adjusted_close', compras que ya no se rechazan
por «cash insuficiente», ventas antes que compras, ventana de estimación previa para el
walk-forward y conversión de divisa de cartera y benchmark con la misma función), sin BD:
- los precios llegan por `historical_data` precargado o por un `cargador_precios`
  (ticker, inicio, fin) -> OHLCV inyectado;
- los tipos de cambio, por un `servicio_cambio` inyectado (ver ypyquant.utils.divisas);
- la persistencia es un gancho (`_guardar_resultado`) que la aplicación sobrescribe.
De la versión 0.2.0 de ypyquant se conservan: comisiones reales por operación en
`total_fees`, PnL de venta con el precio medio previo al cierre y reutilización del motor.

Se relaciona con:
- ypyquant.data.matriz_precios (con_columna_close, CargadorPrecios).
- ypyquant.utils.divisas (ServicioCambio, convertir_equity_a_divisa, simbolo_divisa).
- ypyquant.utils.serializacion (tipos NumPy → nativos).
- scripts/run_fase2.py (usa los alias en español) y GrowSave
  app/services/portafolio/backtest_engine.py (subclases que inyectan BD y tipos de cambio).
"""

import logging
from collections.abc import Callable
from dataclasses import InitVar, asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ypyquant.data.matriz_precios import CargadorPrecios, con_columna_close
from ypyquant.execution.calendario import (
    ALIAS_PANDAS,
    alinear_a_dias_de_trading,
    fechas_programadas,
    validar_frecuencia,
)
from ypyquant.utils.divisas import ServicioCambio, convertir_equity_a_divisa, simbolo_divisa
from ypyquant.utils.serializacion import convertir_numpy_a_nativo

logger = logging.getLogger(__name__)

__all__ = [
    "DIVISA_BASE",
    "VENTANA_ESTIMACION_DIAS",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BacktestTrade",
    "PortfolioBacktestService",
    "con_columna_close",
    "convertir_equity_a_divisa",
    "format_backtest_result",
    "format_backtest_results",
]

# Historia previa al inicio del backtest que se carga para estimar el primer rebalanceo:
# cada optimización usa solo datos anteriores a su fecha (walk-forward), pero sin esta
# ventana la primera tendría apenas unos días de retornos.
VENTANA_ESTIMACION_DIAS = 3 * 365

# Divisa en la que cotizan los precios cargados (yfinance / BD de GrowSave)
DIVISA_BASE = 'USD'

# Días de trading por año para anualizar Sharpe y Sortino
_DIAS_ANUAL = 252

# Frecuencias de rebalanceo admitidas → alias de pandas (la definición vive en execution/calendario.py;
# el nombre se conserva por si alguna subclase de GrowSave lo lee)
_FRECUENCIAS = ALIAS_PANDAS

# Firma de la estrategia: (fecha, {ticker: OHLCV}) -> {ticker: peso}
FuncionEstrategia = Callable[[datetime, dict[str, pd.DataFrame]], dict[str, float]]


# ============================================================================
# DATACLASSES
# ============================================================================


@dataclass
class BacktestConfig:
    """
    Configuración de un backtest.

    Campos (nombres de GrowSave): start_date, end_date ('YYYY-MM-DD'), initial_capital (en
    la divisa del usuario), rebalance_frequency ('daily' | 'weekly' | 'monthly' |
    'quarterly'), commission_rate y slippage_pct (fracciones), max_positions,
    benchmark_ticker, monthly_contribution (DCA, divisa del usuario) y user_currency (ISO 4217).

    Alias de construcción de la API 0.2.0 (solo en el constructor; no aparecen en asdict):
    fecha_inicio, fecha_fin, capital_inicial, frecuencia_rebalanceo, aportacion_mensual.
    Tras construir, también se pueden leer con esos nombres.
    """
    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float | None = None
    rebalance_frequency: str | None = None
    commission_rate: float = 0.001
    slippage_pct: float = 0.001
    max_positions: int = 10
    benchmark_ticker: str = '^GSPC'
    monthly_contribution: float = 0.0
    user_currency: str = DIVISA_BASE
    fecha_inicio: InitVar[str | None] = None
    fecha_fin: InitVar[str | None] = None
    capital_inicial: InitVar[float | None] = None
    frecuencia_rebalanceo: InitVar[str | None] = None
    aportacion_mensual: InitVar[float | None] = None

    def __post_init__(self, fecha_inicio, fecha_fin, capital_inicial, frecuencia_rebalanceo, aportacion_mensual):
        # Los alias en español solo rellenan lo que no se haya pasado con el nombre en inglés
        if self.start_date is None:
            self.start_date = fecha_inicio
        if self.end_date is None:
            self.end_date = fecha_fin
        if self.initial_capital is None:
            self.initial_capital = capital_inicial
        if self.rebalance_frequency is None:
            self.rebalance_frequency = frecuencia_rebalanceo
        if aportacion_mensual is not None and not self.monthly_contribution:
            self.monthly_contribution = aportacion_mensual

        faltan = [
            nombre for nombre in ('start_date', 'end_date', 'initial_capital', 'rebalance_frequency')
            if getattr(self, nombre) is None
        ]
        if faltan:
            raise TypeError(f"BacktestConfig: faltan campos obligatorios {faltan}")
        validar_frecuencia(self.rebalance_frequency)


# Lectura con los nombres en español (los InitVar no se guardan en la instancia)
BacktestConfig.fecha_inicio = property(lambda self: self.start_date)
BacktestConfig.fecha_fin = property(lambda self: self.end_date)
BacktestConfig.capital_inicial = property(lambda self: self.initial_capital)
BacktestConfig.frecuencia_rebalanceo = property(lambda self: self.rebalance_frequency)
BacktestConfig.aportacion_mensual = property(lambda self: self.monthly_contribution)


@dataclass
class BacktestTrade:
    """Operación ejecutada en el backtest (compra o venta), con su comisión."""
    ticker: str
    entry_date: datetime
    exit_date: datetime | None
    entry_price: float
    exit_price: float | None
    quantity: float
    side: str  # 'buy' o 'sell'
    pnl: float | None = None
    pnl_pct: float | None = None
    commission: float = 0.0

    # Alias de la API 0.2.0
    @property
    def fecha_entrada(self) -> datetime:
        return self.entry_date

    @property
    def fecha_salida(self) -> datetime | None:
        return self.exit_date

    @property
    def precio_entrada(self) -> float:
        return self.entry_price

    @property
    def precio_salida(self) -> float | None:
        return self.exit_price

    @property
    def cantidad(self) -> float:
        return self.quantity

    @property
    def lado(self) -> str:
        return self.side

    @property
    def comision(self) -> float:
        return self.commission


@dataclass
class BacktestResult:
    """
    Resultado de un backtest. Importes en la divisa del usuario (`currency`).

    total_return es absoluto y total_return_pct es (final − inicial) / inicial (con DCA
    incluye las aportaciones, como en GrowSave). `trades` contiene solo las operaciones
    cerradas (ventas); total_trades cuenta todas.
    """
    # Métricas básicas
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float

    # Métricas de riesgo
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float

    # Métricas de trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Series temporales
    equity_curve: pd.DataFrame
    drawdown_series: pd.DataFrame
    trades: list[BacktestTrade]

    # Métricas adicionales
    calmar_ratio: float
    recovery_factor: float
    total_fees: float
    total_contributions: float = 0.0
    currency: str = DIVISA_BASE
    # Rotación de cada rebalanceo ejecutado: [(fecha, Σ|valor operado| / (2·valor cartera))].
    # La primera es la asignación inicial (≈1). Mide lo que se opera de verdad, con los
    # pesos a la deriva desde el rebalanceo anterior (no la diferencia entre objetivos).
    rotaciones: list[tuple[datetime, float]] = field(default_factory=list)

    # Alias de la API 0.2.0
    @property
    def capital_inicial(self) -> float:
        return self.initial_capital

    @property
    def capital_final(self) -> float:
        return self.final_capital

    @property
    def retorno_total(self) -> float:
        return self.total_return

    @property
    def retorno_total_pct(self) -> float:
        return self.total_return_pct

    @property
    def trades_ganadores(self) -> int:
        return self.winning_trades

    @property
    def trades_perdedores(self) -> int:
        return self.losing_trades

    @property
    def total_aportaciones(self) -> float:
        return self.total_contributions


# ============================================================================
# MOTOR
# ============================================================================


class BacktestEngine:
    """
    Motor de backtesting de estrategias de portafolio.

    Uso:
        motor = BacktestEngine(config, cargador_precios=mi_cargador)        # carga por tickers
        resultado = motor.run_backtest(estrategia, ['AAPL', 'MSFT'])
        # o con datos precargados ({ticker: DataFrame con 'Close' o 'close'/'adjusted_close'}):
        resultado = BacktestEngine(config).run_backtest(estrategia, historical_data)

    Los precios se asumen en DIVISA_BASE (USD). Si `config.user_currency` es otra, hace falta
    `servicio_cambio`: el capital y las aportaciones se pasan a USD con el tipo de su fecha y
    la curva de equity se convierte de vuelta día a día.
    """

    def __init__(
        self,
        config: BacktestConfig,
        cargador_precios: CargadorPrecios | None = None,
        servicio_cambio: ServicioCambio | None = None,
    ):
        self.config = config
        self.cargador_precios = cargador_precios
        self.exchange_service = servicio_cambio
        self.user_currency = (config.user_currency or DIVISA_BASE).upper()
        self.base_currency = DIVISA_BASE
        if self.user_currency != self.base_currency and servicio_cambio is None:
            raise ValueError(
                f"user_currency={self.user_currency} requiere un servicio_cambio "
                f"(objeto con convert_amount(importe, origen, destino, fecha))"
            )
        self.currency_symbol = simbolo_divisa(self.user_currency)
        self._reset_estado()

        logger.info(f"BacktestEngine: {config.start_date} a {config.end_date}, divisa {self.user_currency}")

    def _reset_estado(self):
        """Inicializa el estado de la simulación (permite reutilizar el motor)."""
        self.initial_capital_user = self.config.initial_capital
        self.initial_capital_base = self._convert_to_base_currency(
            self.config.initial_capital, pd.to_datetime(self.config.start_date).date()
        )
        self.portfolio_value = self.initial_capital_base
        self.cash = self.initial_capital_base
        self.positions: dict[str, dict] = {}  # {ticker: {'quantity', 'avg_price'}}
        self.equity_curve: list[dict] = []
        self.trades: list[BacktestTrade] = []
        self.current_date = None
        self.total_contributions = 0.0       # en divisa base
        self.total_contributions_user = 0.0  # en divisa del usuario
        self.target_weights: dict[str, float] = {}
        self.rotaciones: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------
    # Divisas
    # ------------------------------------------------------------------

    def _convert_to_base_currency(self, amount: float, target_date) -> float:
        """Divisa del usuario → divisa base (USD) con el tipo de `target_date`."""
        if self.user_currency == self.base_currency:
            return amount
        return self.exchange_service.convert_amount(amount, self.user_currency, self.base_currency, target_date)

    def _convert_to_user_currency(self, amount: float, target_date) -> float:
        """Divisa base (USD) → divisa del usuario con el tipo de `target_date`."""
        if self.user_currency == self.base_currency:
            return amount
        return self.exchange_service.convert_amount(amount, self.base_currency, self.user_currency, target_date)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        strategy_func: FuncionEstrategia,
        tickers: list[str] | dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """
        Ejecuta el backtest completo.

        Args:
            strategy_func: (fecha, historical_data) -> {ticker: peso objetivo}. Recibe todo el
                histórico cargado; debe usar solo datos hasta `fecha` (walk-forward).
            tickers: lista de tickers (se cargan con `cargador_precios` desde
                start_date − VENTANA_ESTIMACION_DIAS) o dict {ticker: DataFrame} ya cargado.

        Returns:
            BacktestResult con métricas, equity curve y trades.
        """
        self._reset_estado()

        if isinstance(tickers, dict):
            historical_data = {t: con_columna_close(df) for t, df in tickers.items() if df is not None and not df.empty}
        else:
            historical_data = self._load_historical_data(list(tickers))
        if not historical_data:
            raise ValueError("No se pudieron cargar datos históricos")

        rebalance_dates = self._get_rebalance_dates(
            self.config.start_date, self.config.end_date, self.config.rebalance_frequency
        )

        contribution_dates_set: set = set()
        if self.config.monthly_contribution > 0:
            contribution_dates_set = {
                d.to_pydatetime()
                for d in pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='MS')
            }

        # Días de trading: unión de las fechas de todos los tickers dentro del rango
        inicio = pd.to_datetime(self.config.start_date)
        fin = pd.to_datetime(self.config.end_date)
        all_trading_dates = sorted({
            d for df in historical_data.values() for d in df.index.tolist() if inicio <= d <= fin
        })
        logger.info(f"[BacktestEngine] {len(historical_data)} tickers, {len(all_trading_dates)} días de trading")
        # Las fechas programadas que caen en fin de semana o festivo pasan al siguiente día de
        # trading (convención de bt `RunMonthly` y de zipline `date_rules.month_start`); antes
        # se perdían (un mensual de 3 años hacía ~27 rebalanceos en vez de 36).
        rebalance_dates_set = set(self._alinear_a_dias_de_trading(rebalance_dates, all_trading_dates))

        rebalance_count = 0
        initial_allocation_done = False

        for date in all_trading_dates:
            self.current_date = date
            current_prices = self._get_prices_at_date(historical_data, date)
            if not current_prices:
                continue

            # Asignación inicial en el primer día con precios
            just_did_initial_allocation = False
            if not initial_allocation_done:
                try:
                    initial_weights = strategy_func(date, historical_data)
                    self.target_weights = dict(initial_weights)
                    self._execute_rebalance(initial_weights, current_prices, date)
                    initial_allocation_done = True
                    just_did_initial_allocation = True
                    rebalance_count += 1
                except Exception:  # se reintenta al día siguiente
                    logger.exception(f"Error en asignación inicial ({date})")

            # Aportación mensual (DCA con rebalanceo implícito)
            if (self.config.monthly_contribution > 0 and initial_allocation_done
                    and self._is_contribution_date(date, contribution_dates_set, all_trading_dates)):
                self._execute_contribution(current_prices, date)

            # Valor y equity diarios
            self._update_portfolio_value(current_prices)
            self.equity_curve.append({'date': date, 'equity': self.portfolio_value, 'cash': self.cash})

            # Rebalanceo en las fechas programadas (no el mismo día de la asignación inicial)
            if date in rebalance_dates_set and initial_allocation_done and not just_did_initial_allocation:
                try:
                    target_weights = strategy_func(date, historical_data)
                    self.target_weights = dict(target_weights)
                except Exception:  # una fecha fallida no aborta la simulación
                    logger.exception(f"Error en strategy_func para {date}")
                    continue
                try:
                    self._execute_rebalance(target_weights, current_prices, date)
                    rebalance_count += 1
                except Exception:  # una fecha fallida no aborta la simulación
                    logger.exception(f"Error en rebalanceo para {date}")

        logger.info(f"[BacktestEngine] Simulación completada: {rebalance_count} rebalanceos")
        result = self._calculate_metrics()
        logger.info(
            f"[BacktestEngine] Retorno {result.total_return_pct:+.2%}, Sharpe {result.sharpe_ratio:.2f}, "
            f"MaxDD {result.max_drawdown_pct:.2%}, trades {result.total_trades}"
        )
        return result

    def _load_historical_data(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Carga los OHLCV con el cargador inyectado, desde start_date − VENTANA_ESTIMACION_DIAS."""
        if self.cargador_precios is None:
            raise ValueError("Para pasar una lista de tickers hace falta un cargador_precios")
        inicio_carga = (
            pd.to_datetime(self.config.start_date) - timedelta(days=VENTANA_ESTIMACION_DIAS)
        ).strftime('%Y-%m-%d')
        data, fallidos = {}, []
        for ticker in tickers:
            try:
                df = self.cargador_precios(ticker, inicio_carga, self.config.end_date)
            except Exception as exc:  # noqa: BLE001 — un ticker defectuoso no debe abortar la carga
                logger.error(f"{ticker}: error cargando datos - {exc}")
                df = None
            if df is not None and not df.empty:
                data[ticker] = con_columna_close(df)
            else:
                fallidos.append(ticker)
        if fallidos:
            logger.warning(f"No se pudieron cargar {len(fallidos)} tickers: {fallidos}")
        return data

    @staticmethod
    def _get_rebalance_dates(start_date: str, end_date: str, frequency: str) -> list[datetime]:
        """Fechas de rebalanceo: daily (D), weekly (lunes), monthly (inicio de mes), quarterly (calendario único)."""
        return fechas_programadas(start_date, end_date, frequency)

    @staticmethod
    def _alinear_a_dias_de_trading(programadas: list, dias_trading: list) -> list[pd.Timestamp]:
        """Lleva cada fecha programada al primer día de trading >= ella, sin duplicados (calendario único)."""
        return alinear_a_dias_de_trading(programadas, dias_trading)

    @staticmethod
    def _get_prices_at_date(historical_data: dict[str, pd.DataFrame], date: datetime) -> dict[str, float]:
        """Precio 'Close' de cada ticker en `date` o, si no cotizó, el último anterior."""
        prices = {}
        for ticker, df in historical_data.items():
            try:
                if date in df.index:
                    prices[ticker] = float(df.loc[date, 'Close'])
                else:
                    anteriores = df.index[df.index <= date]
                    if len(anteriores) > 0:
                        prices[ticker] = float(df.loc[anteriores[-1], 'Close'])
            except Exception as exc:  # noqa: BLE001 — sin precio ese día: se ignora el ticker
                logger.debug(f"No se pudo obtener precio de {ticker} en {date}: {exc}")
        return prices

    def _update_portfolio_value(self, current_prices: dict[str, float]):
        """Valor de la cartera = efectivo + posiciones a precio actual."""
        valor_posiciones = sum(
            pos['quantity'] * current_prices[t] for t, pos in self.positions.items() if t in current_prices
        )
        self.portfolio_value = self.cash + valor_posiciones

    @staticmethod
    def _is_contribution_date(date: datetime, contribution_dates_set: set, all_trading_dates: list[datetime]) -> bool:
        """
        True si `date` es fecha de aportación o el primer día de trading de un mes con
        aportación programada (si el día 1 no hubo mercado).
        """
        if date in contribution_dates_set:
            return True
        idx = all_trading_dates.index(date) if date in all_trading_dates else -1
        if idx > 0:
            prev_date = all_trading_dates[idx - 1]
            if date.month != prev_date.month or date.year != prev_date.year:
                for contrib_date in contribution_dates_set:
                    if contrib_date.year == date.year and contrib_date.month == date.month and contrib_date <= date:
                        return True
        return False

    def _execute_contribution(self, current_prices: dict[str, float], date: datetime):
        """
        Aportación mensual con rebalanceo implícito: el dinero se reparte entre los activos
        infraponderados en proporción a lo que les falta para su peso objetivo (sin vender).
        La comisión se descuenta del importe invertido.
        """
        contribution_user = self.config.monthly_contribution
        target_date = date.date() if hasattr(date, 'date') else date
        contribution = self._convert_to_base_currency(contribution_user, target_date)

        self.cash += contribution
        self.total_contributions += contribution
        self.total_contributions_user += contribution_user
        self.portfolio_value += contribution

        if not self.target_weights:
            tickers = list(current_prices.keys())
            self.target_weights = {t: 1.0 / len(tickers) for t in tickers}

        current_weights = {
            t: (self.positions[t]['quantity'] * current_prices[t] / self.portfolio_value)
            if t in self.positions and t in current_prices else 0.0
            for t in self.target_weights
        }
        deviations = {
            t: w - current_weights.get(t, 0.0) for t, w in self.target_weights.items()
            if w - current_weights.get(t, 0.0) > 0
        }
        if not deviations:
            deviations = {t: w for t, w in self.target_weights.items() if w > 0}
        total_deviation = sum(deviations.values())
        allocation_weights = (
            {t: d / total_deviation for t, d in deviations.items()} if total_deviation > 0
            else dict(self.target_weights)
        )

        for ticker, alloc_weight in allocation_weights.items():
            if ticker not in current_prices:
                continue
            amount_to_invest = contribution * alloc_weight
            effective_price = current_prices[ticker] * (1 + self.config.slippage_pct)
            commission = amount_to_invest * self.config.commission_rate
            if effective_price <= 0:
                continue
            shares_to_buy = (amount_to_invest - commission) / effective_price
            if shares_to_buy <= 0:
                continue
            self._add_to_position(ticker, shares_to_buy, effective_price)
            self.cash -= amount_to_invest
            self.trades.append(BacktestTrade(
                ticker=ticker, entry_date=date, exit_date=None, entry_price=effective_price,
                exit_price=None, quantity=shares_to_buy, side='buy', commission=commission,
            ))

    def _execute_rebalance(self, target_weights: dict[str, float], current_prices: dict[str, float], date: datetime):
        """
        Rebalancea hacia los pesos objetivo: normaliza si no suman ~1, cierra las posiciones
        con peso 0 o fuera del objetivo y ajusta el resto ejecutando primero las ventas, para
        que las compras dispongan del efectivo liberado. No opera diferencias < 10 unidades.
        """
        total_weight = sum(target_weights.values())
        if abs(total_weight - 1.0) > 0.01 and total_weight > 0:
            logger.warning(f"Pesos no suman 1.0 (suma: {total_weight}), normalizando")
            target_weights = {t: w / total_weight for t, w in target_weights.items()}

        target_values = {t: self.portfolio_value * w for t, w in target_weights.items()}

        # Rotación: valor que hay que operar (posiciones a la deriva → objetivo) / (2·valor)
        valores_actuales = {
            t: pos['quantity'] * current_prices.get(t, 0.0) for t, pos in self.positions.items()
        }
        operado = sum(
            abs(target_values.get(t, 0.0) - valores_actuales.get(t, 0.0))
            for t in set(target_values) | set(valores_actuales)
        )
        # Lo que quede en efectivo también cuenta (al invertir desde liquidez se opera todo)
        efectivo_objetivo = self.portfolio_value - sum(target_values.values())
        operado += abs(efectivo_objetivo - self.cash)
        if self.portfolio_value > 0:
            self.rotaciones.append((date, float(operado / (2 * self.portfolio_value))))

        for ticker in list(self.positions.keys()):
            if ticker not in target_weights or target_weights[ticker] == 0:
                self._close_position(ticker, current_prices.get(ticker), date)

        def _orden(item):
            ticker, objetivo = item
            precio = current_prices.get(ticker, 0.0)
            actual = self.positions[ticker]['quantity'] * precio if ticker in self.positions else 0.0
            return objetivo - actual  # negativas (ventas) primero

        for ticker, target_value in sorted(target_values.items(), key=_orden):
            if ticker not in current_prices:
                continue
            current_price = current_prices[ticker]
            current_value = self.positions[ticker]['quantity'] * current_price if ticker in self.positions else 0.0
            value_diff = target_value - current_value
            if abs(value_diff) < 10.0:
                continue
            if value_diff > 0:
                self._buy(ticker, value_diff, current_price, date)
            else:
                self._sell(ticker, abs(value_diff), current_price, date)

    def _add_to_position(self, ticker: str, quantity: float, price: float):
        """Suma `quantity` a la posición recalculando el precio medio."""
        if ticker in self.positions:
            old_qty = self.positions[ticker]['quantity']
            old_avg = self.positions[ticker]['avg_price']
            new_qty = old_qty + quantity
            self.positions[ticker] = {'quantity': new_qty, 'avg_price': (old_qty * old_avg + quantity * price) / new_qty}
        else:
            self.positions[ticker] = {'quantity': quantity, 'avg_price': price}

    def _buy(self, ticker: str, amount: float, price: float, date: datetime):
        """
        Compra por `amount`: coste = importe + comisión; el slippage va en el precio de
        ejecución (una sola vez). Si no llega el efectivo, compra lo que se pueda pagar
        (arreglo de GrowSave: antes la última compra de cada rebalanceo nunca cabía).
        """
        maximo_pagable = self.cash / (1 + self.config.commission_rate)
        amount = min(amount, maximo_pagable)
        if amount < 1.0:
            logger.debug(f"Cash insuficiente para comprar {ticker}")
            return
        commission = amount * self.config.commission_rate
        adjusted_price = price * (1 + self.config.slippage_pct)
        quantity = amount / adjusted_price
        self._add_to_position(ticker, quantity, adjusted_price)
        self.cash -= amount + commission
        self.trades.append(BacktestTrade(
            ticker=ticker, entry_date=date, exit_date=None, entry_price=adjusted_price,
            exit_price=None, quantity=quantity, side='buy', commission=commission,
        ))

    def _sell(self, ticker: str, amount: float, price: float, date: datetime):
        """Vende por `amount` (limitado a la posición), con slippage en precio y comisión."""
        if ticker not in self.positions:
            return
        adjusted_price = price * (1 - self.config.slippage_pct)
        quantity_to_sell = min(amount / adjusted_price, self.positions[ticker]['quantity'])
        proceeds = quantity_to_sell * adjusted_price
        commission = proceeds * self.config.commission_rate
        # Precio medio de entrada antes de tocar la posición (si se cierra, ya no existe)
        entry_price = self.positions[ticker]['avg_price']
        pnl = (adjusted_price - entry_price) * quantity_to_sell - commission

        self.positions[ticker]['quantity'] -= quantity_to_sell
        if self.positions[ticker]['quantity'] < 0.001:
            del self.positions[ticker]
        self.cash += proceeds - commission

        self.trades.append(BacktestTrade(
            ticker=ticker, entry_date=date, exit_date=date, entry_price=entry_price,
            exit_price=adjusted_price, quantity=quantity_to_sell, side='sell', pnl=pnl,
            pnl_pct=(adjusted_price - entry_price) / entry_price if entry_price > 0 else 0,
            commission=commission,
        ))

    def _close_position(self, ticker: str, price: float | None, date: datetime):
        """Cierra la posición completa (si hay precio)."""
        if ticker not in self.positions or price is None:
            return
        self._sell(ticker, self.positions[ticker]['quantity'] * price, price, date)

    def _calculate_metrics(self) -> BacktestResult:
        """Métricas del backtest en la divisa del usuario (equity convertida día a día)."""
        equity_df = pd.DataFrame(self.equity_curve)
        if equity_df.empty:
            raise ValueError("El backtest no generó ningún punto de equity (¿sin precios en el rango?)")
        equity_df = equity_df.set_index('date')

        if self.user_currency != self.base_currency:
            equity_df['equity_base'] = equity_df['equity']
            equity_df['equity'] = convertir_equity_a_divisa(
                self.exchange_service, equity_df['equity'], self.base_currency, self.user_currency
            )
            equity_df['cash'] = convertir_equity_a_divisa(
                self.exchange_service, equity_df['cash'], self.base_currency, self.user_currency
            )

        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['cummax']
        equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['cummax']

        initial_capital = self.initial_capital_user
        end_date = pd.to_datetime(self.config.end_date).date()
        final_capital = self._convert_to_user_currency(self.portfolio_value, end_date)
        total_return = final_capital - initial_capital
        total_return_pct = total_return / initial_capital if initial_capital > 0 else 0

        max_drawdown = equity_df['drawdown'].min()
        max_drawdown_pct = equity_df['drawdown_pct'].min()

        # Sharpe y Sortino anualizados con r_f = 0
        returns = equity_df['returns'].dropna()
        sharpe_ratio = (
            (returns.mean() / returns.std()) * np.sqrt(_DIAS_ANUAL) if len(returns) > 0 and returns.std() > 0 else 0.0
        )
        downside = returns[returns < 0]
        sortino_ratio = (
            (returns.mean() / downside.std()) * np.sqrt(_DIAS_ANUAL)
            if len(downside) > 0 and downside.std() > 0 else 0.0
        )

        closed_trades = [t for t in self.trades if t.exit_date is not None and t.pnl is not None]
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl <= 0]
        win_rate = len(winning) / len(closed_trades) if closed_trades else 0
        avg_win = float(np.mean([t.pnl for t in winning])) if winning else 0
        avg_loss = float(np.mean([t.pnl for t in losing])) if losing else 0
        total_losses = abs(sum(t.pnl for t in losing))
        profit_factor = sum(t.pnl for t in winning) / total_losses if total_losses > 0 else float('inf')

        years = (pd.to_datetime(self.config.end_date) - pd.to_datetime(self.config.start_date)).days / 365.25
        annual_return = ((final_capital / initial_capital) ** (1 / years) - 1) if years > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Comisiones reales pagadas (divisa base) convertidas a la del usuario
        total_fees = self._convert_to_user_currency(sum(t.commission for t in self.trades), end_date)
        total_contributions = (
            self.total_contributions_user if self.user_currency != self.base_currency else self.total_contributions
        )

        return BacktestResult(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_df,
            drawdown_series=equity_df[['drawdown', 'drawdown_pct']],
            trades=closed_trades,
            calmar_ratio=calmar_ratio,
            recovery_factor=recovery_factor,
            total_fees=total_fees,
            total_contributions=total_contributions,
            currency=self.user_currency,
            rotaciones=list(self.rotaciones),
        )


def format_backtest_results(result: BacktestResult) -> str:
    """Texto legible del resultado con el símbolo de su divisa."""
    s = simbolo_divisa(result.currency)
    linea = '=' * 52
    return (
        f"{linea}\n         RESULTADOS DEL BACKTEST ({result.currency})\n{linea}\n\n"
        f"PERFORMANCE\n"
        f"   Capital Inicial:     {s}{result.initial_capital:,.2f}\n"
        f"   Capital Final:       {s}{result.final_capital:,.2f}\n"
        f"   Retorno Total:       {s}{result.total_return:,.2f} ({result.total_return_pct:+.2%})\n\n"
        f"MÉTRICAS DE RIESGO\n"
        f"   Sharpe Ratio:        {result.sharpe_ratio:.2f}\n"
        f"   Sortino Ratio:       {result.sortino_ratio:.2f}\n"
        f"   Max Drawdown:        {s}{result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2%})\n"
        f"   Calmar Ratio:        {result.calmar_ratio:.2f}\n"
        f"   Recovery Factor:     {result.recovery_factor:.2f}\n\n"
        f"TRADES\n"
        f"   Total Trades:        {result.total_trades}\n"
        f"   Trades Ganadores:    {result.winning_trades}\n"
        f"   Trades Perdedores:   {result.losing_trades}\n"
        f"   Win Rate:            {result.win_rate:.1%}\n"
        f"   Avg Win:             {s}{result.avg_win:,.2f}\n"
        f"   Avg Loss:            {s}{result.avg_loss:,.2f}\n"
        f"   Profit Factor:       {result.profit_factor:.2f}\n\n"
        f"COSTES\n"
        f"   Total Fees:          {s}{result.total_fees:,.2f}\n"
        f"   Aportaciones:        {s}{result.total_contributions:,.2f}\n"
        f"{linea}"
    )


# Nombre de la API 0.2.0
format_backtest_result = format_backtest_results


# ============================================================================
# SERVICIO: benchmark, comparación de estrategias, métricas rolling
# ============================================================================


class PortfolioBacktestService:
    """
    Backtest de cartera frente a benchmark buy & hold y comparación de estrategias.

    Sin BD: los precios llegan por `cargador_precios` y los tipos de cambio por
    `fabrica_servicio_cambio` (se llama solo si la divisa no es USD). La aplicación puede
    persistir resultados sobrescribiendo `_guardar_resultado` (GrowSave guarda en `testeos`).
    """

    def __init__(
        self,
        cargador_precios: CargadorPrecios | None = None,
        fabrica_servicio_cambio: Callable[[], ServicioCambio] | None = None,
    ):
        self.cargador_precios = cargador_precios
        self.fabrica_servicio_cambio = fabrica_servicio_cambio

    # --- Ganchos para la aplicación -------------------------------------------------

    def _crear_servicio_cambio(self) -> ServicioCambio:
        """Servicio de tipos de cambio para divisas distintas de USD."""
        if self.fabrica_servicio_cambio is None:
            raise ValueError("Divisa distinta de USD sin fabrica_servicio_cambio")
        return self.fabrica_servicio_cambio()

    def _crear_motor(self, config: BacktestConfig) -> BacktestEngine:
        """Motor configurado con el cargador y, si hace falta, el servicio de cambio."""
        servicio = None
        if (config.user_currency or DIVISA_BASE).upper() != DIVISA_BASE:
            servicio = self._crear_servicio_cambio()
        return BacktestEngine(config, cargador_precios=self.cargador_precios, servicio_cambio=servicio)

    def _guardar_resultado(
        self,
        cliente_id: int,
        portfolio_id: int,
        nombre: str,
        tickers: list[str],
        result: dict[str, Any],
        portfolio_result: BacktestResult,
    ) -> int | None:
        """Persistencia opcional; devuelve el id guardado o None (por defecto no guarda)."""
        return None

    def _cargar_benchmark(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """OHLCV del benchmark con columna 'Close'."""
        if self.cargador_precios is None:
            raise ValueError("PortfolioBacktestService necesita un cargador_precios para el benchmark")
        return con_columna_close(self.cargador_precios(ticker, start_date, end_date))

    # --- Flujo principal ----------------------------------------------------------

    def run_backtest_with_benchmark(
        self,
        strategy_func: FuncionEstrategia,
        tickers: list[str],
        config: BacktestConfig,
        portfolio_id: int | None = None,
        cliente_id: int | None = None,
        nombre: str | None = None,
    ) -> dict[str, Any]:
        """
        Backtest de la cartera, benchmark buy & hold con las mismas aportaciones y divisa,
        comparación y curvas listas para graficar.

        Returns:
            {'portfolio': {...}, 'benchmark': {...} | None, 'comparison': {...} | None,
             'config': asdict(config), 'duration_seconds': float, ['testeo_id']}
        """
        inicio = datetime.now()
        portfolio_result = self._crear_motor(config).run_backtest(strategy_func, tickers)
        duration = (datetime.now() - inicio).total_seconds()

        benchmark_result = self._backtest_benchmark(
            config.benchmark_ticker, config.start_date, config.end_date, config.initial_capital,
            config.monthly_contribution, user_currency=config.user_currency,
        )
        comparison = self._compare_with_benchmark(portfolio_result, benchmark_result)

        portfolio_equity = portfolio_result.equity_curve.reset_index()
        portfolio_equity['date'] = portfolio_equity['date'].astype(str)
        portfolio_equity_series = portfolio_equity[['date', 'equity']].to_dict('records')

        benchmark_equity_series = []
        if benchmark_result:
            benchmark_equity = benchmark_result['equity_curve_df'].copy().reset_index()
            benchmark_equity['date'] = benchmark_equity['date'].astype(str)
            benchmark_equity_series = benchmark_equity[['date', 'equity']].to_dict('records')

        result = {
            'portfolio': {
                'initial_capital': portfolio_result.initial_capital,
                'final_capital': portfolio_result.final_capital,
                'total_return': portfolio_result.total_return,
                'total_return_pct': portfolio_result.total_return_pct,
                'sharpe_ratio': portfolio_result.sharpe_ratio,
                'sortino_ratio': portfolio_result.sortino_ratio,
                'max_drawdown': portfolio_result.max_drawdown,
                'max_drawdown_pct': portfolio_result.max_drawdown_pct,
                'calmar_ratio': portfolio_result.calmar_ratio,
                'total_trades': portfolio_result.total_trades,
                'winning_trades': portfolio_result.winning_trades,
                'losing_trades': portfolio_result.losing_trades,
                'win_rate': portfolio_result.win_rate,
                'profit_factor': portfolio_result.profit_factor,
                'total_fees': portfolio_result.total_fees,
                'total_contributions': portfolio_result.total_contributions,
                'equity_curve': portfolio_equity_series,
                'currency': portfolio_result.currency,
            },
            'benchmark': {
                'ticker': benchmark_result['ticker'],
                'initial_capital': benchmark_result['initial_capital'],
                'total_invested': benchmark_result.get('total_invested', benchmark_result['initial_capital']),
                'final_capital': benchmark_result['final_capital'],
                'total_return': benchmark_result['total_return'],
                'total_return_pct': benchmark_result['total_return_pct'],
                'sharpe_ratio': benchmark_result['sharpe_ratio'],
                'sortino_ratio': benchmark_result['sortino_ratio'],
                'max_drawdown': benchmark_result['max_drawdown'],
                'max_drawdown_pct': benchmark_result['max_drawdown_pct'],
                'calmar_ratio': benchmark_result['calmar_ratio'],
                'equity_curve': benchmark_equity_series,
            } if benchmark_result else None,
            'comparison': comparison,
            'config': asdict(config),
            'duration_seconds': duration,
        }

        if cliente_id and portfolio_id:
            testeo_id = self._guardar_resultado(
                cliente_id=cliente_id,
                portfolio_id=portfolio_id,
                nombre=nombre or f"Backtest {config.start_date} - {config.end_date}",
                tickers=tickers,
                result=result,
                portfolio_result=portfolio_result,
            )
            if testeo_id is not None:
                result['testeo_id'] = testeo_id
        return result

    def _backtest_benchmark(
        self,
        benchmark_ticker: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        monthly_contribution: float = 0.0,
        user_currency: str = DIVISA_BASE,
        servicio_cambio: ServicioCambio | None = None,
    ) -> dict[str, Any] | None:
        """
        Benchmark buy & hold (cotiza en USD) con aportaciones mensuales opcionales.

        Si user_currency no es USD, capital y aportaciones (en divisa del usuario) se pasan a
        USD con el tipo de su fecha, igual que la cartera, y la curva resultante se convierte
        a la divisa del usuario con convertir_equity_a_divisa (mismos tipos y fechas).

        Returns:
            Dict con ticker, initial_capital, total_invested (divisa usuario), final_capital,
            total_return(_pct) sobre lo invertido, sharpe, sortino, drawdowns, calmar y
            equity_curve_df; None si no hay datos o hay error.
        """
        try:
            user_currency = user_currency.upper()
            if user_currency != DIVISA_BASE and servicio_cambio is None:
                servicio_cambio = self._crear_servicio_cambio()
            capital_inicial_usuario = initial_capital
            aportacion_usuario = monthly_contribution
            if user_currency != DIVISA_BASE:
                initial_capital = servicio_cambio.convert_amount(
                    capital_inicial_usuario, user_currency, DIVISA_BASE, pd.to_datetime(start_date).date()
                )

            benchmark_data = self._cargar_benchmark(benchmark_ticker, start_date, end_date)
            if benchmark_data is None or benchmark_data.empty:
                logger.warning(f"No se pudieron obtener datos de {benchmark_ticker}")
                return None
            benchmark_data = benchmark_data.copy()

            initial_price = benchmark_data.iloc[0]['Close']
            shares = initial_capital / initial_price
            total_invested_usuario = capital_inicial_usuario

            if monthly_contribution > 0:
                equity_values = []
                contribution_dates = set(pd.date_range(start=start_date, end=end_date, freq='MS'))
                last_contribution_month = None
                for idx, row in benchmark_data.iterrows():
                    current_price = row['Close']
                    current_month = (idx.year, idx.month)
                    if current_month != last_contribution_month:
                        for contrib_date in contribution_dates:
                            if (contrib_date.year == idx.year and contrib_date.month == idx.month
                                    and contrib_date <= idx):
                                aportacion_usd = monthly_contribution
                                if user_currency != DIVISA_BASE:
                                    aportacion_usd = servicio_cambio.convert_amount(
                                        aportacion_usuario, user_currency, DIVISA_BASE,
                                        idx.date() if hasattr(idx, 'date') else idx,
                                    )
                                shares += aportacion_usd / current_price
                                total_invested_usuario += aportacion_usuario
                                last_contribution_month = current_month
                                break
                    equity_values.append(shares * current_price)
                benchmark_data['equity'] = equity_values
            else:
                benchmark_data['equity'] = shares * benchmark_data['Close']

            if user_currency != DIVISA_BASE:
                benchmark_data['equity'] = convertir_equity_a_divisa(
                    servicio_cambio, benchmark_data['equity'], DIVISA_BASE, user_currency
                )
            total_invested = total_invested_usuario
            benchmark_data['returns'] = benchmark_data['equity'].pct_change()

            final_capital = benchmark_data.iloc[-1]['equity']
            total_return = final_capital - total_invested
            total_return_pct = total_return / total_invested

            returns = benchmark_data['returns'].dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(_DIAS_ANUAL) if returns.std() > 0 else 0.0
            downside = returns[returns < 0]
            sortino_ratio = (
                (returns.mean() / downside.std()) * np.sqrt(_DIAS_ANUAL)
                if len(downside) > 0 and downside.std() > 0 else 0.0
            )

            benchmark_data['cummax'] = benchmark_data['equity'].cummax()
            benchmark_data['drawdown'] = benchmark_data['equity'] - benchmark_data['cummax']
            benchmark_data['drawdown_pct'] = benchmark_data['drawdown'] / benchmark_data['cummax']
            max_drawdown = benchmark_data['drawdown'].min()
            max_drawdown_pct = benchmark_data['drawdown_pct'].min()

            years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
            annual_return = ((final_capital / total_invested) ** (1 / years) - 1) if years > 0 else 0
            calmar_ratio = annual_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0

            equity_curve_df = benchmark_data[['equity']].copy()
            equity_curve_df.index.name = 'date'

            return {
                'ticker': benchmark_ticker,
                'initial_capital': capital_inicial_usuario,
                'total_invested': total_invested,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'calmar_ratio': calmar_ratio,
                'equity_curve_df': equity_curve_df,
            }
        except Exception:  # sin benchmark el backtest sigue siendo válido
            logger.exception("Error en backtest de benchmark")
            return None

    @staticmethod
    def _compare_with_benchmark(
        portfolio_result: BacktestResult, benchmark_result: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Diferencias cartera − benchmark (retorno, Sharpe, Sortino, drawdown, Calmar)."""
        if not benchmark_result:
            return None
        return {
            'excess_return': portfolio_result.total_return_pct - benchmark_result['total_return_pct'],
            'sharpe_difference': portfolio_result.sharpe_ratio - benchmark_result['sharpe_ratio'],
            'sortino_difference': portfolio_result.sortino_ratio - benchmark_result['sortino_ratio'],
            'max_drawdown_difference': portfolio_result.max_drawdown_pct - benchmark_result['max_drawdown_pct'],
            'calmar_difference': portfolio_result.calmar_ratio - benchmark_result['calmar_ratio'],
            'outperformed': portfolio_result.total_return_pct > benchmark_result['total_return_pct'],
        }

    @staticmethod
    def _convert_numpy_types(obj):
        """Tipos NumPy → nativos (serialización JSON)."""
        return convertir_numpy_a_nativo(obj)

    def compare_strategies(
        self,
        strategies: dict[str, tuple[Callable, dict]],
        tickers: list[str] | dict[str, pd.DataFrame],
        config: BacktestConfig,
    ) -> dict[str, Any]:
        """
        Ejecuta varias estrategias con la misma configuración y las ordena por Sharpe.

        Args:
            strategies: {nombre: (funcion(fecha, datos, **kwargs), kwargs)}.

        Returns:
            {'individual_results': {nombre: BacktestResult | {'error'}},
             'comparison_table': [...], 'best_strategy': nombre | None}
        """
        results: dict[str, Any] = {}
        for strategy_name, (strategy_func, strategy_kwargs) in strategies.items():
            try:
                def wrapped_strategy(date, data, _f=strategy_func, _kw=strategy_kwargs):
                    return _f(date, data, **_kw)
                results[strategy_name] = self._crear_motor(config).run_backtest(wrapped_strategy, tickers)
            except Exception as exc:  # noqa: BLE001 — una estrategia fallida no aborta la comparación
                logger.error(f"[Compare] Error en {strategy_name}: {exc}")
                results[strategy_name] = {'error': str(exc)}

        comparison_table = [
            {
                'strategy': name,
                'total_return_pct': r.total_return_pct,
                'sharpe_ratio': r.sharpe_ratio,
                'sortino_ratio': r.sortino_ratio,
                'max_drawdown_pct': r.max_drawdown_pct,
                'calmar_ratio': r.calmar_ratio,
                'win_rate': r.win_rate,
                'profit_factor': r.profit_factor,
            }
            for name, r in results.items() if isinstance(r, BacktestResult)
        ]
        comparison_df = pd.DataFrame(comparison_table)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
        return {
            'individual_results': results,
            'comparison_table': comparison_df.to_dict('records'),
            'best_strategy': comparison_df.iloc[0]['strategy'] if not comparison_df.empty else None,
        }

    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """Sharpe (r_f = 0) y volatilidad anualizados en ventana móvil de `window` días."""
        media = returns.rolling(window).mean()
        desviacion = returns.rolling(window).std()
        return pd.DataFrame({
            'rolling_sharpe': media * _DIAS_ANUAL / (desviacion * np.sqrt(_DIAS_ANUAL)),
            'rolling_volatility': desviacion * np.sqrt(_DIAS_ANUAL),
        })
