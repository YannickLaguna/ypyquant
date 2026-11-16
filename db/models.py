"""
Modelos ORM SQLAlchemy para Sistema_trading — Fase 1.

Tablas incluidas: tickers, activos, factor_data, tipos_cambio,
                  portafolios, posiciones, operaciones, signals, backtests.

Tablas excluidas (Fase 2+): clientes, objetivos, reportes, bonos,
                             indicadores, multifactor_models, strategies.
"""

from datetime import datetime
from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Float,
    ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# TICKERS — catálogo de activos financieros
# ---------------------------------------------------------------------------

class Ticker(Base):
    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False, unique=True)
    name = Column(String(500), nullable=True)
    exchange = Column(String(100), nullable=True)        # NYSE, NASDAQ, BME, …
    asset_type = Column(String(50), nullable=True)       # equity, etf, crypto, …
    sector = Column(String(200), nullable=True)
    industry = Column(String(200), nullable=True)
    country = Column(String(100), nullable=True)
    currency = Column(String(10), default="USD")
    is_active = Column(Boolean, default=True)
    index_memberships = Column(JSONB, nullable=True)     # {"SP500": true, "IBEX35": false}
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    activos = relationship("Activo", back_populates="ticker_ref",
                           primaryjoin="Ticker.symbol == foreign(Activo.ticker)",
                           foreign_keys="[Activo.ticker]")


# ---------------------------------------------------------------------------
# ACTIVOS — OHLCV diario por ticker
# ---------------------------------------------------------------------------

class Activo(Base):
    __tablename__ = "activos"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adjusted_close = Column(Float)
    volume = Column(BigInteger)
    ticker = Column(String(50), nullable=False)
    currency = Column(String(10), default="USD")
    source = Column(String(50), nullable=True)   # yfinance | twelvedata | alphavantage
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    ticker_ref = relationship("Ticker", back_populates="activos",
                               primaryjoin="foreign(Activo.ticker) == Ticker.symbol",
                               foreign_keys="[Activo.ticker]")

    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uix_activos_date_ticker"),
        Index("ix_activos_ticker_date", "ticker", "date"),
    )


# ---------------------------------------------------------------------------
# FACTOR DATA — formato largo canónico (idéntico a TFM)
# ---------------------------------------------------------------------------

class FactorData(Base):
    __tablename__ = "factor_data"

    id = Column(Integer, primary_key=True)
    factor_name = Column(String(50), nullable=False)  # 'mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'rf'
    fecha = Column(Date, nullable=False)
    valor = Column(Float, nullable=True)
    source = Column(String(100), default="french_data_library")
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("factor_name", "fecha", "source", name="uix_factor_fecha_source"),
        Index("ix_factor_data_factor_fecha", "factor_name", "fecha"),
    )


# ---------------------------------------------------------------------------
# TIPOS DE CAMBIO — FX diario
# ---------------------------------------------------------------------------

class TipoCambio(Base):
    __tablename__ = "tipos_cambio"

    id = Column(Integer, primary_key=True)
    par = Column(String(10), nullable=False)    # EURUSD, USDJPY, …
    fecha = Column(Date, nullable=False)
    tasa = Column(Float, nullable=False)
    tasa_inversa = Column(Float, nullable=True)
    source = Column(String(50), default="yfinance")
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("par", "fecha", name="uix_tipo_cambio_par_fecha"),
        Index("ix_tipo_cambio_par_fecha", "par", "fecha"),
    )


# ---------------------------------------------------------------------------
# RESEARCH PORTFOLIOS — portfolios de investigación (sin cliente_id)
# Renombrado desde 'portafolios' para distinguir de los portafolios GrowSave
# ---------------------------------------------------------------------------

class ResearchPortfolio(Base):
    __tablename__ = "research_portfolios"

    id = Column(Integer, primary_key=True)
    nombre = Column(String(255), nullable=False)
    descripcion = Column(Text, nullable=True)
    estrategia_id = Column(Integer, nullable=True)
    fecha_creacion = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    tickers = Column(JSONB, nullable=True)            # ["AAPL", "MSFT", …]
    weights = Column(JSONB, nullable=True)            # {"AAPL": 0.3, …}
    metadata_ = Column("metadata", JSONB, nullable=True)

    posiciones = relationship("Posicion", back_populates="portafolio",
                               cascade="all, delete-orphan")
    operaciones = relationship("Operacion", back_populates="portafolio",
                                cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# POSICIONES — holdings actuales por portafolio
# ---------------------------------------------------------------------------

class Posicion(Base):
    __tablename__ = "posiciones"

    id = Column(Integer, primary_key=True)
    portafolio_id = Column(Integer, ForeignKey("research_portfolios.id"), nullable=False)
    ticker = Column(String(50), nullable=False)
    cantidad = Column(Float, nullable=False)
    precio_entrada = Column(Float, nullable=True)
    fecha_entrada = Column(Date, nullable=True)      # renombrado desde fecha_apertura
    peso_objetivo = Column(Float, nullable=True)     # peso target en el portafolio
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    portafolio = relationship("ResearchPortfolio", back_populates="posiciones")

    __table_args__ = (
        Index("ix_posiciones_portafolio_ticker", "portafolio_id", "ticker"),
    )


# ---------------------------------------------------------------------------
# OPERACIONES — log de trades
# ---------------------------------------------------------------------------

class Operacion(Base):
    __tablename__ = "operaciones"

    id = Column(Integer, primary_key=True)
    portafolio_id = Column(Integer, ForeignKey("research_portfolios.id"), nullable=False)
    ticker = Column(String(50), nullable=False)
    tipo = Column(String(10), nullable=False)        # 'compra' | 'venta' (NO usar BUY/SELL)
    cantidad = Column(Float, nullable=False)
    precio = Column(Float, nullable=False)
    comision = Column(Float, default=0.0)
    fecha = Column(DateTime, nullable=False, default=datetime.utcnow)
    notas = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    portafolio = relationship("ResearchPortfolio", back_populates="operaciones")

    __table_args__ = (
        Index("ix_operaciones_portafolio_fecha", "portafolio_id", "fecha"),
    )


# ---------------------------------------------------------------------------
# SIGNALS — señales generadas por modelos/estrategias
# ---------------------------------------------------------------------------

class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(50), nullable=False)
    fecha = Column(Date, nullable=False)
    tipo_signal = Column(String(50), nullable=False)  # BUY | SELL | HOLD | LONG | SHORT
    valor = Column(Float, nullable=True)              # score / probabilidad / z-score
    modelo = Column(String(255), nullable=True)       # nombre del modelo que generó la señal
    confianza = Column(Float, nullable=True)          # 0.0 – 1.0
    metadata_ = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_signals_ticker_fecha", "ticker", "fecha"),
        Index("ix_signals_modelo", "modelo"),
    )


# ---------------------------------------------------------------------------
# BACKTESTS — resultados de backtests
# ---------------------------------------------------------------------------

# DEPRECATED: No escribir nuevos registros aquí.
# Usar la tabla `testeos` de TFM/Supabase con cliente_id del usuario.
class Backtest(Base):
    __tablename__ = "backtests"

    id = Column(Integer, primary_key=True)
    nombre = Column(String(255), nullable=False)
    descripcion = Column(Text, nullable=True)
    estado = Column(String(50), default="pending")   # pending | running | completed | failed
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    universo = Column(JSONB, nullable=True)           # lista de tickers usados
    parametros = Column(JSONB, nullable=True)         # config del backtest
    metricas = Column(JSONB, nullable=True)           # sharpe, sortino, CAGR, …
    resultados = Column(JSONB, nullable=True)         # equity curve, drawdowns, …
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_backtests_estado", "estado"),
    )


# ---------------------------------------------------------------------------
# INDICATOR DATA — tabla unificada long-format para factores, macro y técnicos
# ---------------------------------------------------------------------------

class IndicatorData(Base):
    __tablename__ = "indicator_data"

    id         = Column(Integer, primary_key=True)
    date       = Column(Date, nullable=False)
    entity_id  = Column(String(50), nullable=False)   # FF_USA | FRED | AAPL | SAN.MC
    metric     = Column(String(100), nullable=False)  # MKT_RF | DFF | RSI_14 | MACD
    value      = Column(Float, nullable=True)
    frequency  = Column(String(20), default="daily")  # daily | weekly | monthly
    source     = Column(String(100), nullable=True)   # french | fred | yfinance | computed
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("date", "entity_id", "metric",
                         name="uix_indicator_data"),
        Index("ix_indicator_data_entity_metric_date",
              "entity_id", "metric", "date"),
    )
