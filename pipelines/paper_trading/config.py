"""
config.py — Configuración del paper trading: claves de Alpaca y estrategia.

Responsabilidad:
    Leer variables de entorno (.env en la raíz del repo, ver .env.example) y exponer
    dataclasses tipadas. Las claves NUNCA van en el código. La estrategia se toma de un
    JSON opcional (`--config`) o de valores por defecto; ya no hay base de datos de GrowSave.

Se relaciona con:
    - __main__.py: construye `ConfiguracionServicio` y `ConfiguracionEstrategia`.
    - alpaca_vivo.py y backtest.py: consumen estos objetos.
    - ypyquant.execution.nautilus.constructor_senales: interpreta `config_senales`.
    - ypyquant.estrategias (registro): si `estrategia_registro` está definida, la estrategia sale del
      registro (incluidas y `estrategias_usuario/`) en vez de las señales técnicas.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
TICKERS_POR_DEFECTO = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def cargar_env() -> None:
    """Carga el .env de la raíz si python-dotenv está instalado (extra `trading`)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(RAIZ / ".env")


@dataclass
class ConfiguracionAlpaca:
    clave_api: str
    clave_secreta: str
    paper: bool = True  # siempre paper: este CLI no opera con dinero real

    @classmethod
    def desde_entorno(cls) -> "ConfiguracionAlpaca":
        clave = os.getenv("ALPACA_API_KEY")
        secreta = os.getenv("ALPACA_SECRET_KEY")
        if not clave or not secreta:
            raise OSError("ALPACA_API_KEY y ALPACA_SECRET_KEY deben estar definidas (ver .env.example)")
        return cls(clave_api=clave, clave_secreta=secreta)


@dataclass
class ConfiguracionEstrategia:
    """Estrategia a ejecutar: tickers, señales, riesgo y umbral de operación."""
    nombre: str = "tecnica_por_defecto"
    tickers: list[str] = field(default_factory=lambda: list(TICKERS_POR_DEFECTO))
    config_senales: dict = field(default_factory=lambda: {"generators": [{"name": "technical", "weight": 1.0}]})
    config_riesgo: dict = field(default_factory=lambda: {"max_position_pct": 0.10, "max_drawdown": 0.15})
    frecuencia_rebalanceo: str = "daily"  # "daily" | "weekly"
    umbral_senal: float = 0.3  # |señal| > umbral para operar
    # Modo registro: nombre de una `ypyquant.estrategias.Estrategia` y sus parámetros. Si se define,
    # el bucle rebalancea hacia los pesos de la estrategia (ignora config_senales y umbral_senal).
    estrategia_registro: str | None = None
    parametros_registro: dict = field(default_factory=dict)

    @classmethod
    def desde_json(cls, ruta: Path) -> "ConfiguracionEstrategia":
        """Carga la estrategia desde un JSON con las mismas claves que los campos (ver estrategia_ejemplo.json)."""
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(**datos)


@dataclass
class ConfiguracionServicio:
    """Parámetros del bucle en vivo."""
    alpaca: ConfiguracionAlpaca | None = None
    segundos_entre_iteraciones: int = 60
    dias_historico: int = 252
    ruta_fills: Path = RAIZ / "data" / "paper_trading" / "fills.jsonl"

    @classmethod
    def desde_entorno(cls, con_alpaca: bool = True) -> "ConfiguracionServicio":
        return cls(
            alpaca=ConfiguracionAlpaca.desde_entorno() if con_alpaca else None,
            segundos_entre_iteraciones=int(os.getenv("NAUTILUS_POLL_INTERVAL", "60")),
            dias_historico=int(os.getenv("NAUTILUS_LOOKBACK_DAYS", "252")),
        )
