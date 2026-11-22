"""
config.py — Configuración del paper trading: claves de Alpaca y estrategia.

Responsabilidad:
    Leer variables de entorno (.env en la raíz del repo, ver .env.example) y exponer
    dataclasses tipadas. Las claves NUNCA van en el código ni en el repr de los objetos. La estrategia se toma de un
    JSON opcional (`--config`), de valores por defecto o, en el modo multi-perfil, del VÍNCULO de la cuenta con un
    experimento (`ConfiguracionEstrategia.desde_vinculo`: receta congelada con estrategia, parámetros, tickers y
    frecuencia); ya no hay base de datos de GrowSave.

    Perfiles: `ConfiguracionAlpaca.desde_entorno(perfil)` lee `ALPACA_<PERFIL>_API_KEY` / `_SECRET_KEY` (el perfil
    `DEFECTO`, las variables sin nombre) y `ConfiguracionServicio.para_perfil` coloca todo lo del bot en
    `data/paper_trading/<PERFIL>/` (fills, estado del bot, memoria de la estrategia, instantáneas).

Se relaciona con:
    - __main__.py: construye `ConfiguracionServicio` y `ConfiguracionEstrategia`.
    - alpaca_vivo.py y backtest.py: consumen estos objetos.
    - ypyquant.execution.nautilus.constructor_senales: interpreta `config_senales`.
    - ypyquant.estrategias (registro): si `estrategia_registro` está definida, la estrategia sale del
      registro (incluidas y `estrategias_usuario/`) en vez de las señales técnicas.
    - precios.py (días de historia a descargar) y estado_bot.py (dónde se guarda el estado).
    - ypyquant.perfiles_alpaca (perfiles de claves, almacén por perfil y vínculo con el experimento).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ypyquant.perfiles_alpaca.almacen import AlmacenPerfiles
from ypyquant.perfiles_alpaca.perfiles import NOMBRE_DEFECTO, claves_de, variables_de
from ypyquant.raiz_datos import RaizDatos

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
    clave_api: str = field(repr=False)  # repr=False: que un log o un traceback nunca muestre las claves
    clave_secreta: str = field(repr=False)
    paper: bool = True  # siempre paper: este CLI no opera con dinero real

    @classmethod
    def desde_entorno(cls, perfil: str | None = None) -> "ConfiguracionAlpaca":
        """Claves del perfil (`DEFECTO` si no se indica: `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`); `OSError` si faltan."""
        perfil = perfil or NOMBRE_DEFECTO
        claves = claves_de(perfil)
        if claves is None:
            api, secreto = variables_de(perfil)
            raise OSError(f"Perfil {perfil} sin configurar: {api} y {secreto} deben estar definidas (ver .env.example)")
        return cls(clave_api=claves[0], clave_secreta=claves[1])


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
    # Frecuencia con la que el bot rebalancea en modo registro ("daily" | "weekly" | "monthly" | "quarterly").
    # None = la que sugiere la estrategia (`Estrategia.frecuencia_rebalanceo`; momentum y las del TFM son "monthly").
    frecuencia_rebalanceo_registro: str | None = None
    # Vínculo con un experimento (modo multi-perfil): huella de la receta congelada y origen `{"tipo", "id"}`; la huella
    # entra en la firma del estado y el origen etiqueta los fills
    huella_receta: str | None = None
    origen: dict | None = None

    @classmethod
    def desde_vinculo(cls, vinculo: dict[str, Any]) -> "ConfiguracionEstrategia":
        """Estrategia del bot a partir del `vinculo.json` de un perfil (receta congelada, tickers ya resueltos)."""
        receta = vinculo["receta"]
        return cls(nombre=vinculo["estrategia"], tickers=list(vinculo["tickers"]), estrategia_registro=vinculo["estrategia"],
                   parametros_registro=dict(receta["parametros"]),
                   frecuencia_rebalanceo_registro=vinculo.get("frecuencia_rebalanceo") or receta.get("frecuencia_rebalanceo"),
                   huella_receta=vinculo["huella_receta"], origen=dict(vinculo["origen"]))

    @classmethod
    def desde_json(cls, ruta: Path) -> "ConfiguracionEstrategia":
        """Carga la estrategia desde un JSON con las mismas claves que los campos (ver estrategia_ejemplo.json)."""
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(**datos)


@dataclass
class ConfiguracionServicio:
    """
    Parámetros del bucle en vivo.

    `dias_historico` es el MÍNIMO de días naturales de historia a descargar (env `NAUTILUS_LOOKBACK_DAYS`,
    `--dias-historico` del CLI o `dias_historico` de la API). En modo registro se descargan los que pida la
    estrategia (`Estrategia.sesiones_necesarias`) si son más; ver `precios.resolver_dias_historico`.
    `ruta_estado` guarda la fecha del último rebalanceo y las órdenes pendientes de registrar (ver estado_bot.py).

    Modo multi-perfil (`para_perfil`): `perfil` es el nombre de la cuenta, `ruta_estado_estrategia` guarda la memoria de la
    estrategia entre reinicios, `almacen_perfiles` recibe la instantánea diaria de la cuenta y `etiquetas_fills`
    (perfil, experimento, tipo) acompañan a cada fill registrado. Sin perfil (CLI heredado) nada de eso se usa.
    """
    alpaca: ConfiguracionAlpaca | None = None
    segundos_entre_iteraciones: int = 60
    dias_historico: int = 252
    ruta_fills: Path = RAIZ / "data" / "paper_trading" / "fills.jsonl"
    ruta_estado: Path = RAIZ / "data" / "paper_trading" / "estado_bot.json"
    perfil: str | None = None
    ruta_estado_estrategia: Path | None = None
    almacen_perfiles: AlmacenPerfiles | None = None
    etiquetas_fills: dict = field(default_factory=dict)

    @classmethod
    def desde_entorno(cls, con_alpaca: bool = True, perfil: str | None = None,
                      raiz_datos: RaizDatos | None = None) -> "ConfiguracionServicio":
        """Servicio desde el entorno; con `perfil` usa sus claves y su carpeta de datos (ver `para_perfil`)."""
        alpaca = ConfiguracionAlpaca.desde_entorno(perfil) if con_alpaca else None
        segundos = int(os.getenv("NAUTILUS_POLL_INTERVAL", "60"))
        dias = int(os.getenv("NAUTILUS_LOOKBACK_DAYS", "252"))
        if perfil is None:
            return cls(alpaca=alpaca, segundos_entre_iteraciones=segundos, dias_historico=dias)
        almacen = AlmacenPerfiles((raiz_datos or RaizDatos()).paper_trading)
        return cls(alpaca=alpaca, segundos_entre_iteraciones=segundos, dias_historico=dias, perfil=perfil,
                   ruta_fills=almacen.ruta_fills(perfil), ruta_estado=almacen.ruta_estado_bot(perfil),
                   ruta_estado_estrategia=almacen.documentos.ruta_json(perfil, "estado_estrategia"),
                   almacen_perfiles=almacen, etiquetas_fills={"perfil": perfil, "tipo": "rebalanceo"})
