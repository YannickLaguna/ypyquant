"""
estado_bot.py — Estado persistente del bot de paper trading (JSON en `data/paper_trading/`).

Responsabilidad:
    Recordar entre iteraciones Y entre reinicios del proceso dos cosas que el bot necesita:
    (1) la fecha del último rebalanceo, para rebalancear solo cuando toca según la frecuencia
    (`toca_rebalancear`), y (2) los ids de las órdenes enviadas cuyo fill aún no se ha registrado,
    para anotar el fill real cuando se ejecuten y no duplicarlo. La fecha va asociada a una «firma»
    (estrategia + parámetros + tickers): si cambia la configuración, el estado de rebalanceo se
    reinicia para no heredar el calendario de otra estrategia. Las órdenes pendientes no dependen
    de la firma (siguen siendo órdenes de la cuenta).

Se relaciona con:
    - alpaca_vivo.py: lo carga en `ejecutar_vivo`, lo consulta en `rebalancear_con_estrategia_registro`
      y lo usa en `sincronizar_fills`.
    - config.py: `ConfiguracionServicio.ruta_estado` fija dónde se guarda.
    - ypyquant.estrategias.base: `FRECUENCIAS_VALIDAS` (daily | weekly | monthly | quarterly).
"""

import json
import logging
import os
from datetime import date
from pathlib import Path

from ypyquant.estrategias.base import FRECUENCIAS_VALIDAS

logger = logging.getLogger(__name__)


def periodo_de_rebalanceo(frecuencia: str, fecha: date) -> tuple[int, ...]:
    """
    Periodo de calendario al que pertenece `fecha` según la frecuencia: dos fechas del mismo periodo
    comparten rebalanceo. daily = el día; weekly = semana ISO; monthly = mes; quarterly = trimestre.
    """
    if frecuencia == "daily":
        return (fecha.year, fecha.month, fecha.day)
    if frecuencia == "weekly":
        iso = fecha.isocalendar()
        return (iso.year, iso.week)
    if frecuencia == "monthly":
        return (fecha.year, fecha.month)
    if frecuencia == "quarterly":
        return (fecha.year, (fecha.month - 1) // 3 + 1)
    raise ValueError(f"frecuencia no soportada: {frecuencia!r} (válidas: {', '.join(FRECUENCIAS_VALIDAS)})")


def toca_rebalancear(frecuencia: str, ultimo: date | None, hoy: date) -> bool:
    """
    True si hoy toca rebalancear: nunca se rebalanceó, o hoy cae en un periodo posterior al del último
    rebalanceo (primer día de mercado del periodo, como hace el motor de backtest). Si el reloj va por
    detrás de la última fecha registrada, no se rebalancea.
    """
    if ultimo is None:
        return True
    if hoy <= ultimo:
        return False
    return periodo_de_rebalanceo(frecuencia, hoy) != periodo_de_rebalanceo(frecuencia, ultimo)


class EstadoBot:
    """
    Estado persistente del bot: `ultimo_rebalanceo` (fecha o None) y `ordenes_pendientes` (ids de
    órdenes enviadas sin fill registrado). Cada cambio se guarda al instante (escritura atómica).

    Se crea con `EstadoBot.cargar(ruta, firma)`; un fichero ausente o corrupto da un estado vacío.
    """

    def __init__(self, ruta: Path, firma: str, ultimo_rebalanceo: date | None = None,
                 ordenes_pendientes: list[str] | None = None):
        self.ruta = Path(ruta)
        self.firma = firma
        self.ultimo_rebalanceo = ultimo_rebalanceo
        self.ordenes_pendientes: list[str] = list(ordenes_pendientes or [])

    @classmethod
    def cargar(cls, ruta: Path, firma: str) -> "EstadoBot":
        """Lee el JSON; si falta o está corrupto, estado vacío. La fecha solo se hereda con la misma firma."""
        ruta = Path(ruta)
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            pendientes = [str(i) for i in datos.get("ordenes_pendientes", [])]
            ultimo = None
            if datos.get("firma") == firma and datos.get("ultimo_rebalanceo"):
                ultimo = date.fromisoformat(datos["ultimo_rebalanceo"])
            elif datos.get("ultimo_rebalanceo"):
                logger.info("Cambió la estrategia, los parámetros o los tickers: se ignora el último rebalanceo guardado")
            return cls(ruta, firma, ultimo, pendientes)
        except FileNotFoundError:
            return cls(ruta, firma)
        except (ValueError, TypeError, AttributeError) as error:
            logger.warning("Estado del bot ilegible en %s (%s): se empieza de cero", ruta, error)
            return cls(ruta, firma)

    def guardar(self) -> None:
        """Escribe el estado de forma atómica (fichero temporal + reemplazo) para no dejarlo a medias."""
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        datos = {
            "firma": self.firma,
            "ultimo_rebalanceo": self.ultimo_rebalanceo.isoformat() if self.ultimo_rebalanceo else None,
            "ordenes_pendientes": self.ordenes_pendientes,
        }
        temporal = self.ruta.with_suffix(self.ruta.suffix + ".tmp")
        temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        os.replace(temporal, self.ruta)

    def marcar_rebalanceo(self, fecha: date) -> None:
        """Anota que se rebalanceó en `fecha` y lo persiste."""
        self.ultimo_rebalanceo = fecha
        self.guardar()

    def anadir_pendiente(self, id_orden: str) -> None:
        """Anota una orden enviada cuyo fill aún no se ha registrado (sin repetir ids)."""
        if str(id_orden) not in self.ordenes_pendientes:
            self.ordenes_pendientes.append(str(id_orden))
            self.guardar()

    def quitar_pendiente(self, id_orden: str) -> None:
        """Quita la orden de las pendientes (ya registrada o descartada)."""
        if str(id_orden) in self.ordenes_pendientes:
            self.ordenes_pendientes.remove(str(id_orden))
            self.guardar()
