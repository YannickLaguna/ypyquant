"""
estado_bot.py — Estado persistente del bot de paper trading (JSON en `data/paper_trading/`).

Responsabilidad:
    Recordar entre iteraciones Y entre reinicios del proceso dos cosas que el bot necesita:
    (1) la fecha del último rebalanceo, para rebalancear solo cuando toca según la frecuencia
    (`toca_rebalancear`), y (2) los ids de las órdenes enviadas cuyo fill aún no se ha registrado,
    para anotar el fill real cuando se ejecuten y no duplicarlo (con el precio de referencia con que se decidió
    cada orden, que el fill enriquecido guarda para medir el slippage). La fecha va asociada a una «firma»
    (estrategia + parámetros + tickers): si cambia la configuración, el estado de rebalanceo se
    reinicia para no heredar el calendario de otra estrategia. Las órdenes pendientes no dependen
    de la firma (siguen siendo órdenes de la cuenta).

    También guarda la MEMORIA de la estrategia (`Estrategia.estado`, p. ej. las posiciones y stops de una
    `EstrategiaSenales`) en `estado_estrategia.json`, para que un reinicio del bot no la pierda
    (`guardar_estado_estrategia` / `restaurar_estado_estrategia`). Solo se guardan las entradas serializables
    a JSON (un modelo de optimización en memoria se vuelve a crear solo) y solo se restaura con la misma firma.

Se relaciona con:
    - alpaca_vivo.py: lo carga en `ejecutar_vivo`, lo consulta en `rebalancear_con_estrategia_registro`
      y lo usa en `sincronizar_fills`.
    - config.py: `ConfiguracionServicio.ruta_estado` fija dónde se guarda.
    - ypyquant.almacen_documentos: escritura atómica y lectura de `estado_estrategia.json` (mismo archivo que borra
      `ypyquant.perfiles_alpaca.vinculo.vincular` al cambiar de experimento).
    - ypyquant.execution.calendario: única fuente del calendario (`toca_rebalancear`, `periodo_de_rebalanceo`);
      aquí solo se reexportan para no romper importaciones.
"""

import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ypyquant.almacen_documentos import escribir_json_atomico, leer_json, limpiar_para_json
from ypyquant.execution.calendario import periodo_de_rebalanceo, toca_rebalancear  # noqa: F401  (reexportadas)

logger = logging.getLogger(__name__)


class EstadoBot:
    """
    Estado persistente del bot: `ultimo_rebalanceo` (fecha o None), `ordenes_pendientes` (ids de
    órdenes enviadas sin fill registrado) y `referencias` (`{id_orden: precio de referencia}` con que se decidió
    cada una). Cada cambio se guarda al instante (escritura atómica).

    Se crea con `EstadoBot.cargar(ruta, firma)`; un fichero ausente o corrupto da un estado vacío.
    """

    def __init__(self, ruta: Path, firma: str, ultimo_rebalanceo: date | None = None,
                 ordenes_pendientes: list[str] | None = None, referencias: dict[str, float] | None = None):
        self.ruta = Path(ruta)
        self.firma = firma
        self.ultimo_rebalanceo = ultimo_rebalanceo
        self.ordenes_pendientes: list[str] = list(ordenes_pendientes or [])
        self.referencias: dict[str, float] = dict(referencias or {})

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
            referencias = {str(k): float(v) for k, v in (datos.get("referencias") or {}).items()}
            return cls(ruta, firma, ultimo, pendientes, referencias)
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
            "referencias": self.referencias,
        }
        temporal = self.ruta.with_suffix(self.ruta.suffix + ".tmp")
        temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        os.replace(temporal, self.ruta)

    def marcar_rebalanceo(self, fecha: date) -> None:
        """Anota que se rebalanceó en `fecha` y lo persiste."""
        self.ultimo_rebalanceo = fecha
        self.guardar()

    def anadir_pendiente(self, id_orden: str, precio_referencia: float | None = None) -> None:
        """Anota una orden enviada cuyo fill aún no se ha registrado (sin repetir ids) y el precio con que se decidió."""
        if str(id_orden) not in self.ordenes_pendientes:
            self.ordenes_pendientes.append(str(id_orden))
            if precio_referencia is not None:
                self.referencias[str(id_orden)] = float(precio_referencia)
            self.guardar()

    def quitar_pendiente(self, id_orden: str) -> None:
        """Quita la orden de las pendientes (ya registrada o descartada) y su precio de referencia."""
        if str(id_orden) in self.ordenes_pendientes:
            self.ordenes_pendientes.remove(str(id_orden))
            self.referencias.pop(str(id_orden), None)
            self.guardar()


# ---- memoria de la estrategia ------------------------------------------------------------------


def serializar_estado_estrategia(estado: dict[str, Any]) -> dict[str, Any]:
    """Las entradas de `Estrategia.estado` que se pueden guardar como JSON; el resto (objetos en memoria) se omite."""
    guardables: dict[str, Any] = {}
    for clave, valor in estado.items():
        try:
            json.dumps(limpiar_para_json(valor), allow_nan=False)
        except (TypeError, ValueError):
            logger.debug("El estado «%s» de la estrategia no es serializable: no se guarda", clave)
            continue
        guardables[str(clave)] = limpiar_para_json(valor)
    return guardables


def guardar_estado_estrategia(ruta: Path, firma: str, estrategia: Any) -> None:
    """Guarda `estrategia.estado` (la parte serializable) con la `firma` de la configuración, de forma atómica."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    escribir_json_atomico(Path(ruta), {"firma": firma, "estado": serializar_estado_estrategia(estrategia.estado),
                                       "guardado": datetime.now(UTC).isoformat()})


def restaurar_estado_estrategia(ruta: Path, firma: str, estrategia: Any) -> bool:
    """
    Carga en `estrategia.estado` lo guardado, si existe y su firma coincide. True si se restauró algo.

    Un archivo ilegible o de otra configuración se ignora (estado vacío) con un aviso en el log.
    """
    try:
        datos = leer_json(Path(ruta))
    except (ValueError, OSError) as error:
        logger.warning("Estado de la estrategia ilegible en %s (%s): se empieza con la memoria vacía", ruta, error)
        return False
    if not isinstance(datos, dict) or not isinstance(datos.get("estado"), dict):
        return False
    if datos.get("firma") != firma:
        logger.info("La estrategia, los parámetros o los tickers cambiaron: se ignora la memoria guardada de la estrategia")
        return False
    estrategia.estado.update(datos["estado"])
    logger.info("Memoria de la estrategia restaurada (%d entradas, guardada %s)", len(datos["estado"]), datos.get("guardado"))
    return bool(datos["estado"])
