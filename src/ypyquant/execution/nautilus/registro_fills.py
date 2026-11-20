"""
registro_fills.py — Registro local (JSONL) de fills de Alpaca, sin base de datos.

Responsabilidad:
    Sustituir al `DBWriter` de GrowSave (que escribía en PostgreSQL vía `quant_bridge`):
    normaliza un order de Alpaca a un dict en español y lo añade a un fichero .jsonl.

Se relaciona con:
    - pipelines/paper_trading/alpaca_vivo.py: `sincronizar_fills` llama a `RegistroFills.registrar` cuando una
      orden enviada ya se ha ejecutado (con el fill real, no al enviarla).
    - ypyquant.execution.trade_analytics: puede leer luego estos registros (un dict por línea).
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def parsear_fecha_alpaca(texto: str | None) -> datetime:
    """Parsea un timestamp ISO de Alpaca a datetime con zona horaria; si falta o es inválido, ahora (UTC)."""
    if not texto:
        return datetime.now(UTC)
    try:
        return datetime.fromisoformat(texto.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(UTC)


def normalizar_fill(orden_alpaca: dict) -> dict:
    """Convierte el dict de un order de Alpaca en un registro normalizado (lado compra/venta)."""
    return {
        "ticker": orden_alpaca.get("symbol", ""),
        "lado": "compra" if orden_alpaca.get("side") == "buy" else "venta",
        "cantidad": float(orden_alpaca.get("filled_qty") or 0),
        "precio": float(orden_alpaca.get("filled_avg_price") or 0),
        "comision": 0.0,  # Alpaca no cobra comisión en paper
        "fecha": parsear_fecha_alpaca(orden_alpaca.get("filled_at")).isoformat(),
        "id_orden": orden_alpaca.get("id", ""),
    }


class RegistroFills:
    """Añade fills normalizados a un fichero JSONL (una operación por línea), sin repetir el mismo `id_orden`."""

    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)

    def ids_registrados(self) -> set[str]:
        """Ids de orden que ya constan en el fichero (vacío si no existe; las líneas corruptas se saltan)."""
        if not self.ruta.exists():
            return set()
        ids: set[str] = set()
        for linea in self.ruta.read_text(encoding="utf-8").splitlines():
            try:
                ids.add(json.loads(linea).get("id_orden", ""))
            except (json.JSONDecodeError, AttributeError):
                continue
        return ids

    def registrar(self, orden_alpaca: dict) -> dict | None:
        """
        Guarda el fill; ignora los de cantidad 0 (órdenes aún sin ejecutar) y los de un `id_orden` ya
        registrado (idempotente: reintentar tras un reinicio no duplica). Devuelve el registro o None.
        """
        registro = normalizar_fill(orden_alpaca)
        if registro["cantidad"] <= 0:
            logger.warning("Fill ignorado: cantidad 0 para %s", registro["ticker"])
            return None
        if registro["id_orden"] and registro["id_orden"] in self.ids_registrados():
            logger.info("Fill ya registrado (orden %s): se omite", registro["id_orden"])
            return None
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with self.ruta.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return registro
