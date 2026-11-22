"""
Periodo reservado (hold-out) con contador de usos.

Responsabilidad:
- `RegistroReservado`: lleva en `data/reservado.json` cuántas veces se ha mirado cada periodo
  reservado. Clave = (universo resuelto ordenado, fecha_fin_desarrollo, fecha_fin); valor = lista
  de usos `{fecha, barrido, experimento}`. Es común a todos los barridos: mirar el mismo tramo
  desde otro barrido sobre el mismo universo también cuenta.
- `evaluar_reservado(barrido, referencia, cargador_precios)`: ejecuta UNA combinación del barrido
  (por id de experimento o por etiqueta) en (fecha_fin_desarrollo, base.fecha_fin] con la misma
  historia previa que en desarrollo, la guarda como experimento (para verla en la UI y en
  `/api/comparar`), registra el uso y devuelve el resultado con `usos_previos` y un aviso explícito
  si el periodo ya se había usado.

Por qué un contador: el reservado solo es fuera de muestra la PRIMERA vez. Cada vez que se mira y
se vuelve a elegir o ajustar algo, el reservado pasa a ser otro tramo de desarrollo (sesgo de
selección): el contador no lo impide, lo hace visible.

Se relaciona con:
- ypyquant.estrategias.barridos (`Barrido`, `fecha_fin_desarrollo`) y .experimentos
  (`ejecutar_experimento`, `resolver_universo`), .almacen (guarda el experimento reservado) y
  ypyquant.almacen_documentos (primitivas de E/S atómica de un archivo suelto: `leer_json`, `escribir_json_atomico`).
- app/api/rutas_barridos.py (`POST /api/barridos/{id}/reservado`).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ypyquant.almacen_documentos import escribir_json_atomico, leer_json
from ypyquant.estrategias.barridos import Barrido
from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    ejecutar_experimento,
    resolver_universo,
)

_CERROJO = threading.Lock()  # un único proceso escribe el archivo; los hilos de la API se serializan


def clave_reservado(universo: list[str] | str, fecha_fin_desarrollo: str, fecha_fin: str) -> str:
    """Clave estable (hash de universo resuelto ordenado + fechas); el universo legible va dentro del valor."""
    canonico = json.dumps({"universo": sorted(resolver_universo(universo)), "fecha_fin_desarrollo": fecha_fin_desarrollo,
                           "fecha_fin": fecha_fin}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:16]


class RegistroReservado:
    """Contador de usos del periodo reservado, persistido en un JSON (por defecto `data/reservado.json`)."""

    def __init__(self, ruta: Path | str):
        self.ruta = Path(ruta)

    def usos(self, clave: str) -> list[dict[str, Any]]:
        """Usos registrados para esa clave (lista vacía si nunca se ha mirado)."""
        return (leer_json(self.ruta) or {}).get(clave, {}).get("usos", [])

    def registrar(self, clave: str, universo: list[str], fecha_fin_desarrollo: str, fecha_fin: str,
                  barrido: str, experimento: str) -> None:
        """Añade un uso `{fecha, barrido, experimento}` (escritura atómica)."""
        with _CERROJO:
            datos = leer_json(self.ruta) or {}
            entrada = datos.setdefault(clave, {"universo": sorted(universo), "fecha_fin_desarrollo": fecha_fin_desarrollo,
                                               "fecha_fin": fecha_fin, "usos": []})
            entrada["usos"].append({"fecha": datetime.now().isoformat(timespec="seconds"), "barrido": barrido,
                                    "experimento": experimento})
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            escribir_json_atomico(self.ruta, datos)


def _resolver_combinacion(barrido: Barrido, referencia: str) -> dict[str, Any]:
    """Combinación por id de experimento o por etiqueta exacta (error claro si no existe o es ambigua)."""
    coincidencias = [c for c in barrido.combinaciones if referencia in (c["id"], c["etiqueta"])]
    if not coincidencias:
        raise ValueError(f"'{referencia}' no es ni un id ni una etiqueta de este barrido")
    if len(coincidencias) > 1:
        raise ValueError(f"'{referencia}' es ambigua ({len(coincidencias)} combinaciones); usa el id")
    return coincidencias[0]


def config_reservado(config_desarrollo: ConfiguracionExperimento, fecha_fin_desarrollo: str,
                     fecha_fin: str) -> ConfiguracionExperimento:
    """
    Misma combinación en el reservado: empieza el día siguiente a `fecha_fin_desarrollo` y alarga
    `dias_historia_previa` para cargar los mismos datos previos que en desarrollo (los estimadores
    ven la historia real hasta el día anterior; no se mira nada posterior a cada fecha).
    """
    inicio = pd.Timestamp(fecha_fin_desarrollo) + pd.Timedelta(days=1)
    dias = config_desarrollo.dias_historia_previa + (inicio - pd.Timestamp(config_desarrollo.fecha_inicio)).days
    if dias > 7300:
        raise ValueError("La historia previa necesaria para el reservado supera los 7300 días")
    return config_desarrollo.model_copy(update={
        "fecha_inicio": inicio.strftime("%Y-%m-%d"), "fecha_fin": fecha_fin, "dias_historia_previa": dias,
        "etiqueta": f"reservado: {config_desarrollo.etiqueta}",
    })


def evaluar_reservado(barrido: Barrido, referencia: str, cargador_precios,
                      registro: RegistroReservado | None = None) -> dict[str, Any]:
    """
    Ejecuta la combinación `referencia` (id o etiqueta) en el periodo reservado y registra el uso.

    Guarda el resultado como un experimento más (`experimento_reservado`) y lo añade a
    `reservado.json` del barrido. Devuelve `{experimento_reservado, combinacion, periodo, metricas, usos_previos,
    n_usos, aviso, avisos}`; `aviso` (texto) solo si el periodo ya se había usado antes.

    Raises:
        ValueError: el barrido no tiene `fecha_fin_desarrollo` o la referencia no existe.
    """
    config = barrido.config
    if config.fecha_fin_desarrollo is None:
        raise ValueError("Este barrido no tiene periodo reservado (falta fecha_fin_desarrollo)")
    combinacion = _resolver_combinacion(barrido, referencia)
    base_dev = next(c for c, _, _ in config.combinaciones() if c.identificador() == combinacion["id"])
    fin_dev, fin = config.fecha_fin_desarrollo, config.base.fecha_fin
    reservado = config_reservado(base_dev, fin_dev, fin).normalizada()
    registro = registro or RegistroReservado(barrido.almacen.ruta_reservado)
    clave = clave_reservado(base_dev.universo, fin_dev, fin)
    usos_previos = registro.usos(clave)

    resultado = ejecutar_experimento(reservado, cargador_precios)
    almacen = barrido.almacen.experimentos
    id_reservado = almacen.registrar(reservado)
    almacen.guardar(id_reservado, resultado)
    registro.registrar(clave, resolver_universo(base_dev.universo), fin_dev, fin, barrido.id, combinacion["id"])

    aviso = None
    if usos_previos:
        aviso = (f"Este periodo reservado ({fin_dev} a {fin}, mismo universo) ya se había usado {len(usos_previos)} "
                 f"{'vez' if len(usos_previos) == 1 else 'veces'}: deja de ser fuera de muestra (sesgo de selección). "
                 "Trata este resultado como optimista.")
    salida = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "experimento_reservado": id_reservado, "combinacion": combinacion,
        "periodo": {"inicio": reservado.fecha_inicio, "fin": fin}, "metricas": resultado.metricas,
        "usos_previos": usos_previos, "n_usos": len(usos_previos) + 1, "aviso": aviso, "avisos": resultado.avisos,
    }
    anteriores = barrido.almacen.leer_resultado(barrido.id, "reservado") or []
    barrido.almacen.guardar_resultado(barrido.id, "reservado", [*anteriores, salida])
    return salida
