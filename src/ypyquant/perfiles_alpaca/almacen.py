"""
Almacén de perfiles de Alpaca: lo que se guarda en disco de cada cuenta paper (una carpeta por perfil).

Responsabilidad: `AlmacenPerfiles` es un adaptador fino sobre `AlmacenDocumentos` (ypyquant.almacen_documentos):
`<raiz>/<PERFIL>/` con
- `vinculo.json`: el experimento (o configuración) al que está atada la cuenta, con su receta congelada.
- `bot.json`: el bot en marcha (pid, marca del proceso, modo, intervalo, inicio); lo escribe el gestor de bots.
- `estado_estrategia.json`: la memoria (`estado`) de la estrategia entre pasadas y reinicios del bot.
- `instantaneas.parquet`: una fila por día con equity, efectivo y posiciones de la cuenta (sin ella no se puede
  medir luego el descuadre de cartera: Alpaca no devuelve las posiciones pasadas).
- `fills.jsonl`, `estado_bot.json` y `bot.log`: los escriben el bot (RegistroFills, EstadoBot) y el gestor; aquí solo
  se calculan sus rutas (`ruta_*`) para que nadie recomponga `parent / "..."`.
Nada de este módulo guarda ni lee claves: los perfiles se identifican por su nombre.

Se relaciona con:
- ypyquant.raiz_datos (`RaizDatos.paper_trading`: de dónde cuelga) y ypyquant.almacen_documentos (E/S atómica).
- ypyquant.perfiles_alpaca.vinculo (escribe el vínculo), app/api/gestor_bots.py (bot.json), app/api/rutas_alpaca.py
  (lectura para la UI) y pipelines/paper_trading (vínculo, estado de la estrategia, instantáneas, fills).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ypyquant.almacen_documentos import AlmacenDocumentos, limpiar_para_json
from ypyquant.perfiles_alpaca.perfiles import nombre_valido
from ypyquant.raiz_datos import RaizDatos

COLUMNAS_INSTANTANEAS = ["fecha", "equity", "efectivo", "posiciones"]


class AlmacenPerfiles:
    """Documentos de cada perfil de Alpaca en `<raiz>/<PERFIL>/` (por defecto `data/paper_trading/`)."""

    def __init__(self, raiz: Path | str | None = None):
        self.documentos = AlmacenDocumentos(raiz if raiz is not None else RaizDatos().paper_trading,
                                            nombre="perfil", validar_id=nombre_valido)

    @property
    def raiz(self) -> Path:
        return self.documentos.raiz

    # ---- rutas -----------------------------------------------------------------------------------

    def carpeta(self, perfil: str) -> Path:
        """Carpeta del perfil (la crea)."""
        return self.documentos.crear(perfil)

    def ruta_fills(self, perfil: str) -> Path:
        return self.documentos.carpeta(perfil) / "fills.jsonl"

    def ruta_log(self, perfil: str) -> Path:
        return self.documentos.carpeta(perfil) / "bot.log"

    def ruta_estado_bot(self, perfil: str) -> Path:
        return self.documentos.carpeta(perfil) / "estado_bot.json"

    def perfiles_con_datos(self) -> list[str]:
        """Perfiles que tienen carpeta (válidos), ordenados."""
        return [p for p in self.documentos.ids() if nombre_valido(p)]

    # ---- vínculo, bot y estado de la estrategia (JSON) -------------------------------------------

    def leer_vinculo(self, perfil: str) -> dict[str, Any] | None:
        return self.documentos.leer_json(perfil, "vinculo")

    def guardar_vinculo(self, perfil: str, vinculo: dict[str, Any]) -> None:
        self.carpeta(perfil)
        self.documentos.escribir_json(perfil, "vinculo", limpiar_para_json(vinculo))

    def leer_bot(self, perfil: str) -> dict[str, Any] | None:
        return self.documentos.leer_json(perfil, "bot")

    def guardar_bot(self, perfil: str, meta: dict[str, Any]) -> None:
        self.carpeta(perfil)
        self.documentos.escribir_json(perfil, "bot", limpiar_para_json(meta))

    def borrar_bot(self, perfil: str) -> None:
        ruta = self.documentos.ruta_json(perfil, "bot")
        ruta.unlink(missing_ok=True)

    def leer_estado_estrategia(self, perfil: str) -> dict[str, Any] | None:
        return self.documentos.leer_json(perfil, "estado_estrategia")

    def guardar_estado_estrategia(self, perfil: str, contenido: dict[str, Any]) -> None:
        self.carpeta(perfil)
        self.documentos.escribir_json(perfil, "estado_estrategia", limpiar_para_json(contenido))

    def borrar_estado_estrategia(self, perfil: str) -> None:
        self.documentos.ruta_json(perfil, "estado_estrategia").unlink(missing_ok=True)

    # ---- instantáneas diarias de la cuenta -------------------------------------------------------

    def guardar_instantanea(self, perfil: str, fecha: str, equity: float, efectivo: float,
                            posiciones: dict[str, dict[str, float]]) -> None:
        """
        Guarda (o reemplaza, si ya había una de esa fecha) la fila del día en `instantaneas.parquet`.

        `posiciones` es `{ticker: {"cantidad": q, "precio": p}}`; se guarda como JSON en una columna de texto.
        """
        fila = {"fecha": fecha, "equity": float(equity), "efectivo": float(efectivo),
                "posiciones": json.dumps(limpiar_para_json(posiciones), sort_keys=True)}
        with self.documentos.cerrojo:
            previas = self.leer_instantaneas(perfil)
            if not previas.empty:
                previas = previas[previas["fecha"] != fecha]
            tabla = pd.concat([previas, pd.DataFrame([fila])], ignore_index=True).sort_values("fecha", ignore_index=True)
            self.carpeta(perfil)
            self.documentos.escribir_parquet(perfil, "instantaneas", tabla[COLUMNAS_INSTANTANEAS])

    def leer_instantaneas(self, perfil: str) -> pd.DataFrame:
        """Tabla de instantáneas (columnas `COLUMNAS_INSTANTANEAS`); vacía si aún no hay ninguna."""
        tabla = self.documentos.leer_parquet(perfil, "instantaneas") if self.documentos.hay_carpeta(perfil) else None
        return tabla if tabla is not None else pd.DataFrame(columns=COLUMNAS_INSTANTANEAS)
