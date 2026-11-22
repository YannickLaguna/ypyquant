"""
Marcos de comparación: el mismo entorno para varias estrategias, para comparar solo el efecto de la estrategia
(ceteris paribus).

Responsabilidad:
- `MarcoComparacion` (Pydantic) = `Entorno` (ver `receta.py`: universo, frecuencia, costes, benchmark, semilla, historia
  previa) + id, fechas, capital y `parametros_comunes`; `receta_para(estrategia, parametros)` compone la `Receta`.
  Los `parametros_comunes` son p. ej. `ventana_dias`, `peso_maximo`, `tasa_libre_riesgo`,
  `estimador_sigma`, `frecuencia_estimacion`). Un universo con nombre de archivo se congela como lista (la etiqueta
  conserva el nombre); `sp500_pit` es dinámico y se conserva.
- `aplicar_marco(marco, estrategia, parametros_propios) -> (ConfiguracionExperimento, InformeAplicacion)`: función
  pura. Impone el entorno del marco y aplica los `parametros_comunes` SOLO si la `Parametros` de la estrategia
  declara ese campo (momentum no tiene `ventana_dias`); el informe dice cuáles se aplicaron, cuáles no y si el
  marco pisó un parámetro propio. Los parámetros propios de cada estrategia se eligen aparte.
- `preparar_comparativa(marco, entradas)`: aplica el marco a N estrategias validándolas TODAS antes de lanzar nada.
- `AlmacenMarcos`: `data/marcos/<id>/marco.json` y las comparativas lanzadas en `data/marcos/_comparativas/<id>.json`
  (id del marco + experimentos, con una copia del marco y el informe de cada estrategia). El id de una comparativa
  sale de los experimentos: relanzar lo mismo la reutiliza.
- `marco_desde_configuracion(configuracion, inicio, fin)` y `fechas_por_defecto(...)`: funciones puras que convierten
  una configuración guardada en un marco (su ENTORNO: universo, frecuencia, costes, benchmark, capital, semilla,
  historia previa y los `COMUNES_CONOCIDOS` que traiga) para lanzar la misma receta de entorno contra otras
  estrategias. El id es determinista (configuración + versión + fechas): lanzarlo dos veces reutiliza el marco.
- `diferencias_ceteris_paribus(configs)`: función pura que dice en qué difieren varios experimentos del entorno
  (universo, fechas, frecuencia, costes, benchmark, capital, semilla, historia) o de los parámetros comunes que
  declaran varias estrategias; compara con `receta.valor_comparable` (la misma canonización que la huella). La usa
  `GET /api/comparar` para avisar «no es ceteris paribus».

Los experimentos que crea una comparativa son experimentos normales (mismo almacén, misma huella): los ya
calculados se reutilizan. Los experimentos importados del TFM (`tfm_*`, sin código) no se pueden lanzar en un marco,
pero sí se comparan por los campos de su configuración.

Se relaciona con:
- ypyquant.estrategias.receta (`Entorno`, `Receta`, canonización de `diferencias_ceteris_paribus`), .experimentos
  (ConfiguracionExperimento), .registro (clases y `Parametros`) y .almacen
  (E/S JSON atómica); .configuraciones (una configuración puede apuntar a un marco con `id_marco`).
- app/api/rutas_marcos.py (las expone) y app/web (páginas Marcos y Comparar).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from ypyquant.estrategias.almacen import _escribir_json, _leer_json
from ypyquant.estrategias.configuraciones import crear_slug
from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    es_universo_pit,
    resolver_universo,
)
from ypyquant.estrategias.receta import CAMPOS_ENTORNO as CAMPOS_ENTORNO_RECETA
from ypyquant.estrategias.receta import Entorno, Frecuencia, Receta, valor_comparable
from ypyquant.estrategias.registro import obtener_estrategia

if TYPE_CHECKING:  # solo para anotar: configuraciones.py ya importa este módulo de forma indirecta
    from ypyquant.estrategias.configuraciones import ConfiguracionGuardada

RAIZ_REPO = Path(__file__).resolve().parents[3]
RAIZ_POR_DEFECTO = RAIZ_REPO / "data" / "marcos"
CARPETA_COMPARATIVAS = "_comparativas"
# Parámetros que suelen ser comunes entre estrategias; se sugieren en la UI y se vigilan en `diferencias_ceteris_paribus`
COMUNES_CONOCIDOS = ("ventana_dias", "peso_maximo", "tasa_libre_riesgo", "estimador_sigma", "frecuencia_estimacion")
# Campos de entorno que deben coincidir entre experimentos para que la comparación sea ceteris paribus
# (los del entorno de la receta, más el periodo y el capital del experimento; el orden es el del aviso en la UI)
CAMPOS_ENTORNO = ("universo", "fecha_inicio", "fecha_fin", "frecuencia_rebalanceo", "comision", "slippage",
                  "benchmark", "capital_inicial", "semilla", "dias_historia_previa")
assert set(CAMPOS_ENTORNO) == {*CAMPOS_ENTORNO_RECETA, "fecha_inicio", "fecha_fin", "capital_inicial"}


class MarcoExistente(ValueError):
    """Ya hay un marco con ese id."""


class MarcoComparacion(Entorno):
    """
    Entorno común de una comparación (ver cabecera del módulo): ES un `Entorno` (universo, frecuencia, costes,
    benchmark, semilla, historia previa) más identidad, periodo, capital y parámetros comunes. `frecuencia_rebalanceo=None`
    deja a cada estrategia con la suya (y `aplicar_marco` lo avisa: ya no es ceteris paribus).
    """

    id: str = ""
    nombre: str = Field(min_length=1)
    notas: str = ""
    universo_etiqueta: str | None = None
    fecha_inicio: str
    fecha_fin: str
    frecuencia_rebalanceo: Frecuencia | None = "monthly"
    capital_inicial: float = Field(100_000.0, gt=0)
    parametros_comunes: dict[str, Any] = Field(default_factory=dict)
    creado: str = ""
    # Solo en marcos derivados de una configuración guardada (`marco_desde_configuracion`): de qué receta y versión
    # salen. Es el enlace configuración -> marco SIN tocar la configuración (sus versiones son inmutables).
    id_configuracion: str | None = None
    version_configuracion: int | None = Field(None, ge=1)

    @field_validator("fecha_inicio", "fecha_fin")
    @classmethod
    def _fecha_valida(cls, valor: str) -> str:
        return pd.Timestamp(valor).strftime("%Y-%m-%d")

    @model_validator(mode="after")
    def _coherencia(self) -> MarcoComparacion:
        if self.fecha_inicio >= self.fecha_fin:
            raise ValueError("fecha_inicio debe ser anterior a fecha_fin")
        if not self.id:
            self.id = crear_slug(self.nombre)
        if isinstance(self.universo, list):
            pass  # ya limpio por `Entorno`
        elif not es_universo_pit(self.universo):  # nombre de archivo: se congela
            self.universo_etiqueta = self.universo
            self.universo = resolver_universo(self.universo)
        else:
            self.universo_etiqueta = self.universo
        if not self.creado:
            self.creado = datetime.now().isoformat(timespec="seconds")
        return self

    def receta_para(self, estrategia: str, parametros: Mapping[str, Any] | None = None) -> Receta:
        """La receta de `estrategia` con los parámetros dados dentro de este entorno (sin normalizar)."""
        return Receta(**self.entorno(), estrategia=estrategia, parametros=dict(parametros or {}))


ANIOS_POR_DEFECTO = 5


def marco_desde_configuracion(configuracion: ConfiguracionGuardada, fecha_inicio: str, fecha_fin: str) -> MarcoComparacion:
    """
    Marco con el entorno de una configuración guardada en `[fecha_inicio, fecha_fin]`.

    Copia universo, frecuencia de rebalanceo, comisión, slippage, benchmark, capital, semilla y días de historia
    previa, y de los parámetros de la configuración solo los de `COMUNES_CONOCIDOS` (los propios de su estrategia,
    como `delta` o `top_k`, no son del entorno). Así la propia estrategia dentro del marco reproduce exactamente el
    experimento de la configuración (misma huella) y las demás se evalúan en el mismo entorno. Lanza
    `ValidationError` si las fechas son incoherentes.
    """
    sufijo = hashlib.sha256(json.dumps([configuracion.id, configuracion.version, fecha_inicio, fecha_fin]).encode("utf-8")).hexdigest()[:10]
    return MarcoComparacion(
        **configuracion.entorno(), id=f"cfg-{sufijo}",
        nombre=f"{configuracion.nombre[:80]} (v{configuracion.version}, {fecha_inicio} a {fecha_fin})",
        notas=f"Entorno de la configuración «{configuracion.id}» v{configuracion.version}, para compararla con otras estrategias.",
        fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, capital_inicial=configuracion.capital_inicial,
        parametros_comunes={k: configuracion.parametros[k] for k in COMUNES_CONOCIDOS if k in configuracion.parametros},
        id_configuracion=configuracion.id, version_configuracion=configuracion.version,
    )


def fechas_por_defecto(configuracion: ConfiguracionGuardada, marco: MarcoComparacion | None, hoy: date | None = None) -> tuple[str, str]:
    """
    Fechas sugeridas para comparar una configuración: las de su marco enlazado, si no las del experimento de
    origen (`origen.fechas`) y, si no, los últimos `ANIOS_POR_DEFECTO` años hasta ayer.
    """
    if marco is not None:
        return marco.fecha_inicio, marco.fecha_fin
    if configuracion.origen.fechas and len(configuracion.origen.fechas) == 2:
        return configuracion.origen.fechas[0], configuracion.origen.fechas[1]
    fin = (hoy or date.today()) - timedelta(days=1)
    return (fin - timedelta(days=365 * ANIOS_POR_DEFECTO)).isoformat(), fin.isoformat()


class InformeAplicacion(BaseModel):
    """
    Qué hizo `aplicar_marco` con una estrategia: `aplicados` (comunes que la estrategia declara), `no_aplicados`
    (comunes que no declara), `sobrescritos` (parámetros propios que el marco pisó) y `avisos` legibles.
    """

    estrategia: str
    aplicados: dict[str, Any] = Field(default_factory=dict)
    no_aplicados: list[str] = Field(default_factory=list)
    sobrescritos: list[str] = Field(default_factory=list)
    avisos: list[str] = Field(default_factory=list)


def aplicar_marco(marco: MarcoComparacion, estrategia: str,
                  parametros_propios: Mapping[str, Any] | None = None) -> tuple[ConfiguracionExperimento, InformeAplicacion]:
    """
    Configuración de experimento de `estrategia` dentro de `marco` y el informe de qué se aplicó.

    El marco prevalece sobre un parámetro propio con el mismo nombre que un común (es lo que pide la comparación
    ceteris paribus) y lo anota en `informe.sobrescritos`. Lanza `ValueError` si la estrategia no está registrada
    (p. ej. un `tfm_*` importado, sin código) y `ValidationError` si algún parámetro no es válido para ella.
    """
    try:
        clase = obtener_estrategia(estrategia)
    except KeyError as error:
        raise ValueError(f"{estrategia}: no se puede lanzar en un marco, no hay código registrado "
                         "(¿experimento importado del TFM?)") from error
    campos = clase.Parametros.model_fields
    propios = dict(parametros_propios or {})
    aplicados = {k: v for k, v in marco.parametros_comunes.items() if k in campos}
    no_aplicados = [k for k in marco.parametros_comunes if k not in campos]
    sobrescritos = [k for k, v in aplicados.items() if k in propios and propios[k] != v]
    receta = marco.receta_para(estrategia, {**propios, **aplicados})
    config = ConfiguracionExperimento.desde_receta(
        receta, marco.fecha_inicio, marco.fecha_fin, marco.capital_inicial, etiqueta=f"marco {marco.id}").normalizada()
    avisos = []
    if no_aplicados:
        avisos.append(f"No se aplicó en {estrategia}: {', '.join(sorted(no_aplicados))} (la estrategia no lo declara)")
    if sobrescritos:
        avisos.append(f"El marco pisó el valor propio de {', '.join(sorted(sobrescritos))} en {estrategia}")
    if marco.frecuencia_rebalanceo is None:
        avisos.append(f"{estrategia} usa su frecuencia propia ({config.frecuencia_rebalanceo}): el marco no fija "
                      "frecuencia, la comparación no es ceteris paribus")
    informe = InformeAplicacion(estrategia=estrategia, aplicados=aplicados, no_aplicados=sorted(no_aplicados),
                                sobrescritos=sorted(sobrescritos), avisos=avisos)
    return config, informe


def preparar_comparativa(marco: MarcoComparacion,
                         entradas: Sequence[Mapping[str, Any]]) -> list[tuple[ConfiguracionExperimento, InformeAplicacion]]:
    """
    Aplica el marco a cada entrada `{estrategia, parametros}` y devuelve `[(configuración, informe)]`.

    Valida TODAS antes de devolver (así no se lanza una comparativa a medias): si alguna falla, `ValueError`
    con los fallos de todas, precedidos por el nombre de la estrategia.
    """
    if not entradas:
        raise ValueError("Indica al menos una estrategia")
    resultado, errores = [], []
    for entrada in entradas:
        nombre = entrada.get("estrategia", "?")
        try:
            resultado.append(aplicar_marco(marco, nombre, entrada.get("parametros") or {}))
        except ValueError as error:  # incluye ValidationError de Pydantic
            errores.append(f"{nombre}: {_resumen_error(error)}")
    if errores:
        raise ValueError("; ".join(errores))
    return resultado


def _resumen_error(error: ValueError) -> str:
    """Texto corto de un error de validación (campo y mensaje) o el propio mensaje."""
    errores = getattr(error, "errors", None)
    if callable(errores):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in errores(include_url=False, include_context=False))
    return str(error)


class AlmacenMarcos:
    """Persistencia de marcos y comparativas (ver cabecera del módulo). Escritura atómica por archivo."""

    def __init__(self, raiz: Path | str | None = None):
        self.raiz = Path(raiz) if raiz else RAIZ_POR_DEFECTO
        self.raiz.mkdir(parents=True, exist_ok=True)
        (self.raiz / CARPETA_COMPARATIVAS).mkdir(exist_ok=True)
        self._cerrojo = threading.Lock()

    def _carpeta(self, id_marco: str) -> Path:
        if not id_marco or not re.fullmatch(r"[a-z0-9][a-z0-9\-_.]*", id_marco) or ".." in id_marco:
            raise ValueError(f"Identificador de marco inválido: {id_marco!r}")
        return self.raiz / id_marco

    def guardar(self, marco: MarcoComparacion) -> MarcoComparacion:
        """Crea el marco. `MarcoExistente` si el id ya existe."""
        carpeta = self._carpeta(marco.id)
        with self._cerrojo:
            if (carpeta / "marco.json").exists():
                raise MarcoExistente(f"Ya existe el marco '{marco.id}'")
            carpeta.mkdir(parents=True, exist_ok=True)
            _escribir_json(carpeta / "marco.json", marco.model_dump())
        return marco

    def actualizar(self, marco: MarcoComparacion) -> MarcoComparacion:
        """Reemplaza un marco existente (`KeyError` si no existe). Los experimentos ya lanzados no cambian."""
        carpeta = self._carpeta(marco.id)
        with self._cerrojo:
            if not (carpeta / "marco.json").exists():
                raise KeyError(marco.id)
            anterior = MarcoComparacion(**_leer_json(carpeta / "marco.json"))
            cambios: dict = {"creado": anterior.creado}
            # El formulario de marcos no envía el vínculo con la configuración de origen: si falta, se conserva.
            if marco.id_configuracion is None:
                cambios |= {
                    "id_configuracion": anterior.id_configuracion,
                    "version_configuracion": anterior.version_configuracion,
                }
            marco = marco.model_copy(update=cambios)
            _escribir_json(carpeta / "marco.json", marco.model_dump())
        return marco

    def leer(self, id_marco: str) -> MarcoComparacion:
        datos = _leer_json(self._carpeta(id_marco) / "marco.json")
        if datos is None:
            raise KeyError(id_marco)
        return MarcoComparacion(**datos)

    def listar(self) -> list[MarcoComparacion]:
        """Todos los marcos, los más recientes primero."""
        marcos = []
        for carpeta in self.raiz.iterdir():
            datos = _leer_json(carpeta / "marco.json") if carpeta.is_dir() and carpeta.name != CARPETA_COMPARATIVAS else None
            if datos is not None:
                marcos.append(MarcoComparacion(**datos))
        return sorted(marcos, key=lambda m: m.creado, reverse=True)

    def borrar(self, id_marco: str) -> bool:
        """Borra el marco (las comparativas ya lanzadas se conservan con su copia del marco)."""
        carpeta = self._carpeta(id_marco)
        with self._cerrojo:
            if not carpeta.exists():
                return False
            shutil.rmtree(carpeta)
        return True

    # ---- comparativas ---------------------------------------------------------------------

    def registrar_comparativa(self, marco: MarcoComparacion, experimentos: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """
        Guarda la comparativa `{id, id_marco, marco, creada, experimentos}`. `experimentos` son
        `{id_experimento, estrategia, parametros, informe, reutilizado}`. El id depende solo del marco y de los
        ids de experimento (sin orden): lanzar lo mismo otra vez la sobrescribe con idéntico contenido.
        """
        ids = sorted(e["id_experimento"] for e in experimentos)
        identificador = "cmp-" + hashlib.sha256(json.dumps([marco.id, ids]).encode("utf-8")).hexdigest()[:10]
        comparativa = {"id": identificador, "id_marco": marco.id, "marco": marco.model_dump(),
                       "creada": datetime.now().isoformat(timespec="seconds"), "experimentos": [dict(e) for e in experimentos]}
        with self._cerrojo:
            _escribir_json(self.raiz / CARPETA_COMPARATIVAS / f"{identificador}.json", comparativa)
        return comparativa

    def leer_comparativa(self, id_comparativa: str) -> dict[str, Any]:
        if not re.fullmatch(r"cmp-[0-9a-f]{10}", id_comparativa):
            raise KeyError(id_comparativa)
        datos = _leer_json(self.raiz / CARPETA_COMPARATIVAS / f"{id_comparativa}.json")
        if datos is None:
            raise KeyError(id_comparativa)
        return datos

    def listar_comparativas(self, id_marco: str | None = None) -> list[dict[str, Any]]:
        """Comparativas (las más recientes primero), de un marco o de todos."""
        todas = [_leer_json(ruta) for ruta in (self.raiz / CARPETA_COMPARATIVAS).glob("cmp-*.json")]
        todas = [c for c in todas if c is not None and (id_marco is None or c["id_marco"] == id_marco)]
        return sorted(todas, key=lambda c: c["creada"], reverse=True)


# ---- ceteris paribus ---------------------------------------------------------------------


def diferencias_ceteris_paribus(configs: Mapping[str, Mapping[str, Any]],
                                comunes: Iterable[str] = COMUNES_CONOCIDOS) -> list[dict[str, Any]]:
    """
    En qué difieren varios experimentos del entorno compartido. `configs` es `{id_experimento: configuración}`
    (el `config.json` de cada experimento). Devuelve una lista de `{campo, tipo, valores: {id: valor}}`:
    - `tipo='entorno'`: campos de `CAMPOS_ENTORNO` con algún valor distinto.
    - `tipo='comun'`: parámetros de `comunes` que declaran al menos dos experimentos y con algún valor distinto
      (un común que una estrategia no declara no cuenta: no es aplicable, no es una diferencia).
    Los parámetros propios de cada estrategia (top_k, preset...) no se comparan: es justo lo que se quiere variar.
    Lista vacía = comparación ceteris paribus (o un solo experimento).
    """
    if len(configs) < 2:
        return []
    diferencias: list[dict[str, Any]] = []
    for campo in CAMPOS_ENTORNO:
        valores = {i: c.get(campo) for i, c in configs.items()}
        if len({valor_comparable(v) for v in valores.values()}) > 1:
            diferencias.append({"campo": campo, "tipo": "entorno", "valores": valores})
    for campo in comunes:
        valores = {i: c["parametros"][campo] for i, c in configs.items() if campo in (c.get("parametros") or {})}
        if len(valores) > 1 and len({valor_comparable(v) for v in valores.values()}) > 1:
            diferencias.append({"campo": campo, "tipo": "comun", "valores": valores})
    return diferencias
