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
  sale de los experimentos: relanzar lo mismo la reutiliza. Es un adaptador de dominio sobre `AlmacenDocumentos` (los
  marcos) y `ColeccionJson` (las comparativas, archivos planos); `comparativas_de_configuracion` las consulta por
  configuración con un índice en memoria.
- `marco_desde_configuracion(configuracion, inicio, fin)`, `resolver_fechas` (la regla ÚNICA de fechas por defecto de
  backtest y comparativa: explícitas > marco enlazado > experimento de origen > 5 años hasta el último día hábil) y
  `fechas_por_defecto(...)`: funciones puras que convierten
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
  y ypyquant.almacen_documentos (E/S JSON atómica); .configuraciones (una configuración puede apuntar a un marco con
  `id_marco`).
- app/api/rutas_marcos.py (las expone) y app/web (páginas Marcos y Comparar).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from ypyquant.almacen_documentos import AlmacenDocumentos, ColeccionJson, id_slug
from ypyquant.estrategias.configuraciones import crear_slug
from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    es_universo_pit,
    resolver_universo,
)
from ypyquant.estrategias.receta import CAMPOS_ENTORNO as CAMPOS_ENTORNO_RECETA
from ypyquant.estrategias.receta import Entorno, Frecuencia, Receta, valor_comparable
from ypyquant.estrategias.registro import obtener_estrategia
from ypyquant.raiz_datos import RaizDatos

if TYPE_CHECKING:  # solo para anotar: configuraciones.py ya importa este módulo de forma indirecta
    from ypyquant.estrategias.configuraciones import ConfiguracionGuardada

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


def _ultimo_dia_habil(hoy: date) -> date:
    """Último día hábil (lunes a viernes) ANTERIOR a `hoy`: ayer, o el viernes si ayer fue fin de semana."""
    dia = hoy - timedelta(days=1)
    while dia.weekday() >= 5:
        dia -= timedelta(days=1)
    return dia


def resolver_fechas(configuracion: ConfiguracionGuardada, marco: MarcoComparacion | None,
                    fecha_inicio: str | None = None, fecha_fin: str | None = None,
                    hoy: date | None = None) -> tuple[str, str]:
    """
    Fechas `(inicio, fin)` con que se lanza una configuración guardada (backtest o comparativa): UNA sola regla.

    Cada fecha sale, por orden de prioridad, de: 1) la explícita; 2) el marco enlazado; 3) el experimento de origen
    (`origen.fechas`); 4) los últimos `ANIOS_POR_DEFECTO` años hasta el último día hábil anterior a `hoy`. El
    par de las fuentes 2 y 3 se toma de una sola (la primera disponible); si solo falta el inicio y hay un fin
    explícito sin marco ni origen, el inicio son `ANIOS_POR_DEFECTO` años antes de ese fin.
    """
    if marco is not None:
        base: tuple[str, str] | None = (marco.fecha_inicio, marco.fecha_fin)
    elif configuracion.origen.fechas and len(configuracion.origen.fechas) == 2:
        base = (configuracion.origen.fechas[0], configuracion.origen.fechas[1])
    else:
        base = None
    fin = fecha_fin or (base[1] if base else _ultimo_dia_habil(hoy or date.today()).isoformat())
    inicio = fecha_inicio or (base[0] if base else (date.fromisoformat(fin) - timedelta(days=365 * ANIOS_POR_DEFECTO)).isoformat())
    return inicio, fin


def fechas_por_defecto(configuracion: ConfiguracionGuardada, marco: MarcoComparacion | None, hoy: date | None = None) -> tuple[str, str]:
    """Fechas sugeridas sin explícitas: `resolver_fechas` (marco enlazado > origen > 5 años hasta el último día hábil)."""
    return resolver_fechas(configuracion, marco, hoy=hoy)


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
    """
    Persistencia de marcos y comparativas (ver cabecera del módulo). Escritura atómica por archivo.

    `self.documentos` guarda los marcos (`<raiz>/<id>/marco.json`) y `self.comparativas` las comparativas
    (`<raiz>/_comparativas/cmp-*.json`). El índice `{id_configuracion: ids de comparativa}` es **en memoria**: se construye
    en la primera consulta (o con `indexar_comparativas`) y se mantiene en `registrar_comparativa`; las comparativas que
    otro proceso añada después no se ven hasta reindexar (las consultas descartan las que ya no apuntan a la configuración).
    """

    def __init__(self, raiz: Path | str | None = None):
        self.documentos = AlmacenDocumentos(raiz or RaizDatos().marcos, nombre="marco", validar_id=id_slug, marca="marco")
        self.raiz = self.documentos.raiz
        self.comparativas = ColeccionJson(self.raiz / CARPETA_COMPARATIVAS, prefijo="cmp-")
        self._cerrojo = threading.Lock()
        self._por_configuracion: dict[str, set[str]] | None = None

    def guardar(self, marco: MarcoComparacion) -> MarcoComparacion:
        """Crea el marco. `MarcoExistente` si el id ya existe."""
        with self._cerrojo:
            if self.documentos.existe(marco.id):
                raise MarcoExistente(f"Ya existe el marco '{marco.id}'")
            self.documentos.crear(marco.id)
            self.documentos.escribir_json(marco.id, "marco", marco.model_dump())
        return marco

    def actualizar(self, marco: MarcoComparacion) -> MarcoComparacion:
        """Reemplaza un marco existente (`KeyError` si no existe). Los experimentos ya lanzados no cambian."""
        with self._cerrojo:
            if not self.documentos.existe(marco.id):
                raise KeyError(marco.id)
            anterior = MarcoComparacion(**self.documentos.leer_json(marco.id, "marco"))
            cambios: dict = {"creado": anterior.creado}
            # El formulario de marcos no envía el vínculo con la configuración de origen: si falta, se conserva.
            if marco.id_configuracion is None:
                cambios |= {
                    "id_configuracion": anterior.id_configuracion,
                    "version_configuracion": anterior.version_configuracion,
                }
            marco = marco.model_copy(update=cambios)
            self.documentos.escribir_json(marco.id, "marco", marco.model_dump())
        return marco

    def leer(self, id_marco: str) -> MarcoComparacion:
        datos = self.documentos.leer_json(id_marco, "marco")
        if datos is None:
            raise KeyError(id_marco)
        return MarcoComparacion(**datos)

    def listar(self) -> list[MarcoComparacion]:
        """Todos los marcos, los más recientes primero."""
        marcos = []
        for id_marco in self.documentos.ids():
            datos = self.documentos.leer_json(id_marco, "marco")
            if datos is not None:
                marcos.append(MarcoComparacion(**datos))
        return sorted(marcos, key=lambda m: m.creado, reverse=True)

    def borrar(self, id_marco: str) -> bool:
        """Borra el marco (las comparativas ya lanzadas se conservan con su copia del marco)."""
        with self._cerrojo:
            return self.documentos.borrar(id_marco)

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
            self.comparativas.escribir(identificador, comparativa)
            if self._por_configuracion is not None:
                for ids in self._por_configuracion.values():
                    ids.discard(identificador)  # el mismo id puede reasignarse si el marco cambió de configuración
                if marco.id_configuracion:
                    self._por_configuracion.setdefault(marco.id_configuracion, set()).add(identificador)
        return comparativa

    def leer_comparativa(self, id_comparativa: str) -> dict[str, Any]:
        if not re.fullmatch(r"cmp-[0-9a-f]{10}", id_comparativa):
            raise KeyError(id_comparativa)
        datos = self.comparativas.leer(id_comparativa)
        if datos is None:
            raise KeyError(id_comparativa)
        return datos

    def listar_comparativas(self, id_marco: str | None = None) -> list[dict[str, Any]]:
        """Comparativas (las más recientes primero), de un marco o de todos."""
        todas = [c for c in self.comparativas.listar() if id_marco is None or c["id_marco"] == id_marco]
        return sorted(todas, key=lambda c: c["creada"], reverse=True)

    def indexar_comparativas(self) -> None:
        """(Re)construye el índice por configuración leyendo todas las comparativas (una vez al arrancar basta)."""
        indice: dict[str, set[str]] = {}
        for comparativa in self.comparativas.listar():
            id_configuracion = comparativa["marco"].get("id_configuracion")
            if id_configuracion:
                indice.setdefault(id_configuracion, set()).add(comparativa["id"])
        with self._cerrojo:
            self._por_configuracion = indice

    def comparativas_de_configuracion(self, id_configuracion: str) -> list[dict[str, Any]]:
        """
        Comparativas (las más recientes primero) lanzadas con un marco creado desde esa configuración
        (`marco.id_configuracion`), sin recorrer todas: usa el índice en memoria y solo lee esas comparativas.
        """
        if self._por_configuracion is None:
            self.indexar_comparativas()
        ids = sorted(self._por_configuracion.get(id_configuracion, ()))
        leidas = [c for i in ids if (c := self.comparativas.leer(i)) is not None]
        propias = [c for c in leidas if c["marco"].get("id_configuracion") == id_configuracion]
        return sorted(propias, key=lambda c: c["creada"], reverse=True)


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
