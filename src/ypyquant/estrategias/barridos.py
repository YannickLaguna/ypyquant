"""
Barridos como entidad: una rejilla de configuraciones con identidad, estado y resultados propios.

Responsabilidad:
- `ConfiguracionBarrido` (Pydantic): configuración base + rejilla `{parametro_o_campo: [valores]}` +
  `fecha_fin_desarrollo` opcional + métrica objetivo. Si se da `fecha_fin_desarrollo`, todas las
  combinaciones se ejecutan hasta esa fecha y el tramo (fecha_fin_desarrollo, base.fecha_fin] es el
  **periodo reservado** (no se toca hasta `reservado.evaluar_reservado`). `huella()` /
  `identificador()` (`barrido-<10 hex>`): misma configuración, mismo id.
- `AlmacenBarridos`: persiste en `data/barridos/<id>/` (raíz configurable, igual que
  `AlmacenExperimentos`): `barrido.json` (configuración, combinaciones con sus ids de experimento y
  posiciones en la rejilla, estado, progreso y tareas largas) y los resultados de validación
  (`validacion.json`, `walk_forward.json`, `permutacion_*.json`, `reservado.json` + Parquet de
  series). Los experimentos viven en `AlmacenExperimentos`; el barrido solo guarda sus ids.
- `Barrido`: vista de un barrido guardado (config, combinaciones, almacenes) que consumen las
  funciones `matriz_retornos` y `superficie` y la validación.
- `matriz_retornos(barrido)`: DataFrame fechas x configuraciones (columna = id del experimento).
- `superficie(barrido)`: tabla parámetros -> métricas con la puntuación de **meseta**.

Decisión sobre NaN en la matriz de retornos (`alinear_retornos`): si las configuraciones no cubren
las mismas fechas (p. ej. `fecha_inicio` varía en la rejilla) se **recorta al tramo común** por
defecto (`relleno="recortar"`: no se inventan datos y todas las columnas se comparan sobre lo mismo);
`relleno="cero"` conserva la unión de fechas con retorno 0 los días sin posición (caja sin
rentabilidad), que penaliza a las que empiezan tarde. Con la misma `fecha_inicio` y el mismo
calendario (lo habitual) no hay NaN y ambas opciones coinciden.

Meseta: para cada combinación, media de la métrica objetivo de sus vecinos en la rejilla (± 1
posición en UN eje, el resto igual; la propia combinación no cuenta). Un máximo aislado rodeado de
valores malos (pico) puntúa bajo; una meseta amplia de buenos resultados, alto. Es la regla
práctica de «elige el centro de una zona estable, no el pico».

Se relaciona con:
- ypyquant.estrategias.experimentos (`expandir_rejilla`, configuración base) y .almacen (lee
  métricas y `equity.parquet` de los experimentos).
- ypyquant.estrategias.validacion_barridos (DSR, PBO, permutación, walk-forward sobre la matriz) y
  .reservado (periodo reservado con contador).
- app/api/rutas_barridos.py (los expone).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ypyquant.estrategias.almacen import AlmacenExperimentos, _escribir_json, _escribir_parquet, _leer_json
from ypyquant.estrategias.experimentos import (
    MAX_COMBINACIONES_REJILLA,
    ConfiguracionExperimento,
    _limpiar,
    expandir_rejilla_con_omitidas,
)

RAIZ_REPO = Path(__file__).resolve().parents[3]
RAIZ_POR_DEFECTO = RAIZ_REPO / "data" / "barridos"

# Métricas por las que se puede elegir la «mejor» combinación: todas con la convención «mayor es mejor»
# (max_drawdown es <= 0, así que el menos negativo gana). No se admiten volatilidad, turnover ni costes.
METRICAS_OBJETIVO = ("sharpe", "sortino", "calmar", "martin", "cagr", "retorno_total", "max_drawdown",
                     "information_ratio", "alfa", "tasa_acierto")
ESTADOS_TAREA = ("pendiente", "en_curso", "hecho", "error", "cancelado")


class ConfiguracionBarrido(BaseModel):
    """
    Barrido de parámetros sobre una configuración base.

    Campos:
        base: configuración de partida (sus `fecha_inicio`/`fecha_fin` delimitan TODO el periodo).
        rejilla: `{parametro_de_la_estrategia | campo_de_la_configuracion: [valores]}`; producto cartesiano.
        fecha_fin_desarrollo: si se da, las combinaciones se ejecutan con `fecha_fin = fecha_fin_desarrollo`
            y el tramo (fecha_fin_desarrollo, base.fecha_fin] queda reservado para `evaluar_reservado`.
        metrica_objetivo: una de `METRICAS_OBJETIVO` (por defecto `sharpe`).
        etiqueta: nota libre; NO entra en la huella.
    """

    model_config = ConfigDict(extra="forbid")

    base: ConfiguracionExperimento
    rejilla: dict[str, list[Any]]
    fecha_fin_desarrollo: str | None = None
    metrica_objetivo: str = "sharpe"
    etiqueta: str = ""

    @field_validator("fecha_fin_desarrollo")
    @classmethod
    def _fecha_valida(cls, valor: str | None) -> str | None:
        return None if valor is None else pd.Timestamp(valor).strftime("%Y-%m-%d")

    @model_validator(mode="after")
    def _coherencia(self) -> ConfiguracionBarrido:
        if self.metrica_objetivo not in METRICAS_OBJETIVO:
            raise ValueError(f"metrica_objetivo '{self.metrica_objetivo}' no válida. Admitidas: {', '.join(METRICAS_OBJETIVO)}")
        if not self.rejilla or any(not isinstance(v, list) or not v for v in self.rejilla.values()):
            raise ValueError("la rejilla debe tener al menos un eje y cada eje al menos un valor")
        total = math.prod(len(v) for v in self.rejilla.values())
        if total > MAX_COMBINACIONES_REJILLA:
            raise ValueError(f"La rejilla tiene {total} combinaciones (máximo {MAX_COMBINACIONES_REJILLA})")
        if self.fecha_fin_desarrollo is not None and not (self.base.fecha_inicio < self.fecha_fin_desarrollo < self.base.fecha_fin):
            raise ValueError("fecha_fin_desarrollo debe estar estrictamente entre base.fecha_inicio y base.fecha_fin")
        return self

    def base_desarrollo(self) -> ConfiguracionExperimento:
        """Base con `fecha_fin = fecha_fin_desarrollo` (o la base tal cual si no hay periodo reservado)."""
        if self.fecha_fin_desarrollo is None:
            return self.base
        return self.base.model_copy(update={"fecha_fin": self.fecha_fin_desarrollo})

    def huella(self) -> str:
        """SHA-256 estable (sin la etiqueta): base, rejilla, fecha_fin_desarrollo y métrica objetivo."""
        canonico = json.dumps(
            {"base": self.base.huella(), "rejilla": self.rejilla, "fecha_fin_desarrollo": self.fecha_fin_desarrollo,
             "metrica_objetivo": self.metrica_objetivo},
            sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
        return hashlib.sha256(canonico.encode("utf-8")).hexdigest()

    def identificador(self) -> str:
        """`barrido-<10 primeros caracteres de la huella>`."""
        return f"barrido-{self.huella()[:10]}"

    def _expandir_tolerante(self) -> tuple[list[tuple[ConfiguracionExperimento, dict[str, Any], dict[str, int]]],
                                           list[dict[str, Any]]]:
        """Combinaciones válidas (con su posición en cada eje) y omitidas `{valores, motivo}`, en orden de producto."""
        base = self.base_desarrollo()
        claves = list(self.rejilla)
        unicas: dict[str, tuple[ConfiguracionExperimento, dict[str, Any], dict[str, int]]] = {}
        omitidas: list[dict[str, Any]] = []
        for combinacion in itertools.product(*(self.rejilla[k] for k in claves)):
            valores = dict(zip(claves, combinacion, strict=True))
            configs, omitidas_combinacion = expandir_rejilla_con_omitidas(base, {k: [v] for k, v in valores.items()})
            if not configs:
                omitidas.extend(omitidas_combinacion)
                continue
            posiciones = {k: self.rejilla[k].index(v) for k, v in valores.items()}
            unicas.setdefault(configs[0].huella(), (configs[0], valores, posiciones))
        return list(unicas.values()), omitidas

    def combinaciones(self) -> list[tuple[ConfiguracionExperimento, dict[str, Any], dict[str, int]]]:
        """
        Combinaciones únicas `(configuración normalizada, valores por eje, posición por eje)` en orden de producto.

        La posición es el índice del valor en la lista del eje (la primera aparición si se repite);
        la usa la superficie para encontrar vecinos. Las combinaciones duplicadas se descartan y las
        que no validan (p. ej. `rapida >= lenta`) se omiten dejando un hueco en la superficie
        (ver `omitidas`).
        """
        return self._expandir_tolerante()[0]

    def omitidas(self) -> list[dict[str, Any]]:
        """Combinaciones de la rejilla descartadas por no validar: `{valores, motivo}`."""
        return self._expandir_tolerante()[1]


@dataclass
class Barrido:
    """
    Vista de un barrido guardado.

    Atributos: `id`, `config` (ConfiguracionBarrido), `combinaciones` (lista de dicts `{id, etiqueta, valores,
    posiciones}` en el orden de la rejilla) y `almacen` (AlmacenBarridos, que da acceso a los experimentos).
    """

    id: str
    config: ConfiguracionBarrido
    combinaciones: list[dict[str, Any]]
    almacen: AlmacenBarridos = field(repr=False)

    @property
    def ids(self) -> list[str]:
        return [c["id"] for c in self.combinaciones]

    @property
    def etiquetas(self) -> dict[str, str]:
        """Mapeo id de experimento -> etiqueta legible (`param=valor, ...`): las columnas de la matriz son los ids."""
        return {c["id"]: c["etiqueta"] for c in self.combinaciones}

    @property
    def ejes(self) -> dict[str, list[Any]]:
        return self.config.rejilla

    def experimentos_hechos(self) -> list[str]:
        """Ids de las combinaciones cuyo experimento terminó bien (en el orden de la rejilla)."""
        return [i for i in self.ids if self.almacen.experimentos.estado_de(i) == "hecho"]


# ----------------------------------------------------------------------------
# Almacén
# ----------------------------------------------------------------------------


class AlmacenBarridos:
    """
    Persistencia de barridos en `data/barridos/<id>/`.

    Args:
        raiz: carpeta del almacén (por defecto `data/barridos/` del repo). Su carpeta padre guarda
            `reservado.json` (contador de uso del periodo reservado, común a todos los barridos).
        experimentos: `AlmacenExperimentos` donde viven los backtests de las combinaciones.
    """

    def __init__(self, raiz: Path | str | None = None, experimentos: AlmacenExperimentos | None = None):
        self.raiz = Path(raiz) if raiz else RAIZ_POR_DEFECTO
        self.raiz.mkdir(parents=True, exist_ok=True)
        self.experimentos = experimentos or AlmacenExperimentos()
        self.ruta_reservado = self.raiz.parent / "reservado.json"
        self._cerrojo = threading.RLock()

    def _carpeta(self, id_barrido: str) -> Path:
        if not id_barrido or "/" in id_barrido or "\\" in id_barrido or id_barrido.startswith("."):
            raise ValueError(f"Identificador de barrido inválido: {id_barrido!r}")
        return self.raiz / id_barrido

    # ---- creación y borrado ----------------------------------------------------

    def crear(self, config: ConfiguracionBarrido, forzar: bool = False) -> tuple[str, bool]:
        """
        Registra el barrido (sin ejecutar nada) y devuelve `(id, creado)`.

        Si ya existe y no se fuerza se conserva con sus resultados (`creado=False`); con `forzar`
        se reinicia (se borran sus resultados de validación).
        """
        id_barrido = config.identificador()
        carpeta = self._carpeta(id_barrido)
        with self._cerrojo:
            if (carpeta / "barrido.json").exists() and not forzar:
                return id_barrido, False
            if carpeta.exists():
                shutil.rmtree(carpeta)
            carpeta.mkdir(parents=True)
            validas, omitidas = config._expandir_tolerante()
            combinaciones = [{"id": c.identificador(), "etiqueta": c.etiqueta, "valores": valores,
                              "posiciones": posiciones} for c, valores, posiciones in validas]
            ahora = datetime.now().isoformat()
            _escribir_json(carpeta / "barrido.json", {
                "id": id_barrido, "config": config.model_dump(), "combinaciones": combinaciones, "omitidas": omitidas,
                "estado": "pendiente", "progreso": {"total": len(combinaciones), "hechos": 0, "errores": 0},
                "tareas": {}, "creado": ahora, "actualizado": ahora,
            })
        return id_barrido, True

    def borrar(self, id_barrido: str, con_experimentos: bool = False) -> bool:
        """Elimina el barrido (y, si se pide, los experimentos de sus combinaciones). False si no existía."""
        carpeta = self._carpeta(id_barrido)
        with self._cerrojo:
            datos = _leer_json(carpeta / "barrido.json")
            if datos is None:
                return False
            if con_experimentos:
                for combinacion in datos["combinaciones"]:
                    self.experimentos.borrar(combinacion["id"])
            shutil.rmtree(carpeta)
        return True

    def existe(self, id_barrido: str) -> bool:
        return (self._carpeta(id_barrido) / "barrido.json").exists()

    # ---- lectura ---------------------------------------------------------------

    def cargar(self, id_barrido: str) -> Barrido:
        """Vista `Barrido` del barrido guardado (KeyError si no existe)."""
        datos = _leer_json(self._carpeta(id_barrido) / "barrido.json")
        if datos is None:
            raise KeyError(id_barrido)
        return Barrido(id_barrido, ConfiguracionBarrido(**datos["config"]), datos["combinaciones"], self)

    def estado_experimentos(self, barrido: Barrido) -> dict[str, Any]:
        """
        Estado agregado a partir de los experimentos: `pendiente` (ninguno ha empezado), `en_curso` (alguno
        pendiente o en curso), `hecho` (todos bien) o `error` (terminó con algún fallo), y el progreso.
        """
        cuentas = {"pendiente": 0, "en_curso": 0, "hecho": 0, "error": 0}
        for id_experimento in barrido.ids:
            cuentas[self.experimentos.estado_de(id_experimento) or "error"] += 1  # experimento borrado = fallo
        total = len(barrido.ids)
        if cuentas["pendiente"] + cuentas["en_curso"] == 0:
            estado = "error" if cuentas["error"] else "hecho"
        elif cuentas["pendiente"] == total:
            estado = "pendiente"
        else:
            estado = "en_curso"
        return {"estado": estado, "progreso": {"total": total, "hechos": cuentas["hecho"], "errores": cuentas["error"],
                                              "en_curso": cuentas["en_curso"], "pendientes": cuentas["pendiente"]}}

    def leer(self, id_barrido: str) -> dict[str, Any]:
        """
        `barrido.json` con el estado y el progreso al día (recalculados desde los experimentos) y los
        resultados de validación disponibles (nombres de archivo sin extensión).
        """
        carpeta = self._carpeta(id_barrido)
        with self._cerrojo:
            datos = _leer_json(carpeta / "barrido.json")
            if datos is None:
                raise KeyError(id_barrido)
            nuevo = self.estado_experimentos(self.cargar(id_barrido))
            if nuevo["estado"] != datos.get("estado") or nuevo["progreso"] != datos.get("progreso"):
                datos.update(nuevo, actualizado=datetime.now().isoformat())
                _escribir_json(carpeta / "barrido.json", datos)
        datos["resultados"] = sorted(p.stem for p in carpeta.glob("*.json") if p.name != "barrido.json")
        return datos

    def listar(self) -> list[dict[str, Any]]:
        """Resúmenes (más recientes primero): id, etiqueta, estrategia, estado, progreso, métrica objetivo..."""
        resumenes = []
        for carpeta in self.raiz.iterdir():
            if not (carpeta / "barrido.json").exists():
                continue
            datos = self.leer(carpeta.name)
            config = datos["config"]
            resumenes.append({
                "id": datos["id"], "etiqueta": config.get("etiqueta", ""), "estrategia": config["base"]["estrategia"],
                "universo": config["base"]["universo"], "ejes": list(config["rejilla"]),
                "n_combinaciones": len(datos["combinaciones"]), "metrica_objetivo": config["metrica_objetivo"],
                "fecha_fin_desarrollo": config.get("fecha_fin_desarrollo"), "estado": datos["estado"],
                "progreso": datos["progreso"], "resultados": datos["resultados"], "creado": datos["creado"],
            })
        return sorted(resumenes, key=lambda r: r["creado"] or "", reverse=True)

    # ---- resultados de validación ----------------------------------------------

    def guardar_resultado(self, id_barrido: str, nombre: str, contenido: Any) -> None:
        """Guarda `<nombre>.json` en la carpeta del barrido (escritura atómica)."""
        _escribir_json(self._carpeta(id_barrido) / f"{nombre}.json", contenido)

    def leer_resultado(self, id_barrido: str, nombre: str) -> Any:
        """Contenido de `<nombre>.json` (None si no existe)."""
        return _leer_json(self._carpeta(id_barrido) / f"{nombre}.json")

    def guardar_parquet(self, id_barrido: str, nombre: str, tabla: pd.DataFrame) -> None:
        _escribir_parquet(tabla, self._carpeta(id_barrido) / f"{nombre}.parquet", index=True)

    # ---- tareas largas (permutación) -------------------------------------------

    def actualizar_tarea(self, id_barrido: str, nombre: str, **campos: Any) -> dict[str, Any]:
        """Crea o actualiza `tareas[nombre]` en `barrido.json` (estado, progreso, error, parámetros...)."""
        if "estado" in campos and campos["estado"] not in ESTADOS_TAREA:
            raise ValueError(f"Estado de tarea inválido: {campos['estado']}")
        carpeta = self._carpeta(id_barrido)
        with self._cerrojo:
            datos = _leer_json(carpeta / "barrido.json")
            if datos is None:
                raise KeyError(id_barrido)
            tarea = datos["tareas"].setdefault(nombre, {})
            tarea.update(campos, actualizado=datetime.now().isoformat())
            _escribir_json(carpeta / "barrido.json", datos)
            return tarea

    def tarea(self, id_barrido: str, nombre: str) -> dict[str, Any] | None:
        datos = _leer_json(self._carpeta(id_barrido) / "barrido.json")
        return (datos or {}).get("tareas", {}).get(nombre)

    def marcar_interrumpidos(self) -> int:
        """Pasa a `error` las tareas que quedaron `en_curso`/`pendiente` (el proceso anterior murió)."""
        cuenta = 0
        for carpeta in self.raiz.iterdir():
            datos = _leer_json(carpeta / "barrido.json") if carpeta.is_dir() else None
            for nombre, tarea in (datos or {}).get("tareas", {}).items():
                if tarea.get("estado") in ("pendiente", "en_curso"):
                    self.actualizar_tarea(carpeta.name, nombre, estado="error",
                                          error="Interrumpida: el servidor se reinició durante la ejecución")
                    cuenta += 1
        return cuenta


# ----------------------------------------------------------------------------
# Matriz de retornos
# ----------------------------------------------------------------------------


def alinear_retornos(series: dict[str, pd.Series], relleno: Literal["recortar", "cero"] = "recortar") -> pd.DataFrame:
    """
    DataFrame fechas x columnas a partir de series de retornos diarios (ver la decisión de NaN en la cabecera).

    `relleno="recortar"`: solo las fechas donde TODAS las series tienen dato (tramo común);
    `relleno="cero"`: unión de fechas con 0 donde falta el dato.
    """
    if relleno not in ("recortar", "cero"):
        raise ValueError("relleno debe ser 'recortar' o 'cero'")
    matriz = pd.concat(series, axis=1).sort_index()
    matriz = matriz.dropna(how="any") if relleno == "recortar" else matriz.fillna(0.0)
    matriz.index.name = "fecha"
    return matriz.astype(float)


def serie_retornos(equity: pd.DataFrame) -> pd.Series:
    """Serie de retornos diarios indexada por fecha desde el DataFrame de equity de un experimento."""
    return equity.set_index(pd.DatetimeIndex(pd.to_datetime(equity["fecha"])))["retorno"].astype(float)


def matriz_retornos(barrido: Barrido, relleno: Literal["recortar", "cero"] = "recortar") -> pd.DataFrame:
    """
    Matriz de retornos diarios del barrido: índice = fechas, columnas = ids de los experimentos terminados.

    Las etiquetas legibles están en `barrido.etiquetas` (id -> etiqueta). Los experimentos que no
    terminaron bien se omiten. Lanza `ValueError` si no hay ninguno. Ver `alinear_retornos` para NaN.
    """
    series = {}
    for id_experimento in barrido.experimentos_hechos():
        equity = barrido.almacen.experimentos.leer_equity(id_experimento)
        if equity is not None:
            series[id_experimento] = serie_retornos(equity)
    if not series:
        raise ValueError("El barrido no tiene ningún experimento terminado")
    return alinear_retornos(series, relleno)


# ----------------------------------------------------------------------------
# Superficie y meseta
# ----------------------------------------------------------------------------


def _vecinos(posiciones: dict[str, int], por_posicion: dict[tuple, float], claves: list[str]) -> list[float]:
    """Valores de la métrica objetivo en las combinaciones a ±1 posición en un solo eje (las que existan)."""
    valores = []
    actual = tuple(posiciones[k] for k in claves)
    for i in range(len(claves)):
        for delta in (-1, 1):
            vecino = actual[:i] + (actual[i] + delta,) + actual[i + 1:]
            valor = por_posicion.get(vecino)
            if valor is not None and math.isfinite(valor):
                valores.append(valor)
    return valores


def superficie_registros(barrido: Barrido) -> list[dict[str, Any]]:
    """
    Una fila por combinación terminada: `{id, etiqueta, valores, posiciones, metricas, objetivo, meseta, n_vecinos}`.

    `objetivo` es la métrica objetivo del barrido; `meseta` la media de ese objetivo en sus vecinos
    (None sin vecinos válidos) y `n_vecinos` cuántos entran. `metricas` son las numéricas de nivel
    superior del experimento. Es el formato que sirve la API.
    """
    claves = list(barrido.ejes)
    objetivo = barrido.config.metrica_objetivo
    filas = []
    for combinacion in barrido.combinaciones:
        metricas = barrido.almacen.experimentos.leer_metricas(combinacion["id"])
        if metricas is None or barrido.almacen.experimentos.estado_de(combinacion["id"]) != "hecho":
            continue
        planas = {k: v for k, v in metricas.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        valor = metricas.get(objetivo)
        filas.append({"id": combinacion["id"], "etiqueta": combinacion["etiqueta"], "valores": combinacion["valores"],
                      "posiciones": combinacion["posiciones"], "metricas": planas,
                      "objetivo": valor if isinstance(valor, (int, float)) else None})
    por_posicion = {tuple(f["posiciones"][k] for k in claves): f["objetivo"] for f in filas if f["objetivo"] is not None}
    for fila in filas:
        vecinos = _vecinos(fila["posiciones"], por_posicion, claves)
        fila["meseta"] = sum(vecinos) / len(vecinos) if vecinos else None
        fila["n_vecinos"] = len(vecinos)
    return _limpiar(filas)


def superficie(barrido: Barrido) -> pd.DataFrame:
    """
    Tabla parámetros -> métricas con la meseta, lista para un mapa de calor.

    Columnas: `id`, `etiqueta`, una `eje_<parametro>` por eje de la rejilla, `objetivo`, `meseta`,
    `n_vecinos` y todas las métricas numéricas del experimento (`sharpe`, `cagr`, ...). Solo las
    combinaciones terminadas. Ver `superficie_registros` para el formato anidado.
    """
    filas = []
    for r in superficie_registros(barrido):
        fila = {"id": r["id"], "etiqueta": r["etiqueta"], **{f"eje_{k}": v for k, v in r["valores"].items()},
                "objetivo": r["objetivo"], "meseta": r["meseta"], "n_vecinos": r["n_vecinos"]}
        fila.update({k: v for k, v in r["metricas"].items() if k not in fila})
        filas.append(fila)
    return pd.DataFrame(filas)
