"""
membresias_sp500.py — Mantiene verificado y al día `universos/sp500_membresias.json` desde fuentes públicas gratuitas.

Responsabilidad:
- ADAPTADORES por fuente (costura: una clase por fuente, todas con la misma interfaz; el adaptador `Sintetico*` sirve
  a los tests): `AdaptadorWikipediaActual` (tabla «S&P 500 component stocks»), `AdaptadorWikipediaCambios` (tabla de
  cambios históricos con fecha efectiva, motivo y tickers) y `AdaptadorSpdrSpy` (holdings diarios del ETF SPY, xlsx).
  Descargan con `descargar_con_cache` (User-Agent genérico, una petición por fuente y comprobación, caché local).
- RECONCILIADOR (`reconciliar`): cambios propuestos = diferencia entre la lista de la fuente PRIMARIA y los vigentes del
  fichero. Un cambio se APLICA solo si la fuente de VERIFICACIÓN lo confirma; si discrepan, queda «pendiente de
  revisar» y no se toca el fichero. La fecha efectiva sale de la tabla de cambios; si falta, es la primera fecha
  observada (hoy) y el cambio se marca `fecha_aproximada`.
- ACTUALIZACIÓN (`actualizar`): aplica los cambios confirmados con escritura atómica, guarda copia de la versión anterior
  (`sp500_membresias.json.anterior`), añade cada cambio a `sp500_membresias_cambios.jsonl` (con fuente y fecha de
  comprobación) y deja el estado de la última comprobación en `sp500_membresias_estado.json` (lo consulta
  `GET /api/universos/membresias/estado`). `ServicioMembresias` es la fachada con reloj y descarga inyectables que
  usa el planificador de la API (como máximo una vez al día, tras el cierre de Nueva York).
- CONTRASTE CON LA FECHA DEL DATO DE SPY (`ObservacionSpy`, `_resolver_con_spy`): la fecha de la tabla de Wikipedia puede ser
  la del anuncio. Cada comprobación guarda la observación de SPY (fecha de holdings + símbolos) en
  `sp500_membresias_observaciones_spy.json` (últimas `MAX_OBSERVACIONES_SPY`). Un cambio solo se aplica cuando SPY lo
  refleja y su fecha efectiva es la PRIMERA fecha de holdings de SPY que lo refleja (con aviso y `fecha_wikipedia` si
  difiere de la tabla). Si SPY aún no lo refleja, queda pendiente aunque la fecha de Wikipedia ya haya pasado (posible
  anuncio) y se reintenta en las siguientes comprobaciones. Sin transición observada (primera observación ya lo refleja)
  se acepta la fecha de Wikipedia si no es posterior al dato de SPY, con aviso de que no se pudo contrastar.
- CLI: `python -m ypyquant.data.membresias_sp500 actualizar [--simular]` y `... estado`.

Límites conocidos: un cambio de ticker (BK -> BNY) aparece como baja + alta del símbolo nuevo; si el valor ya figura
con el símbolo nuevo (convención del fichero) solo se cierra el antiguo. La fecha de la tabla de Wikipedia puede ser la
del anuncio y no la efectiva (caso OKTA, 2026-10-30 frente a 2026-11-04): lo contrasta la regla de SPY descrita arriba. Límite:
la fecha de holdings de SPY puede ir un día hábil antes de la efectiva si el índice rebalancea al cierre previo.

Se relaciona con: ypyquant.data.universos (lee el fichero; `avisos_membresias` usa la fecha de comprobación de aquí),
ypyquant.almacen_documentos (`escribir_json_atomico`), ypyquant.paper.planificador (lo invoca a diario),
app/api/rutas_universos.py (estado), app/web (tarjeta «Membresías del S&P 500»), scripts/exportar_membresias_sp500.py
(origen histórico, ya no es la fuente de verdad) y docs/planes/2026-11-22-membresias-sp500.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree

from lxml import html as lxml_html

from ypyquant.almacen_documentos import escribir_json_atomico
from ypyquant.data.universos import CARPETA_UNIVERSOS, RUTA_MEMBRESIAS, a_simbolo_canonico

logger = logging.getLogger(__name__)

RUTA_HISTORIAL = CARPETA_UNIVERSOS / "sp500_membresias_cambios.jsonl"
RUTA_ESTADO = CARPETA_UNIVERSOS / "sp500_membresias_estado.json"
RUTA_PENDIENTES = CARPETA_UNIVERSOS / "sp500_membresias_pendientes.json"
RUTA_OBSERVACIONES_SPY = CARPETA_UNIVERSOS / "sp500_membresias_observaciones_spy.json"
MAX_OBSERVACIONES_SPY = 40  # holdings diarios guardados para fijar la primera fecha en que SPY refleja un cambio
CARPETA_CACHE = CARPETA_UNIVERSOS / ".cache_fuentes"
URL_WIKIPEDIA_ACTUAL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
URL_WIKIPEDIA_CAMBIOS = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
URL_SPDR_SPY = "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
USER_AGENT = "ypyquant-membresias/1.0 (verificacion de datos publicos del S&P 500)"
TTL_CACHE_S = 6 * 3600
MAX_BYTES = 10_000_000
MIN_VIGENTES_FUENTE, MAX_VIGENTES_FUENTE = 480, 520  # una lista fuera de este rango se considera rota
TOPE_CAMBIOS_POR_COMPROBACION = 15  # más cambios de golpe que esto indican una fuente rota: se piden a revisión
REINTENTO_TRAS_ERROR_S = 2 * 3600
HORA_COMPROBACION_NY = (16, 30)  # tras el cierre de Nueva York
VARIABLE_ACTIVAR = "YPYQUANT_MEMBRESIAS_AUTO"  # "0" desactiva la comprobación automática de la API

_MESES = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"], 1)}
_SIMBOLO_VALIDO = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")


class ErrorFuente(RuntimeError):
    """La fuente no devolvió lo esperado (descarga fallida, formato cambiado, tabla ausente)."""


# ----------------------------------------------------------------------------
# Tipos
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ListaVigente:
    """Símbolos (canónicos) que una fuente da como miembros actuales, con su fecha de dato y lo que ignoró (efectivo, etc.)."""

    fuente: str
    simbolos: frozenset[str]
    fecha_dato: str | None = None
    ignorados: tuple[str, ...] = ()


@dataclass(frozen=True)
class CambioPublicado:
    """Una fila de la tabla de cambios: fecha efectiva publicada, alta, baja (símbolos canónicos o None) y motivo."""

    fecha: str
    alta: str | None
    baja: str | None
    motivo: str = ""


@dataclass(frozen=True)
class ObservacionSpy:
    """Holdings de SPY en una fecha de dato: símbolos canónicos que figuraban ese día (sirve para fijar fechas efectivas)."""

    fecha: str
    simbolos: frozenset[str]


@dataclass
class CambioPropuesto:
    """Un alta o una baja que sale de comparar la fuente primaria con el fichero, con su confirmación y su fecha."""

    tipo: str  # "alta" | "baja"
    simbolo: str
    fecha: str
    fecha_aproximada: bool
    fuente: str
    motivo: str = ""
    confirmado_por: str | None = None
    motivo_pendiente: str | None = None
    fecha_wikipedia: str | None = None  # fecha de la tabla de Wikipedia si se descartó a favor de la de SPY

    def a_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


@dataclass
class Reconciliacion:
    """Resultado de `reconciliar`: lo que se puede aplicar, lo pendiente de revisar y los avisos."""

    aplicables: list[CambioPropuesto] = field(default_factory=list)
    pendientes: list[CambioPropuesto] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


class FuenteLista(Protocol):
    """Interfaz de un adaptador de lista actual: `obtener()` devuelve la `ListaVigente` de la fuente."""

    nombre: str

    def obtener(self) -> ListaVigente: ...


class FuenteCambios(Protocol):
    """Interfaz de un adaptador de tabla de cambios: `obtener()` devuelve los `CambioPublicado`, el más reciente primero."""

    nombre: str

    def obtener(self) -> list[CambioPublicado]: ...


Descargador = Callable[[str], bytes]


# ----------------------------------------------------------------------------
# Descarga
# ----------------------------------------------------------------------------


def descargar_con_cache(url: str, carpeta_cache: Path = CARPETA_CACHE, ttl_s: float = TTL_CACHE_S,
                        ahora: Callable[[], float] = time.time) -> bytes:
    """Descarga `url` (User-Agent genérico, tope de tamaño) y la guarda en caché; reutiliza la caché hasta `ttl_s`."""
    carpeta_cache.mkdir(parents=True, exist_ok=True)
    archivo = carpeta_cache / (re.sub(r"[^A-Za-z0-9]+", "_", url)[-120:] + ".bin")
    if archivo.is_file() and ahora() - archivo.stat().st_mtime < ttl_s:
        return archivo.read_bytes()
    peticion = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(peticion, timeout=60) as respuesta:  # noqa: S310 - URLs fijas https
            contenido = respuesta.read(MAX_BYTES + 1)
    except OSError as fallo:
        raise ErrorFuente(f"no se pudo descargar {url}: {fallo}") from fallo
    if len(contenido) > MAX_BYTES:
        raise ErrorFuente(f"{url} supera {MAX_BYTES} bytes")
    archivo.write_bytes(contenido)
    return contenido


# ----------------------------------------------------------------------------
# Adaptadores
# ----------------------------------------------------------------------------


def _tabla(contenido: bytes, id_tabla: str):
    arbol = lxml_html.fromstring(contenido)
    tablas = arbol.xpath(f'//table[@id="{id_tabla}"]')
    if not tablas:
        raise ErrorFuente(f"no hay la tabla #{id_tabla} (¿cambió el formato de la página?)")
    return tablas[0]


def _filas_de_datos(tabla) -> list[list[str]]:
    return [[celda.text_content().strip() for celda in fila.xpath("./td")]
            for fila in tabla.xpath(".//tr") if not fila.xpath("./th") and fila.xpath("./td")]


def _fecha_inglesa(texto: str) -> str:
    """`'October 30, 2026'` / `'19-Nov-2026'` -> `'2026-10-30'` sin depender del locale del sistema."""
    m = re.match(r"^\s*([A-Za-z]+)\.? (\d{1,2}),? (\d{4})\s*$", texto)
    if m:
        mes, dia, anio = m.group(1), m.group(2), m.group(3)
    else:
        m = re.match(r"^\s*(\d{1,2})-([A-Za-z]+)-(\d{4})\s*$", texto)
        if not m:
            raise ErrorFuente(f"fecha no reconocida: {texto!r}")
        dia, mes, anio = m.group(1), m.group(2), m.group(3)
    numero = next((n for nombre, n in _MESES.items() if nombre.startswith(mes.lower()[:3])), None)
    if numero is None:
        raise ErrorFuente(f"mes no reconocido: {texto!r}")
    return f"{int(anio):04d}-{numero:02d}-{int(dia):02d}"


def _simbolo_o_none(texto: str) -> str | None:
    texto = texto.strip()
    return a_simbolo_canonico(texto) if texto else None


class AdaptadorWikipediaActual:
    """Lista actual del S&P 500 desde la tabla `#constituents` de «List of S&P 500 companies» (primera columna: símbolo)."""

    nombre = "wikipedia_lista"

    def __init__(self, descargar: Descargador = descargar_con_cache, url: str = URL_WIKIPEDIA_ACTUAL):
        self.descargar, self.url = descargar, url

    def obtener(self) -> ListaVigente:
        filas = _filas_de_datos(_tabla(self.descargar(self.url), "constituents"))
        simbolos = {a_simbolo_canonico(f[0]) for f in filas if f and f[0]}
        return ListaVigente(self.nombre, frozenset(simbolos))


class AdaptadorWikipediaCambios:
    """Cambios históricos con fecha efectiva (tabla `#changes` de «Historical components of the S&P 500»), más recientes primero."""

    nombre = "wikipedia_cambios"

    def __init__(self, descargar: Descargador = descargar_con_cache, url: str = URL_WIKIPEDIA_CAMBIOS):
        self.descargar, self.url = descargar, url

    def obtener(self) -> list[CambioPublicado]:
        cambios = []
        for f in _filas_de_datos(_tabla(self.descargar(self.url), "changes")):
            if len(f) < 6:
                continue
            try:
                fecha = _fecha_inglesa(f[0])
            except ErrorFuente:
                continue
            cambios.append(CambioPublicado(fecha, _simbolo_o_none(f[1]), _simbolo_o_none(f[3]), f[5]))
        if not cambios:
            raise ErrorFuente("la tabla de cambios no tiene filas legibles")
        return sorted(cambios, key=lambda c: c.fecha, reverse=True)


_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _leer_xlsx(contenido: bytes) -> list[dict[str, str]]:
    """Filas `{columna: texto}` de la primera hoja de un xlsx con solo la biblioteca estándar (sin dependencias nuevas)."""
    try:
        z = zipfile.ZipFile(BytesIO(contenido))
        cadenas = ["".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t"))
                   for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS)]
        hoja = ElementTree.fromstring(z.read("xl/worksheets/sheet1.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as fallo:
        raise ErrorFuente(f"el archivo no es un xlsx legible: {fallo}") from fallo
    filas = []
    for fila in hoja.iter(f"{{{_NS['m']}}}row"):
        datos = {}
        for celda in fila.findall("m:c", _NS):
            valor = celda.find("m:v", _NS)
            if valor is not None and valor.text is not None:
                datos[re.match(r"[A-Z]+", celda.get("r")).group()] = (
                    cadenas[int(valor.text)] if celda.get("t") == "s" else valor.text)
        filas.append(datos)
    return filas


class AdaptadorSpdrSpy:
    """
    Holdings diarios del ETF SPY (State Street, xlsx): columna B = ticker. Ignora efectivo (`-`) y marcadores internos
    con dígitos (p. ej. `2602335D`); la fecha del dato sale de la cabecera «As of 19-Nov-2026».
    """

    nombre = "spdr_spy"

    def __init__(self, descargar: Descargador = descargar_con_cache, url: str = URL_SPDR_SPY):
        self.descargar, self.url = descargar, url

    def obtener(self) -> ListaVigente:
        filas = _leer_xlsx(self.descargar(self.url))
        fecha = next((_fecha_inglesa(f["B"].split("As of")[1]) for f in filas[:6]
                      if "B" in f and "As of" in f["B"]), None)
        indice = next((i for i, f in enumerate(filas) if f.get("B") == "Ticker"), None)
        if indice is None:
            raise ErrorFuente("no se encontró la cabecera «Ticker» del archivo de SPY")
        simbolos, ignorados = set(), []
        for f in filas[indice + 1:]:
            ticker = f.get("B", "").strip()
            if not ticker:
                continue
            if _SIMBOLO_VALIDO.match(ticker):
                simbolos.add(a_simbolo_canonico(ticker))
            else:
                ignorados.append(ticker)
        return ListaVigente(self.nombre, frozenset(simbolos), fecha, tuple(ignorados))


class SinteticoLista:
    """Adaptador de lista para tests: devuelve lo que se le da (o lanza `error`)."""

    def __init__(self, nombre: str, simbolos: Iterable[str], fecha_dato: str | None = None, error: Exception | None = None):
        self.nombre, self._simbolos, self._fecha, self._error = nombre, frozenset(simbolos), fecha_dato, error

    def obtener(self) -> ListaVigente:
        if self._error:
            raise self._error
        return ListaVigente(self.nombre, self._simbolos, self._fecha)


class SinteticoCambios:
    """Adaptador de cambios para tests."""

    nombre = "sintetico_cambios"

    def __init__(self, cambios: Iterable[CambioPublicado] = ()):
        self._cambios = list(cambios)

    def obtener(self) -> list[CambioPublicado]:
        return list(self._cambios)


# ----------------------------------------------------------------------------
# Reconciliación
# ----------------------------------------------------------------------------


def simbolos_vigentes(documento: Mapping[str, Any]) -> set[str]:
    """Símbolos con algún intervalo abierto (`date_removed` nulo) en el documento del fichero de membresías."""
    return {s for s, ivs in documento["miembros"].items() if any(i.get("date_removed") is None for i in ivs)}


def _fecha_publicada(tipo: str, simbolo: str, cambios: Iterable[CambioPublicado], hoy: str) -> str | None:
    """Fecha efectiva más reciente publicada para el alta/baja de `simbolo` que no sea futura; None si no hay."""
    fechas = [c.fecha for c in cambios if (c.alta if tipo == "alta" else c.baja) == simbolo and c.fecha <= hoy]
    return max(fechas) if fechas else None


def _resolver_con_spy(c: CambioPropuesto, verificacion: ListaVigente, observaciones: list[ObservacionSpy],
                      avisos: list[str]) -> None:
    """
    Contrasta la fecha de Wikipedia con la fecha del dato de SPY (ver cabecera). Modifica `c`: lo deja pendiente
    (`motivo_pendiente`) o fija su fecha efectiva; las notas van a `avisos`. `observaciones` ya incluye la actual y
    llega ordenada, con fechas <= la del dato de SPY.
    """
    fs, fw = verificacion.fecha_dato, (None if c.fecha_aproximada else c.fecha)
    refleja = [(c.simbolo in o.simbolos) == (c.tipo == "alta") for o in observaciones]
    if not refleja[-1]:
        c.motivo_pendiente = f"{verificacion.fuente} discrepa: {'no incluye' if c.tipo == 'alta' else 'todavía incluye'} {c.simbolo}"
        if fw and fw <= fs:
            c.motivo_pendiente += (f" (Wikipedia da {fw} y el dato de SPY es {fs}: posible fecha de anuncio, "
                                   "se reintenta en la próxima comprobación)")
        return
    inicio = len(refleja) - 1
    while inicio > 0 and refleja[inicio - 1]:
        inicio -= 1
    if inicio > 0:  # transición observada: la primera fecha de SPY que lo refleja es la efectiva
        primera = observaciones[inicio].fecha
        if fw and fw != primera:
            c.fecha_wikipedia = fw
            avisos.append(f"{c.tipo} de {c.simbolo}: Wikipedia da {fw} pero SPY lo refleja por primera vez el {primera}; "
                          f"se usa {primera}.")
        c.fecha, c.fecha_aproximada = primera, False
    elif fw and fw > fs:
        c.motivo_pendiente = (f"{verificacion.fuente} ya lo refleja (dato de {fs}) antes de la fecha de Wikipedia ({fw}); "
                              "se reintenta cuando el dato de SPY alcance esa fecha")
    elif fw:
        avisos.append(f"{c.tipo} de {c.simbolo}: sin observación previa de SPY, la fecha de Wikipedia ({fw}) no se pudo contrastar.")


def reconciliar(documento: Mapping[str, Any], primaria: ListaVigente, verificacion: ListaVigente | None,
                cambios: Iterable[CambioPublicado], hoy: str,
                tope_cambios: int = TOPE_CAMBIOS_POR_COMPROBACION,
                observaciones_spy: Iterable[ObservacionSpy] = ()) -> Reconciliacion:
    """
    Compara la lista de la fuente primaria con los vigentes del fichero y separa lo aplicable de lo pendiente.

    Aplicable = confirmado por la fuente de verificación (alta: también figura en ella; baja: ya no figura). Todo pasa a
    pendiente si la lista primaria es implausible (fuera de 480-520) o hay demasiados cambios a la vez (fuente rota).
    Si la verificación trae fecha de dato, la fecha efectiva se contrasta con `observaciones_spy` (`_resolver_con_spy`).
    """
    cambios = list(cambios)
    resultado = Reconciliacion()
    vigentes = simbolos_vigentes(documento)
    observaciones = []
    if verificacion is not None and verificacion.fecha_dato:
        actual = ObservacionSpy(verificacion.fecha_dato, verificacion.simbolos)
        observaciones = sorted([o for o in observaciones_spy if o.fecha < actual.fecha] + [actual], key=lambda o: o.fecha)
    propuestos = []
    for tipo, simbolos in (("baja", sorted(vigentes - primaria.simbolos)), ("alta", sorted(primaria.simbolos - vigentes))):
        for simbolo in simbolos:
            fecha = _fecha_publicada(tipo, simbolo, cambios, hoy)
            propuestos.append(CambioPropuesto(tipo, simbolo, fecha or hoy, fecha is None, primaria.fuente))
            if fecha is None:
                resultado.avisos.append(
                    f"{tipo} de {simbolo}: sin fecha en la tabla de cambios; se usa la primera fecha observada ({hoy}) "
                    "como aproximada.")
    global_pendiente = None
    if not MIN_VIGENTES_FUENTE <= len(primaria.simbolos) <= MAX_VIGENTES_FUENTE:
        global_pendiente = f"la lista de {primaria.fuente} tiene {len(primaria.simbolos)} símbolos (se esperan 480-520)"
    elif len(propuestos) > tope_cambios:
        global_pendiente = f"{len(propuestos)} cambios a la vez (tope {tope_cambios}): fuente probablemente rota"
    for c in propuestos:
        if global_pendiente:
            c.motivo_pendiente = global_pendiente
        elif verificacion is None:
            c.motivo_pendiente = "sin fuente de verificación disponible"
        else:
            confirma = (c.simbolo in verificacion.simbolos) == (c.tipo == "alta")
            if observaciones:
                _resolver_con_spy(c, verificacion, observaciones, resultado.avisos)
                if not c.motivo_pendiente:
                    c.confirmado_por = verificacion.fuente
            elif confirma:
                c.confirmado_por = verificacion.fuente
            else:
                c.motivo_pendiente = (f"{verificacion.fuente} discrepa: "
                                      f"{'no incluye' if c.tipo == 'alta' else 'todavía incluye'} {c.simbolo}")
        (resultado.pendientes if c.motivo_pendiente else resultado.aplicables).append(c)
    if global_pendiente:
        resultado.avisos.append(global_pendiente)
    return resultado


def aplicar_cambios(documento: dict[str, Any], aplicables: Iterable[CambioPropuesto], hoy: str) -> dict[str, Any]:
    """Aplica altas (nuevo intervalo abierto) y bajas (cierra el abierto) con su procedencia; devuelve el documento (mutado)."""
    miembros = documento["miembros"]
    for c in aplicables:
        extra = {"fecha_aproximada": True} if c.fecha_aproximada else {}
        if c.tipo == "alta":
            miembros.setdefault(c.simbolo, []).append(
                {"date_added": c.fecha, "date_removed": None, "fuente_alta": c.fuente, **extra})
        else:
            abierto = next(i for i in reversed(miembros[c.simbolo]) if i.get("date_removed") is None)
            abierto["date_removed"] = c.fecha
            abierto["fuente_baja"] = c.fuente
            if extra:
                abierto["fecha_baja_aproximada"] = True
    documento["miembros"] = dict(sorted(miembros.items()))
    documento["n_tickers"] = len(documento["miembros"])
    documento["n_vigentes"] = len(simbolos_vigentes(documento))
    documento["comprobado"] = hoy
    return documento


# ----------------------------------------------------------------------------
# Actualización (E/S)
# ----------------------------------------------------------------------------


def _leer_json(ruta: Path) -> Any:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else None


def leer_estado(ruta_estado: Path = RUTA_ESTADO) -> dict[str, Any] | None:
    """Estado guardado de la última comprobación (None si nunca se ha comprobado)."""
    return _leer_json(ruta_estado)


def fecha_ultima_comprobacion(ruta_estado: Path = RUTA_ESTADO, ruta: Path = RUTA_MEMBRESIAS) -> date | None:
    """
    Fecha de la última comprobación SATISFACTORIA (sin error de fuente), del estado o, si falta, del campo
    `comprobado` del propio fichero. La usa `avisos_membresias` para no avisar de antigüedad si no hubo cambios.
    """
    candidatas = []
    estado = _leer_json(ruta_estado) or {}
    if estado.get("ultima_comprobacion_correcta"):
        candidatas.append(estado["ultima_comprobacion_correcta"][:10])
    documento = _leer_json(ruta) or {}
    if documento.get("comprobado"):
        candidatas.append(documento["comprobado"][:10])
    return date.fromisoformat(max(candidatas)) if candidatas else None


def leer_observaciones_spy(ruta: Path = RUTA_OBSERVACIONES_SPY) -> list[ObservacionSpy]:
    """Observaciones de SPY guardadas, de la más antigua a la más reciente (lista vacía si no hay o el fichero es ilegible)."""
    try:
        crudo = (_leer_json(ruta) or {}).get("observaciones", [])
        return sorted((ObservacionSpy(o["fecha"], frozenset(o["simbolos"])) for o in crudo), key=lambda o: o.fecha)
    except (ValueError, KeyError, TypeError, AttributeError):
        logger.warning("Observaciones de SPY ilegibles en %s: se ignoran", ruta)
        return []


def guardar_observaciones_spy(ruta: Path, observaciones: Iterable[ObservacionSpy]) -> None:
    """Guarda las últimas `MAX_OBSERVACIONES_SPY` observaciones (sin repetir fecha), escritura atómica."""
    por_fecha = {o.fecha: o for o in observaciones}
    ultimas = [por_fecha[f] for f in sorted(por_fecha)][-MAX_OBSERVACIONES_SPY:]
    escribir_json_atomico(ruta, {"observaciones": [{"fecha": o.fecha, "simbolos": sorted(o.simbolos)} for o in ultimas]})


def _cambios_aplicados(ruta_historial: Path) -> list[dict[str, Any]]:
    if not ruta_historial.is_file():
        return []
    return [json.loads(linea) for linea in ruta_historial.read_text(encoding="utf-8").splitlines() if linea.strip()]


def actualizar(primaria: FuenteLista, verificacion: FuenteLista | None, fuente_cambios: FuenteCambios | None,
               ruta: Path = RUTA_MEMBRESIAS, ruta_historial: Path = RUTA_HISTORIAL, ruta_estado: Path = RUTA_ESTADO,
               ruta_pendientes: Path = RUTA_PENDIENTES, simular: bool = False,
               ruta_observaciones: Path | None = None, hoy: str | None = None, ahora: Callable[[], datetime] = lambda: datetime.now().astimezone()) -> dict[str, Any]:
    """
    Una comprobación completa: descarga, reconcilia y (si no es `simular`) aplica lo confirmado.

    Devuelve el resumen `{resultado: al_dia|actualizado|pendiente|error, aplicados, pendientes, avisos, n_vigentes, ...}`
    y lo guarda como estado. Un fallo de la fuente primaria o de la de cambios deja el fichero intacto (`error`); si
    falla solo la de verificación, todo cambio queda pendiente.
    """
    ruta_observaciones = ruta_observaciones or ruta.with_name(RUTA_OBSERVACIONES_SPY.name)
    momento = ahora()
    hoy = hoy or momento.date().isoformat()
    resumen: dict[str, Any] = {"hora": momento.isoformat(timespec="seconds"), "simulacion": simular, "aplicados": [],
                               "pendientes": [], "avisos": [], "error": None}
    documento = json.loads(ruta.read_text(encoding="utf-8"))
    try:
        lista = primaria.obtener()
        try:
            lista_verificacion = verificacion.obtener() if verificacion else None
        except ErrorFuente as fallo:
            lista_verificacion = None
            resumen["avisos"].append(f"fuente de verificación no disponible: {fallo}")
        try:
            tabla = fuente_cambios.obtener() if fuente_cambios else []
        except ErrorFuente as fallo:
            tabla = []
            resumen["avisos"].append(f"tabla de cambios no disponible: {fallo}")
        previas = leer_observaciones_spy(ruta_observaciones)
        reconciliacion = reconciliar(documento, lista, lista_verificacion, tabla, hoy, observaciones_spy=previas)
    except ErrorFuente as fallo:
        resumen.update(resultado="error", error=str(fallo), n_vigentes=len(simbolos_vigentes(documento)))
        _guardar_estado(resumen, documento, ruta_estado, ruta_historial, simular, correcta=False)
        return resumen
    resumen["avisos"] += reconciliacion.avisos
    resumen["aplicados"] = [c.a_dict() for c in reconciliacion.aplicables]
    resumen["pendientes"] = [c.a_dict() for c in reconciliacion.pendientes]
    resumen["fuente_primaria"] = lista.fuente
    resumen["fuente_verificacion"] = lista_verificacion.fuente if lista_verificacion else None
    resumen["fecha_dato_verificacion"] = lista_verificacion.fecha_dato if lista_verificacion else None
    if lista_verificacion is not None and lista_verificacion.fecha_dato and not simular:
        guardar_observaciones_spy(ruta_observaciones, [*previas, ObservacionSpy(lista_verificacion.fecha_dato,
                                                                               lista_verificacion.simbolos)])
    if reconciliacion.aplicables and not simular:
        ruta_anterior = ruta.with_name(ruta.name + ".anterior")
        shutil.copyfile(ruta, ruta_anterior)
        aplicar_cambios(documento, reconciliacion.aplicables, hoy)
        escribir_json_atomico(ruta, documento)
        with ruta_historial.open("a", encoding="utf-8", newline="\n") as f:
            for c in reconciliacion.aplicables:
                f.write(json.dumps({"tipo": c.tipo, "simbolo": c.simbolo, "fecha": c.fecha,
                                    "fecha_aproximada": c.fecha_aproximada, "fuente": c.fuente,
                                    "verificado_por": c.confirmado_por, "comprobado": hoy,
                                    **({"fecha_wikipedia": c.fecha_wikipedia} if c.fecha_wikipedia else {})},
                                   ensure_ascii=False) + "\n")
    resumen["n_vigentes"] = len(simbolos_vigentes(documento))
    resumen["resultado"] = ("pendiente" if reconciliacion.pendientes else
                            "actualizado" if reconciliacion.aplicables else "al_dia")
    _guardar_estado(resumen, documento, ruta_estado, ruta_historial, simular, correcta=True)
    if not simular:
        if reconciliacion.pendientes:
            escribir_json_atomico(ruta_pendientes, {"comprobado": hoy, "pendientes": resumen["pendientes"]})
        elif ruta_pendientes.is_file():
            ruta_pendientes.unlink()
    return resumen


def _guardar_estado(resumen: dict[str, Any], documento: Mapping[str, Any], ruta_estado: Path, ruta_historial: Path,
                    simular: bool, correcta: bool) -> None:
    """Persiste el estado de la comprobación (las simulaciones no tocan el estado)."""
    if simular:
        return
    anterior = _leer_json(ruta_estado) or {}
    estado = {**anterior, "ultima_comprobacion": resumen["hora"], "resultado": resumen["resultado"],
              "error": resumen["error"], "n_vigentes": resumen["n_vigentes"], "pendientes": resumen["pendientes"],
              "avisos": resumen["avisos"]}
    if correcta:
        estado["ultima_comprobacion_correcta"] = resumen["hora"]
        estado["fuente_primaria"] = resumen.get("fuente_primaria")
        estado["fuente_verificacion"] = resumen.get("fuente_verificacion")
    escribir_json_atomico(ruta_estado, estado)


# ----------------------------------------------------------------------------
# Servicio para la API
# ----------------------------------------------------------------------------


def fuentes_por_defecto(descargar: Descargador = descargar_con_cache) -> tuple[FuenteLista, FuenteLista, FuenteCambios]:
    """(primaria, verificación, cambios): Wikipedia (lista + tabla de cambios) y SPDR SPY como verificación cruzada."""
    return AdaptadorWikipediaActual(descargar), AdaptadorSpdrSpy(descargar), AdaptadorWikipediaCambios(descargar)


def comprobacion_automatica_activa(entorno: Mapping[str, str]) -> bool:
    """False si `YPYQUANT_MEMBRESIAS_AUTO` vale 0/false/no/off."""
    return entorno.get(VARIABLE_ACTIVAR, "1").strip().lower() not in {"0", "false", "no", "off"}


class ServicioMembresias:
    """
    Fachada para el planificador y la API: decide cuándo toca comprobar (una vez por fecha de Nueva York, tras el cierre)
    y expone el estado. `ejecutar` es inyectable (tests): por defecto `actualizar` con las fuentes por defecto.
    """

    def __init__(self, ejecutar: Callable[[], dict[str, Any]] | None = None, ruta: Path = RUTA_MEMBRESIAS,
                 ruta_estado: Path = RUTA_ESTADO, ruta_historial: Path = RUTA_HISTORIAL,
                 ruta_pendientes: Path = RUTA_PENDIENTES, activo: bool = True):
        self.ruta, self.ruta_estado, self.ruta_historial, self.ruta_pendientes = ruta, ruta_estado, ruta_historial, ruta_pendientes
        self.activo = activo
        self._ejecutar = ejecutar or self._ejecutar_real

    def _ejecutar_real(self) -> dict[str, Any]:
        primaria, verificacion, cambios = fuentes_por_defecto()
        return actualizar(primaria, verificacion, cambios, self.ruta, self.ruta_historial, self.ruta_estado, self.ruta_pendientes)

    def toca_comprobar(self, ahora_ny: datetime) -> bool:
        """True si está activo, es posterior a las 16:30 de NY y aún no hay comprobación hecha en esa fecha de NY."""
        if not self.activo or (ahora_ny.hour, ahora_ny.minute) < HORA_COMPROBACION_NY:
            return False
        estado = leer_estado(self.ruta_estado) or {}
        if not estado.get("ultima_comprobacion"):
            return True
        ultima = datetime.fromisoformat(estado["ultima_comprobacion"]).astimezone(ahora_ny.tzinfo)
        if estado.get("resultado") == "error":  # fallo de red o de formato: reintenta pasadas unas horas, no al día siguiente
            return (ahora_ny - ultima).total_seconds() >= REINTENTO_TRAS_ERROR_S
        return ultima.date() < ahora_ny.date()

    def comprobar(self) -> dict[str, Any]:
        """Ejecuta una comprobación; un fallo inesperado se registra y se devuelve como `error` sin propagarse."""
        try:
            return self._ejecutar()
        except Exception as fallo:  # noqa: BLE001 - el planificador no debe caerse por esto
            logger.exception("Falló la comprobación de membresías del S&P 500")
            return {"resultado": "error", "error": f"{type(fallo).__name__}: {fallo}"}

    def estado(self, ahora: datetime | None = None) -> dict[str, Any]:
        """Estado consultable: última comprobación, nº de vigentes, cambios aplicados, pendientes y antigüedad en días."""
        documento = _leer_json(self.ruta) or {"miembros": {}}
        estado = leer_estado(self.ruta_estado) or {}
        aplicados = _cambios_aplicados(self.ruta_historial)
        correcta = fecha_ultima_comprobacion(self.ruta_estado, self.ruta)
        hoy = (ahora or datetime.now().astimezone()).date()
        pendientes = (_leer_json(self.ruta_pendientes) or {}).get("pendientes", [])
        return {"activo": self.activo, "ultima_comprobacion": estado.get("ultima_comprobacion"),
                "resultado": estado.get("resultado"), "error": estado.get("error"),
                "n_vigentes": len(simbolos_vigentes(documento)),
                "ultima_comprobacion_correcta": correcta.isoformat() if correcta else None,
                "antiguedad_dias": (hoy - correcta).days if correcta else None,
                "cambios_aplicados_total": len(aplicados), "cambios_aplicados_recientes": aplicados[-10:][::-1],
                "pendientes_de_revisar": pendientes, "avisos": estado.get("avisos", []),
                "fuentes": {"primaria": estado.get("fuente_primaria"), "verificacion": estado.get("fuente_verificacion")}}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(prog="python -m ypyquant.data.membresias_sp500",
                                         description="Mantiene al día las membresías del S&P 500.")
    sub = analizador.add_subparsers(dest="orden", required=True)
    act = sub.add_parser("actualizar", help="comprueba las fuentes y aplica los cambios confirmados")
    act.add_argument("--simular", action="store_true", help="no escribe nada: solo muestra lo que haría")
    act.add_argument("--sin-cache", action="store_true", help="ignora la caché local de descargas")
    sub.add_parser("estado", help="muestra el estado de la última comprobación")
    argumentos = analizador.parse_args(argv)
    if argumentos.orden == "estado":
        print(json.dumps(ServicioMembresias().estado(), ensure_ascii=False, indent=2))
        return 0
    descargar = (lambda url: descargar_con_cache(url, ttl_s=0)) if argumentos.sin_cache else descargar_con_cache
    primaria, verificacion, cambios = fuentes_por_defecto(descargar)
    resumen = actualizar(primaria, verificacion, cambios, simular=argumentos.simular)
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    return 1 if resumen["resultado"] in ("error", "pendiente") else 0


if __name__ == "__main__":
    sys.exit(main())
