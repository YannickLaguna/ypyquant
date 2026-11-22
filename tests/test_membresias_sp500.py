"""
test_membresias_sp500.py — Mantenimiento verificado de las membresías del S&P 500 (sin red).

Responsabilidad:
    - adaptadores contra fixtures recortados de las fuentes reales (`tests/datos/`: Wikipedia lista y cambios, SPDR SPY);
    - reconciliador: confirmado por la segunda fuente -> aplicable; discrepancia, lista implausible o demasiados
      cambios -> pendiente de revisar; fecha de la tabla o, si falta, la primera observada marcada aproximada;
    - `actualizar`: escritura atómica con copia anterior, historial `.jsonl`, estado, simulación sin efectos y fallos de fuente;
    - `ServicioMembresias.toca_comprobar` (una vez al día tras el cierre de NY, reintento tras error) y el planificador;
    - el fichero versionado: entre 495 y 510 vigentes, sin avisos de calidad y legible por `universos`;
    - los experimentos guardados con `sp500_pit` mantienen su huella (si hay `data/` real; ruta en `YPYQUANT_DATOS_REALES`).

Se relaciona con: ypyquant.data.membresias_sp500, ypyquant.data.universos, ypyquant.paper.planificador,
app/api/rutas_universos.py y universos/sp500_membresias.json.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ypyquant.data import membresias_sp500 as m
from ypyquant.data.universos import avisos_membresias, cargar_membresias_sp500, miembros_vigentes_en

DATOS = Path(__file__).parent / "datos"
NY = ZoneInfo("America/New_York")


def _bytes(nombre: str):
    return lambda _url: (DATOS / nombre).read_bytes()


def _documento(**miembros):
    """Documento mínimo: `T=[(alta, baja)]`."""
    return {"miembros": {t: [{"date_added": a, "date_removed": b} for a, b in ivs] for t, ivs in miembros.items()},
            "n_tickers": len(miembros)}


# ---- adaptadores -------------------------------------------------------------------------------


def test_wikipedia_lista_lee_la_tabla_de_constituyentes():
    lista = m.AdaptadorWikipediaActual(_bytes("wikipedia_sp500_actual.html")).obtener()
    assert len(lista.simbolos) == 30 and {"MMM", "AOS", "ABT"} <= lista.simbolos


def test_wikipedia_cambios_lee_fechas_efectivas_sin_depender_del_locale():
    cambios = m.AdaptadorWikipediaCambios(_bytes("wikipedia_sp500_cambios.html")).obtener()
    okta = next(c for c in cambios if c.alta == "OKTA")
    assert (okta.fecha, okta.baja) == ("2026-10-30", "KVUE")  # la tabla pone la fecha del anuncio (ver cabecera del módulo)
    assert next(c for c in cambios if c.alta == "VYLR").baja is None  # una escisión: solo alta
    assert cambios == sorted(cambios, key=lambda c: c.fecha, reverse=True)


def test_spdr_spy_ignora_efectivo_y_marcadores_y_lee_la_fecha_del_dato():
    lista = m.AdaptadorSpdrSpy(_bytes("spdr_spy_holdings.xlsx")).obtener()
    assert lista.fecha_dato == "2026-11-19"
    assert 495 <= len(lista.simbolos) <= 510 and {"AAPL", "BRK-B", "BF-B", "GOOG", "GOOGL"} <= lista.simbolos
    assert {"-", "2602335D"} <= set(lista.ignorados) and not {"-", "2602335D"} & lista.simbolos


def test_un_archivo_que_no_es_el_esperado_da_error_de_fuente():
    with pytest.raises(m.ErrorFuente):
        m.AdaptadorSpdrSpy(lambda _u: b"<html>bloqueado</html>").obtener()
    with pytest.raises(m.ErrorFuente):
        m.AdaptadorWikipediaCambios(lambda _u: b"<html><body>sin tabla</body></html>").obtener()


# ---- reconciliador -----------------------------------------------------------------------------

BASE = {f"S{i:03d}" for i in range(500)}


def _doc_base():
    return _documento(**{s: [("2000-01-03", None)] for s in BASE})


def _lista(nombre, simbolos):
    return m.ListaVigente(nombre, frozenset(simbolos))


def test_cambio_confirmado_por_las_dos_fuentes_es_aplicable_con_la_fecha_de_la_tabla():
    nueva = BASE - {"S000"} | {"NEW"}
    tabla = [m.CambioPublicado("2026-11-03", "NEW", "S000", "capitalización")]
    r = m.reconciliar(_doc_base(), _lista("a", nueva), _lista("b", nueva), tabla, "2026-11-05")
    assert {(c.tipo, c.simbolo, c.fecha, c.fecha_aproximada) for c in r.aplicables} == {
        ("alta", "NEW", "2026-11-03", False), ("baja", "S000", "2026-11-03", False)}
    assert not r.pendientes and all(c.confirmado_por == "b" for c in r.aplicables)


def test_si_la_verificacion_discrepa_el_cambio_queda_pendiente():
    nueva = BASE - {"S000"} | {"NEW"}
    r = m.reconciliar(_doc_base(), _lista("a", nueva), _lista("b", BASE), [], "2026-11-05")
    assert not r.aplicables and len(r.pendientes) == 2
    assert all("b discrepa" in c.motivo_pendiente for c in r.pendientes)


def test_sin_fecha_en_la_tabla_usa_la_primera_observada_como_aproximada_y_avisa():
    nueva = BASE | {"NEW"}
    r = m.reconciliar(_doc_base(), _lista("a", nueva), _lista("b", nueva), [], "2026-11-05")
    (c,) = r.aplicables
    assert (c.fecha, c.fecha_aproximada) == ("2026-11-05", True) and any("NEW" in a for a in r.avisos)


def test_una_fecha_futura_de_la_tabla_no_se_usa():
    nueva = BASE | {"NEW"}
    tabla = [m.CambioPublicado("2026-12-01", "NEW", None)]
    (c,) = m.reconciliar(_doc_base(), _lista("a", nueva), _lista("b", nueva), tabla, "2026-11-05").aplicables
    assert c.fecha_aproximada


def test_lista_primaria_implausible_o_demasiados_cambios_no_se_aplican():
    r = m.reconciliar(_doc_base(), _lista("a", list(BASE)[:100]), _lista("b", list(BASE)[:100]), [], "2026-11-05")
    assert not r.aplicables and r.pendientes
    muchos = {f"X{i}" for i in range(20)}
    r = m.reconciliar(_doc_base(), _lista("a", BASE | muchos), _lista("b", BASE | muchos), [], "2026-11-05")
    assert not r.aplicables and len(r.pendientes) == 20


def test_sin_fuente_de_verificacion_nada_se_aplica():
    r = m.reconciliar(_doc_base(), _lista("a", BASE | {"NEW"}), None, [], "2026-11-05")
    assert not r.aplicables and r.pendientes[0].motivo_pendiente == "sin fuente de verificación disponible"


def test_aplicar_cierra_el_intervalo_abierto_y_reabre_con_procedencia():
    doc = _documento(ILMN=[("2015-11-19", "2024-06-24")], KVUE=[("2023-08-25", None)])
    cambios = [m.CambioPropuesto("alta", "ILMN", "2026-09-21", False, "wiki", confirmado_por="spy"),
               m.CambioPropuesto("baja", "KVUE", "2026-11-04", True, "wiki", confirmado_por="spy")]
    m.aplicar_cambios(doc, cambios, "2026-11-22")
    assert doc["miembros"]["ILMN"][-1] == {"date_added": "2026-09-21", "date_removed": None, "fuente_alta": "wiki"}
    assert doc["miembros"]["KVUE"][0]["date_removed"] == "2026-11-04" and doc["miembros"]["KVUE"][0]["fecha_baja_aproximada"]
    assert doc["n_vigentes"] == 1 and doc["comprobado"] == "2026-11-22"


# ---- actualizar (E/S) --------------------------------------------------------------------------


def _rutas(tmp_path):
    ruta = tmp_path / "sp500_membresias.json"
    ruta.write_text(json.dumps(_doc_base()), encoding="utf-8")
    return dict(ruta=ruta, ruta_historial=tmp_path / "cambios.jsonl", ruta_estado=tmp_path / "estado.json",
                ruta_pendientes=tmp_path / "pendientes.json")


def AHORA():
    return datetime(2026, 11, 5, 17, 0, tzinfo=NY)


def test_actualizar_aplica_lo_confirmado_con_copia_anterior_historial_y_estado(tmp_path):
    rutas = _rutas(tmp_path)
    original = rutas["ruta"].read_text(encoding="utf-8")
    nueva = BASE - {"S000"} | {"NEW"}
    tabla = m.SinteticoCambios([m.CambioPublicado("2026-11-03", "NEW", "S000")])
    resumen = m.actualizar(m.SinteticoLista("a", nueva), m.SinteticoLista("b", nueva, "2026-11-04"), tabla, ahora=AHORA, **rutas)
    assert resumen["resultado"] == "actualizado" and resumen["n_vigentes"] == 500
    assert (tmp_path / "sp500_membresias.json.anterior").read_text(encoding="utf-8") == original
    doc = json.loads(rutas["ruta"].read_text(encoding="utf-8"))
    assert doc["miembros"]["S000"][0]["date_removed"] == "2026-11-03" and doc["miembros"]["NEW"][0]["date_added"] == "2026-11-03"
    historial = [json.loads(x) for x in rutas["ruta_historial"].read_text(encoding="utf-8").splitlines()]
    assert {(h["tipo"], h["simbolo"], h["verificado_por"], h["comprobado"]) for h in historial} == {
        ("alta", "NEW", "b", "2026-11-05"), ("baja", "S000", "b", "2026-11-05")}
    estado = json.loads(rutas["ruta_estado"].read_text(encoding="utf-8"))
    assert estado["resultado"] == "actualizado" and estado["ultima_comprobacion_correcta"].startswith("2026-11-05")
    de_nuevo = m.actualizar(m.SinteticoLista("a", nueva), m.SinteticoLista("b", nueva), tabla, ahora=AHORA, **rutas)
    assert de_nuevo["resultado"] == "al_dia"  # idempotente


def test_simular_no_escribe_nada(tmp_path):
    rutas = _rutas(tmp_path)
    nueva = BASE | {"NEW"}
    resumen = m.actualizar(m.SinteticoLista("a", nueva), m.SinteticoLista("b", nueva), None, simular=True, ahora=AHORA, **rutas)
    assert resumen["aplicados"] and not rutas["ruta_historial"].exists() and not rutas["ruta_estado"].exists()
    assert "NEW" not in rutas["ruta"].read_text(encoding="utf-8")


def test_discrepancia_deja_pendiente_y_no_toca_el_fichero(tmp_path):
    rutas = _rutas(tmp_path)
    antes = rutas["ruta"].read_bytes()
    resumen = m.actualizar(m.SinteticoLista("a", BASE | {"NEW"}), m.SinteticoLista("b", BASE), None, ahora=AHORA, **rutas)
    assert resumen["resultado"] == "pendiente" and rutas["ruta"].read_bytes() == antes
    assert json.loads(rutas["ruta_pendientes"].read_text(encoding="utf-8"))["pendientes"][0]["simbolo"] == "NEW"
    m.actualizar(m.SinteticoLista("a", BASE), m.SinteticoLista("b", BASE), None, ahora=AHORA, **rutas)
    assert not rutas["ruta_pendientes"].exists()  # resuelto: desaparece


def test_fallo_de_la_primaria_es_error_sin_tocar_nada_y_el_de_verificacion_deja_todo_pendiente(tmp_path):
    rutas = _rutas(tmp_path)
    antes = rutas["ruta"].read_bytes()
    caida = m.ErrorFuente("sin red")
    r = m.actualizar(m.SinteticoLista("a", [], error=caida), m.SinteticoLista("b", BASE), None, ahora=AHORA, **rutas)
    assert r["resultado"] == "error" and "sin red" in r["error"] and rutas["ruta"].read_bytes() == antes
    assert json.loads(rutas["ruta_estado"].read_text(encoding="utf-8")).get("ultima_comprobacion_correcta") is None
    r = m.actualizar(m.SinteticoLista("a", BASE | {"NEW"}), m.SinteticoLista("b", [], error=caida), None, ahora=AHORA, **rutas)
    assert r["resultado"] == "pendiente" and rutas["ruta"].read_bytes() == antes


# ---- servicio y planificador -------------------------------------------------------------------


def _servicio(tmp_path, ejecutar=None, activo=True):
    return m.ServicioMembresias(ejecutar, **_rutas(tmp_path), activo=activo)


def test_toca_comprobar_una_vez_al_dia_tras_el_cierre_y_reintenta_tras_error(tmp_path):
    s = _servicio(tmp_path)
    assert not s.toca_comprobar(datetime(2026, 11, 5, 16, 0, tzinfo=NY))  # antes del cierre
    assert s.toca_comprobar(datetime(2026, 11, 5, 16, 30, tzinfo=NY))
    s.ruta_estado.write_text(json.dumps({"ultima_comprobacion": "2026-11-05T17:00:00-05:00", "resultado": "al_dia"}))
    assert not s.toca_comprobar(datetime(2026, 11, 5, 20, 0, tzinfo=NY))
    assert s.toca_comprobar(datetime(2026, 11, 6, 17, 0, tzinfo=NY))
    s.ruta_estado.write_text(json.dumps({"ultima_comprobacion": "2026-11-05T17:00:00-05:00", "resultado": "error"}))
    assert not s.toca_comprobar(datetime(2026, 11, 5, 18, 0, tzinfo=NY))
    assert s.toca_comprobar(datetime(2026, 11, 5, 19, 5, tzinfo=NY))
    assert not _servicio(tmp_path, activo=False).toca_comprobar(datetime(2026, 11, 6, 17, 0, tzinfo=NY))


def test_comprobar_no_propaga_errores_inesperados(tmp_path):
    def roto():
        raise ValueError("boom")

    assert _servicio(tmp_path, roto).comprobar()["resultado"] == "error"


def test_estado_resume_lo_consultable(tmp_path):
    rutas = _rutas(tmp_path)
    nueva = BASE | {"NEW"}
    m.actualizar(m.SinteticoLista("a", nueva), m.SinteticoLista("b", nueva), None, ahora=AHORA, **rutas)
    e = m.ServicioMembresias(**rutas).estado(AHORA())
    assert (e["n_vigentes"], e["cambios_aplicados_total"], e["antiguedad_dias"], e["pendientes_de_revisar"]) == (501, 1, 0, [])
    assert e["cambios_aplicados_recientes"][0]["simbolo"] == "NEW" and e["ultima_comprobacion"]


def test_el_planificador_lanza_la_comprobacion_de_membresias_sin_bloquear(tmp_path):
    from tests.test_planificador_paper import _planificador
    from tests.test_planificador_paper import _servicio as _servicio_paper

    llamadas = []
    servicio_m = _servicio(tmp_path, lambda: llamadas.append(1) or {"resultado": "al_dia"})
    servicio_p, _ = _servicio_paper(tmp_path, "2021-03-10")
    servicio_p.ahora = lambda: datetime(2021, 3, 10, 17, 0, tzinfo=NY)
    p = _planificador(servicio_p, membresias=servicio_m)
    p.comprobar()
    p._hilo_membresias.join(5)
    assert llamadas == [1]


def test_variable_de_entorno_desactiva_la_comprobacion_automatica():
    assert m.comprobacion_automatica_activa({}) and not m.comprobacion_automatica_activa({m.VARIABLE_ACTIVAR: "0"})


# ---- el fichero versionado ---------------------------------------------------------------------


def test_el_fichero_arreglado_tiene_unos_503_vigentes_y_ningun_aviso_de_calidad():
    membresias = cargar_membresias_sp500()
    vigentes = miembros_vigentes_en(membresias, "2026-11-22")
    assert 495 <= len(vigentes) <= 510
    assert avisos_membresias(membresias, "2026-11-22", comprobado="2026-11-22") == []
    assert {"BNY", "SKYD", "VMRK", "ILMN", "OKTA", "ECHO"} <= set(vigentes)
    assert not {"BK", "PSKY", "EQR", "KVUE", "WBD", "AVB", "DAY"} & set(vigentes)


def test_los_intervalos_del_fichero_son_coherentes():
    membresias = cargar_membresias_sp500()
    for simbolo, intervalos in membresias.items():
        for i in intervalos:
            assert i["date_removed"] is None or i["date_added"] is None or i["date_added"] <= i["date_removed"], simbolo
    assert [(i["date_added"], i["date_removed"]) for i in membresias["ILMN"]][-2:] == [
        ("2015-11-19", "2024-06-24"), ("2026-09-21", None)]


def test_la_antiguedad_sin_cambios_no_avisa_si_hay_comprobacion_reciente():
    membresias = cargar_membresias_sp500()
    assert any("antigüedad" in a for a in avisos_membresias(membresias, "2027-06-01"))
    assert not [a for a in avisos_membresias(membresias, "2027-06-01", comprobado="2027-05-30") if "antigüedad" in a]


# ---- backtests guardados -----------------------------------------------------------------------

DATOS_REALES = Path(os.environ.get("YPYQUANT_DATOS_REALES", Path(__file__).resolve().parents[1] / "data"))


@pytest.mark.skipif(not (DATOS_REALES / "experimentos").is_dir(), reason="no hay data/experimentos local")
def test_los_experimentos_guardados_con_sp500_pit_mantienen_su_huella():
    from ypyquant.estrategias.experimentos import ConfiguracionExperimento

    con_pit = [c for c in (DATOS_REALES / "experimentos").iterdir() if (c / "config.json").is_file()
               and "sp500_pit" in (c / "config.json").read_text(encoding="utf-8") and not c.name.startswith("tfm-")]
    for carpeta in con_pit:  # la huella depende de la configuración, no del contenido del fichero de membresías
        config = ConfiguracionExperimento(**json.loads((carpeta / "config.json").read_text(encoding="utf-8")))
        assert config.identificador() == carpeta.name


# ---- API ---------------------------------------------------------------------------------------


def test_endpoint_de_estado_de_las_membresias(tmp_path):
    from fastapi.testclient import TestClient

    from app.api.main import crear_app

    rutas = _rutas(tmp_path)
    nueva = BASE | {"NEW"}
    m.actualizar(m.SinteticoLista("a", nueva), m.SinteticoLista("b", BASE), None, ahora=AHORA, **rutas)  # queda pendiente
    app = crear_app(raiz_experimentos=tmp_path / "exp", cargar_env=False, catchup_paper_al_arrancar=False,
                    intervalo_paper_min=0, servicio_membresias=m.ServicioMembresias(**rutas))
    cuerpo = TestClient(app).get("/api/universos/membresias/estado").json()
    assert cuerpo["n_vigentes"] == 500 and cuerpo["resultado"] == "pendiente" and cuerpo["activo"]
    assert cuerpo["pendientes_de_revisar"][0]["simbolo"] == "NEW" and "avisos_calidad" in cuerpo
