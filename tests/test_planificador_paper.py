"""
test_planificador_paper.py — El temporizador en segundo plano que pone al día el paper interno mientras la API vive.

Responsabilidad (precios sintéticos, reloj fijo; sin red):
    - `comprobar()` no hace nada si las sesiones están al día, avanza si hay días cerrados pendientes y deja la
      sesión al día; guarda la última ejecución (hora, resultado, días procesados);
    - no se solapa: con un avance en curso (`ControlAvance`) no lanza otro;
    - un día sin datos (festivo) no se reintenta indefinidamente: tras unos intentos sin novedad deja de descargar
      hasta que cambia la sesión cerrada objetivo;
    - el hilo comprueba cada intervalo, informa de la próxima comprobación y se detiene limpio;
    - `GET /api/paper/interno/planificador` expone el estado y el ciclo de vida de la API (lifespan) lo arranca y lo para.

Se relaciona con: ypyquant.paper.planificador, ypyquant.paper.servicio (`hay_dias_pendientes`, `ControlAvance`),
app/api/main.py (lifespan) y app/api/rutas_paper_interno.py.
"""

import time

from fastapi.testclient import TestClient

from app.api.main import crear_app
from app.api.paper import BotControl, ServicioAlpaca
from tests._paper_utiles import FuenteManual, barras, reloj
from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.receta import Receta
from ypyquant.paper.almacen import AlmacenPaper
from ypyquant.paper.planificador import PlanificadorPaper
from ypyquant.paper.servicio import ControlAvance, ServicioPaper


def _receta() -> Receta:
    return Receta(estrategia="pesos_iguales", parametros={}, universo=["AAA", "BBB"], frecuencia_rebalanceo="monthly",
                  comision=0.001, slippage=0.001, dias_historia_previa=60).normalizada()


def _servicio(tmp_path, dia, fin="2021-03-31"):
    series = {t: barras("2019-01-01", fin, 50 + 25 * i) for i, t in enumerate(["AAA", "BBB", "SPY"])}
    fuente = FuenteManual(series)
    return ServicioPaper(AlmacenPaper(tmp_path / "paper"), fuente, ahora=reloj(dia)), fuente


def _planificador(servicio, **opciones):
    return PlanificadorPaper(servicio, opciones.pop("control", ControlAvance()), intervalo_s=opciones.pop("intervalo_s", 60), **opciones)


def test_sin_dias_pendientes_no_avanza(tmp_path):
    servicio, fuente = _servicio(tmp_path, "2021-03-10")
    servicio.crear_sesion(_receta(), fecha_alta="2021-03-01", modo_ejecucion="cierre")
    servicio.avanzar()
    peticiones = dict(fuente.peticiones)
    planificador = _planificador(servicio)
    assert planificador.comprobar()["resultado"] == "al_dia"
    assert fuente.peticiones == peticiones  # ni siquiera se descargó nada


def test_con_dias_pendientes_avanza_y_deja_la_sesion_al_dia(tmp_path):
    servicio, _ = _servicio(tmp_path, "2021-03-10")
    sesion = servicio.crear_sesion(_receta(), fecha_alta="2021-03-01", modo_ejecucion="cierre")
    assert servicio.hay_dias_pendientes()
    planificador = _planificador(servicio)
    ultima = planificador.comprobar()
    assert ultima["resultado"] == "avanzado" and ultima["dias_procesados"] == 8 and ultima["hora"]
    assert not servicio.hay_dias_pendientes() and servicio.resumen(sesion.id)["al_dia"]
    assert planificador.estado()["ultima"] == ultima


def test_no_se_solapa_con_un_avance_en_curso(tmp_path):
    servicio, _ = _servicio(tmp_path, "2021-03-10")
    sesion = servicio.crear_sesion(_receta(), fecha_alta="2021-03-01", modo_ejecucion="cierre")
    control = ControlAvance()
    assert control.iniciar()  # alguien (el botón «Avanzar ahora») ya está avanzando
    assert _planificador(servicio, control=control).comprobar()["resultado"] == "avance_en_curso"
    assert servicio.resumen(sesion.id)["dias"] == 0


def test_un_dia_sin_datos_no_se_reintenta_indefinidamente(tmp_path):
    servicio, fuente = _servicio(tmp_path, "2021-03-15", fin="2021-03-12")  # el lunes 15 ya cerró pero la fuente no tiene barra
    servicio.crear_sesion(_receta(), fecha_alta="2021-03-01", modo_ejecucion="cierre")
    servicio.avanzar()
    planificador = _planificador(servicio, max_intentos_sin_novedad=3)
    resultados = [planificador.comprobar()["resultado"] for _ in range(5)]
    assert resultados == ["sin_datos", "sin_datos", "sin_datos", "sin_novedades", "sin_novedades"]
    antes = dict(fuente.peticiones)
    planificador.comprobar()
    assert fuente.peticiones == antes
    servicio.ahora = servicio.simulador.ahora = reloj("2021-03-16")  # otro día objetivo: se vuelve a intentar
    assert planificador.comprobar()["resultado"] == "sin_datos"


def test_el_hilo_comprueba_informa_de_la_proxima_y_se_detiene_limpio(tmp_path):
    servicio, _ = _servicio(tmp_path, "2021-03-10")
    servicio.crear_sesion(_receta(), fecha_alta="2021-03-01", modo_ejecucion="cierre")
    planificador = _planificador(servicio, intervalo_s=0.05)
    assert planificador.estado()["activo"] is False and planificador.estado()["ultima"] is None
    planificador.iniciar()
    try:
        limite = time.time() + 20
        while planificador.estado()["ultima"] is None and time.time() < limite:
            time.sleep(0.02)
        estado = planificador.estado()
        assert estado["activo"] and estado["ultima"]["resultado"] == "avanzado" and estado["proxima_comprobacion"]
    finally:
        planificador.detener()
    assert planificador.estado()["activo"] is False and planificador.estado()["proxima_comprobacion"] is None


def _cliente(tmp_path, **opciones) -> TestClient:
    app = crear_app(FuenteSintetica(), tmp_path / "exp", ServicioAlpaca(), BotControl(tmp_path / "bot"), cargar_env=False,
                    catchup_paper_al_arrancar=False, reloj_paper=reloj("2021-03-12"), **opciones)
    return TestClient(app)


def test_la_api_expone_el_planificador_y_su_ciclo_de_vida(tmp_path):
    with _cliente(tmp_path, intervalo_paper_min=30) as cliente:
        estado = cliente.get("/api/paper/interno/planificador").json()
        assert estado["activo"] is True and estado["intervalo_min"] == 30 and estado["proxima_comprobacion"]
        planificador = cliente.app.state.ctx.planificador_paper
    assert planificador.estado()["activo"] is False  # el lifespan lo detuvo al cerrar


def test_el_planificador_se_puede_desactivar(tmp_path):
    with _cliente(tmp_path, intervalo_paper_min=0) as cliente:
        estado = cliente.get("/api/paper/interno/planificador").json()
        assert estado["activo"] is False and estado["intervalo_min"] == 0 and estado["proxima_comprobacion"] is None
