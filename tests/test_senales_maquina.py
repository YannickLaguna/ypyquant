"""
Tests de la máquina de estados de señales (`MaquinaPosiciones`): largos, cortos, stops simétricos, giro y límites.

Responsabilidad: fijar con casos construidos a mano el comportamiento de cada regla (entrada, stop de pérdida, take profit,
trailing, tiempo, señal contraria, salida de universo, límites de exposición) en ambos lados y comprobar con Hypothesis las
invariantes que no pueden romperse nunca: el signo del peso coincide con el lado, el stop está del lado correcto (en un corto
por encima de la entrada) y un valor que lo cruzó se cierra, las posiciones respetan los topes y las entradas respetan los
límites de exposición bruta y neta, y el estado es JSON-serializable.

Se relaciona con: src/ypyquant/estrategias/senales_maquina.py, senales_modelo.py, senales_tamano.py y tests/test_senales.py
(adaptador de pandas).
"""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ypyquant.estrategias.senales_maquina import MaquinaPosiciones, Mercado
from ypyquant.estrategias.senales_modelo import (
    CORTO,
    LARGO,
    Dimensionamiento,
    GestionPosicion,
    LimitesExposicion,
    Senal,
    nivel_stop,
    nivel_take_profit,
    nivel_trailing,
)

DIAS = [f"2022-01-{d:02d}" for d in range(3, 29) if d not in (8, 9, 15, 16, 22, 23)]  # días hábiles
AMBOS = (LARGO, CORTO)


def _maquina(lados=AMBOS, max_posiciones=2, peso_maximo=0.5, bruta=2.0, neta_max=1.0, neta_min=-1.0, max_largas=None,
             max_cortas=None, ponderacion="iguales", **gestion) -> MaquinaPosiciones:
    return MaquinaPosiciones(
        GestionPosicion(lados=lados, **{"enfriamiento_dias": 0, **gestion}),
        LimitesExposicion(max_posiciones, max_largas, max_cortas, bruta, neta_max, neta_min),
        Dimensionamiento(ponderacion=ponderacion, max_posiciones=max_posiciones, peso_maximo=peso_maximo,
                         stop_perdida=gestion.get("stop_perdida")))


class Simulador:
    """Recorre días con precios dados `{ticker: [precios]}` y señales por día; guarda estado y eventos como la estrategia."""

    def __init__(self, maquina, precios, universo=None):
        self.maquina, self.precios = maquina, precios
        self.universo = set(precios) if universo is None else set(universo)
        self.posiciones, self.enfriamiento, self.eventos, self.dia = {}, {}, [], -1

    def mercado(self) -> Mercado:
        i = self.dia
        precios = self.precios
        previa = i > 0

        def extremos(t):  # llamadas diarias: el tramo desde la llamada anterior es solo el cierre de hoy
            return (precios[t][i], precios[t][i])

        return Mercado(
            fecha=DIAS[i], universo=frozenset(self.universo), precio=lambda t: precios[t][i], hay_previa=previa,
            crecimiento=lambda t: precios[t][i] / precios[t][i - 1],
            extremos_tramo=extremos, sesiones_desde=lambda f: i - DIAS.index(f))

    def paso(self, senales=()):
        self.dia += 1
        eventos = self.maquina.avanzar(self.posiciones, self.enfriamiento, list(senales), self.mercado())
        self.eventos += eventos
        return eventos


def _entra(ticker, lado, fuerza=None):
    return Senal(ticker, lado, "entrada", fuerza)


def _sale(ticker, lado):
    return Senal(ticker, lado, "salida")


def _motivos(eventos, tipo="salida"):
    return [(e["ticker"], e["motivo"]) for e in eventos if e["tipo"] == tipo]


# ---- entradas ---------------------------------------------------------------------------------------------


def test_una_entrada_corta_abre_una_posicion_con_peso_negativo_y_minimo():
    sim = Simulador(_maquina(max_posiciones=4, peso_maximo=0.25), {"AAA": [100.0] * 5})
    eventos = sim.paso([_entra("AAA", CORTO)])
    assert _motivos(eventos, "entrada") == [("AAA", "entrada_corta")]
    posicion = sim.posiciones["AAA"]
    assert posicion["lado"] == CORTO and posicion["peso"] == pytest.approx(-0.25) and posicion["minimo"] == 100.0
    assert "maximo" not in posicion


def test_un_lado_no_habilitado_no_abre_posiciones():
    sim = Simulador(_maquina(lados=(LARGO,)), {"AAA": [100.0] * 5})
    assert sim.paso([_entra("AAA", CORTO)]) == [] and sim.posiciones == {}


def test_senales_contradictorias_en_un_ticker_no_abren_ni_largo_ni_corto():
    sim = Simulador(_maquina(), {"AAA": [100.0] * 5})
    assert sim.paso([_entra("AAA", LARGO), _entra("AAA", CORTO)]) == [] and sim.posiciones == {}


def test_entrada_y_salida_del_mismo_lado_el_mismo_dia_no_entra():
    sim = Simulador(_maquina(), {"AAA": [100.0] * 5})
    assert sim.paso([_entra("AAA", CORTO), _sale("AAA", CORTO)]) == []


def test_la_fuerza_ordena_los_candidatos_cuando_hay_menos_huecos_que_senales():
    sim = Simulador(_maquina(max_posiciones=1, peso_maximo=1.0), {"AAA": [100.0] * 3, "BBB": [100.0] * 3})
    sim.paso([_entra("AAA", CORTO, 1.0), _entra("BBB", CORTO, 2.0)])
    assert set(sim.posiciones) == {"BBB"}


def test_ponderacion_por_fuerza_da_mas_peso_a_la_senal_mas_fuerte():
    sim = Simulador(_maquina(max_posiciones=2, peso_maximo=1.0, ponderacion="fuerza"), {"AAA": [100.0] * 3, "BBB": [100.0] * 3})
    sim.paso([_entra("AAA", LARGO, 1.0), _entra("BBB", LARGO, 3.0)])
    assert sim.posiciones["BBB"]["peso"] == pytest.approx(3 * sim.posiciones["AAA"]["peso"])


def test_ponderacion_por_riesgo_pierde_el_riesgo_por_operacion_al_saltar_el_stop():
    maquina = MaquinaPosiciones(
        GestionPosicion(stop_perdida=0.05, lados=AMBOS), LimitesExposicion(10, None, None, 2.0, 1.0, -1.0),
        Dimensionamiento("riesgo", 10, 0.5, riesgo_por_operacion=0.01, stop_perdida=0.05))
    sim = Simulador(maquina, {"AAA": [100.0] * 3, "BBB": [100.0] * 3})
    sim.paso([_entra("AAA", LARGO), _entra("BBB", CORTO)])
    assert sim.posiciones["AAA"]["peso"] == pytest.approx(0.2) and sim.posiciones["BBB"]["peso"] == pytest.approx(-0.2)


# ---- stops y take profit simétricos -------------------------------------------------------------------------


def test_el_stop_de_perdida_de_un_corto_salta_cuando_el_precio_sube_y_no_cuando_cae():
    sube = Simulador(_maquina(stop_perdida=0.1), {"AAA": [100.0, 105.0, 110.0, 110.0]})
    sube.paso([_entra("AAA", CORTO)])
    assert sube.paso() == []                                  # +5 %: aguanta
    assert _motivos(sube.paso()) == [("AAA", "stop_loss")]    # +10 %: sale
    cae = Simulador(_maquina(stop_perdida=0.1), {"AAA": [100.0, 80.0, 60.0, 40.0]})
    cae.paso([_entra("AAA", CORTO)])
    assert cae.paso() == [] and cae.paso() == [] and "AAA" in cae.posiciones


def test_el_stop_de_un_largo_esta_debajo_y_el_de_un_corto_encima_de_la_entrada():
    largo, corto = {"lado": LARGO, "precio_entrada": 100.0}, {"lado": CORTO, "precio_entrada": 100.0}
    assert nivel_stop(largo, 0.1) == pytest.approx(90.0) and nivel_stop(corto, 0.1) == pytest.approx(110.0)
    assert nivel_take_profit(largo, 0.2) == pytest.approx(120.0) and nivel_take_profit(corto, 0.2) == pytest.approx(80.0)


def test_take_profit_de_un_corto_salta_cuando_el_precio_cae():
    sim = Simulador(_maquina(take_profit=0.2), {"AAA": [100.0, 90.0, 79.0, 79.0]})
    sim.paso([_entra("AAA", CORTO)])
    assert sim.paso() == []
    assert _motivos(sim.paso()) == [("AAA", "take_profit")]


def test_trailing_de_un_corto_sigue_al_minimo_y_sale_cuando_rebota():
    # Entra a 100, cae a 80 (mínimo) y rebota a 86 (+7,5 %: aguanta) y a 88 (+10 %: sale)
    sim = Simulador(_maquina(stop_trailing=0.1, take_profit=None), {"AAA": [100.0, 90.0, 80.0, 86.0, 88.0, 88.0]})
    sim.paso([_entra("AAA", CORTO)])
    sim.paso(), sim.paso()
    assert sim.posiciones["AAA"]["minimo"] == 80.0
    assert sim.paso() == []
    assert _motivos(sim.paso()) == [("AAA", "trailing")]
    assert nivel_trailing({"lado": CORTO, "minimo": 80.0}, 0.1) == pytest.approx(88.0)
    assert nivel_trailing({"lado": LARGO, "maximo": 120.0}, 0.1) == pytest.approx(108.0)


def test_trailing_y_stop_de_perdida_activan_enfriamiento_y_el_tiempo_no():
    sim = Simulador(_maquina(stop_perdida=0.05, enfriamiento_dias=3), {"AAA": [100.0, 106.0, 106.0, 106.0, 106.0, 106.0]})
    sim.paso([_entra("AAA", CORTO)])
    assert _motivos(sim.paso([_entra("AAA", CORTO)])) == [("AAA", "stop_loss")]
    assert sim.enfriamiento == {"AAA": "2022-01-07"}
    sim.paso([_entra("AAA", CORTO)])
    assert "AAA" not in sim.posiciones                          # sigue enfriando
    sim.paso([_entra("AAA", CORTO)])
    sim.paso([_entra("AAA", CORTO)])
    assert sim.posiciones["AAA"]["lado"] == CORTO               # reentra pasado el enfriamiento


def test_salida_por_tiempo_y_por_universo_en_un_corto():
    sim = Simulador(_maquina(max_dias_posicion=2), {"AAA": [100.0] * 5})
    sim.paso([_entra("AAA", CORTO)])
    sim.paso()
    assert _motivos(sim.paso()) == [("AAA", "tiempo")]
    fuera = Simulador(_maquina(), {"AAA": [100.0] * 5, "BBB": [100.0] * 5})
    fuera.paso([_entra("AAA", CORTO)])
    fuera.universo = {"BBB"}
    assert _motivos(fuera.paso()) == [("AAA", "salida_universo")]


def test_la_salida_del_corto_es_la_regla_de_recompra_del_corto_no_la_del_largo():
    sim = Simulador(_maquina(), {"AAA": [100.0] * 5})
    sim.paso([_entra("AAA", CORTO)])
    assert sim.paso([_sale("AAA", LARGO)]) == []
    assert _motivos(sim.paso([_sale("AAA", CORTO)])) == [("AAA", "senal_salida")]


# ---- señal contraria y giro --------------------------------------------------------------------------------


def test_la_senal_contraria_cierra_la_posicion_y_sin_giro_no_abre_la_opuesta_ese_dia():
    sim = Simulador(_maquina(), {"AAA": [100.0] * 5})
    sim.paso([_entra("AAA", LARGO)])
    eventos = sim.paso([_entra("AAA", CORTO)])
    assert _motivos(eventos) == [("AAA", "senal_contraria")] and sim.posiciones == {}
    sim.paso([_entra("AAA", CORTO)])
    assert sim.posiciones["AAA"]["lado"] == CORTO              # al día siguiente, plano, entra


def test_con_giro_la_senal_contraria_cierra_y_abre_la_opuesta_el_mismo_dia():
    sim = Simulador(_maquina(permitir_giro=True), {"AAA": [100.0] * 5})
    sim.paso([_entra("AAA", LARGO)])
    eventos = sim.paso([_entra("AAA", CORTO)])
    assert [(e["tipo"], e["motivo"]) for e in eventos] == [("salida", "senal_contraria"), ("entrada", "entrada_corta")]
    assert sim.posiciones["AAA"]["lado"] == CORTO and sim.posiciones["AAA"]["peso"] < 0


def test_sin_lado_opuesto_habilitado_no_hay_senal_contraria():
    sim = Simulador(_maquina(lados=(LARGO,)), {"AAA": [100.0] * 5})
    sim.paso([_entra("AAA", LARGO)])
    assert sim.paso([_entra("AAA", CORTO)]) == [] and "AAA" in sim.posiciones


# ---- límites ------------------------------------------------------------------------------------------------


def test_los_cortos_no_pasan_de_la_exposicion_bruta_y_el_largo_ya_abierto_la_consume():
    sim = Simulador(_maquina(max_posiciones=2, peso_maximo=0.5, bruta=0.8), {"AAA": [100.0] * 3, "BBB": [100.0] * 3})
    sim.paso([_entra("AAA", LARGO), _entra("BBB", CORTO)])
    assert sim.posiciones["AAA"]["peso"] == pytest.approx(0.5) and sim.posiciones["BBB"]["peso"] == pytest.approx(-0.3)


def test_la_exposicion_neta_minima_acota_los_cortos_y_la_maxima_los_largos():
    sim = Simulador(_maquina(max_posiciones=2, peso_maximo=0.5, neta_min=-0.2), {"AAA": [100.0] * 3, "BBB": [100.0] * 3})
    sim.paso([_entra("AAA", CORTO), _entra("BBB", CORTO)])
    assert sum(p["peso"] for p in sim.posiciones.values()) == pytest.approx(-0.2)
    largos = Simulador(_maquina(max_posiciones=2, peso_maximo=0.8, neta_max=0.6), {"AAA": [100.0] * 3, "BBB": [100.0] * 3})
    largos.paso([_entra("AAA", LARGO), _entra("BBB", LARGO)])
    assert sum(p["peso"] for p in largos.posiciones.values()) == pytest.approx(0.6)


def test_los_topes_por_lado_reservan_huecos_para_el_otro_lado():
    maquina = _maquina(max_posiciones=3, peso_maximo=0.3, max_largas=1, max_cortas=2)
    sim = Simulador(maquina, {t: [100.0] * 3 for t in ("AAA", "BBB", "CCC", "DDD")})
    sim.paso([_entra("AAA", LARGO), _entra("BBB", LARGO), _entra("CCC", CORTO), _entra("DDD", CORTO)])
    assert {t: p["lado"] for t, p in sim.posiciones.items()} == {"AAA": LARGO, "CCC": CORTO, "DDD": CORTO}


def test_la_deriva_mueve_los_pesos_del_corto_en_contra_cuando_el_precio_sube():
    sim = Simulador(_maquina(max_posiciones=4, peso_maximo=0.25), {"AAA": [100.0, 110.0, 110.0], "BBB": [100.0, 100.0, 100.0]})
    sim.paso([_entra("AAA", CORTO), _entra("BBB", LARGO)])
    sim.paso()
    efectivo = 1.0 - (-0.25 + 0.25)
    total = efectivo + (-0.25 * 1.1) + 0.25
    assert sim.posiciones["AAA"]["peso"] == pytest.approx(-0.25 * 1.1 / total) and sim.posiciones["AAA"]["peso"] < -0.25


def test_el_estado_es_json_y_las_posiciones_guardadas_sin_lado_se_leen_como_largas():
    sim = Simulador(_maquina(), {"AAA": [100.0] * 4, "BBB": [100.0] * 4})
    sim.paso([_entra("AAA", LARGO), _entra("BBB", CORTO)])
    assert json.loads(json.dumps(sim.posiciones)) == sim.posiciones
    antigua = {"AAA": {"fecha_entrada": DIAS[0], "precio_entrada": 100.0, "maximo": 100.0, "peso": 0.5}}
    sim.posiciones = antigua
    sim.dia = 0
    assert _motivos(sim.paso([_sale("AAA", LARGO)])) == [("AAA", "senal_salida")]


# ---- invariantes con Hypothesis ------------------------------------------------------------------------------

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


@st.composite
def _escenarios(draw):
    n = len(DIAS)
    precios = {t: [100.0] for t in TICKERS}
    for t in TICKERS:
        for _ in range(n - 1):
            precios[t].append(precios[t][-1] * (1 + draw(st.floats(-0.08, 0.08))))
    senales = [[Senal(t, draw(st.sampled_from(AMBOS)), draw(st.sampled_from(["entrada", "salida", "entrada"])),
                      draw(st.one_of(st.none(), st.floats(0, 5))))
                for t in TICKERS if draw(st.booleans())] for _ in range(n)]
    universo = [draw(st.sets(st.sampled_from(TICKERS), min_size=3)) for _ in range(n)]
    opcional = lambda fraccion: st.one_of(st.none(), st.floats(0.02, fraccion))
    maquina = _maquina(
        lados=draw(st.sampled_from([(LARGO,), (CORTO,), AMBOS])), max_posiciones=draw(st.integers(1, 4)),
        peso_maximo=draw(st.floats(0.1, 1.0)), bruta=draw(st.floats(0.3, 2.0)), neta_max=draw(st.floats(0.3, 1.5)),
        neta_min=draw(st.floats(-1.0, 0.3)), max_largas=draw(st.one_of(st.none(), st.integers(1, 3))),
        max_cortas=draw(st.one_of(st.none(), st.integers(1, 3))),
        ponderacion=draw(st.sampled_from(["iguales", "fuerza"])), stop_perdida=draw(opcional(0.2)),
        take_profit=draw(opcional(0.3)), stop_trailing=draw(opcional(0.2)),
        max_dias_posicion=draw(st.one_of(st.none(), st.integers(1, 6))), permitir_giro=draw(st.booleans()),
        enfriamiento_dias=draw(st.integers(0, 3)))
    return maquina, precios, senales, universo


@settings(max_examples=150, deadline=None)
@given(_escenarios())
def test_invariantes_de_la_maquina_en_cualquier_secuencia(escenario):
    maquina, precios, senales, universos = escenario
    gestion, limites = maquina.gestion, maquina.limites
    sim = Simulador(maquina, precios)
    for dia, senales_del_dia in enumerate(senales):
        sim.universo = universos[dia]
        eventos = sim.paso(senales_del_dia)
        posiciones = sim.posiciones
        entradas = [e for e in eventos if e["tipo"] == "entrada"]
        # 1) El lado y el signo del peso coinciden, el extremo correcto existe y solo se operan los lados habilitados
        for ticker, p in posiciones.items():
            assert (p["lado"] == LARGO) == (p["peso"] > 0) and p["lado"] in gestion.lados
            assert ("maximo" in p) == (p["lado"] == LARGO) and ("minimo" in p) == (p["lado"] == CORTO)
            assert ticker in sim.universo
        # 2) Nunca largo y corto a la vez en un ticker, ni dos entradas del mismo ticker en un día
        assert len({e["ticker"] for e in entradas}) == len(entradas)
        # 3) Topes de posiciones
        assert len(posiciones) <= limites.max_posiciones
        for lado, tope in ((LARGO, limites.max_largas), (CORTO, limites.max_cortas)):
            if tope is not None:
                assert sum(1 for p in posiciones.values() if p["lado"] == lado) <= tope
        # 4) El stop está del lado correcto y un valor que lo cruzó ya no sigue abierto
        for ticker, p in posiciones.items():
            precio, entrada = precios[ticker][dia], p["precio_entrada"]
            if gestion.stop_perdida is not None:
                if p["lado"] == LARGO:
                    assert nivel_stop(p, gestion.stop_perdida) < entrada and precio / entrada - 1.0 > -gestion.stop_perdida
                else:
                    assert nivel_stop(p, gestion.stop_perdida) > entrada and precio / entrada - 1.0 < gestion.stop_perdida
            if gestion.stop_trailing is not None:
                if p["lado"] == LARGO:
                    assert precio / p["maximo"] - 1.0 > -gestion.stop_trailing
                else:
                    assert precio / p["minimo"] - 1.0 < gestion.stop_trailing
        # 5) Si hubo entradas, la exposición bruta y la neta respetan sus límites (la deriva puede romperlos entre entradas)
        if entradas:
            bruta = sum(abs(p["peso"]) for p in posiciones.values())
            neta = sum(p["peso"] for p in posiciones.values())
            assert bruta <= limites.exposicion_bruta_max + 1e-9
            if any(e["lado"] == LARGO for e in entradas):
                assert neta <= limites.exposicion_neta_max + 1e-9
            if any(e["lado"] == CORTO for e in entradas):
                assert neta >= limites.exposicion_neta_min - 1e-9
        # 6) El estado se serializa a JSON y vuelve igual
        guardado = {"posiciones": posiciones, "enfriamiento": sim.enfriamiento}
        assert json.loads(json.dumps(guardado)) == guardado
