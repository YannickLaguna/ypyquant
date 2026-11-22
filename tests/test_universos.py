"""
test_universos.py — Tests del módulo de universos (`ypyquant.data.universos`) y de los deslistados explícitos.

Responsabilidad:
- Símbolos canónicos (`a_simbolo_canonico`: BRK.B -> BRK-B) y su uso al leer archivos de universo.
- `resolver_seleccion` / `universo_de_indice`: la resolución que antes duplicaba `AssetSelectionService`.
- `experimentos.resolver_universo` y `listar_universos` siguen disponibles y delegan en este módulo.
- Deslistados EXPLÍCITOS de punta a punta: una `FuenteSintetica` que declara un deslistado con motivo llega a
  `DatosExperimento.deslistados` y de ahí al aviso de la liquidación, con el motivo; un cargador suelto sigue
  recibiendo el aviso inferido (sin motivo).

Se relaciona con: ypyquant.data.universos, ypyquant.data.fuente_precios, estrategias.experimentos/simulacion.
"""

from datetime import datetime

import pytest

from ypyquant.data import universos
from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.data.universos import a_simbolo_canonico, resolver_seleccion, universo_de_indice
from ypyquant.estrategias import experimentos
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, cargar_datos_experimento, ejecutar_con_datos


def test_simbolo_canonico_es_el_de_yfinance():
    assert a_simbolo_canonico("brk.b") == "BRK-B" and a_simbolo_canonico(" BF-B ") == "BF-B"


def test_los_archivos_de_universo_se_leen_con_simbolos_canonicos(tmp_path):
    (tmp_path / "mini.txt").write_text("# comentario\nbrk.b\nAAPL\n\n", encoding="utf-8")
    assert universos.resolver_universo("mini", tmp_path) == ["BRK-B", "AAPL"]


def test_experimentos_reexporta_la_resolucion_de_universos():
    assert experimentos.resolver_universo is universos.resolver_universo
    assert experimentos.listar_universos is universos.listar_universos


def test_resolver_seleccion_usa_los_simbolos_explicitos_o_el_indice():
    assert resolver_seleccion(["A", "B"], None, None, None, True, None) == ["A", "B"]
    vistos = []

    def proveedor(indice, inicio, fin, solo_supervivientes):
        vistos.append((indice, inicio, fin, solo_supervivientes))
        return ["X"]

    assert resolver_seleccion(None, "2021-01-01", "2022-01-01", "SP500", True, proveedor) == ["X"]
    assert vistos == [("SP500", datetime(2021, 1, 1), datetime(2022, 1, 1), False)]  # evitar sesgo = incluir los que salieron


def test_resolver_seleccion_valida_sus_entradas():
    with pytest.raises(ValueError):
        resolver_seleccion(None, None, None, "SP500", True, lambda *a: ["X"])  # índice sin fechas
    with pytest.raises(ValueError):
        resolver_seleccion(None, None, None, None, True, None)  # ni símbolos ni índice
    with pytest.raises(RuntimeError):
        universo_de_indice("SP500", "2021-01-01", "2022-01-01", True, None)  # sin proveedor


# ---- deslistados explícitos de la fuente hasta el aviso -----------------------------------------------------


@pytest.fixture
def membresias_de_prueba(monkeypatch):
    miembros = {"AAA": [{"index": "SP500", "date_added": "2015-01-02", "date_removed": None}],
                "CCC": [{"index": "SP500", "date_added": "2015-01-02", "date_removed": "2020-07-08"}]}
    monkeypatch.setattr(universos, "cargar_membresias_sp500", lambda: miembros)


def _config():
    return ConfiguracionExperimento(estrategia="pesos_iguales", universo="sp500_pit", fecha_inicio="2020-01-02",
                                    fecha_fin="2020-12-31", comision=0.0, slippage=0.0)


def test_el_deslistado_explicito_de_la_fuente_llega_al_aviso_con_su_motivo(membresias_de_prueba):
    fuente = FuenteSintetica(deslistados={"CCC": ("2020-07-03", "quiebra")})
    config = _config()
    datos = cargar_datos_experimento(config, fuente)
    assert datos.deslistados["CCC"].motivo == "quiebra" and not datos.deslistados["CCC"].inferido
    assert "AAA" not in datos.deslistados
    avisos = ejecutar_con_datos(config, datos).avisos
    assert any(a.startswith("CCC se liquidó a su último precio conocido (2020-07-03)") and "(deslistado (quiebra))" in a
               for a in avisos)


def test_un_cargador_suelto_mantiene_el_aviso_inferido_sin_motivo(membresias_de_prueba):
    sintetica = FuenteSintetica(deslistados={"CCC": ("2020-07-03", "quiebra")})
    config = _config()
    datos = cargar_datos_experimento(config, lambda t, i, f: sintetica.ohlcv(t, i, f))  # callable suelto: no sabe de motivos
    assert datos.deslistados["CCC"].inferido and datos.deslistados["CCC"].motivo is None
    avisos = ejecutar_con_datos(config, datos).avisos
    assert any(a.startswith("CCC se liquidó a su último precio conocido (2020-07-03)") and "quiebra" not in a for a in avisos)
