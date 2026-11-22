"""
Garantía de compatibilidad de las huellas con los datos reales ya guardados.

Responsabilidad: recalcula, SIN modificar nada, la huella de las configuraciones de experimento, de las
configuraciones guardadas y los marcos que hay en `data/` del repo principal (`Sistema_trading`) y comprueba que
coincide con lo persistido: id de carpeta de cada experimento (`<estrategia>-<10 primeros de la huella>`),
`huella_estrategia` de cada versión de configuración, y que las comparativas siguen apuntando a experimentos que
existen con el id que les corresponde. Si el cálculo canónico de `receta.py` cambiase una sola coma, este test falla.

Se relaciona con: ypyquant.estrategias.receta/experimentos/configuraciones/marcos (lo que protege) y
tests/test_estrategias.py (huella fija de un caso sintético). Se omite si el repo principal no está en la ruta.
Los experimentos importados del TFM (`tfm-*`, estrategia `tfm_*`) tienen otro esquema de id (`tfm-<slug>`): se excluyen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ypyquant.estrategias.configuraciones import ConfiguracionGuardada
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.estrategias.marcos import MarcoComparacion

DATOS_REALES = Path("C:/Users/yanni/Documents/Codigo/Proyectos/Sistema_trading/data")
pytestmark = pytest.mark.skipif(not DATOS_REALES.is_dir(), reason="no está el repo principal con sus datos")


def _json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8"))


def _experimentos_propios() -> list[Path]:
    """Carpetas de experimentos con id `<estrategia>-<hash>` (sin los `tfm-*` importados)."""
    return sorted(c for c in (DATOS_REALES / "experimentos").iterdir()
                  if (c / "config.json").is_file() and not c.name.startswith("tfm-"))


def test_los_experimentos_importados_del_tfm_tienen_otro_esquema_de_id():
    importados = [c for c in (DATOS_REALES / "experimentos").iterdir() if c.name.startswith("tfm-")]
    assert importados, "se esperaba al menos un experimento importado del TFM"
    for carpeta in importados:
        assert _json(carpeta / "config.json")["estrategia"].startswith("tfm_")


def test_la_huella_de_todos_los_experimentos_guardados_coincide_con_su_carpeta():
    carpetas = _experimentos_propios()
    assert len(carpetas) > 10
    for carpeta in carpetas:
        config = ConfiguracionExperimento(**_json(carpeta / "config.json"))
        assert config.identificador() == carpeta.name, carpeta.name
        assert config.normalizada().identificador() == carpeta.name, carpeta.name


def test_la_huella_de_receta_de_las_configuraciones_guardadas_no_cambia():
    versiones = sorted((DATOS_REALES / "configuraciones").glob("*/v*.json"))
    assert versiones
    for ruta in versiones:
        datos = _json(ruta)
        guardada = ConfiguracionGuardada(**datos)
        recalculada = guardada.a_experimento("2000-01-01", "2000-01-02").huella_receta()
        assert recalculada == datos["huella_estrategia"], ruta
        assert guardada.receta().huella() == datos["huella_estrategia"], ruta


def test_los_marcos_guardados_se_cargan_y_sus_comparativas_apuntan_a_los_mismos_ids():
    for ruta in sorted((DATOS_REALES / "marcos").glob("*/marco.json")):
        if ruta.parent.name != "_comparativas":
            MarcoComparacion(**_json(ruta))
    for ruta in sorted((DATOS_REALES / "marcos" / "_comparativas").glob("cmp-*.json")):
        comparativa = _json(ruta)
        marco = MarcoComparacion(**comparativa["marco"])
        assert marco.id == comparativa["id_marco"]
        ids = sorted(e["id_experimento"] for e in comparativa["experimentos"])
        esperado = "cmp-" + hashlib.sha256(json.dumps([marco.id, ids]).encode("utf-8")).hexdigest()[:10]
        assert comparativa["id"] == esperado == ruta.stem
        for id_experimento in ids:
            carpeta = DATOS_REALES / "experimentos" / id_experimento
            if carpeta.is_dir():  # un experimento borrado después no invalida el registro histórico
                assert ConfiguracionExperimento(**_json(carpeta / "config.json")).identificador() == id_experimento
