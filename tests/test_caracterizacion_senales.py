"""
Caracterización de las estrategias de señales en modo solo largo (compatibilidad del rediseño largo-corto).

Responsabilidad: fijar que, con los parámetros por defecto (solo largo), `cruce_medias`, `canal_rsi_volatilidad` y
`senales_tecnicas` producen EXACTAMENTE la misma equity, pesos, operaciones, eventos, estado y huella (id) que el marco
anterior. Los valores salen de `tests/_generar_caracterizacion_senales.py`, ejecutado sobre el código previo al rediseño.
Los motivos se comparan con el vocabulario anterior (`senal`, `stop_perdida`...) mediante `MOTIVOS_A_LEGADO`.

Se relaciona con: tests/datos/caracterizacion_senales.json, ypyquant.estrategias.senales y tests/test_huellas_guardadas.py.
"""

import hashlib
import json
from pathlib import Path

import pytest

from tests._generar_caracterizacion_senales import CASOS, huella_caso, huella_estado

ORACULO = json.loads((Path(__file__).parent / "datos" / "caracterizacion_senales.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("nombre", sorted(CASOS))
def test_el_modo_solo_largo_reproduce_exactamente_el_marco_anterior(nombre):
    esperado = ORACULO[nombre]
    obtenido = {**huella_caso(*CASOS[nombre]), **huella_estado(*CASOS[nombre])}
    for clave in ("id", "equity_final", "equity", "pesos", "n_operaciones", "operaciones", "estado", "eventos"):
        assert obtenido[clave] == esperado[clave], f"{nombre}: cambió {clave}"


@pytest.mark.parametrize("nombre", sorted(CASOS))
def test_los_motivos_actuales_equivalen_a_los_anteriores_operacion_a_operacion(nombre):
    """El desglose de motivos solo cambia de nombre: traducido al vocabulario anterior coincide con el guardado."""
    import json as _json

    from tests._generar_caracterizacion_senales import UNIVERSO
    from ypyquant.data.fuentes import FuenteSintetica
    from ypyquant.estrategias.experimentos import ConfiguracionExperimento, cargar_datos_experimento, ejecutar_con_datos
    from ypyquant.estrategias.senales_modelo import MOTIVOS, a_motivo_legado

    estrategia, parametros = CASOS[nombre]
    config = ConfiguracionExperimento(estrategia=estrategia, universo=UNIVERSO, fecha_inicio="2020-01-01",
                                      fecha_fin="2021-12-31", parametros=parametros)
    motivos = ejecutar_con_datos(config, cargar_datos_experimento(config, FuenteSintetica())).operaciones["motivo"]
    assert set(motivos) <= set(MOTIVOS)
    conteo = sorted(motivos.map(a_motivo_legado).value_counts().to_dict().items())
    assert hashlib.sha256(_json.dumps(conteo, sort_keys=True, default=str).encode()).hexdigest() == ORACULO[nombre]["motivos_legado"]
