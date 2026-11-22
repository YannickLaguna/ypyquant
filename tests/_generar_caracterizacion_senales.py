"""
Generador de la huella de caracterización de las estrategias de señales (no es un test).

Responsabilidad: ejecutar las estrategias de señales incluidas con sus parámetros por defecto (modo solo largo) sobre la
fuente sintética y volcar a `tests/datos/caracterizacion_senales.json` las huellas de equity, pesos, operaciones, eventos y
estado. Se ejecutó UNA VEZ sobre el código previo al rediseño del marco de señales; `tests/test_caracterizacion_senales.py`
comprueba que el marco nuevo reproduce exactamente esos valores. No regenerar salvo decisión explícita de cambiar resultados.

Se relaciona con: tests/test_caracterizacion_senales.py, ypyquant.estrategias.senales y estrategias_usuario/.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, cargar_datos_experimento, ejecutar_con_datos
from ypyquant.estrategias import registro

UNIVERSO = ["AAA", "BBB", "CCC", "DDD", "EEE"]
CASOS = {
    "cruce_defecto": ("cruce_medias", {"rapida": 10, "lenta": 40}),
    "cruce_stops": ("cruce_medias", {"rapida": 10, "lenta": 40, "stop_perdida": 0.05, "stop_trailing": 0.08,
                                     "take_profit": 0.15, "max_dias_posicion": 30, "max_posiciones": 3}),
    "cruce_inv_vol": ("cruce_medias", {"rapida": 8, "lenta": 30, "ponderacion": "inversa_volatilidad", "max_posiciones": 4}),
    "canal_defecto": ("canal_rsi_volatilidad", {"ventana_canal": 20, "ventana_salida": 8, "ventana_rsi": 7}),
    "canal_iguales_stop": ("canal_rsi_volatilidad", {"ventana_canal": 20, "ventana_salida": 8, "ventana_rsi": 7,
                                                     "ponderacion": "iguales", "stop_perdida": 0.06, "max_posiciones": 3}),
    "tecnicas": ("senales_tecnicas", {"max_posiciones": 3, "umbral": 0.2}),
}
SALIDAS_LEGADO = ("senal", "stop_perdida", "stop_trailing", "take_profit", "tiempo", "fuera_universo", "regla_propia")


def _hash(objeto) -> str:
    return hashlib.sha256(json.dumps(objeto, sort_keys=True, default=str).encode()).hexdigest()


def huella_caso(estrategia: str, parametros: dict) -> dict:
    """Huellas de un caso: id (huella de config), equity, pesos, operaciones (sin motivo), eventos y estado final."""
    config = ConfiguracionExperimento(estrategia=estrategia, universo=UNIVERSO, fecha_inicio="2020-01-01",
                                      fecha_fin="2021-12-31", parametros=parametros)
    resultado = ejecutar_con_datos(config, cargar_datos_experimento(config, FuenteSintetica()))
    operaciones = resultado.operaciones.drop(columns=["motivo"], errors="ignore")
    return {
        "id": config.normalizada().identificador(),
        "equity_final": float(resultado.equity["equity"].iloc[-1]),
        "equity": _hash([round(float(v), 10) for v in resultado.equity["equity"]]),
        "pesos": _hash([(f"{f:%Y-%m-%d}", t, round(float(w), 12)) for f, t, w in resultado.pesos[["fecha", "ticker", "peso"]].itertuples(index=False)]),
        "n_operaciones": int(len(operaciones)),
        "operaciones": _hash([(f"{r.fecha:%Y-%m-%d}", r.ticker, r.lado, round(float(r.cantidad), 8), round(float(r.precio), 8),
                               None if pd.isna(r.pnl) else round(float(r.pnl), 8)) for r in operaciones.itertuples()]),
        "motivos_legado": _hash(sorted(resultado.operaciones["motivo"].value_counts().to_dict().items())),
    }


def huella_estado(estrategia: str, parametros: dict) -> dict:
    """Huella del estado JSON (sin la clave `lado`, posterior) tras recorrer 2021 día a día con `como_funcion`."""
    config = ConfiguracionExperimento(estrategia=estrategia, universo=UNIVERSO, fecha_inicio="2020-01-01",
                                      fecha_fin="2021-12-31", parametros=parametros)
    datos = cargar_datos_experimento(config, FuenteSintetica()).precios
    datos = {t: datos[t] for t in UNIVERSO}
    instancia = registro.obtener_estrategia(estrategia)(parametros)
    funcion = instancia.como_funcion()
    for fecha in pd.bdate_range("2021-01-04", "2021-12-31"):
        funcion(fecha.to_pydatetime(), datos)
    posiciones = {t: {k: v for k, v in p.items() if k != "lado"} for t, p in instancia.estado["posiciones"].items()}
    return {"estado": _hash({"posiciones": posiciones, "enfriamiento": instancia.estado["enfriamiento"],
                             "fecha": instancia.estado["fecha"], "pesos": instancia.estado["pesos"]}),
            "eventos": _hash([(e["fecha"], e["ticker"], e["tipo"], e["precio"]) for e in instancia.eventos])}


if __name__ == "__main__":
    golden = {nombre: {**huella_caso(*caso), **huella_estado(*caso)} for nombre, caso in CASOS.items()}
    destino = Path(__file__).parent / "datos" / "caracterizacion_senales.json"
    destino.write_text(json.dumps(golden, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: (v["n_operaciones"], v["equity_final"]) for k, v in golden.items()}, indent=1))
