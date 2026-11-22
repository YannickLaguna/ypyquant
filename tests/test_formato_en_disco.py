"""
Garantía del formato en disco de los almacenes de documentos.

Responsabilidad: reproduce A MANO (con `pathlib`, sin pasar por ningún almacén) la estructura exacta que el código anterior
dejó en `data/experimentos`, `data/barridos`, `data/configuraciones`, `data/marcos` y `data/reservado.json`, y comprueba que
los almacenes actuales la leen, listan y borran igual; y, a la inversa, que lo que escriben tiene los mismos nombres de
archivo y el mismo texto JSON (`json.dumps(..., ensure_ascii=False, indent=2)`, sin salto de línea final; el final de línea lo decide `write_text` según la plataforma, como siempre). Así la refactorización
sobre `AlmacenDocumentos` no puede cambiar el formato sin que este test falle (el `data/` real no está en los worktrees).

Se relaciona con: ypyquant.almacen_documentos (el módulo común), ypyquant.estrategias.almacen/barridos/configuraciones/marcos/
reservado (los adaptadores) y tests/test_huellas_guardadas.py (lo mismo contra los datos reales locales).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.barridos import AlmacenBarridos, ConfiguracionBarrido
from ypyquant.estrategias.configuraciones import AlmacenConfiguraciones, ConfiguracionGuardada
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.estrategias.marcos import AlmacenMarcos, MarcoComparacion
from ypyquant.estrategias.reservado import RegistroReservado

TICKERS = ["AAPL", "MSFT", "JPM"]


def _json_canonico(contenido) -> str:
    """El formato de los JSON del almacén: indentado con 2 espacios, sin escapar la eñe, sin salto final."""
    return json.dumps(contenido, ensure_ascii=False, indent=2)


def _poner(ruta: Path, contenido) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(_json_canonico(contenido).encode("utf-8"))


def _config(**cambios) -> ConfiguracionExperimento:
    base = {"estrategia": "momentum", "universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31",
            "parametros": {"top_k": 2}}
    return ConfiguracionExperimento(**{**base, **cambios}).normalizada()


def _equity() -> pd.DataFrame:
    return pd.DataFrame({"fecha": pd.to_datetime(["2020-01-01", "2020-01-02"]), "equity": [100.0, 101.0],
                         "benchmark": [100.0, 100.5], "drawdown": [0.0, 0.0], "retorno": [0.0, 0.01]})


def _experimento_en_disco(raiz: Path, config: ConfiguracionExperimento, estado: str = "hecho") -> str:
    """Carpeta de experimento tal como la dejaba `registrar` + `guardar` (a mano)."""
    id_experimento = config.identificador()
    carpeta = raiz / id_experimento
    _poner(carpeta / "config.json", config.model_dump())
    _poner(carpeta / "estado.json", {"estado": estado, "creado": "2026-11-01T10:00:00", "actualizado": "2026-11-01T10:01:00",
                                     "error": None, "traza": None})
    if estado == "hecho":
        _poner(carpeta / "metricas.json", {"sharpe": 1.25, "cagr": 0.1, "avisos": ["aviso de prueba"]})
        _poner(carpeta / "procedencia.json", {"commit": "abc123"})
        _equity().to_parquet(carpeta / "equity.parquet", index=False)
    return id_experimento


# ---- experimentos ----------------------------------------------------------------------------


def test_experimentos_existentes_se_leen_listan_y_borran(tmp_path):
    hecho = _experimento_en_disco(tmp_path, _config())
    pendiente = _experimento_en_disco(tmp_path, _config(parametros={"top_k": 3}), estado="pendiente")
    almacen = AlmacenExperimentos(tmp_path)

    assert almacen.existe(hecho) and almacen.estado_de(hecho) == "hecho" and almacen.estado_de(pendiente) == "pendiente"
    assert {r["id"] for r in almacen.listar()} == {hecho, pendiente}
    assert [r["id"] for r in almacen.listar(estado="hecho")] == [hecho]
    assert almacen.leer_metricas(hecho)["avisos"] == ["aviso de prueba"]
    assert almacen.leer_config(hecho)["estrategia"] == "momentum"
    assert list(almacen.leer_equity(hecho)["equity"]) == [100.0, 101.0]
    assert almacen.leer_pesos(hecho) is None
    detalle = almacen.leer(hecho)
    assert detalle["metricas"]["sharpe"] == 1.25 and detalle["equity"][0]["fecha"] == "2020-01-01"
    assert almacen.comparar([hecho])["ids"] == [hecho]
    assert almacen.marcar_interrumpidos() == 1 and almacen.estado_de(pendiente) == "error"

    assert almacen.borrar(hecho) is True and almacen.borrar(hecho) is False
    assert not (tmp_path / hecho).exists() and almacen.estado_de(hecho) is None
    with pytest.raises(ValueError):
        almacen.leer_config("../fuera")


def test_experimentos_escriben_los_mismos_archivos_y_bytes(tmp_path):
    almacen = AlmacenExperimentos(tmp_path)
    config = _config()
    id_experimento = almacen.registrar(config)
    carpeta = tmp_path / id_experimento
    assert {p.name for p in carpeta.iterdir()} == {"config.json", "estado.json"}
    assert (carpeta / "config.json").read_text(encoding="utf-8") == _json_canonico(config.model_dump())
    estado = json.loads((carpeta / "estado.json").read_text(encoding="utf-8"))
    assert list(estado) == ["estado", "creado", "actualizado", "error"] and estado["estado"] == "pendiente"
    almacen.marcar_estado(id_experimento, "error", "falló")
    estado = json.loads((carpeta / "estado.json").read_text(encoding="utf-8"))
    assert list(estado) == ["estado", "creado", "actualizado", "error", "traza"] and estado["error"] == "falló"
    assert not list(carpeta.glob("*.tmp"))


# ---- barridos --------------------------------------------------------------------------------


def _barrido_en_disco(raiz: Path) -> tuple[str, ConfiguracionBarrido]:
    config = ConfiguracionBarrido(base=_config(), rejilla={"top_k": [1, 2]})
    id_barrido = config.identificador()
    combinaciones = [{"id": c.identificador(), "etiqueta": c.etiqueta, "valores": v, "posiciones": p}
                     for c, v, p in config.combinaciones()]
    _poner(raiz / id_barrido / "barrido.json", {
        "id": id_barrido, "config": config.model_dump(), "combinaciones": combinaciones, "omitidas": [],
        "estado": "pendiente", "progreso": {"total": 2, "hechos": 0, "errores": 0}, "tareas": {},
        "creado": "2026-11-01T10:00:00", "actualizado": "2026-11-01T10:00:00"})
    _poner(raiz / id_barrido / "validacion.json", {"dsr": 0.5})
    pd.DataFrame({"a": [1.0, 2.0]}).to_parquet(raiz / id_barrido / "matriz.parquet", index=True)
    return id_barrido, config


def test_barridos_existentes_se_leen_listan_y_borran(tmp_path):
    id_barrido, config = _barrido_en_disco(tmp_path / "barridos")
    almacen = AlmacenBarridos(tmp_path / "barridos", AlmacenExperimentos(tmp_path / "experimentos"))

    assert almacen.existe(id_barrido) and almacen.crear(config) == (id_barrido, False)
    datos = almacen.leer(id_barrido)
    assert datos["resultados"] == ["validacion"] and datos["estado"] == "error"  # sus experimentos no existen
    assert almacen.leer_resultado(id_barrido, "validacion") == {"dsr": 0.5}
    assert almacen.leer_resultado(id_barrido, "walk_forward") is None
    assert [r["id"] for r in almacen.listar()] == [id_barrido]
    assert almacen.cargar(id_barrido).ids == [c["id"] for c in datos["combinaciones"]]
    almacen.actualizar_tarea(id_barrido, "permutacion_in_sample", estado="en_curso")
    assert almacen.tarea(id_barrido, "permutacion_in_sample")["estado"] == "en_curso"
    assert almacen.marcar_interrumpidos() == 1
    almacen.guardar_resultado(id_barrido, "reservado", [{"x": 1}])
    assert (tmp_path / "barridos" / id_barrido / "reservado.json").read_text(encoding="utf-8") == _json_canonico([{"x": 1}])
    almacen.guardar_parquet(id_barrido, "otra", pd.DataFrame({"b": [1.0]}))
    assert (tmp_path / "barridos" / id_barrido / "otra.parquet").is_file()

    assert almacen.borrar(id_barrido) is True and almacen.borrar(id_barrido) is False
    assert not (tmp_path / "barridos" / id_barrido).exists()
    with pytest.raises(KeyError):
        almacen.leer(id_barrido)
    assert almacen.ruta_reservado == tmp_path / "reservado.json"


# ---- configuraciones -------------------------------------------------------------------------


def _guardada() -> ConfiguracionGuardada:
    return ConfiguracionGuardada.desde_experimento(_config(), nombre="Momentum top 2")


def test_configuraciones_existentes_se_leen_listan_y_borran(tmp_path):
    v1 = _guardada()
    v2 = ConfiguracionGuardada(**{**v1.model_dump(), "version": 2, "deriva_de": 1, "notas": "segunda"})
    _poner(tmp_path / v1.id / "v1.json", v1.model_dump())
    _poner(tmp_path / v1.id / "v2.json", v2.model_dump())
    _poner(tmp_path / v1.id / "indice.json", {"ultima_version": 2, "archivada": False})
    almacen = AlmacenConfiguraciones(tmp_path)

    assert almacen.leer(v1.id).version == 2 and almacen.leer(v1.id, version=1).notas == ""
    assert [v.version for v in almacen.versiones(v1.id)] == [1, 2]
    assert [c.id for c in almacen.listar()] == [v1.id]
    almacen.archivar(v1.id)
    assert almacen.listar() == [] and [c.id for c in almacen.listar(archivadas=True)] == [v1.id]
    assert json.loads((tmp_path / v1.id / "indice.json").read_text(encoding="utf-8")) == {"ultima_version": 2, "archivada": True}
    nueva = almacen.nueva_version(v1.id, {"notas": "tercera"})
    assert nueva.version == 3 and (tmp_path / v1.id / "v3.json").is_file()
    assert {p.name for p in (tmp_path / v1.id).iterdir()} == {"v1.json", "v2.json", "v3.json", "indice.json"}
    assert almacen.borrar(v1.id) is True and almacen.borrar(v1.id) is False
    with pytest.raises(KeyError):
        almacen.leer(v1.id)
    with pytest.raises(ValueError):
        almacen.leer("../x")


# ---- marcos y comparativas -------------------------------------------------------------------


def _marco(**cambios) -> MarcoComparacion:
    base = {"id": "marco-prueba", "nombre": "Marco", "universo": TICKERS, "fecha_inicio": "2020-01-01",
            "fecha_fin": "2021-12-31", "frecuencia_rebalanceo": "monthly", "id_configuracion": "cfg-1"}
    return MarcoComparacion(**{**base, **cambios})


def test_marcos_y_comparativas_existentes_se_leen_listan_y_borran(tmp_path):
    marco = _marco()
    _poner(tmp_path / marco.id / "marco.json", marco.model_dump())
    comparativa = {"id": "cmp-0123456789", "id_marco": marco.id, "marco": marco.model_dump(),
                   "creada": "2026-11-02T10:00:00", "experimentos": []}
    _poner(tmp_path / "_comparativas" / "cmp-0123456789.json", comparativa)
    almacen = AlmacenMarcos(tmp_path)

    assert almacen.leer(marco.id).id == marco.id and [m.id for m in almacen.listar()] == [marco.id]
    assert almacen.leer_comparativa("cmp-0123456789") == comparativa
    assert almacen.listar_comparativas() == [comparativa] and almacen.listar_comparativas("otro") == []
    assert almacen.comparativas_de_configuracion("cfg-1") == [comparativa]
    assert almacen.comparativas_de_configuracion("otra") == []
    nueva = almacen.registrar_comparativa(marco, [{"id_experimento": "a-1", "estrategia": "x", "parametros": {},
                                                  "informe": {}, "reutilizado": False}])
    assert (tmp_path / "_comparativas" / f"{nueva['id']}.json").read_text(encoding="utf-8") == _json_canonico(nueva)
    assert {c["id"] for c in almacen.comparativas_de_configuracion("cfg-1")} == {"cmp-0123456789", nueva["id"]}
    assert almacen.borrar(marco.id) is True and almacen.borrar(marco.id) is False
    assert almacen.listar() == [] and len(almacen.listar_comparativas()) == 2  # las comparativas sobreviven al marco
    with pytest.raises(KeyError):
        almacen.leer(marco.id)


# ---- reservado -------------------------------------------------------------------------------


def test_reservado_existente_se_lee_y_se_amplia(tmp_path):
    uso = {"fecha": "2026-11-01T10:00:00", "barrido": "barrido-aaaaaaaaaa", "experimento": "momentum-bbbbbbbbbb"}
    _poner(tmp_path / "reservado.json", {"k1": {"universo": ["AAPL"], "fecha_fin_desarrollo": "2021-06-30",
                                                "fecha_fin": "2021-12-31", "usos": [uso]}})
    registro = RegistroReservado(tmp_path / "reservado.json")
    assert registro.usos("k1") == [uso] and registro.usos("nueva") == []
    registro.registrar("k1", ["AAPL"], "2021-06-30", "2021-12-31", "barrido-cccccccccc", "momentum-dddddddddd")
    contenido = json.loads((tmp_path / "reservado.json").read_text(encoding="utf-8"))
    assert len(contenido["k1"]["usos"]) == 2 and list(contenido["k1"]) == ["universo", "fecha_fin_desarrollo", "fecha_fin", "usos"]
