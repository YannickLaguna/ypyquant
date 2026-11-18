"""
Almacén de experimentos sobre Parquet + DuckDB.

Responsabilidad: persistir cada experimento en `data/experimentos/<id>/`:
    config.json      configuración (reproducible; su huella da el id)
    estado.json      pendiente | en_curso | hecho | error (+ traza, fechas)
    metricas.json    métricas finales (incluye `validacion`, vacío por ahora)
    equity.parquet   fecha, equity, benchmark, drawdown, retorno
    pesos.parquet    fecha, ticker, peso (pesos objetivo por rebalanceo)
    operaciones.parquet  fecha, ticker, lado, cantidad, precio, comision, pnl
y ofrecer `listar`, `leer`, `borrar`, `comparar` (DuckDB lee los Parquet de varios experimentos
a la vez) y `ejecutar_sql` (vistas `equity`, `pesos`, `operaciones` con la columna `id`).

Se relaciona con:
- ypyquant.estrategias.experimentos (ConfiguracionExperimento, ResultadoExperimento).
- app/api (el lanzador en segundo plano escribe el estado; las rutas leen y comparan).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    ResultadoExperimento,
    _limpiar,
)

RAIZ_REPO = Path(__file__).resolve().parents[3]
RAIZ_POR_DEFECTO = RAIZ_REPO / "data" / "experimentos"
ESTADOS = ("pendiente", "en_curso", "hecho", "error")
CLAVES_RESUMEN = ("retorno_total", "cagr", "volatilidad", "sharpe", "sortino", "calmar", "max_drawdown",
                  "turnover_anual", "alfa", "beta", "n_operaciones")


def _escribir_json(ruta: Path, contenido: Any) -> None:
    """Escritura atómica (archivo temporal + reemplazo): la API puede leer mientras otro hilo escribe."""
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json.dumps(_limpiar(contenido), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporal, ruta)


def _leer_json(ruta: Path) -> Any:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def _a_registros(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame a lista de dicts JSON-friendly (fechas ISO 'YYYY-MM-DD', NaN -> None)."""
    df = df.copy()
    for columna in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[columna]):
            df[columna] = df[columna].dt.strftime("%Y-%m-%d")
    return _limpiar(df.astype(object).where(df.notna(), None).to_dict("records"))


class AlmacenExperimentos:
    """
    Persistencia de experimentos en disco (Parquet + JSON) con consultas DuckDB.

    Args:
        raiz: carpeta del almacén (por defecto `data/experimentos/` del repo).
    """

    def __init__(self, raiz: Path | str | None = None):
        self.raiz = Path(raiz) if raiz else RAIZ_POR_DEFECTO
        self.raiz.mkdir(parents=True, exist_ok=True)
        self._cerrojo = threading.Lock()

    # ---- escritura -----------------------------------------------------------

    def _carpeta(self, id_experimento: str) -> Path:
        if not id_experimento or "/" in id_experimento or "\\" in id_experimento or id_experimento.startswith("."):
            raise ValueError(f"Identificador de experimento inválido: {id_experimento!r}")
        return self.raiz / id_experimento

    def registrar(self, config: ConfiguracionExperimento) -> str:
        """Crea la carpeta del experimento en estado `pendiente` (sobrescribe un intento anterior) y devuelve su id."""
        id_experimento = config.identificador()
        carpeta = self._carpeta(id_experimento)
        with self._cerrojo:
            if carpeta.exists():
                shutil.rmtree(carpeta)
            carpeta.mkdir(parents=True)
            _escribir_json(carpeta / "config.json", config.model_dump())
            _escribir_json(carpeta / "estado.json", {"estado": "pendiente", "creado": datetime.now().isoformat(),
                                                     "actualizado": datetime.now().isoformat(), "error": None})
        return id_experimento

    def marcar_estado(self, id_experimento: str, estado: str, error: BaseException | str | None = None) -> None:
        """Actualiza `estado.json`; con una excepción guarda su traza."""
        if estado not in ESTADOS:
            raise ValueError(f"Estado inválido: {estado}")
        carpeta = self._carpeta(id_experimento)
        actual = _leer_json(carpeta / "estado.json") or {"creado": datetime.now().isoformat()}
        traza = None
        if isinstance(error, BaseException):
            traza = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        elif error:
            traza = str(error)
        mensaje = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else (str(error) if error else None)
        actual.update({"estado": estado, "actualizado": datetime.now().isoformat(), "error": mensaje, "traza": traza})
        _escribir_json(carpeta / "estado.json", actual)

    def guardar(self, id_experimento: str, resultado: ResultadoExperimento) -> None:
        """Escribe métricas y Parquet del resultado y marca el experimento como `hecho`."""
        carpeta = self._carpeta(id_experimento)
        carpeta.mkdir(parents=True, exist_ok=True)
        resultado.equity.to_parquet(carpeta / "equity.parquet", index=False)
        resultado.pesos.to_parquet(carpeta / "pesos.parquet", index=False)
        resultado.operaciones.to_parquet(carpeta / "operaciones.parquet", index=False)
        _escribir_json(carpeta / "metricas.json", {**resultado.metricas, "avisos": resultado.avisos})
        self.marcar_estado(id_experimento, "hecho")

    def borrar(self, id_experimento: str) -> bool:
        """Elimina el experimento; False si no existía."""
        carpeta = self._carpeta(id_experimento)
        if not carpeta.exists():
            return False
        shutil.rmtree(carpeta)
        return True

    def marcar_interrumpidos(self) -> int:
        """Pasa a `error` los experimentos que quedaron `pendiente` o `en_curso` (el proceso anterior murió)."""
        cuenta = 0
        for carpeta in self.raiz.iterdir():
            estado = _leer_json(carpeta / "estado.json") if carpeta.is_dir() else None
            if estado and estado["estado"] in ("pendiente", "en_curso"):
                self.marcar_estado(carpeta.name, "error", "Interrumpido: el servidor se reinició durante la ejecución")
                cuenta += 1
        return cuenta

    # ---- lectura -------------------------------------------------------------

    def existe(self, id_experimento: str) -> bool:
        return (self._carpeta(id_experimento) / "estado.json").exists()

    def _resumen(self, carpeta: Path) -> dict[str, Any] | None:
        config, estado = _leer_json(carpeta / "config.json"), _leer_json(carpeta / "estado.json")
        if config is None or estado is None:
            return None
        metricas = _leer_json(carpeta / "metricas.json") or {}
        return {
            "id": carpeta.name, "estrategia": config["estrategia"], "parametros": config["parametros"],
            "universo": config["universo"], "fecha_inicio": config["fecha_inicio"], "fecha_fin": config["fecha_fin"],
            "benchmark": config["benchmark"], "frecuencia_rebalanceo": config["frecuencia_rebalanceo"],
            "etiqueta": config.get("etiqueta", ""), "estado": estado["estado"], "error": estado.get("error"),
            "creado": estado.get("creado"), "actualizado": estado.get("actualizado"),
            "metricas": {k: metricas.get(k) for k in CLAVES_RESUMEN} if metricas else {},
        }

    def listar(self, estrategia: str | None = None, estado: str | None = None,
               texto: str | None = None) -> list[dict[str, Any]]:
        """Resúmenes (más recientes primero), filtrables por estrategia, estado y texto libre (id, etiqueta, universo)."""
        resumenes = []
        for carpeta in self.raiz.iterdir():
            if not carpeta.is_dir():
                continue
            resumen = self._resumen(carpeta)
            if resumen is None:
                continue
            if estrategia and resumen["estrategia"] != estrategia:
                continue
            if estado and resumen["estado"] != estado:
                continue
            if texto and texto.lower() not in json.dumps([resumen["id"], resumen["etiqueta"], resumen["universo"]]).lower():
                continue
            resumenes.append(resumen)
        return sorted(resumenes, key=lambda r: r["creado"] or "", reverse=True)

    def leer(self, id_experimento: str) -> dict[str, Any]:
        """Experimento completo: resumen, configuración, estado, métricas y series (equity, pesos, operaciones)."""
        carpeta = self._carpeta(id_experimento)
        resumen = self._resumen(carpeta)
        if resumen is None:
            raise KeyError(id_experimento)
        detalle: dict[str, Any] = {
            **resumen, "config": _leer_json(carpeta / "config.json"),
            "traza": (_leer_json(carpeta / "estado.json") or {}).get("traza"),
            "metricas_completas": _leer_json(carpeta / "metricas.json"),
            "equity": [], "pesos": [], "operaciones": [],
        }
        for clave in ("equity", "pesos", "operaciones"):
            ruta = carpeta / f"{clave}.parquet"
            if ruta.exists():
                detalle[clave] = _a_registros(pd.read_parquet(ruta))
        return detalle

    # ---- DuckDB --------------------------------------------------------------

    def _rutas_parquet(self, nombre: str, ids: list[str] | None = None) -> list[str]:
        carpetas = [self._carpeta(i) for i in ids] if ids is not None else sorted(p for p in self.raiz.iterdir() if p.is_dir())
        return [str(c / f"{nombre}.parquet").replace("\\", "/") for c in carpetas if (c / f"{nombre}.parquet").exists()]

    def comparar(self, ids: list[str]) -> dict[str, Any]:
        """
        Compara experimentos terminados con DuckDB: equity normalizada a base 100 sobre el eje de
        fechas común (intersección), benchmark del primero y tabla de métricas lado a lado.
        """
        validos = [i for i in ids if (self._carpeta(i) / "equity.parquet").exists()]
        if not validos:
            raise KeyError("Ninguno de los experimentos pedidos tiene resultados")
        consulta = " UNION ALL ".join(
            f"SELECT '{i}' AS id, fecha, equity / FIRST_VALUE(equity) OVER (ORDER BY fecha) * 100 AS valor, "
            f"benchmark / FIRST_VALUE(benchmark) OVER (ORDER BY fecha) * 100 AS bench "
            f"FROM read_parquet('{self._rutas_parquet('equity', [i])[0]}')" for i in validos
        )
        con = duckdb.connect()
        try:
            largo = con.execute(
                f"WITH t AS ({consulta}), comunes AS (SELECT fecha FROM t GROUP BY fecha HAVING COUNT(*) = {len(validos)}) "
                "SELECT t.* FROM t JOIN comunes USING (fecha) ORDER BY id, fecha"
            ).df()
        finally:
            con.close()
        fechas = sorted(largo["fecha"].unique())
        series = {i: largo[largo["id"] == i].sort_values("fecha")["valor"].tolist() for i in validos}
        bench = largo[largo["id"] == validos[0]].sort_values("fecha")["bench"].tolist()
        return _limpiar({
            "fechas": [pd.Timestamp(f).strftime("%Y-%m-%d") for f in fechas],
            "series": series, "benchmark": bench, "ids": validos,
            "metricas": {i: (_leer_json(self._carpeta(i) / "metricas.json") or {}) for i in validos},
            "resumenes": {i: self._resumen(self._carpeta(i)) for i in validos},
        })

    def ejecutar_sql(self, sql: str) -> pd.DataFrame:
        """
        Consulta DuckDB de solo lectura sobre todos los experimentos: vistas `equity`, `pesos` y
        `operaciones` (cada una con la columna `id`). Base para validaciones y análisis futuros.
        """
        con = duckdb.connect()
        try:
            for nombre in ("equity", "pesos", "operaciones"):
                rutas = self._rutas_parquet(nombre)
                if rutas:
                    lista = ", ".join(f"'{r}'" for r in rutas)
                    con.execute(
                        f"CREATE VIEW {nombre} AS SELECT regexp_extract(filename, '([^/]+)/{nombre}\\.parquet$', 1) AS id, * "
                        f"FROM read_parquet([{lista}], filename = true)"
                    )
            return con.execute(sql).df()
        finally:
            con.close()
