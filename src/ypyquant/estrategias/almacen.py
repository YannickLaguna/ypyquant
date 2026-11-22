"""
Almacén de experimentos sobre Parquet + DuckDB.

Responsabilidad: persistir cada experimento en `data/experimentos/<id>/`:
    config.json      configuración (reproducible; su huella da el id)
    estado.json      pendiente | en_curso | hecho | error (+ traza, fechas)
    metricas.json    métricas finales (incluye `validacion`: salida de los validadores pedidos)
    procedencia.json commit git, versiones, hash de los precios y fecha de ejecución
    equity.parquet   fecha, equity, benchmark, drawdown, retorno
    pesos.parquet    fecha, ticker, peso (pesos objetivo por rebalanceo)
    operaciones.parquet  fecha, ticker, lado, cantidad, precio, comision, pnl
    rotaciones.parquet   fecha, rotacion (lo operado en cada rebalanceo; permite `reevaluar` sin repetir el backtest)
y ofrecer `listar`, `leer`, `borrar`, `comparar` (DuckDB lee los Parquet de varios experimentos
a la vez), `ejecutar_sql` (vistas `equity`, `pesos`, `operaciones` con la columna `id`) y `reevaluar`
(recalcula `metricas.json` desde los Parquet con `evaluacion.reevaluar`, sin repetir el backtest).

Es un adaptador de dominio fino sobre `ypyquant.almacen_documentos.AlmacenDocumentos` (carpeta por id, JSON y Parquet
atómicos, ids, borrado): aquí solo viven el formato de cada archivo, los estados, los resúmenes y las consultas DuckDB.

Se relaciona con:
- ypyquant.almacen_documentos (persistencia por id) y ypyquant.raiz_datos (dónde vive `data/experimentos`).
- ypyquant.estrategias.indice_recetas (índice en memoria receta -> experimentos: `indexar`, `resumenes_de_receta`),
  .experimentos (ConfiguracionExperimento, ResultadoExperimento), .evaluacion (`reevaluar`) y
  .procedencia (qué contiene `procedencia.json`).
- ypyquant.estrategias.barridos (`AlmacenBarridos` lee las métricas y `equity.parquet` de aquí).
- app/api (el lanzador en segundo plano escribe el estado; las rutas leen y comparan).
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ypyquant.almacen_documentos import AlmacenDocumentos, id_sin_separadores, limpiar_para_json
from ypyquant.estrategias.evaluacion import desglose_por_motivo
from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    ResultadoExperimento,
)
from ypyquant.estrategias.indice_recetas import IndiceRecetas
from ypyquant.raiz_datos import RaizDatos

ESTADOS = ("pendiente", "en_curso", "hecho", "error")
CLAVES_RESUMEN = ("retorno_total", "cagr", "volatilidad", "sharpe", "sortino", "calmar", "max_drawdown",
                  "turnover_anual", "alfa", "beta", "n_operaciones")


def _a_registros(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame a lista de dicts JSON-friendly (fechas ISO 'YYYY-MM-DD', NaN -> None)."""
    df = df.copy()
    for columna in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[columna]):
            df[columna] = df[columna].dt.strftime("%Y-%m-%d")
    return limpiar_para_json(df.astype(object).where(df.notna(), None).to_dict("records"))


class AlmacenExperimentos:
    """
    Persistencia de experimentos en disco (Parquet + JSON) con consultas DuckDB.

    Adaptador de dominio sobre un `AlmacenDocumentos` (`self.documentos`): el id del experimento es el de su carpeta.

    Args:
        raiz: carpeta del almacén (por defecto `data/experimentos/` del repo, según `RaizDatos`).
    """

    def __init__(self, raiz: Path | str | None = None):
        self.documentos = AlmacenDocumentos(raiz or RaizDatos().experimentos, nombre="experimento",
                                            validar_id=id_sin_separadores)
        self.raiz = self.documentos.raiz
        self._indice = IndiceRecetas()  # receta -> ids; se construye en la primera consulta (o con `indexar`)

    # ---- escritura -----------------------------------------------------------

    def registrar(self, config: ConfiguracionExperimento) -> str:
        """Crea la carpeta del experimento en estado `pendiente` (sobrescribe un intento anterior) y devuelve su id."""
        id_experimento = config.identificador()
        with self.documentos.cerrojo:
            self.documentos.crear(id_experimento, reemplazar=True)
            self.documentos.escribir_json(id_experimento, "config", config.model_dump())
            self.documentos.escribir_json(id_experimento, "estado", {
                "estado": "pendiente", "creado": datetime.now().isoformat(),
                "actualizado": datetime.now().isoformat(), "error": None})
        self._indice.anadir(id_experimento, config.model_dump())  # también sin construir: una construcción en curso lo vería
        return id_experimento

    def marcar_estado(self, id_experimento: str, estado: str, error: BaseException | str | None = None) -> None:
        """Actualiza `estado.json`; con una excepción guarda su traza."""
        if estado not in ESTADOS:
            raise ValueError(f"Estado inválido: {estado}")
        actual = self.documentos.leer_json(id_experimento, "estado") or {"creado": datetime.now().isoformat()}
        traza = None
        if isinstance(error, BaseException):
            traza = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        elif error:
            traza = str(error)
        mensaje = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else (str(error) if error else None)
        actual.update({"estado": estado, "actualizado": datetime.now().isoformat(), "error": mensaje, "traza": traza})
        self.documentos.escribir_json(id_experimento, "estado", actual)

    def guardar(self, id_experimento: str, resultado: ResultadoExperimento) -> None:
        """Escribe métricas y Parquet del resultado y marca el experimento como `hecho`."""
        self.documentos.crear(id_experimento)
        self.documentos.escribir_parquet(id_experimento, "equity", resultado.equity)
        self.documentos.escribir_parquet(id_experimento, "pesos", resultado.pesos)
        self.documentos.escribir_parquet(id_experimento, "operaciones", resultado.operaciones)
        if resultado.rotaciones is not None:
            self.documentos.escribir_parquet(id_experimento, "rotaciones", resultado.rotaciones)
        self.documentos.escribir_json(id_experimento, "metricas", {**resultado.metricas, "avisos": resultado.avisos})
        self.documentos.escribir_json(id_experimento, "procedencia", resultado.procedencia)
        self.marcar_estado(id_experimento, "hecho")

    def reevaluar(self, id_experimento: str, guardar: bool = True) -> dict[str, Any]:
        """
        Recalcula las métricas de un experimento guardado solo con sus Parquet (equity, pesos, operaciones y,
        si existe, rotaciones), sin repetir el backtest; devuelve las métricas nuevas.

        Con `guardar=True` reescribe `metricas.json` conservando sus `avisos`; la procedencia y los Parquet no se
        tocan. Lanza `ValueError` si el experimento no está `hecho`, es una importación del TFM (sus métricas no
        salen de operaciones) o le faltan tablas.
        """
        from ypyquant.estrategias.evaluacion import reevaluar  # diferido: evaluacion importa de experimentos

        if self.estado_de(id_experimento) != "hecho":
            raise ValueError(f"El experimento {id_experimento} no está terminado")
        procedencia = self.documentos.leer_json(id_experimento, "procedencia") or {}
        if procedencia.get("origen"):
            raise ValueError(f"El experimento {id_experimento} es una importación ({procedencia['origen']}): no se reevalúa")
        config, anteriores = self.leer_config(id_experimento), self.leer_metricas(id_experimento) or {}
        nombres = ("equity", "pesos", "operaciones")
        if config is None or not all(self.documentos.tiene_parquet(id_experimento, n) for n in nombres):
            raise ValueError(f"Al experimento {id_experimento} le faltan config o tablas Parquet para reevaluarlo")
        evaluacion = reevaluar(
            ConfiguracionExperimento(**config), *(self.documentos.leer_parquet(id_experimento, n) for n in nombres),
            rotaciones=self.documentos.leer_parquet(id_experimento, "rotaciones"),
            avisos=anteriores.get("avisos", []),
        )
        if guardar:
            self.documentos.escribir_json(id_experimento, "metricas",
                                          {**evaluacion.metricas, "avisos": anteriores.get("avisos", [])})
        return evaluacion.metricas

    def borrar(self, id_experimento: str) -> bool:
        """Elimina el experimento; False si no existía."""
        if not self.documentos.borrar(id_experimento):
            return False
        self._indice.quitar(id_experimento)
        return True

    def marcar_interrumpidos(self) -> int:
        """Pasa a `error` los experimentos que quedaron `pendiente` o `en_curso` (el proceso anterior murió)."""
        cuenta = 0
        for id_experimento in self.documentos.ids():
            estado = self.documentos.leer_json(id_experimento, "estado")
            if estado and estado["estado"] in ("pendiente", "en_curso"):
                self.marcar_estado(id_experimento, "error", "Interrumpido: el servidor se reinició durante la ejecución")
                cuenta += 1
        return cuenta

    # ---- lectura -------------------------------------------------------------

    def existe(self, id_experimento: str) -> bool:
        return self.documentos.tiene_json(id_experimento, "estado")

    def indexar(self) -> None:
        """(Re)construye el índice por receta leyendo los `config.json` del disco (una vez al arrancar basta)."""
        configs = ((i, config) for i in self.documentos.ids()
                   if (config := self.documentos.leer_json(i, "config")) is not None)
        self._indice.construir(configs)

    def resumenes_de_receta(self, huella_receta: str) -> list[dict[str, Any]]:
        """
        Resúmenes (más recientes primero) de los experimentos de una receta (`ConfiguracionExperimento.huella_receta`),
        sin recorrer el almacén: usa el índice en memoria (ver `indice_recetas.py`) y solo lee esos experimentos.
        """
        if not self._indice.construido:
            self.indexar()
        resumenes = []
        for id_experimento in self._indice.ids(huella_receta):
            resumen = self._resumen(id_experimento) if self.documentos.hay_carpeta(id_experimento) else None
            if resumen is not None:
                resumenes.append(resumen)
        return sorted(resumenes, key=lambda r: r["creado"] or "", reverse=True)

    def estado_de(self, id_experimento: str) -> str | None:
        """Estado (`pendiente|en_curso|hecho|error`) del experimento; None si no existe (lectura ligera)."""
        estado = self.documentos.leer_json(id_experimento, "estado")
        return estado["estado"] if estado else None

    def leer_config(self, id_experimento: str) -> dict[str, Any] | None:
        """`config.json` del experimento (None si no existe): la configuración tal como se guardó, sin series."""
        return self.documentos.leer_json(id_experimento, "config")

    def leer_metricas(self, id_experimento: str) -> dict[str, Any] | None:
        """Contenido de `metricas.json` (None si el experimento no ha terminado)."""
        return self.documentos.leer_json(id_experimento, "metricas")

    def leer_equity(self, id_experimento: str) -> pd.DataFrame | None:
        """DataFrame de `equity.parquet` (`fecha, equity, benchmark, drawdown, retorno`); None si no existe."""
        return self.documentos.leer_parquet(id_experimento, "equity")

    def leer_pesos(self, id_experimento: str) -> pd.DataFrame | None:
        """DataFrame de `pesos.parquet` (`fecha, ticker, peso`); None si no existe (p. ej. experimentos del TFM)."""
        return self.documentos.leer_parquet(id_experimento, "pesos")

    def _resumen(self, id_experimento: str) -> dict[str, Any] | None:
        config = self.documentos.leer_json(id_experimento, "config")
        estado = self.documentos.leer_json(id_experimento, "estado", reintentar_si_falta=config is not None)
        if config is None or estado is None:
            return None
        metricas = self.documentos.leer_json(id_experimento, "metricas") or {}
        return {
            "id": id_experimento, "estrategia": config["estrategia"], "parametros": config["parametros"],
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
        for id_experimento in self.documentos.ids():
            resumen = self._resumen(id_experimento)
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
        """Experimento completo: resumen, configuración, estado, métricas, procedencia y series (equity, pesos, operaciones)."""
        resumen = self._resumen(id_experimento)
        if resumen is None:
            raise KeyError(id_experimento)
        detalle: dict[str, Any] = {
            **resumen, "config": self.documentos.leer_json(id_experimento, "config"),
            "traza": (self.documentos.leer_json(id_experimento, "estado") or {}).get("traza"),
            "metricas_completas": self.documentos.leer_json(id_experimento, "metricas"),
            "procedencia": self.documentos.leer_json(id_experimento, "procedencia"),
            "equity": [], "pesos": [], "operaciones": [],
        }
        for clave in ("equity", "pesos", "operaciones"):
            # Las series solo son definitivas cuando el experimento está `hecho` (se escriben antes)
            if resumen["estado"] == "hecho" and self.documentos.tiene_parquet(id_experimento, clave):
                detalle[clave] = _a_registros(self.documentos.leer_parquet(id_experimento, clave))
        metricas = detalle["metricas_completas"]
        if isinstance(metricas, dict) and "motivos_salida" not in metricas and detalle["operaciones"]                 and "motivo" in detalle["operaciones"][0]:
            # Estrategia de señales guardada antes de esta métrica: se calcula desde sus operaciones (no se reescribe el archivo)
            metricas["motivos_salida"] = desglose_por_motivo(pd.DataFrame(detalle["operaciones"]).assign(
                fecha=lambda t: pd.to_datetime(t["fecha"])))
        return detalle

    # ---- DuckDB --------------------------------------------------------------

    def _rutas_parquet(self, nombre: str, ids: list[str] | None = None) -> list[str]:
        ids = self.documentos.ids() if ids is None else ids
        return [str(self.documentos.ruta_parquet(i, nombre)).replace("\\", "/") for i in ids
                if self.documentos.tiene_parquet(i, nombre)]

    def comparar(self, ids: list[str]) -> dict[str, Any]:
        """
        Compara experimentos terminados con DuckDB: equity normalizada a base 100 sobre el eje de
        fechas común (intersección), benchmark del primero y tabla de métricas lado a lado.
        """
        validos = [i for i in ids if self.documentos.tiene_parquet(i, "equity")]
        if not validos:
            raise KeyError("Ninguno de los experimentos pedidos tiene resultados")
        # Se rebasa a 100 en la primera fecha COMÚN (no en la primera de cada serie): así series con fechas
        # de inicio distintas (p. ej. un experimento TFM importado desde 2009-12 y un backtest que empieza
        # más tarde) arrancan juntas en 100 sobre el eje compartido.
        consulta = " UNION ALL ".join(
            f"SELECT '{i}' AS id, fecha, equity, benchmark "
            f"FROM read_parquet('{self._rutas_parquet('equity', [i])[0]}')" for i in validos
        )
        con = duckdb.connect()
        try:
            largo = con.execute(
                f"WITH t AS ({consulta}), comunes AS (SELECT fecha FROM t GROUP BY fecha HAVING COUNT(*) = {len(validos)}) "
                "SELECT t.id, t.fecha, "
                "t.equity / FIRST_VALUE(t.equity) OVER (PARTITION BY t.id ORDER BY t.fecha) * 100 AS valor, "
                "t.benchmark / FIRST_VALUE(t.benchmark) OVER (PARTITION BY t.id ORDER BY t.fecha) * 100 AS bench "
                "FROM t JOIN comunes USING (fecha) ORDER BY t.id, t.fecha"
            ).df()
        finally:
            con.close()
        fechas = sorted(largo["fecha"].unique())
        series = {i: largo[largo["id"] == i].sort_values("fecha")["valor"].tolist() for i in validos}
        bench = largo[largo["id"] == validos[0]].sort_values("fecha")["bench"].tolist()
        return limpiar_para_json({
            "fechas": [pd.Timestamp(f).strftime("%Y-%m-%d") for f in fechas],
            "series": series, "benchmark": bench, "ids": validos,
            "metricas": {i: (self.documentos.leer_json(i, "metricas") or {}) for i in validos},
            "resumenes": {i: self._resumen(i) for i in validos},
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
                        f"FROM read_parquet([{lista}], filename = true, union_by_name = true)"
                    )
            return con.execute(sql).df()
        finally:
            con.close()
