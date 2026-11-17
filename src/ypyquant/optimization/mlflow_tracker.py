"""
MLflow tracker para experimentos walk-forward.
==============================================

Diseño:
- enabled=False (default) → todos los métodos son no-ops, sin dependencia de mlflow.
- enabled=True → requiere mlflow>=2.10 instalado.
- Un MLflow run  = un ConfigExperimento.
- Un MLflow experiment = todos los 21 experimentos ("walk_forward_factorial").

Tracking URI: file:///[PROJECT_ROOT]/data/mlruns  (Fase 3 lo migra a servidor).
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ypyquant.optimization.walk_forward import (
        ConfigExperimento,
        ResultadoExperimento,
        ResultadoVentana,
    )


class MLflowTracker:
    """
    Wrapper de MLflow para tracking de walk-forward.

    Si enabled=False (default), todos los métodos son no-ops — cero overhead,
    sin importar mlflow. Walk-forward funciona igual sin la librería.

    Uso típico::

        tracker = MLflowTracker(enabled=True, tracking_uri="file:///data/mlruns")
        tracker.start_run(config_exp)
        for i, ventana in enumerate(ventanas):
            tracker.log_window(ventana, step=i)
        tracker.end_run(resultado)
    """

    def __init__(
        self,
        enabled: bool = False,
        experiment_name: str = "walk_forward_factorial",
        tracking_uri: str | None = None,
    ):
        """
        Args:
            enabled: Si False, todos los métodos son no-ops.
            experiment_name: Nombre del experimento MLflow (agrupa todos los runs).
            tracking_uri: URI del tracking server. None → MLflow default (~/.mlruns).
                          Usar "file:///ruta/absoluta/data/mlruns" para almacenamiento local.
        """
        self.enabled = enabled
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._run = None
        self._mlflow = None

        if enabled:
            try:
                import mlflow as _mlflow
                self._mlflow = _mlflow
                if tracking_uri:
                    _mlflow.set_tracking_uri(tracking_uri)
                _mlflow.set_experiment(experiment_name)
                logger.info(
                    f"[MLflow] Tracking habilitado — experimento: '{experiment_name}' | "
                    f"URI: {_mlflow.get_tracking_uri()}"
                )
            except ImportError:
                logger.warning(
                    "[MLflow] mlflow no instalado — tracking deshabilitado. "
                    "Instala con: pip install 'mlflow>=2.10'"
                )
                self.enabled = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_run(self, config_exp: ConfigExperimento) -> None:
        """
        Inicia un run y loguea todos los campos de ConfigExperimento como params.

        Un run = un experimento del diseño factorial.
        """
        if not self.enabled or self._mlflow is None:
            return
        try:
            self._run = self._mlflow.start_run(run_name=config_exp.nombre)
            self._mlflow.log_params({
                "id_experimento":  config_exp.id_experimento,
                "nombre":          config_exp.nombre,
                "tipo":            config_exp.tipo.value,
                "metodo_sigma":    config_exp.metodo_sigma.value,
                "metodo_mu":       config_exp.metodo_mu.value,
                "metodo_riesgo":   config_exp.metodo_riesgo.value,
                "regularizacion":  config_exp.regularizacion.value,
                "metodo_bl":       config_exp.metodo_bl.value,
            })
        except Exception as exc:
            logger.warning(f"[MLflow] start_run error: {exc}")

    def log_window(self, resultado: ResultadoVentana, step: int) -> None:
        """
        Loguea métricas de una ventana OOS individual.

        Args:
            resultado: ResultadoVentana del mes procesado.
            step: Índice de la ventana (0-based).
        """
        if not self.enabled or self._mlflow is None:
            return
        try:
            ret = resultado.retorno_oos
            if ret is not None and not math.isnan(ret):
                self._mlflow.log_metric("retorno_oos", ret, step=step)
            self._mlflow.log_metric("tiempo_segundos", resultado.tiempo_segundos, step=step)
            self._mlflow.log_metric("n_tickers", len(resultado.tickers_seleccionados), step=step)
            self._mlflow.log_metric("convergente", int(resultado.optimizacion_convergente), step=step)
        except Exception as exc:
            logger.warning(f"[MLflow] log_window(step={step}) error: {exc}")

    def end_run(self, resultado: ResultadoExperimento) -> None:
        """
        Loguea métricas agregadas + equity curve como artifact CSV y cierra el run.

        Args:
            resultado: ResultadoExperimento finalizado.
        """
        if not self.enabled or self._mlflow is None:
            return
        try:
            # -- Métricas agregadas --
            metricas = {
                "sharpe_ratio":      resultado.metricas.get("sharpe_ratio"),
                "sortino_ratio":     resultado.metricas.get("sortino_ratio"),
                "retorno_anualizado": resultado.metricas.get("retorno_anualizado"),
                "volatilidad":       resultado.metricas.get("volatilidad"),
                "max_drawdown":      resultado.metricas.get("max_drawdown"),
                "cvar_95":           resultado.metricas.get("cvar_95"),
                "calmar_ratio":      resultado.metricas.get("calmar_ratio"),
                "alpha_jensen":      resultado.alpha_jensen,
                "alpha_pvalue":      resultado.alpha_pvalue,
            }
            for nombre, valor in metricas.items():
                if valor is not None and not (isinstance(valor, float) and math.isnan(valor)):
                    self._mlflow.log_metric(nombre, float(valor))

            # -- Equity curve como artifact CSV --
            if resultado.curva_equity is not None and len(resultado.curva_equity) > 0:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    suffix=f"_{resultado.config.id_experimento}_equity.csv"
                )
                try:
                    with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                        resultado.curva_equity.to_csv(f, header=True)
                    self._mlflow.log_artifact(
                        tmp_path,
                        artifact_path=f"equity_{resultado.config.id_experimento}",
                    )
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            self._mlflow.end_run()
            self._run = None

        except Exception as exc:
            logger.warning(f"[MLflow] end_run error: {exc}")
            try:
                self._mlflow.end_run()
            except Exception:
                pass
            self._run = None
