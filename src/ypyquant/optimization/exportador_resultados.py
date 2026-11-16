"""
Exportador de resultados del experimento factorial.
====================================================

Responsabilidad:
- Exportar resultados de walk-forward a CSV, LaTeX, gráficos PNG y pickle.
- Generar equity curves comparativas, heatmaps de factores y distribuciones bootstrap.
- Producir tablas de Jensen's alpha con significancia estadística.

Se relaciona con:
- walk_forward.py: ResultadoExperimento, WalkForwardEngine, TipoExperimento, enums.
- matplotlib/seaborn: generación de gráficos estáticos.
- pandas: tablas comparativas y exportación a CSV/LaTeX.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ypyquant.optimization.walk_forward import (
    ResultadoExperimento,
    TipoExperimento,
    WalkForwardEngine,
)

logger = logging.getLogger(__name__)


class ExportadorResultados:
    """
    Exporta los resultados del experimento factorial a múltiples formatos.

    Genera:
    - CSV con métricas y retornos OOS mensuales.
    - Tablas LaTeX por etapa y de Jensen's alpha.
    - Gráficos PNG: equity curves, heatmaps de factores, violin plots bootstrap.
    - Pickle con los objetos ResultadoExperimento completos.
    """

    def __init__(
        self,
        directorio_salida,
        resultados,
    ):
        """
        Args:
            directorio_salida: Ruta al directorio base de reportes (str o Path).
            resultados: Diccionario {id_experimento: ResultadoExperimento}
                        o lista de ResultadoExperimento (se convierte automáticamente).
        """
        self.directorio_salida = Path(directorio_salida)
        if isinstance(resultados, list):
            self.resultados = {r.config.id_experimento: r for r in resultados}
        else:
            self.resultados = resultados

        # Subdirectorios de salida
        self.dir_latex = self.directorio_salida / "tablas_latex"
        self.dir_graficos = self.directorio_salida / "graficos"

        # Crear directorios necesarios
        self.directorio_salida.mkdir(parents=True, exist_ok=True)
        self.dir_latex.mkdir(parents=True, exist_ok=True)
        self.dir_graficos.mkdir(parents=True, exist_ok=True)

        # Clasificar resultados por etapa
        self._etapa1 = {k: v for k, v in self.resultados.items()
                        if v.config.tipo == TipoExperimento.ETAPA_1}
        self._etapa2a = {k: v for k, v in self.resultados.items()
                         if v.config.tipo == TipoExperimento.ETAPA_2A}
        self._etapa2b = {k: v for k, v in self.resultados.items()
                         if v.config.tipo == TipoExperimento.ETAPA_2B}
        self._benchmarks = {k: v for k, v in self.resultados.items()
                            if v.config.tipo in (
                                TipoExperimento.BENCHMARK_EQUAL,
                                TipoExperimento.BENCHMARK_SP500,
                            )}

    # ------------------------------------------------------------------
    # Exportación completa
    # ------------------------------------------------------------------

    def exportar_todo(self) -> None:
        """Ejecuta todas las exportaciones: CSV, LaTeX, gráficos y pickle."""
        logger.info("Iniciando exportación completa de resultados")
        self.exportar_csv()
        self.exportar_latex()
        self.exportar_graficos()
        self.exportar_pickle()
        logger.info(f"Exportación completa en {self.directorio_salida}")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def exportar_csv(self) -> None:
        """Exporta tabla resumen y retornos OOS mensuales a CSV."""
        # Tabla resumen con métricas
        df_resumen = WalkForwardEngine.generar_tabla_comparativa(self.resultados)
        ruta_resumen = self.directorio_salida / "resultados_factorial.csv"
        df_resumen.to_csv(ruta_resumen, index=False)
        logger.info(f"CSV resumen: {ruta_resumen}")

        # Retornos OOS mensuales de cada experimento
        retornos_dict = {}
        for id_exp, res in self.resultados.items():
            if res.retornos_oos_mensuales is not None:
                retornos_dict[res.config.nombre] = res.retornos_oos_mensuales

        if retornos_dict:
            df_retornos = pd.DataFrame(retornos_dict)
            ruta_retornos = self.directorio_salida / "retornos_oos_mensuales.csv"
            df_retornos.to_csv(ruta_retornos)
            logger.info(f"CSV retornos OOS: {ruta_retornos}")

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------

    def exportar_latex(self) -> None:
        """Genera tablas LaTeX por etapa, Jensen's alpha y resumen global."""
        columnas_metricas = [
            'nombre', 'retorno_anualizado', 'volatilidad', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'cvar_95',
        ]
        renombrar = {
            'nombre': 'Experimento',
            'retorno_anualizado': 'Ret. Anual.',
            'volatilidad': 'Vol. Anual.',
            'sharpe_ratio': 'Sharpe',
            'sortino_ratio': 'Sortino',
            'max_drawdown': 'Max DD',
            'cvar_95': 'CVaR 95\\%',
        }

        # --- Tabla por etapa ---
        etapas = [
            (self._etapa1, "etapa1", "Evaluación del estimador de covarianza (Etapa 1)"),
            (self._etapa2a, "etapa2a", "Marco frecuentista (Etapa 2a)"),
            (self._etapa2b, "etapa2b", "Marco bayesiano Black-Litterman (Etapa 2b)"),
        ]
        for datos_etapa, nombre_archivo, caption in etapas:
            if not datos_etapa:
                continue
            tabla = WalkForwardEngine.generar_tabla_comparativa(datos_etapa)
            cols_existentes = [c for c in columnas_metricas if c in tabla.columns]
            tabla_display = tabla[cols_existentes].rename(columns=renombrar)
            latex = tabla_display.to_latex(
                index=False, float_format='%.4f',
                caption=caption,
                label=f'tab:{nombre_archivo}',
            )
            ruta = self.dir_latex / f"{nombre_archivo}.tex"
            ruta.write_text(latex, encoding='utf-8')
            logger.info(f"LaTeX {nombre_archivo}: {ruta}")

        # --- Tabla Jensen's alpha ---
        df_alpha = self._tabla_jensen_alpha()
        if df_alpha is not None:
            latex_alpha = df_alpha.to_latex(
                index=False, float_format='%.4f',
                caption="Jensen's Alpha por experimento",
                label='tab:jensen_alpha',
            )
            ruta_alpha = self.dir_latex / "jensen_alpha.tex"
            ruta_alpha.write_text(latex_alpha, encoding='utf-8')
            logger.info(f"LaTeX Jensen's alpha: {ruta_alpha}")

        # --- Tabla resumen global ---
        tabla_global = WalkForwardEngine.generar_tabla_comparativa(self.resultados)
        cols_global = [c for c in columnas_metricas + ['alpha_jensen', 'alpha_pvalue']
                       if c in tabla_global.columns]
        renombrar_global = {**renombrar, 'alpha_jensen': "Jensen's α", 'alpha_pvalue': 'p-value'}
        tabla_global_display = tabla_global[cols_global].rename(columns=renombrar_global)
        latex_global = tabla_global_display.to_latex(
            index=False, float_format='%.4f',
            caption='Resumen global de todos los experimentos',
            label='tab:resumen_global',
        )
        ruta_global = self.dir_latex / "resumen_global.tex"
        ruta_global.write_text(latex_global, encoding='utf-8')
        logger.info(f"LaTeX resumen global: {ruta_global}")

    def _tabla_jensen_alpha(self) -> Optional[pd.DataFrame]:
        """Construye tabla de Jensen's alpha con significancia estadística."""
        filas = []
        for id_exp, res in self.resultados.items():
            filas.append({
                'Experimento': res.config.nombre,
                'Etapa': res.config.tipo.value,
                "Jensen's α": res.alpha_jensen,
                'β (mercado)': res.metricas.get('beta'),
                't-stat': res.alpha_tstat,
                'p-value': res.alpha_pvalue,
                'Signif.': (
                    '***' if res.alpha_pvalue is not None and res.alpha_pvalue < 0.05
                    else '**' if res.alpha_pvalue is not None and res.alpha_pvalue < 0.10
                    else ''
                ),
            })
        if not filas:
            return None
        df = pd.DataFrame(filas)
        df = df.sort_values("Jensen's α", ascending=False, na_position='last')
        return df

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------

    def exportar_graficos(self) -> None:
        """Genera todos los gráficos PNG."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (14, 7)
        plt.rcParams['font.size'] = 12

        for nombre, fn in [
            ("equity_curves", lambda: self._grafico_equity_curves(plt, sns)),
            ("heatmaps",      lambda: self._grafico_heatmaps(plt, sns)),
            ("bootstrap",     lambda: self._grafico_bootstrap(plt, sns)),
        ]:
            try:
                fn()
            except Exception as exc:
                logger.warning(f"Grafico '{nombre}' omitido: {exc}")

        logger.info(f"Graficos exportados en {self.dir_graficos}")

    def _grafico_equity_curves(self, plt, sns) -> None:
        """Panel 2x2: equity curves por etapa y top 3 vs benchmarks."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Panel 1: Etapa 1
        ax = axes[0, 0]
        for id_exp, res in self._etapa1.items():
            if res.curva_equity is not None:
                ax.plot(res.curva_equity.index, res.curva_equity.values,
                        label=res.config.nombre, linewidth=1.5)
        ax.set_title('Etapa 1: Evaluación de Σ', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylabel('Equity (base 1)')
        ax.grid(True, alpha=0.3)

        # Panel 2: Top 4 frecuentistas por Sharpe
        ax = axes[0, 1]
        ranking_2a = sorted(
            self._etapa2a.items(),
            key=lambda x: x[1].metricas.get('sharpe_ratio', -999),
            reverse=True,
        )
        for id_exp, res in ranking_2a[:4]:
            if res.curva_equity is not None:
                ax.plot(res.curva_equity.index, res.curva_equity.values,
                        label=res.config.nombre, linewidth=1.5)
        ax.set_title('Etapa 2a: Top 4 Frecuentistas (por Sharpe)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylabel('Equity (base 1)')
        ax.grid(True, alpha=0.3)

        # Panel 3: Etapa 2b (bayesianos)
        ax = axes[1, 0]
        for id_exp, res in self._etapa2b.items():
            if res.curva_equity is not None:
                ax.plot(res.curva_equity.index, res.curva_equity.values,
                        label=res.config.nombre, linewidth=1.5)
        ax.set_title('Etapa 2b: Black-Litterman', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylabel('Equity (base 1)')
        ax.grid(True, alpha=0.3)

        # Panel 4: Top 3 globales vs benchmarks
        ax = axes[1, 1]
        res_sin_bm = {k: v for k, v in self.resultados.items()
                      if v.config.tipo not in (
                          TipoExperimento.BENCHMARK_EQUAL,
                          TipoExperimento.BENCHMARK_SP500,
                      )}
        top3 = sorted(
            res_sin_bm.items(),
            key=lambda x: x[1].metricas.get('sharpe_ratio', -999),
            reverse=True,
        )[:3]
        for id_exp, res in top3:
            if res.curva_equity is not None:
                ax.plot(res.curva_equity.index, res.curva_equity.values,
                        label=res.config.nombre, linewidth=2)
        for id_exp, res in self._benchmarks.items():
            if res.curva_equity is not None:
                ax.plot(res.curva_equity.index, res.curva_equity.values,
                        label=res.config.nombre, linewidth=1.5,
                        linestyle='--', alpha=0.7)
        ax.set_title('Top 3 vs Benchmarks', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylabel('Equity (base 1)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        ruta = self.dir_graficos / 'equity_curves_factorial.png'
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Gráfico equity curves: {ruta}")

    def _grafico_heatmaps(self, plt, sns) -> None:
        """Heatmaps del efecto de factores (μ, regularización, riesgo) sobre Sharpe y MaxDD."""
        if not self._etapa2a:
            logger.warning("Sin resultados de Etapa 2a, saltando heatmaps")
            return

        filas = []
        for id_exp, res in self._etapa2a.items():
            cfg = res.config
            filas.append({
                'mu': cfg.metodo_mu.value,
                'reg': cfg.regularizacion.value,
                'riesgo': cfg.metodo_riesgo.value,
                'Sharpe': res.metricas.get('sharpe_ratio', np.nan),
                'MaxDD': res.metricas.get('max_drawdown', np.nan),
            })
        df_hm = pd.DataFrame(filas)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Heatmap 1: μ × Regularización → Sharpe
        pivot_sharpe = df_hm.pivot_table(
            values='Sharpe', index='mu', columns='reg', aggfunc='mean')
        sns.heatmap(pivot_sharpe, annot=True, fmt='.3f', cmap='RdYlGn',
                    ax=axes[0], linewidths=0.5, center=0)
        axes[0].set_title('Sharpe: Est. μ × Regularización\n(promedio MPT/PMPT)',
                          fontweight='bold')

        # Heatmap 2: μ × Modelo riesgo → Sharpe
        pivot_sharpe2 = df_hm.pivot_table(
            values='Sharpe', index='mu', columns='riesgo', aggfunc='mean')
        sns.heatmap(pivot_sharpe2, annot=True, fmt='.3f', cmap='RdYlGn',
                    ax=axes[1], linewidths=0.5, center=0)
        axes[1].set_title('Sharpe: Est. μ × Modelo Riesgo\n(promedio Reg./No Reg.)',
                          fontweight='bold')

        # Heatmap 3: μ × Regularización → Max Drawdown
        pivot_dd = df_hm.pivot_table(
            values='MaxDD', index='mu', columns='reg', aggfunc='mean')
        sns.heatmap(pivot_dd, annot=True, fmt='.3f', cmap='RdYlGn_r',
                    ax=axes[2], linewidths=0.5)
        axes[2].set_title('Max DD: Est. μ × Regularización\n(promedio MPT/PMPT)',
                          fontweight='bold')

        plt.tight_layout()
        ruta = self.dir_graficos / 'heatmaps_factorial.png'
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Gráfico heatmaps: {ruta}")

    def _grafico_bootstrap(self, plt, sns) -> None:
        """Violin plots con distribución bootstrap del Sharpe Ratio."""
        # Seleccionar top 5 por Sharpe + benchmarks
        ranking = sorted(
            self.resultados.items(),
            key=lambda x: x[1].metricas.get('sharpe_ratio', -999),
            reverse=True,
        )[:5]
        # Agregar benchmarks si no están ya incluidos
        ids_seleccionados = {k for k, _ in ranking}
        for id_exp, res in self._benchmarks.items():
            if id_exp not in ids_seleccionados:
                ranking.append((id_exp, res))

        fig, ax = plt.subplots(figsize=(14, 6))
        posiciones = []
        labels = []

        for i, (id_exp, res) in enumerate(ranking):
            if res.retornos_oos_mensuales is None or len(res.retornos_oos_mensuales) < 12:
                continue

            dist_sharpe = _bootstrap_bloques_sharpe(
                res.retornos_oos_mensuales, n_bootstrap=1000, seed=42,
            )
            ax.violinplot([dist_sharpe], positions=[i], showmeans=True, showmedians=True)
            labels.append(res.config.nombre[:30])
            posiciones.append(i)

        if posiciones:
            ax.set_xticks(posiciones)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

        ax.set_ylabel('Sharpe Ratio (Bootstrap)')
        ax.set_title('Distribución Bootstrap del Sharpe (1000 iteraciones)',
                      fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        ruta = self.dir_graficos / 'bootstrap_sharpe.png'
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Gráfico bootstrap: {ruta}")

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def exportar_a_db(
        self,
        session,
        nombre_experimento: str,
        sobrescribir: bool = False,
    ) -> list:
        """
        Persiste los resultados del experimento factorial en la tabla ``backtests``.

        Cada ``ResultadoExperimento`` se guarda como una fila individual con:
        - ``parametros``  JSONB ← campos de ConfigExperimento
        - ``metricas``    JSONB ← resultado.metricas + alpha_jensen / alpha_pvalue
        - ``resultados``  JSONB ← equity curve [[date, nav], ...] + pesos por ventana
        - ``universo``    JSONB ← lista de tickers únicos del experimento
        - ``start_date``  / ``end_date`` ← primera / última ventana OOS
        - ``estado``      ← "completed" si hay métricas, "failed" si no

        Idempotente: si ya existe un backtest con el mismo ``nombre`` y
        ``sobrescribir=False``, se omite y no se lanza excepción.

        Args:
            session: Sesión SQLAlchemy activa.
            nombre_experimento: Prefijo de agrupación (ej. "walk_forward_factorial_2025").
            sobrescribir: Si True, elimina filas previas con el mismo nombre antes de insertar.

        Returns:
            Lista de IDs de filas insertadas.
        """
        try:
            from db.models import Backtest
        except ImportError:
            logger.error(
                "No se puede importar db.models.Backtest — "
                "asegúrate de ejecutar desde la raíz del proyecto con el .venv activo."
            )
            return []

        ids_insertados = []

        for id_exp, res in self.resultados.items():
            nombre_fila = f"{nombre_experimento}/{res.config.nombre}"

            # Idempotencia: lookup por nombre
            existente = session.query(Backtest).filter_by(nombre=nombre_fila).first()
            if existente is not None:
                if not sobrescribir:
                    logger.debug(f"[DB] '{nombre_fila}' ya existe, omitiendo.")
                    continue
                session.delete(existente)
                session.flush()

            # --- parametros JSONB ---
            parametros = {
                "id_experimento": res.config.id_experimento,
                "tipo":           res.config.tipo.value,
                "metodo_sigma":   res.config.metodo_sigma.value,
                "metodo_mu":      res.config.metodo_mu.value,
                "metodo_riesgo":  res.config.metodo_riesgo.value,
                "regularizacion": res.config.regularizacion.value,
                "metodo_bl":      res.config.metodo_bl.value,
            }

            # --- metricas JSONB ---
            metricas = dict(res.metricas) if res.metricas else {}
            if res.alpha_jensen is not None:
                metricas["alpha_jensen"] = res.alpha_jensen
            if res.alpha_tstat is not None:
                metricas["alpha_tstat"] = res.alpha_tstat
            if res.alpha_pvalue is not None:
                metricas["alpha_pvalue"] = res.alpha_pvalue

            # --- resultados JSONB (compacto) ---
            equity_list = []
            if res.curva_equity is not None:
                equity_list = [
                    [str(dt.date()), float(nav)]
                    for dt, nav in res.curva_equity.items()
                ]

            pesos_por_ventana = []
            for v in res.ventanas:
                if not v.ventana_fallida and v.pesos:
                    pesos_por_ventana.append({
                        "fecha": str(v.fecha_inicio_oos.date()),
                        "pesos": {k: round(float(w), 6) for k, w in v.pesos.items()},
                    })

            resultados_json = {
                "equity_curve": equity_list,
                "pesos_historicos": pesos_por_ventana,
            }

            # --- universo JSONB ---
            todos_tickers: set = set()
            for v in res.ventanas:
                todos_tickers.update(v.tickers_seleccionados)
            universo = sorted(todos_tickers)

            # --- start_date / end_date ---
            ventanas_ok = [v for v in res.ventanas if not v.ventana_fallida]
            start_date = ventanas_ok[0].fecha_inicio_oos.date() if ventanas_ok else None
            end_date   = ventanas_ok[-1].fecha_fin_oos.date()   if ventanas_ok else None

            # --- estado ---
            estado = "completed" if metricas.get("sharpe_ratio") is not None else "failed"

            fila = Backtest(
                nombre=nombre_fila,
                descripcion=res.config.nombre,
                estado=estado,
                start_date=start_date,
                end_date=end_date,
                universo=universo,
                parametros=parametros,
                metricas=metricas,
                resultados=resultados_json,
            )
            session.add(fila)
            session.flush()
            ids_insertados.append(fila.id)
            logger.info(f"[DB] Insertado backtest id={fila.id} '{nombre_fila}' ({estado})")

        session.commit()
        logger.info(f"[DB] {len(ids_insertados)} backtests persistidos en Supabase.")
        return ids_insertados

    # ------------------------------------------------------------------
    # Pickle
    # ------------------------------------------------------------------

    def exportar_pickle(self) -> None:
        """Persiste todos los ResultadoExperimento como pickle."""
        ruta = self.directorio_salida / "resultados_factorial_completos.pkl"
        with open(ruta, 'wb') as f:
            pickle.dump(self.resultados, f)
        logger.info(f"Pickle: {ruta}")


# ======================================================================
# Función auxiliar: bootstrap de bloques para Sharpe
# ======================================================================

def _bootstrap_bloques_sharpe(
    retornos: pd.Series,
    n_bootstrap: int = 1000,
    tamano_bloque: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """
    Bootstrap de bloques para estimar la distribución del Sharpe Ratio.

    Remuestrea bloques consecutivos de retornos mensuales para construir
    la distribución bootstrap del Sharpe. Bloques de 3 meses preservan
    la dependencia serial trimestral.

    Args:
        retornos: Serie de retornos mensuales OOS
        n_bootstrap: Número de remuestreos
        tamano_bloque: Meses por bloque (3 ~ 1 trimestre)
        seed: Semilla para reproducibilidad

    Returns:
        Array con n_bootstrap estimaciones de Sharpe
    """
    rng = np.random.RandomState(seed)
    T = len(retornos)
    n_bloques = T // tamano_bloque
    valores = retornos.values

    sharpes = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        indices = rng.randint(0, T - tamano_bloque, size=n_bloques)
        muestra = np.concatenate([valores[i:i + tamano_bloque] for i in indices])
        media = np.mean(muestra)
        std = np.std(muestra, ddof=1)
        sharpes[b] = (media / std * np.sqrt(12)) if std > 0 else 0.0

    return sharpes
