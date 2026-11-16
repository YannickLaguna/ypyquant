"""
FactorSignalGenerator — señal de alpha estimado por modelo de factores.
=======================================================================

Wraps FamaFrenchModel (FF5+MOM) para estimar el alpha de Jensen por ticker,
luego normaliza cross-seccional por rank percentil → [-1, +1].

Tickers con mayor alpha estimado → señal positiva (sobreponderar).
Tickers con menor alpha estimado → señal negativa (subponderar).

Requiere:
- factor_data: DataFrame con columnas MKT, SMB, HML, RMW, CMA, MOM, RF
  (misma estructura que FrenchDataFetcher.get_factor_data()).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from ypyquant.signals.base import BaseSignalGenerator, SignalFrame

logger = logging.getLogger(__name__)


class FactorSignalGenerator(BaseSignalGenerator):
    """
    Señal basada en alpha de Jensen estimado por Fama-French FF5+MOM.

    Para cada fecha, ajusta el modelo FF sobre una ventana rolling de
    entrenamiento y asigna la señal del alpha cross-seccional al siguiente
    período.

    Input:
        prices:      adjusted_close (DatetimeIndex × tickers)
        factor_data: DataFrame(DatetimeIndex, columnas MKT/SMB/HML/RMW/CMA/MOM/RF)

    Output:
        SignalFrame: rank(alpha) → [-1, +1]
    """

    @property
    def name(self) -> str:
        return "factor"

    def generate(
        self,
        prices: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        factor_data: Optional[pd.DataFrame] = None,
        train_window_days: int = 252,
        step_days: int = 21,
        factores: Optional[List[str]] = None,
        **kwargs,
    ) -> SignalFrame:
        """
        Args:
            prices: adjusted_close diario (DatetimeIndex × tickers).
            tickers: Subconjunto de tickers. None → todos.
            factor_data: Factores FF (MKT, SMB, HML, RMW, CMA, MOM, RF).
                         Si None, retorna señal cero.
            train_window_days: Ventana de entrenamiento en días (default: 252).
            step_days: Frecuencia de re-estimación en días (default: 21 ~ mensual).
            factores: Lista de factores a usar. None → FF5+MOM por defecto.

        Returns:
            SignalFrame con señal alpha rank ∈ [-1, +1].
        """
        try:
            from ypyquant.factors.factor_models import FamaFrenchModel, FACTORES_FF5_MOM
        except ImportError as e:
            logger.warning(f"[FactorSignal] Dependencia no disponible: {e}. Señal=0.")
            cols = self._select_tickers(prices, tickers)
            return pd.DataFrame(0.0, index=prices.index, columns=cols)

        cols = self._select_tickers(prices, tickers)

        if factor_data is None:
            logger.warning("[FactorSignal] factor_data no proporcionado. Señal=0.")
            return pd.DataFrame(0.0, index=prices.index, columns=cols)

        factores_a_usar = factores or list(FACTORES_FF5_MOM)
        # Filtrar factores disponibles en factor_data
        factores_disponibles = [f for f in factores_a_usar if f in factor_data.columns]
        if not factores_disponibles:
            logger.warning(
                f"[FactorSignal] Ningún factor de {factores_a_usar} en factor_data. Señal=0."
            )
            return pd.DataFrame(0.0, index=prices.index, columns=cols)

        retornos = prices[cols].pct_change().dropna(how="all")
        signal_df = pd.DataFrame(np.nan, index=prices.index, columns=cols)

        # Estimación rolling: cada step_days días re-ajustamos el modelo
        fechas = retornos.index
        indices_reestimacion = list(range(train_window_days, len(fechas), step_days))

        if not indices_reestimacion:
            return pd.DataFrame(0.0, index=prices.index, columns=cols)

        ultima_señal: Optional[pd.Series] = None

        for idx in indices_reestimacion:
            train_slice = retornos.iloc[idx - train_window_days: idx]
            fecha_señal = fechas[idx]  # aplicar señal a partir de esta fecha

            # Factores del mismo período de entrenamiento
            ff_slice = factor_data[factores_disponibles].reindex(train_slice.index).dropna()
            ret_aligned = train_slice.reindex(ff_slice.index).dropna(how="all")

            if ret_aligned.empty or len(ff_slice) < 60:
                continue

            try:
                modelo = FamaFrenchModel(factores=factores_disponibles)
                modelo.fit(ret_aligned, ff_slice)
                alphas = pd.Series(modelo.alphas)

                # Rank cross-seccional del alpha → [-1, +1]
                ranked_alphas = alphas.rank(pct=True).apply(lambda x: 2 * x - 1)
                ultima_señal = ranked_alphas
            except Exception as exc:
                logger.debug(f"[FactorSignal] Estimación en {fecha_señal}: {exc}")
                # Mantener señal anterior si falla

            # Asignar señal a fechas entre este punto y el siguiente paso
            fecha_fin_señal_idx = min(idx + step_days, len(fechas))
            fechas_aplicar = fechas[idx:fecha_fin_señal_idx]

            if ultima_señal is not None:
                for ticker in cols:
                    val = ultima_señal.get(ticker, np.nan)
                    signal_df.loc[fechas_aplicar, ticker] = val

        return signal_df.fillna(0.0).clip(-1.0, 1.0)
