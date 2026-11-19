"""
Estrategias incluidas en ypyquant (se registran al importar este módulo).

Responsabilidad: ofrecer un catálogo base sobre el contrato `Estrategia`, con parámetros
tipados que la UI convierte en formulario:
- `pesos_iguales`: 1/N sobre el universo (referencia mínima).
- `momentum`: top-k por retorno acumulado 12-1 (salta el último mes), pesos iguales.
- `bayes_stein`, `michaud`, `bl_prior`: los estimadores del TFM (`optimization.estimacion_robusta`)
  con Σ Ledoit-Wolf y max-Sharpe QP; tope de peso y penalización de turnover como parámetros
  (Michaud no admite penalización: promedia pesos de réplicas). Con `volatilidad_objetivo` > 0
  el objetivo pasa de max-Sharpe a máxima rentabilidad con σ <= objetivo (frontera eficiente).
- `preset_ypyquant`: adaptador genérico que envuelve cualquier preset de `optimization/presets`
  (markowitz, risk_parity, pmpt_*, tfm_*...). Algunos presets (capm, fama_french_4) necesitan
  datos externos (tasa libre, factores) y pueden fallar sin red.

Se relaciona con:
- ypyquant.estrategias.base y .registro.
- ypyquant.optimization.estimacion_robusta y ypyquant.optimization.presets.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from ypyquant.estrategias.base import Estrategia, ParametrosBase, matriz_cierres, matriz_volumen
from ypyquant.estrategias.registro import registrar_estrategia
from ypyquant.optimization.estimacion_robusta import (
    estimar_mu_bayes_stein,
    estimar_mu_bl_prior,
    estimar_mu_michaud,
    estimar_sigma_ledoit_wolf,
    optimizar_max_sharpe,
    optimizar_volatilidad_objetivo,
    pesos_mercado,
)
from ypyquant.optimization.presets import PRESETS_REGISTRADOS

logger = logging.getLogger(__name__)

DIAS_ANIO = 252
DIAS_MES = 21


def pesos_iguales_entre(tickers) -> dict[str, float]:
    """Reparto 1/N entre los tickers dados (vacío si no hay ninguno)."""
    tickers = list(tickers)
    return {t: 1.0 / len(tickers) for t in tickers} if tickers else {}


def retornos_ventana(datos: dict[str, pd.DataFrame], ventana_dias: int) -> pd.DataFrame:
    """
    Retornos simples diarios de los últimos `ventana_dias` días, solo de los tickers con historia
    completa en la ventana (sin huecos), para poder estimar µ y Σ sin imputar.
    """
    cierres = matriz_cierres(datos).iloc[-(ventana_dias + 1):]
    cierres = cierres.dropna(axis=1, how="any")
    return cierres.pct_change().dropna(how="any")


@registrar_estrategia
class PesosIguales(Estrategia):
    """Equiponderada: 1/N sobre los tickers con precio en la fecha."""

    nombre = "pesos_iguales"
    descripcion = "Equiponderada 1/N sobre el universo; referencia mínima para comparar cualquier otra estrategia."
    frecuencia_rebalanceo = "monthly"

    def generar_pesos(self, fecha: datetime, datos, estado: dict[str, Any]) -> dict[str, float]:
        return pesos_iguales_entre(datos)


@registrar_estrategia
class Momentum(Estrategia):
    """Momentum de sección cruzada: top-k por retorno acumulado de `meses_retorno` saltando `meses_salto`."""

    nombre = "momentum"
    descripcion = "Top-k por retorno acumulado 12-1 (salta el último mes), equiponderado entre los k ganadores."
    frecuencia_rebalanceo = "monthly"

    class Parametros(ParametrosBase):
        top_k: int = Field(5, ge=1, le=200, description="Número de activos ganadores en cartera")
        meses_retorno: int = Field(12, ge=2, le=36, description="Meses de la ventana de retorno")
        meses_salto: int = Field(1, ge=0, le=6, description="Meses recientes que se saltan (reversión a corto plazo)")

    def generar_pesos(self, fecha: datetime, datos, estado: dict[str, Any]) -> dict[str, float]:
        p = self.parametros
        cierres = matriz_cierres(datos)
        fin = len(cierres) - 1 - p.meses_salto * DIAS_MES
        inicio = len(cierres) - 1 - p.meses_retorno * DIAS_MES
        if inicio < 0 or fin <= inicio:
            return pesos_iguales_entre(cierres.columns)  # sin historia suficiente: neutral
        retorno = cierres.iloc[fin] / cierres.iloc[inicio] - 1.0
        ganadores = retorno.dropna().nlargest(p.top_k).index
        return pesos_iguales_entre(ganadores)


class _ParametrosEstimador(ParametrosBase):
    """Parámetros comunes de las estrategias de optimización robusta."""

    ventana_dias: int = Field(756, ge=60, le=2520, description="Días de historia para estimar µ y Σ (756 ≈ 3 años)")
    peso_maximo: float = Field(0.30, gt=0.0, le=1.0, description="Tope de peso por activo")
    penalizacion_turnover: float = Field(0.0, ge=0.0, le=100.0, description="λ de penalización de turnover (0 = sin)")
    tasa_libre_riesgo: float = Field(0.0, ge=0.0, le=0.2, description="Tasa libre anual para el max-Sharpe")
    volatilidad_objetivo: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Volatilidad anual objetivo (0,12 = 12 %): máxima rentabilidad con esa σ. 0 = max-Sharpe",
    )


class _EstimadorRobusto(Estrategia):
    """
    Base de las estrategias del TFM: estima µ y Σ en una ventana diaria y resuelve max-Sharpe long-only.

    Las subclases implementan `estimar_pesos` (devuelve el vector de pesos alineado con `activos`).
    Si no hay historia suficiente o hay menos de 2 activos válidos, reparte 1/N.
    """

    frecuencia_rebalanceo = "monthly"
    Parametros = _ParametrosEstimador

    def estimar_pesos(self, retornos: pd.DataFrame, datos, fecha, pesos_previos: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError

    def optimizar(self, mu: np.ndarray, sigma: np.ndarray, rf: float, pesos_previos: np.ndarray | None) -> np.ndarray:
        """Max-Sharpe (con penalización de turnover) o, si hay `volatilidad_objetivo`, máx. µ'w con σ <= objetivo."""
        p = self.parametros
        if p.volatilidad_objetivo > 0:
            pesos, _ = optimizar_volatilidad_objetivo(mu - rf, sigma, p.volatilidad_objetivo, peso_maximo=p.peso_maximo)
            return pesos
        pesos, _, _ = optimizar_max_sharpe(
            mu, sigma, rf=rf, peso_maximo=p.peso_maximo,
            pesos_previos=pesos_previos, penalizacion_turnover=p.penalizacion_turnover,
        )
        return pesos

    def generar_pesos(self, fecha: datetime, datos, estado: dict[str, Any]) -> dict[str, float]:
        p = self.parametros
        retornos = retornos_ventana(datos, p.ventana_dias)
        minimo = max(60, min(p.ventana_dias, 120))
        if retornos.shape[1] < 2 or len(retornos) < minimo:
            return pesos_iguales_entre(matriz_cierres(datos).columns)
        previos = estado.get("pesos_previos") or {}
        pesos_previos = np.array([previos.get(t, 0.0) for t in retornos.columns])
        pesos = self.estimar_pesos(retornos, datos, fecha, pesos_previos if pesos_previos.sum() > 0 else None)
        return dict(zip(retornos.columns, pesos, strict=True))


@registrar_estrategia
class BayesStein(_EstimadorRobusto):
    """µ de Bayes-Stein (Jorion 1986) + Σ Ledoit-Wolf + max-Sharpe QP."""

    nombre = "bayes_stein"
    descripcion = "Bayes-Stein sobre µ (contrae hacia la GMV), Σ Ledoit-Wolf y max-Sharpe long-only exacto (TFM)."

    def estimar_pesos(self, retornos, datos, fecha, pesos_previos):
        p = self.parametros
        sigma = estimar_sigma_ledoit_wolf(retornos, DIAS_ANIO)
        mu = estimar_mu_bayes_stein(retornos, sigma, DIAS_ANIO, si_t_menor_n2="ignorar")
        return self.optimizar(mu, sigma.values, p.tasa_libre_riesgo, pesos_previos)


@registrar_estrategia
class Michaud(_EstimadorRobusto):
    """Remuestreo eficiente de Michaud: promedia los pesos max-Sharpe de réplicas bootstrap."""

    nombre = "michaud"
    descripcion = "Remuestreo eficiente de Michaud: bootstrap del histórico y promedio de pesos max-Sharpe (TFM)."

    class Parametros(_ParametrosEstimador):
        n_replicas: int = Field(50, ge=5, le=500, description="Réplicas bootstrap (más = más lento y más estable)")

    def estimar_pesos(self, retornos, datos, fecha, pesos_previos):
        p = self.parametros
        # La penalización de turnover no aplica: el promedio de réplicas ya es una cartera estable
        rng = self.estado.setdefault("rng", np.random.default_rng(self.semilla))
        return estimar_mu_michaud(
            retornos, n_replicas=p.n_replicas, rf=p.tasa_libre_riesgo / DIAS_ANIO,
            peso_maximo=p.peso_maximo, rng=rng, frecuencia=DIAS_ANIO,
            volatilidad_objetivo=p.volatilidad_objetivo or None,
        )


@registrar_estrategia
class BLPrior(_EstimadorRobusto):
    """Prior de equilibrio de Black-Litterman sin views (w_mkt por capitalización o proxy de volumen)."""

    nombre = "bl_prior"
    descripcion = "Black-Litterman prior de equilibrio Π = δΣw_mkt (w_mkt por proxy de $-volumen) + max-Sharpe QP (TFM)."

    class Parametros(_ParametrosEstimador):
        delta: float = Field(2.5, gt=0.0, le=20.0, description="Aversión al riesgo δ del mercado")

    def estimar_pesos(self, retornos, datos, fecha, pesos_previos):
        p = self.parametros
        sigma = estimar_sigma_ledoit_wolf(retornos, DIAS_ANIO)
        activos = list(retornos.columns)
        cierres, volumen = matriz_cierres(datos), matriz_volumen(datos)
        pesos_mkt, _fuente = pesos_mercado(
            activos, pd.Timestamp(fecha), precios=cierres, volumen=volumen if not volumen.empty else None
        )
        mu = estimar_mu_bl_prior(sigma, pesos_mkt, p.delta)
        return self.optimizar(mu, sigma.values, 0.0, pesos_previos)


@registrar_estrategia
class PresetYpyquant(Estrategia):
    """Adaptador genérico: ejecuta cualquier preset de `ypyquant.optimization.presets` en cada rebalanceo."""

    nombre = "preset_ypyquant"
    descripcion = "Envuelve un preset de optimización de ypyquant (markowitz, risk_parity, pmpt_*, tfm_*...)."
    frecuencia_rebalanceo = "monthly"

    class Parametros(ParametrosBase):
        preset: str = Field(
            "markowitz", description=f"Nombre del preset. Disponibles: {', '.join(sorted(PRESETS_REGISTRADOS))}"
        )
        ventana_dias: int = Field(756, ge=60, le=2520, description="Días de historia para estimar")
        peso_maximo: float = Field(0.30, gt=0.0, le=1.0, description="Tope de peso por activo")
        tasa_libre_riesgo: float = Field(0.02, ge=0.0, le=0.2, description="Tasa libre anual del preset")

    def generar_pesos(self, fecha: datetime, datos, estado: dict[str, Any]) -> dict[str, float]:
        from ypyquant.optimization.presets import obtener_preset

        p = self.parametros
        retornos = retornos_ventana(datos, p.ventana_dias)
        if retornos.shape[1] < 2 or len(retornos) < 60:
            return pesos_iguales_entre(matriz_cierres(datos).columns)
        if "modelo" not in estado:
            estado["modelo"] = obtener_preset(p.preset, risk_free_rate=p.tasa_libre_riesgo)
        volumen = matriz_volumen(datos)
        resultado = estado["modelo"].optimize(
            retornos, constraints={"max_weight": p.peso_maximo},
            precios=matriz_cierres(datos), volumen=volumen if not volumen.empty else None,
        )
        return {t: float(w) for t, w in resultado["weights"].items()}
