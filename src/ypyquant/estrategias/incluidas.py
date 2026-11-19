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
- `robusta_elipsoide`, `robusta_caja`: media-varianza robusta a la incertidumbre en µ (conjunto
  elipsoidal de Goldfarb-Iyengar o de caja, `estimacion_robusta.elipsoide`); µ̂ muestral, Bayes-Stein
  o BL-prior y Σ Ledoit-Wolf. El radio sale de una `confianza` o se fija directo (`kappa` / `z`).
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
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ypyquant.estrategias.base import Estrategia, ParametrosBase, matriz_cierres, matriz_volumen
from ypyquant.estrategias.registro import registrar_estrategia
from ypyquant.optimization.estimacion_robusta import (
    delta_caja,
    estimar_mu_bayes_stein,
    estimar_mu_bl_prior,
    estimar_mu_michaud,
    estimar_mu_muestral,
    estimar_sigma_ledoit_wolf,
    kappa_confianza,
    matriz_incertidumbre,
    optimizar_max_sharpe,
    optimizar_robusto_caja,
    optimizar_robusto_elipsoide,
    optimizar_volatilidad_objetivo,
    pesos_mercado,
    z_confianza,
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


class _ParametrosIncertidumbre(ParametrosBase):
    """Parámetros comunes de las estrategias con incertidumbre en µ (elipsoide y caja)."""

    estimador_mu: Literal["muestral", "bayes_stein", "bl_prior"] = Field(
        "muestral",
        description="Estimador de µ̂: media muestral, Bayes-Stein (contrae hacia la GMV) o prior de equilibrio de Black-Litterman",
    )
    confianza: float = Field(
        0.9, ge=0.0, le=0.999,
        description="Nivel de confianza del conjunto de incertidumbre (0,9 = 90 %). Solo se usa si el radio directo es 0. "
        "0 = sin robustez (media-varianza)",
    )
    aversion: float = Field(2.5, ge=0.0, le=100.0, description="Aversión al riesgo λ del término de varianza (0 = solo robusto)")
    volatilidad_objetivo: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Volatilidad anual objetivo (0,12 = 12 %): añade la restricción σ <= objetivo. 0 = sin restricción",
    )
    delta: float = Field(2.5, gt=0.0, le=20.0, description="Aversión δ del mercado del prior (solo con estimador_mu = bl_prior)")
    ventana_dias: int = Field(756, ge=60, le=2520, description="Días de historia para estimar µ y Σ (756 ≈ 3 años)")
    peso_maximo: float = Field(0.30, gt=0.0, le=1.0, description="Tope de peso por activo")


class _EstimadorIncertidumbre(_EstimadorRobusto):
    """
    Base de las estrategias robustas a la incertidumbre en µ: estima µ̂ y Σ (Ledoit-Wolf) en la
    ventana, construye el conjunto de incertidumbre con T = nº de observaciones de la ventana y
    resuelve el SOCP. Las subclases implementan `resolver(mu, sigma, n_observaciones)`.
    """

    def resolver(self, mu: np.ndarray, sigma: np.ndarray, n_observaciones: int) -> np.ndarray:
        raise NotImplementedError

    def estimar_mu(self, retornos: pd.DataFrame, sigma: pd.DataFrame, datos, fecha) -> np.ndarray:
        """µ̂ anualizado según `estimador_mu`."""
        p = self.parametros
        if p.estimador_mu == "bayes_stein":
            return estimar_mu_bayes_stein(retornos, sigma, DIAS_ANIO, si_t_menor_n2="ignorar")
        if p.estimador_mu == "bl_prior":
            cierres, volumen = matriz_cierres(datos), matriz_volumen(datos)
            pesos_mkt, _fuente = pesos_mercado(
                list(retornos.columns), pd.Timestamp(fecha), precios=cierres,
                volumen=volumen if not volumen.empty else None,
            )
            return estimar_mu_bl_prior(sigma, pesos_mkt, p.delta)
        return estimar_mu_muestral(retornos, DIAS_ANIO)

    def estimar_pesos(self, retornos, datos, fecha, pesos_previos):
        sigma = estimar_sigma_ledoit_wolf(retornos, DIAS_ANIO)
        mu = self.estimar_mu(retornos, sigma, datos, fecha)
        return self.resolver(mu, sigma.values, len(retornos))


class _ParametrosElipsoide(_ParametrosIncertidumbre):
    """Parámetros de `robusta_elipsoide`: añade el radio directo κ y la forma de Θ."""

    kappa: float = Field(
        0.0, ge=0.0, le=50.0,
        description="Radio κ del elipsoide, directo. Si es > 0 prevalece sobre la confianza; 0 = usar la confianza. "
        "Con muchos activos κ=√χ²_N(confianza) es muy grande: un κ pequeño (0,5-3) regulariza más",
    )
    tipo_theta: Literal["sigma", "diagonal"] = Field(
        "sigma", description="Incertidumbre de µ̂: Θ=Σ/T (con correlaciones) o diag(Σ)/T (errores independientes)"
    )


@registrar_estrategia
class RobustaElipsoide(_EstimadorIncertidumbre):
    """Media-varianza robusta con incertidumbre elipsoidal en µ (Goldfarb-Iyengar 2003)."""

    nombre = "robusta_elipsoide"
    descripcion = (
        "Media-varianza robusta: máx. µ̂'w - κ‖Θ^½w‖ - λ/2·w'Σw con elipsoide de incertidumbre en µ "
        "(Goldfarb-Iyengar 2003); κ por confianza χ² o directo, long-only."
    )
    Parametros = _ParametrosElipsoide

    def resolver(self, mu, sigma, n_observaciones):
        p = self.parametros
        theta = matriz_incertidumbre(sigma, n_observaciones, p.tipo_theta, frecuencia=DIAS_ANIO)
        kappa = p.kappa if p.kappa > 0 else kappa_confianza(p.confianza, len(mu))
        pesos, _estado, _diagnostico = optimizar_robusto_elipsoide(
            mu, sigma, theta, kappa, aversion=p.aversion,
            volatilidad_objetivo=p.volatilidad_objetivo or None, peso_maximo=max(p.peso_maximo, 1.0 / len(mu)),
        )
        return pesos


class _ParametrosCaja(_ParametrosIncertidumbre):
    """Parámetros de `robusta_caja`: añade el z directo."""

    z: float = Field(
        0.0, ge=0.0, le=10.0,
        description="Múltiplo z del error típico (δ_i = z·σ_i/√T), directo. Si es > 0 prevalece sobre la confianza; "
        "0 = usar la confianza (z bilateral normal)",
    )


@registrar_estrategia
class RobustaCaja(_EstimadorIncertidumbre):
    """Media-varianza robusta con incertidumbre de caja |µ_i - µ̂_i| <= δ_i."""

    nombre = "robusta_caja"
    descripcion = (
        "Media-varianza robusta con incertidumbre de caja: cada µ̂_i se rebaja z errores típicos "
        "(δ_i = z·σ_i/√T) y se optimiza long-only."
    )
    Parametros = _ParametrosCaja

    def resolver(self, mu, sigma, n_observaciones):
        p = self.parametros
        z = p.z if p.z > 0 else z_confianza(p.confianza)
        delta = delta_caja(sigma, n_observaciones, z, frecuencia=DIAS_ANIO)
        pesos, _estado, _diagnostico = optimizar_robusto_caja(
            mu, sigma, delta, aversion=p.aversion,
            volatilidad_objetivo=p.volatilidad_objetivo or None, peso_maximo=max(p.peso_maximo, 1.0 / len(mu)),
        )
        return pesos


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
