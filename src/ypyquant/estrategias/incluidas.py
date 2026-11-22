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
- Estimadores comunes (para comparar estrategias ceteris paribus, ver `marcos.py`): `estimador_sigma`
  (`ledoit_wolf`, defecto, o `muestral`) y `frecuencia_estimacion` (`diaria`, defecto, o `mensual`: remuestrea
  los cierres de la ventana a fin de mes y estima con retornos mensuales y 12 periodos/año, como el TFM; el mes en
  curso se descarta si la fecha no es su último día hábil, así no hay look-ahead). Son parámetros posteriores a los
  experimentos guardados: se omiten de la huella mientras valgan su defecto.
- `robusta_elipsoide`, `robusta_caja`: media-varianza robusta a la incertidumbre en µ (conjunto
  elipsoidal de Goldfarb-Iyengar o de caja, `estimacion_robusta.elipsoide`); µ̂ muestral, Bayes-Stein
  o BL-prior y Σ Ledoit-Wolf. El radio sale de una `confianza` o se fija directo (`kappa` / `z`).
- `senales_tecnicas` (en `senales_tecnicas.py`): puente con las señales técnicas heredadas, como `EstrategiaSenales`.
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
from pydantic import Field, model_validator

from ypyquant.estrategias import senales_tecnicas  # noqa: F401 - registra `senales_tecnicas` al importar este módulo
from ypyquant.estrategias.base import MARGEN_SESIONES, Estrategia, ParametrosBase, matriz_cierres, matriz_volumen
from ypyquant.estrategias.registro import registrar_estrategia
from ypyquant.execution.ejecutor import TICKER_EFECTIVO
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
    porcentaje_es,
    z_confianza,
)
from ypyquant.optimization.presets import PRESETS_REGISTRADOS

logger = logging.getLogger(__name__)

DIAS_ANIO = 252
DIAS_MES = 21
MESES_ANIO = 12
MINIMO_RETORNOS_MENSUALES = 12  # por debajo se reparte 1/N
AVISO_RETORNOS_MENSUALES = 24  # por debajo se avisa de que la estimación mensual es poco fiable


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


def retornos_ventana_mensual(datos: dict[str, pd.DataFrame], ventana_dias: int, fecha: pd.Timestamp) -> pd.DataFrame:
    """
    Retornos simples MENSUALES de la ventana de los últimos `ventana_dias` días (tickers sin huecos en ella).

    Remuestrea los cierres a fin de mes (último cierre disponible de cada mes natural). El mes de `fecha` solo
    cuenta si `fecha` es su último día hábil (calculado con el calendario, sin mirar datos posteriores); si no,
    es un mes incompleto y se descarta para no mezclar un retorno parcial con retornos de mes completo. La
    función solo usa filas <= `fecha`, así que no introduce look-ahead.
    """
    fecha = pd.Timestamp(fecha)
    cierres = matriz_cierres(datos).iloc[-(ventana_dias + 1):]
    cierres = cierres.dropna(axis=1, how="any")
    cierres = cierres.loc[cierres.index <= fecha]
    fin_de_mes = cierres.groupby(cierres.index.to_period("M")).tail(1)
    mes_completo = (fecha + pd.offsets.BDay(1)).to_period("M") != fecha.to_period("M")
    if not mes_completo and len(fin_de_mes) and fin_de_mes.index[-1].to_period("M") == fecha.to_period("M"):
        fin_de_mes = fin_de_mes.iloc[:-1]
    return fin_de_mes.pct_change().dropna(how="any")


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
    universo_sugerido = "sp500_pit"  # sección cruzada: sin sesgo de supervivencia por defecto

    class Parametros(ParametrosBase):
        top_k: int = Field(5, ge=1, le=200, description="Número de activos ganadores en cartera")
        meses_retorno: int = Field(12, ge=2, le=36, description="Meses de la ventana de retorno")
        meses_salto: int = Field(1, ge=0, le=6, description="Meses recientes que se saltan (reversión a corto plazo)")

    def sesiones_necesarias(self) -> int:
        """Retorno de `meses_retorno` meses (de `DIAS_MES` sesiones) más la sesión de partida y el margen."""
        return self.parametros.meses_retorno * DIAS_MES + 1 + MARGEN_SESIONES

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


ESTIMADORES_COMUNES = ("estimador_sigma", "frecuencia_estimacion")
# Parámetros de apalancamiento, posteriores a los experimentos guardados: fuera de la huella mientras valgan su defecto
PARAMETROS_APALANCAMIENTO = ("apalancamiento_maximo", "diferencial_financiacion")


class _ParametrosEstimacion(ParametrosBase):
    """Cómo se estiman Σ y la frecuencia de los retornos; común a las dos familias de estimadores robustos."""

    estimador_sigma: Literal["ledoit_wolf", "muestral"] = Field(
        "ledoit_wolf", description="Estimador de Σ: shrinkage Ledoit-Wolf (defecto) o covarianza muestral")
    frecuencia_estimacion: Literal["diaria", "mensual"] = Field(
        "diaria",
        description="Retornos con que se estiman µ y Σ: diarios (252 periodos/año) o mensuales a fin de mes (12 "
        "periodos/año, como el TFM; necesita >= 12 retornos mensuales en la ventana)",
    )


class _ParametrosApalancamiento(_ParametrosEstimacion):
    """
    Apalancamiento opcional de las estrategias con `volatilidad_objetivo` (las demás no lo declaran).

    Con `apalancamiento_maximo` = 1 (defecto) no hay apalancamiento y nada cambia. Con L > 1, si la σ objetivo es
    mayor que la que alcanza la cartera long-only totalmente invertida, la cartera se escala por
    min(L, σ*/σ) y el resto es un préstamo de margen (peso negativo en `CASH`) que paga la tasa libre de riesgo
    más `diferencial_financiacion`. Exige `volatilidad_objetivo` > 0 (sin objetivo no hay a qué escalar).
    """

    apalancamiento_maximo: float = Field(
        1.0, ge=1.0, le=2.0,
        description="Exposición bruta máxima sobre el patrimonio (1 = sin apalancamiento; 2 = límite Reg T overnight). "
        "Solo actúa si la σ objetivo es inalcanzable sin apalancar; exige volatilidad_objetivo > 0",
    )
    diferencial_financiacion: float = Field(
        0.01, ge=0.0, le=0.10,
        description="Diferencial anual sobre la tasa libre que cuesta el préstamo de margen (0,01 = 1 %). "
        "Solo se usa con apalancamiento_maximo > 1",
    )

    @model_validator(mode="after")
    def _apalancamiento_exige_volatilidad_objetivo(self):
        if self.apalancamiento_maximo > 1.0 and getattr(self, "volatilidad_objetivo", 0.0) <= 0:
            raise ValueError("apalancamiento_maximo > 1 exige volatilidad_objetivo > 0 (sin objetivo no hay a qué escalar)")
        return self


class _ParametrosEstimador(_ParametrosApalancamiento):
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
    parametros_fuera_de_huella_si_defecto = ESTIMADORES_COMUNES + PARAMETROS_APALANCAMIENTO
    Parametros = _ParametrosEstimador

    def estimar_pesos(self, retornos: pd.DataFrame, datos, fecha, pesos_previos: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError

    @property
    def periodos_anio(self) -> int:
        """Periodos por año de los retornos de estimación: 12 con `frecuencia_estimacion='mensual'`, 252 con 'diaria'."""
        return MESES_ANIO if self.parametros.frecuencia_estimacion == "mensual" else DIAS_ANIO

    def estimar_sigma(self, retornos: pd.DataFrame) -> pd.DataFrame:
        """Σ anualizada según `estimador_sigma` (Ledoit-Wolf de `estimacion_robusta` o covarianza muestral)."""
        if self.parametros.estimador_sigma == "muestral":
            return retornos.cov() * self.periodos_anio
        return estimar_sigma_ledoit_wolf(retornos, self.periodos_anio)

    def retornos_estimacion(self, datos, fecha) -> tuple[pd.DataFrame, int]:
        """Retornos de la ventana (diarios o mensuales según el parámetro) y el mínimo de observaciones exigido."""
        p = self.parametros
        if p.frecuencia_estimacion == "mensual":
            retornos = retornos_ventana_mensual(datos, p.ventana_dias, fecha)
            if MINIMO_RETORNOS_MENSUALES <= len(retornos) < AVISO_RETORNOS_MENSUALES:
                self._avisar(f"estimación mensual con solo {len(retornos)} retornos mensuales: poco fiable "
                             f"(amplía `ventana_dias`; se recomiendan >= {AVISO_RETORNOS_MENSUALES})")
            return retornos, MINIMO_RETORNOS_MENSUALES
        return retornos_ventana(datos, p.ventana_dias), max(60, min(p.ventana_dias, 120))

    def _avisar(self, mensaje: str) -> None:
        """Anota un aviso una sola vez por estrategia (se repetiría en cada rebalanceo)."""
        if mensaje not in self.avisos:
            self.avisos.append(mensaje)

    def _actualizar_aviso(self, clave: str, mensaje: str) -> None:
        """Mantiene UN aviso por `clave` (se reescribe en cada rebalanceo: agregado, no uno por fecha)."""
        indices = self.__dict__.setdefault("_indices_avisos", {})
        if clave in indices:
            self.avisos[indices[clave]] = mensaje
        else:
            indices[clave] = len(self.avisos)
            self.avisos.append(mensaje)

    def _anotar_vol_inalcanzable(self, vol_minima: float, fraccion_replicas: float | None = None) -> None:
        """
        Registra que en el rebalanceo actual la `volatilidad_objetivo` era inalcanzable y se usó la mínima varianza.

        Acumula fechas y σ mínimas y reescribe un único aviso: «en N de M rebalanceos (primero, último): σ objetivo X %
        < σ mínima alcanzable (entre a % y b %): se usa la cartera de mínima varianza». En Michaud `fraccion_replicas`
        es la parte de réplicas bootstrap que cayó en la GMV (la media de pesos mezcla réplicas, no es toda la GMV).
        """
        registro = self.__dict__.setdefault("_vol_inalcanzable", {"fechas": [], "vols": [], "fracciones": []})
        registro["fechas"].append(self._fecha_actual)
        registro["vols"].append(vol_minima)
        if fraccion_replicas is not None:
            registro["fracciones"].append(fraccion_replicas)
        n, total = len(registro["fechas"]), self._n_rebalanceos
        minima, maxima = min(registro["vols"]), max(registro["vols"])
        rango = porcentaje_es(minima) if abs(maxima - minima) < 5e-4 else f"entre {porcentaje_es(minima)} y {porcentaje_es(maxima)}"
        cuando = f"primero {registro['fechas'][0]:%Y-%m-%d}, último {registro['fechas'][-1]:%Y-%m-%d}"
        objetivo = porcentaje_es(self.parametros.volatilidad_objetivo)
        if registro["fracciones"]:
            sujeto = (f"En {n} de {total} rebalanceos hubo réplicas del bootstrap con la volatilidad objetivo inalcanzable "
                      f"(de media el {np.mean(registro['fracciones']):.0%} de las réplicas; {cuando})")
            cola = "esas réplicas usaron la cartera de mínima varianza"
        else:
            sujeto = f"Volatilidad objetivo inalcanzable en {n} de {total} rebalanceos ({cuando})"
            cola = "en esos rebalanceos se usó la cartera de mínima varianza"
        self._actualizar_aviso("vol_inalcanzable",
                               f"{sujeto}: σ objetivo {objetivo} < σ mínima alcanzable {rango}; {cola}.")

    def sesiones_necesarias(self) -> int:
        """Ventana completa de estimación (`ventana_dias` retornos = `ventana_dias` + 1 cierres) más el margen."""
        return self.parametros.ventana_dias + 1 + MARGEN_SESIONES

    def optimizar(self, mu: np.ndarray, sigma: np.ndarray, rf: float, pesos_previos: np.ndarray | None) -> np.ndarray:
        """Max-Sharpe (con penalización de turnover) o, si hay `volatilidad_objetivo`, máx. µ'w con σ <= objetivo."""
        p = self.parametros
        if p.volatilidad_objetivo > 0:
            pesos, estado = optimizar_volatilidad_objetivo(mu - rf, sigma, p.volatilidad_objetivo, peso_maximo=p.peso_maximo)
            if estado == "gmv_volatilidad_inalcanzable":
                self._anotar_vol_inalcanzable(float(np.sqrt(pesos @ sigma @ pesos)))
            return pesos
        pesos, _, _ = optimizar_max_sharpe(
            mu, sigma, rf=rf, peso_maximo=p.peso_maximo,
            pesos_previos=pesos_previos, penalizacion_turnover=p.penalizacion_turnover,
        )
        return pesos

    def generar_pesos(self, fecha: datetime, datos, estado: dict[str, Any]) -> dict[str, float]:
        retornos, minimo = self.retornos_estimacion(datos, fecha)
        if retornos.shape[1] < 2 or len(retornos) < minimo:
            if self.parametros.frecuencia_estimacion == "mensual":
                self._avisar(f"estimación mensual con menos de {minimo} retornos mensuales o menos de 2 activos "
                             "válidos en algún rebalanceo: se reparte 1/N")
            return pesos_iguales_entre(matriz_cierres(datos).columns)
        previos = estado.get("pesos_previos") or {}
        pesos_previos = np.array([previos.get(t, 0.0) for t in retornos.columns])
        self._fecha_actual = fecha  # para los avisos agregados (qué rebalanceos cayeron en un caso límite)
        self._n_rebalanceos = getattr(self, "_n_rebalanceos", 0) + 1
        pesos = self.estimar_pesos(retornos, datos, fecha, pesos_previos if pesos_previos.sum() > 0 else None)
        resultado = dict(zip(retornos.columns, pesos, strict=True))
        return self._apalancar(retornos, pesos, resultado)

    def _apalancar(self, retornos: pd.DataFrame, pesos: np.ndarray, resultado: dict[str, float]) -> dict[str, float]:
        """
        Escala la cartera por min(L, σ*/σ) si `apalancamiento_maximo` > 1 y la σ de la cartera queda por debajo del
        objetivo; el préstamo es un peso NEGATIVO en `CASH` (posiciones suman `factor`, CASH = 1 - factor). Sin
        apalancamiento (o con σ >= σ*) devuelve `resultado` sin tocar. El tope de peso se aplica antes de apalancar.
        """
        p = self.parametros
        maximo = getattr(p, "apalancamiento_maximo", 1.0)
        if maximo <= 1.0 or p.volatilidad_objetivo <= 0:
            return resultado
        sigma = self.estimar_sigma(retornos).values
        total = float(np.sum(pesos))
        if total <= 0:
            return resultado
        normalizados = pesos / total
        vol = float(np.sqrt(normalizados @ sigma @ normalizados))
        if vol <= 0:
            return resultado
        factor = min(maximo, p.volatilidad_objetivo / vol)
        if factor <= 1.0 + 1e-9:
            return resultado
        self._anotar_apalancamiento(factor)
        return {**{t: w / total * factor for t, w in resultado.items()}, TICKER_EFECTIVO: 1.0 - factor}

    def _anotar_apalancamiento(self, factor: float) -> None:
        """Aviso único y agregado: en cuántos rebalanceos se tomó préstamo, exposición media y máxima, y su coste."""
        factores = self.__dict__.setdefault("_factores_apalancamiento", [])
        factores.append(factor)
        p = self.parametros
        self._actualizar_aviso(
            "apalancamiento",
            f"Apalancamiento (máximo {p.apalancamiento_maximo:.2f}x): en {len(factores)} de {self._n_rebalanceos} rebalanceos "
            f"se tomó préstamo de margen (exposición bruta media {np.mean(factores):.2f}x, máxima {max(factores):.2f}x); "
            f"coste de financiación = tasa libre + {porcentaje_es(p.diferencial_financiacion)}.")


@registrar_estrategia
class BayesStein(_EstimadorRobusto):
    """µ de Bayes-Stein (Jorion 1986) + Σ Ledoit-Wolf + max-Sharpe QP."""

    nombre = "bayes_stein"
    descripcion = "Bayes-Stein sobre µ (contrae hacia la GMV), Σ Ledoit-Wolf y max-Sharpe long-only exacto (TFM)."

    def estimar_pesos(self, retornos, datos, fecha, pesos_previos):
        p = self.parametros
        sigma = self.estimar_sigma(retornos)
        mu = estimar_mu_bayes_stein(retornos, sigma, self.periodos_anio, si_t_menor_n2="ignorar")
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
        rng = self.generador_azar()  # generador propio de la ejecución (misma secuencia que default_rng(semilla))
        pesos, fraccion_gmv = estimar_mu_michaud(
            retornos, n_replicas=p.n_replicas, rf=p.tasa_libre_riesgo / self.periodos_anio,
            peso_maximo=p.peso_maximo, rng=rng, frecuencia=self.periodos_anio,
            volatilidad_objetivo=p.volatilidad_objetivo or None, estimador_sigma=p.estimador_sigma,
            devolver_diagnostico=True,
        )
        if p.volatilidad_objetivo > 0 and fraccion_gmv > 0:  # NaN (ninguna réplica convergió) no entra
            sigma = self.estimar_sigma(retornos).values
            self._anotar_vol_inalcanzable(float(np.sqrt(pesos @ sigma @ pesos)), float(fraccion_gmv))
        return pesos


@registrar_estrategia
class BLPrior(_EstimadorRobusto):
    """Prior de equilibrio de Black-Litterman sin views (w_mkt por capitalización o proxy de volumen)."""

    nombre = "bl_prior"
    descripcion = "Black-Litterman prior de equilibrio Π = δΣw_mkt (w_mkt por proxy de $-volumen) + max-Sharpe QP (TFM)."

    class Parametros(_ParametrosEstimador):
        delta: float = Field(2.5, gt=0.0, le=20.0, description="Aversión al riesgo δ del mercado")

    def estimar_pesos(self, retornos, datos, fecha, pesos_previos):
        p = self.parametros
        sigma = self.estimar_sigma(retornos)
        activos = list(retornos.columns)
        cierres, volumen = matriz_cierres(datos), matriz_volumen(datos)
        pesos_mkt, _fuente = pesos_mercado(
            activos, pd.Timestamp(fecha), precios=cierres, volumen=volumen if not volumen.empty else None
        )
        mu = estimar_mu_bl_prior(sigma, pesos_mkt, p.delta)
        return self.optimizar(mu, sigma.values, 0.0, pesos_previos)


class _ParametrosIncertidumbre(_ParametrosApalancamiento):
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
            return estimar_mu_bayes_stein(retornos, sigma, self.periodos_anio, si_t_menor_n2="ignorar")
        if p.estimador_mu == "bl_prior":
            cierres, volumen = matriz_cierres(datos), matriz_volumen(datos)
            pesos_mkt, _fuente = pesos_mercado(
                list(retornos.columns), pd.Timestamp(fecha), precios=cierres,
                volumen=volumen if not volumen.empty else None,
            )
            return estimar_mu_bl_prior(sigma, pesos_mkt, p.delta)
        return estimar_mu_muestral(retornos, self.periodos_anio)

    def estimar_pesos(self, retornos, datos, fecha, pesos_previos):
        sigma = self.estimar_sigma(retornos)
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
        theta = matriz_incertidumbre(sigma, n_observaciones, p.tipo_theta, frecuencia=self.periodos_anio)
        kappa = p.kappa if p.kappa > 0 else kappa_confianza(p.confianza, len(mu))
        pesos, estado, _diagnostico = optimizar_robusto_elipsoide(
            mu, sigma, theta, kappa, aversion=p.aversion,
            volatilidad_objetivo=p.volatilidad_objetivo or None, peso_maximo=max(p.peso_maximo, 1.0 / len(mu)),
        )
        if estado == "gmv_volatilidad_inalcanzable":
            self._anotar_vol_inalcanzable(float(np.sqrt(pesos @ sigma @ pesos)))
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
        delta = delta_caja(sigma, n_observaciones, z, frecuencia=self.periodos_anio)
        pesos, estado, _diagnostico = optimizar_robusto_caja(
            mu, sigma, delta, aversion=p.aversion,
            volatilidad_objetivo=p.volatilidad_objetivo or None, peso_maximo=max(p.peso_maximo, 1.0 / len(mu)),
        )
        if estado == "gmv_volatilidad_inalcanzable":
            self._anotar_vol_inalcanzable(float(np.sqrt(pesos @ sigma @ pesos)))
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

    def sesiones_necesarias(self) -> int:
        """Ventana completa de estimación (`ventana_dias` retornos = `ventana_dias` + 1 cierres) más el margen."""
        return self.parametros.ventana_dias + 1 + MARGEN_SESIONES

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
