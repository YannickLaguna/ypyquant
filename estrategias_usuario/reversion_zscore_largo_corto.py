"""
Plantilla de estrategia de señales LARGO-CORTO: reversión a la media por z-score con stops simétricos.

Responsabilidad: ejemplo completo para copiar al escribir una estrategia que opera en los dos lados. Solo se escriben las reglas
(`indicadores`, las cuatro señales y las dos puntuaciones); la base se ocupa de las posiciones largas y cortas, los stops
(pérdida, trailing, take profit, tiempo), el enfriamiento, la señal contraria, el tamaño, los límites de exposición y el efectivo
(`CASH`, negativo si hay préstamo de margen). Al dejar el archivo en `estrategias_usuario/` se descubre solo y aparece en la UI.

Regla: el z-score es la distancia del cierre a su media móvil en desviaciones típicas. Compra (largo) cuando está MUY por debajo
(z < -`z_entrada`, sobrevendido) y cierra al volver cerca de la media (z > -`z_salida`). Vende en corto cuando está MUY por encima
(z > `z_entrada`) y recompra al volver (z < `z_salida`). Entre candidatos elige el más extremo y, con `ponderacion="fuerza"`, le da
más peso. Todo es causal (`rolling` hacia atrás): `verificar_sin_lookahead` lo comprueba en tests/test_senales_ejemplos.py.

Por defecto opera largo-corto con exposición bruta máxima 1,0 y neta entre -0,5 y 0,5 (casi neutral al mercado). Los cortos exigen
cuenta de margen; con `lados="largo"` queda como una reversión solo larga.

Se relaciona con: ypyquant.estrategias.senales (base, lados y límites), ypyquant.estrategias.senales_modelo (motivos),
estrategias_usuario/cruce_medias.py y canal_rsi_volatilidad.py (otras plantillas) y docs/estrategias_senales.md (guía).
"""

from typing import Literal

from pydantic import Field

from ypyquant.estrategias import registrar_estrategia
from ypyquant.estrategias.base import MARGEN_SESIONES
from ypyquant.estrategias.senales import (
    PARAMETROS_LARGO_CORTO_SENALES,
    EstrategiaSenales,
    ParametrosSenales,
    Ponderacion,
)


@registrar_estrategia
class ReversionZscoreLargoCorto(EstrategiaSenales):
    """Reversión a la media: compra lo muy por debajo de su media, vende en corto lo muy por encima."""

    nombre = "reversion_zscore_largo_corto"
    descripcion = "Reversión a la media por z-score, largo-corto con stops simétricos (plantilla de señales largo-corto)."
    universo_sugerido = "sp500_pit"
    parametros_fuera_de_huella_si_defecto = PARAMETROS_LARGO_CORTO_SENALES  # los nuevos de la base, por si cambia su defecto

    class Parametros(ParametrosSenales):
        """Parámetros de la base (largo-corto, fuerza y stop por defecto) más la ventana y los umbrales del z-score."""

        lados: Literal["largo", "corto", "largo_corto"] = "largo_corto"
        ponderacion: Ponderacion = "fuerza"
        stop_perdida: float | None = Field(0.08, gt=0, lt=1, description="Sale si pierde esta fracción (en un corto, si sube)")
        exposicion_neta_min: float = Field(-0.5, ge=-2.0, le=1.0, description="Exposición neta mínima")
        exposicion_neta_max: float = Field(0.5, gt=0, le=2.0, description="Exposición neta máxima")
        ventana: int = Field(20, ge=5, le=250, description="Sesiones de la media y la desviación del z-score")
        z_entrada: float = Field(2.0, gt=0, le=5, description="Entra cuando |z| supera este valor")
        z_salida: float = Field(0.5, ge=0, le=3, description="Cierra cuando z vuelve a esta distancia de la media")

    def indicadores(self, cierres, datos):
        """Z-score del cierre frente a su media y desviación móviles (hacia atrás, sin mirar el futuro)."""
        ventana = self.parametros.ventana
        media, desviacion = cierres.rolling(ventana).mean(), cierres.rolling(ventana).std()
        return {"z": (cierres - media) / desviacion}

    def senal_entrada(self, cierres, indicadores):
        """Largo: muy por debajo de la media."""
        return (indicadores["z"] < -self.parametros.z_entrada).fillna(False)

    def senal_salida(self, cierres, indicadores):
        """Cierra el largo al volver cerca de la media."""
        return (indicadores["z"] > -self.parametros.z_salida).fillna(False)

    def senal_entrada_corta(self, cierres, indicadores):
        """Corto: muy por encima de la media."""
        return (indicadores["z"] > self.parametros.z_entrada).fillna(False)

    def senal_salida_corta(self, cierres, indicadores):
        """Recompra al volver cerca de la media."""
        return (indicadores["z"] < self.parametros.z_salida).fillna(False)

    def puntuacion(self, cierres, indicadores):
        """Fuerza del largo: lo más sobrevendido primero y con más peso."""
        return -indicadores["z"]

    def puntuacion_corta(self, cierres, indicadores):
        """Fuerza del corto: lo más sobrecomprado primero y con más peso."""
        return indicadores["z"]

    def sesiones_necesarias(self) -> int:
        """Historia mínima: la ventana del z-score más el margen estándar."""
        return self.parametros.ventana + MARGEN_SESIONES
