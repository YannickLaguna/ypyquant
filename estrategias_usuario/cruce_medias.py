"""
Ejemplo de estrategia de señales: cruce de medias móviles con stops (plantilla para escribir la tuya).

Responsabilidad: mostrar la forma mínima de una `EstrategiaSenales`. Solo se escriben las reglas
(`indicadores`, `senal_entrada`, `senal_salida`, `puntuacion`); la base se ocupa de las posiciones,
los stops (pérdida, trailing, take profit, tiempo), el enfriamiento, el tamaño y el efectivo (`CASH`).
Al dejar este archivo en `estrategias_usuario/` se descubre solo y aparece en la UI con su formulario.

Regla: largo mientras la media rápida esté por encima de la lenta. Entra al cruzar al alza; sale al
cruzar a la baja o por stop. Entre varios candidatos elige los de mayor distancia relativa entre medias.
Todas las operaciones son CAUSALES (medias móviles hacia atrás): `verificar_sin_lookahead` lo comprueba
en tests/test_senales_ejemplos.py.

Se relaciona con: ypyquant.estrategias.senales (base y stops), el registro (descubrimiento automático),
docs/estrategias_senales.md (guía) y estrategias_usuario/canal_rsi_volatilidad.py (otra plantilla).
"""

from pydantic import Field, model_validator

from ypyquant.estrategias import registrar_estrategia
from ypyquant.estrategias.base import MARGEN_SESIONES
from ypyquant.estrategias.senales import EstrategiaSenales, ParametrosSenales


@registrar_estrategia
class CruceMedias(EstrategiaSenales):
    """Compra cuando la media rápida supera a la lenta; vende al revés o por stop."""

    nombre = "cruce_medias"
    descripcion = "Compra cuando la media rápida supera a la lenta; vende al revés o por stop (ejemplo de señales)."
    universo_sugerido = "sp500_pit"

    class Parametros(ParametrosSenales):
        """Parámetros de la base de señales más las dos ventanas de las medias."""

        rapida: int = Field(20, ge=2, le=200, description="Sesiones de la media rápida")
        lenta: int = Field(100, ge=5, le=400, description="Sesiones de la media lenta")

        @model_validator(mode="after")
        def _rapida_menor_que_lenta(self):
            if self.rapida >= self.lenta:
                raise ValueError("rapida debe ser menor que lenta")
            return self

    def indicadores(self, cierres, datos):
        """Medias móviles simples de cierre (hacia atrás, sin mirar el futuro)."""
        p = self.parametros
        return {"rapida": cierres.rolling(p.rapida).mean(), "lenta": cierres.rolling(p.lenta).mean()}

    def senal_entrada(self, cierres, indicadores):
        """Rápida por encima de la lenta (False mientras no hay historia para la lenta)."""
        return (indicadores["rapida"] > indicadores["lenta"]).fillna(False)

    def senal_salida(self, cierres, indicadores):
        """Rápida por debajo de la lenta."""
        return (indicadores["rapida"] < indicadores["lenta"]).fillna(False)

    def puntuacion(self, cierres, indicadores):
        """Entre candidatos, el de mayor distancia relativa entre medias."""
        return indicadores["rapida"] / indicadores["lenta"] - 1.0

    def sesiones_necesarias(self) -> int:
        """Historia mínima: la media lenta más el margen estándar."""
        return self.parametros.lenta + MARGEN_SESIONES
