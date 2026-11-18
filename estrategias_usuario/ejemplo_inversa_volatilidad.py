"""
Ejemplo de estrategia de usuario: ponderación por inversa de la volatilidad.

Responsabilidad: mostrar cómo se escribe una estrategia propia. Basta con dejar un `.py` en esta
carpeta (`estrategias_usuario/`): al arrancar la API (o llamar a `listar_estrategias()`) se
descubre sola y aparece en la UI con su formulario de parámetros. Los archivos que empiezan por `_`
se ignoran.

Se relaciona con: ypyquant.estrategias.base (contrato `Estrategia`), ypyquant.estrategias.registro
(`registrar_estrategia`, descubrimiento) y docs/ui.md (guía en 5 pasos).
"""

from pydantic import Field

from ypyquant.estrategias import Estrategia, ParametrosBase, matriz_cierres, registrar_estrategia


@registrar_estrategia
class InversaVolatilidad(Estrategia):
    """Cada activo pesa proporcionalmente a 1/volatilidad reciente: menos riesgo, más peso."""

    nombre = "inversa_volatilidad"
    descripcion = "Pondera por la inversa de la volatilidad de los últimos N días (ejemplo de estrategia de usuario)."
    frecuencia_rebalanceo = "monthly"

    class Parametros(ParametrosBase):
        ventana_dias: int = Field(60, ge=10, le=500, description="Días para medir la volatilidad")

    def generar_pesos(self, fecha, datos, estado):
        retornos = matriz_cierres(datos).pct_change().iloc[-self.parametros.ventana_dias:]
        volatilidad = retornos.std().replace(0, float("nan")).dropna()
        if volatilidad.empty:
            return {}
        inversa = 1.0 / volatilidad
        return (inversa / inversa.sum()).to_dict()
