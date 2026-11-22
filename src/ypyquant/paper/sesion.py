"""
sesion.py — Modelos de una sesión de paper interno: su receta congelada y el estado de su cartera.

Responsabilidad:
    - `SesionPaper` (Pydantic): lo que NO cambia con el tiempo salvo el estado de control: la `Receta`
      congelada (estrategia + parámetros + entorno, con el universo ya resuelto a lista de tickers y los costes
      efectivos de la sesión), el origen (`id_experimento` o `id_configuracion`), `fecha_alta`,
      `capital_inicial`, `modo_ejecucion` (`cierre` | `apertura_siguiente`) y el `estado` de control
      (`activa` | `pausada` | `detenida`). `config_evaluacion` y `config_espejo` la convierten en una
      `ConfiguracionExperimento` (para evaluar lo ocurrido con `evaluar` o lanzar el backtest espejo).
    - `EstadoCartera` (Pydantic): lo que SÍ cambia cada día y se persiste como `estado.json`: efectivo, posiciones
      (`cantidad` y `precio_medio`), memoria de la estrategia (`estado_estrategia`, solo JSON) y su generador
      aleatorio (`estado_rng`), pesos pendientes de ejecutar, último rebalanceo, último cierre de cada posición
      (para detectar re-bases por ajuste) y la `ultima_fecha_procesada`, que es el COMMIT del día: se escribe al
      final, así que una caída a mitad de día se repite entero al reanudar.
    - Modos de ejecución: `cierre` decide y ejecuta con el cierre del mismo día (idéntico al backtest, para la
      prueba de paridad; optimista) y `apertura_siguiente` decide con el cierre de D y ejecuta a la apertura de
      D+1 (lo que haría una orden de mercado de Alpaca).

Se relaciona con:
    - ypyquant.estrategias.receta (`Receta`) y .experimentos (`ConfiguracionExperimento`).
    - ypyquant.paper.almacen (persistencia) y ypyquant.paper.simulador (quien muta el `EstadoCartera`).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.estrategias.receta import Receta

ModoEjecucion = Literal["cierre", "apertura_siguiente"]
EstadoSesion = Literal["activa", "pausada", "detenida"]
MODOS_EJECUCION = ("cierre", "apertura_siguiente")


class SesionPaper(BaseModel):
    """Una sesión de paper interno: receta congelada + parámetros de la simulación forward (ver cabecera)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    nombre: str
    receta: Receta
    fecha_alta: str
    capital_inicial: float = Field(100_000.0, gt=0)
    modo_ejecucion: ModoEjecucion = "apertura_siguiente"
    estado: EstadoSesion = "activa"
    id_experimento: str | None = None
    id_configuracion: str | None = None
    creada: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def tickers(self) -> list[str]:
        """Universo congelado (lista de tickers)."""
        return list(self.receta.universo)  # type: ignore[arg-type]

    def config_experimento(self, fecha_inicio: str, fecha_fin: str) -> ConfiguracionExperimento:
        """El experimento de esta receta (con los costes de la sesión y su capital) en `[fecha_inicio, fecha_fin]`."""
        return ConfiguracionExperimento.desde_receta(
            self.receta, fecha_inicio, fecha_fin, self.capital_inicial, etiqueta=f"paper interno {self.id}")

    def config_evaluacion(self, primera_fecha: date | str, ultima_fecha: date | str) -> ConfiguracionExperimento | None:
        """Configuración con la que evaluar lo ocurrido; None si aún no hay dos fechas distintas."""
        inicio, fin = str(primera_fecha)[:10], str(ultima_fecha)[:10]
        return self.config_experimento(inicio, fin) if inicio < fin else None

    def config_espejo(self, ultima_fecha: date | str) -> ConfiguracionExperimento:
        """Backtest espejo: la misma receta de `fecha_alta` a la última fecha procesada (se ejecuta al cierre)."""
        return self.config_experimento(self.fecha_alta, str(ultima_fecha)[:10])


class PosicionPaper(BaseModel):
    """Posición larga: `cantidad` de acciones (fraccionable) y `precio_medio` de entrada (con slippage)."""

    cantidad: float
    precio_medio: float


class EstadoCartera(BaseModel):
    """Estado de la cartera de una sesión al terminar `ultima_fecha_procesada` (ver cabecera del módulo)."""

    model_config = ConfigDict(extra="forbid")

    ultima_fecha_procesada: str
    efectivo: float
    posiciones: dict[str, PosicionPaper] = Field(default_factory=dict)
    asignacion_inicial_hecha: bool = False
    ultimo_rebalanceo: str | None = None
    pesos_pendientes: dict[str, float] | None = None
    fecha_decision_pendiente: str | None = None
    estado_estrategia: dict[str, Any] = Field(default_factory=dict)
    estado_rng: dict[str, Any] | None = None
    ultimos_cierres: dict[str, float] = Field(default_factory=dict)
    dias_sin_barra: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def inicial(cls, capital: float, fecha_alta: str) -> EstadoCartera:
        """Cartera de un alta: todo en efectivo y el día anterior al alta como último procesado."""
        anterior = (date.fromisoformat(fecha_alta) - timedelta(days=1)).isoformat()
        return cls(ultima_fecha_procesada=anterior, efectivo=capital)
