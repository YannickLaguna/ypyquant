"""
Paquete `ypyquant.paper` — paper trading interno: simulación forward de estrategias con precios reales.

Responsabilidad:
    Cada sesión congela la receta de un experimento terminado y la hace avanzar día a día (sesiones cerradas) con la
    misma decisión de órdenes (Ejecutor) y los mismos costes que el backtest, para ver cómo se comporta fuera de
    muestra. Interfaz pequeña:
        ServicioPaper(AlmacenPaper(raiz), fuente, ahora).crear_sesion(receta, modo_ejecucion=...) / .avanzar() / .listar()
    Módulos: `sesion` (modelos), `almacen` (disco y bloqueo), `simulador` (el avance), `calendario` (sesiones
    cerradas), `servicio` (interfaz de dominio) y `__main__` (CLI: `python -m ypyquant.paper avanzar`).

Se relaciona con:
    - ypyquant.execution (ejecutor, broker_simulado, calendario), ypyquant.estrategias (recetas, evaluar) y
      ypyquant.data (fuentes de precios).
    - app/api/rutas_paper_interno.py y app/web (página «Paper interno»); plan
      docs/planes/2026-11-20-configuraciones-y-paper-interno.md (fases 3 y 4).
"""

from ypyquant.paper.almacen import AlmacenPaper, SesionBloqueada, SesionInexistente
from ypyquant.paper.servicio import ControlAvance, ServicioPaper, TransicionInvalida
from ypyquant.paper.sesion import EstadoCartera, SesionPaper
from ypyquant.paper.simulador import ResultadoAvance, SimuladorPaper

__all__ = [
    "AlmacenPaper", "ControlAvance", "EstadoCartera", "ResultadoAvance", "ServicioPaper", "SesionBloqueada",
    "SesionInexistente", "SesionPaper", "SimuladorPaper", "TransicionInvalida",
]
