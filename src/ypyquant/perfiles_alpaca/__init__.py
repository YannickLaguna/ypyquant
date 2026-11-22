"""
Perfiles de Alpaca (multi-cuenta paper): descubrimiento de claves, vínculo con un experimento, estado en disco y redacción.

Responsabilidad: reunir lo que no depende ni de FastAPI ni del bot, para que ambos lo compartan.
- `perfiles`: qué cuentas hay (sin exponer secretos). `redaccion`: ocultar claves en logs.
- `almacen`: documentos de cada perfil en `data/paper_trading/<PERFIL>/`. `vinculo`: cuenta <-> experimento con receta congelada.
- `procesos`: pid vivo / terminar, para rehidratar los bots.
Se relaciona con: app/api (ServicioAlpaca, GestorBots, rutas_alpaca) y pipelines/paper_trading (el bot).
Se llama `perfiles_alpaca` y no `paper` porque `ypyquant.paper` es el paper trading interno.
"""
