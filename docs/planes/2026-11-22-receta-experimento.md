# Plan: Receta de experimento + servicio de experimentos (2026-11-22)

Candidato 2 de la revisión de arquitectura. Rama `arq/receta-experimento`.

## Problema

Ir de «configuración -> backtest» obligaba a saltar entre ocho módulos: la huella vivía en `experimentos.py`, la copia de
8 campos con fechas falsas (2000-01-01) en `configuraciones.py`, otra canonización en `marcos.py` (`_clave`), las reglas de
dominio en las rutas (`_backtests_de_la_receta`, fechas por defecto, borrado múltiple, comparativas) y el almacén recorría
todo el disco en cada GET (O(N)).

## Qué se hizo

- `estrategias/receta.py` (nuevo): `Entorno` (universo, frecuencia, costes, benchmark, semilla, historia previa),
  `Receta` = `Entorno` + estrategia + parámetros, y la canonización única (`huella_de`, `valor_comparable`).
- `ConfiguracionExperimento` hereda de `Receta` (receta + periodo + capital + validaciones + etiqueta; sigue plana, así que
  `config.json` y el cuerpo de la API no cambian). `ConfiguracionGuardada` hereda de `Receta` (adiós copia de 8 campos y
  fechas falsas); `MarcoComparacion` hereda de `Entorno` y compone la receta con `receta_para`. El ceteris paribus usa
  `valor_comparable`, la misma canonización que la huella.
- `estrategias/servicio_experimentos.py` (nuevo, sin FastAPI): `ServicioExperimentos` (lanzar, reutilizar por huella, forzar,
  estados, barridos, comparativas, backtest y comparativa de una configuración, borrado múltiple, backtests de una receta).
  Lanza excepciones del dominio; las rutas solo las traducen a 404/409/422. `contexto.py` ya no lanza nada.
- Índice por receta: **en memoria** (`indice_recetas.py`), construido al arrancar (`AlmacenExperimentos.indexar()`) y
  mantenido en `registrar`/`borrar`. Se descartó un fichero índice: duplica la verdad del disco y exige coherencia ante
  escrituras concurrentes y sincronización; construirlo cuesta una lectura de cada `config.json`, una vez por proceso.
  Límite: cambios hechos fuera del proceso (copiar o borrar carpetas a mano) requieren `indexar()` o reiniciar; las
  consultas ya descartan los ids cuya carpeta no existe.

## Garantías

- `tests/test_huellas_guardadas.py` recalcula (sin modificar) la huella de todos los experimentos propios de
  `data/experimentos` (`<estrategia>-<hash10>`; los `tfm-*` tienen otro esquema y se excluyen), la `huella_estrategia` de las
  configuraciones guardadas y comprueba que marcos y comparativas se cargan y apuntan a los mismos ids.
- `tests/test_receta.py` fija huellas calculadas con el código anterior.
- La forma OpenAPI es idéntica salvo docstrings y el orden de `required`.

## Fuera de alcance

Carga de datos y resolución de universo (candidato 3, otro agente); unificar los almacenes de documentos (candidato 5).
`lanzar_permutacion` sigue en `contexto.py`.
