# Plan: almacén de documentos por id (2026-11-22)

Candidato 5 de la revisión de arquitectura. Rama `arq/almacen-documentos`.

## Problema

Cuatro almacenes reimplementaban lo mismo: carpeta por id, escribir/leer JSON atómico (con los reintentos de `PermissionError`
de Windows), listar, borrar, validar ids, y `contexto.py` recomponía a mano las rutas hermanas (`almacen.raiz.parent /
"barridos"`...). Las primitivas vivían en `estrategias/almacen.py` con nombre privado (`_escribir_json`) y los demás
almacenes las importaban de ahí; `reservado.py` las usaba para un archivo suelto.

## Qué se hizo

- `ypyquant/almacen_documentos.py` (nuevo, sin dominio): **interfaz** pequeña, **profundidad** en lo que esconde (escritura
  atómica con temporal único por proceso e hilo, `os.replace` con reintentos, lectura tolerante a bloqueos, validación de id,
  limpieza NumPy/NaN):
  - `AlmacenDocumentos(raiz, nombre, validar_id, marca)`: `crear`, `existe`, `ids`, `borrar`, `escribir_json`/`leer_json`/
    `tiene_json`/`nombres_json`, `escribir_parquet`/`leer_parquet`/`tiene_parquet`/`ruta_parquet`, `cerrojo`.
  - `ColeccionJson(raiz, prefijo)`: archivos planos `<id>.json` (comparativas).
  - `escribir_json_atomico` / `leer_json`: el archivo suelto (`reservado.json`).
  - Validadores de id: `id_sin_separadores` (ids por huella) e `id_slug` (ids elegidos por el usuario).
- `ypyquant/raiz_datos.py` (nuevo): `RaizDatos(base)` construye `experimentos`, `barridos`, `configuraciones`, `marcos` y
  `reservado`; `RaizDatos.de_experimentos(ruta)` para tests y scripts que eligen la carpeta de experimentos.
- Los cuatro almacenes y `RegistroReservado` son **adaptadores de dominio** sobre ese módulo (costura: el módulo, que cada
  adaptador recibe por composición en `self.documentos`). Conservan nombres, firmas, excepciones y formato.
- Consulta por configuración: `AlmacenMarcos.comparativas_de_configuracion(id_configuracion)` con índice **en memoria**
  (`id_configuracion -> ids de comparativa`, mismo criterio que el índice por receta: no duplica la verdad del disco), construido
  en el arranque (`indexar_comparativas`) o en la primera consulta, y mantenido en `registrar_comparativa`.
  `detalle_configuracion` ya no recorre todas las comparativas.
- `lanzar_permutacion`, `cancelar_permutacion` y el hilo de la prueba pasan de `contexto.py` a `ServicioExperimentos`
  (que recibe `ejecutor_barridos` y guarda sus cancelaciones). `contexto.py` queda en el dataclass de dependencias y construye
  los almacenes desde `RaizDatos`.

## Garantías

- `tests/test_formato_en_disco.py`: fixtures escritas **a mano con `pathlib`** (sin usar ningún almacén) con la estructura exacta
  de `data/experimentos|barridos|configuraciones|marcos|reservado.json`; comprueba que los almacenes las leen, listan,
  actualizan y borran, y que lo que escriben tiene los mismos nombres de archivo y el mismo texto JSON (`indent=2`,
  `ensure_ascii=False`). Se escribió **antes** de refactorizar y pasó contra el código anterior.
- `tests/test_huellas_guardadas.py` sigue leyendo el `data/` real local del repo principal (sin cambios).
- `tests/test_almacen_documentos.py`: interfaz del módulo, reintentos ante `PermissionError`, escritura concurrente, `RaizDatos`.
- La forma de la API HTTP no cambia (los 3 ficheros de rutas solo cambian cómo llaman al servicio).

## Límites y decisiones

- El índice de comparativas es en memoria: comparativas añadidas por otro proceso no se ven hasta `indexar_comparativas()`;
  las consultas descartan las que ya no apuntan a la configuración.
- `AlmacenBarridos(..., ruta_reservado=None)` mantiene el reservado junto a su carpeta (`raiz.parent`); `crear_app` y `Contexto`
  le pasan `RaizDatos.reservado`.
- El paper interno (sesiones, órdenes, fills) se construye como un adaptador más: un `AlmacenDocumentos` con su `nombre`,
  su validador de id y su `marca`, y una propiedad nueva en `RaizDatos`. Sin tocar el módulo.
- Fuera de alcance: DuckDB y las vistas Parquet de `AlmacenExperimentos` (dominio de experimentos, no de documentos).
