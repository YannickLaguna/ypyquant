# Membresías del S&P 500: arreglo y mantenimiento verificado (2026-11-22)

Rama `datos/membresias-sp500`. Fichero: `universos/sp500_membresias.json`; módulo: `src/ypyquant/data/membresias_sp500.py`.

## Problema

El fichero exportado de `TFM.db` daba 525 vigentes (el índice son 503 símbolos) y su último cambio era 2026-11-04: faltaban
las bajas de las altas recientes y no había forma de mantenerlo. Causas reales encontradas al compararlo con las fuentes:

- 19 bajas sin registrar desde 2026-02 (DAY, MTCH, MOH, LW, PAYC, HOLX, CTRA, EPAM, CPB, POOL, CAG, EA, AVB, BLDR, TAP, TTD,
  WBD, CTVA, KVUE).
- 3 cambios de ticker con el símbolo viejo y el nuevo ambos abiertos (BK -> BNY, EQR -> VMRK, PSKY -> SKYD).
- ILMN: el intervalo 2015-2024 nunca se cerró (salió del índice el 2024-06-24 y volvió el 2026-09-21).

## Fuentes evaluadas (descargadas y comprobadas el 2026-11-22)

| Fuente | Qué da | Formato real | Latencia | Automatizable sin clave | Veredicto |
|---|---|---|---|---|---|
| [Wikipedia: List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) | Lista actual (503 símbolos), fecha de alta | HTML, `table#constituents` (CC BY-SA) | mismo día o pocos días (edición voluntaria) | sí (User-Agent identificable) | **primaria** (lista) |
| [Wikipedia: Historical components of the S&P 500](https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500) | Tabla de cambios con fecha, altas, bajas y motivo desde 1976 | HTML, `table#changes`, cabecera de dos filas | igual | sí | **primaria** (fechas) |
| [SPDR SPY (State Street)](https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx) | Holdings diarios del ETF (505 filas: 503 símbolos + efectivo `-` + un marcador `2602335D`) | xlsx de una hoja, cabecera «As of 19-Nov-2026», columna Ticker | 1 día hábil | sí (200 con User-Agent genérico) | **verificación cruzada** |
| iShares IVV (`ishares.com`, CSV `.ajax?fileType=csv`) | Holdings diarios | devuelve una página HTML de bloqueo sin cabeceras de navegador | 1 día | **no** (anti-bot) | descartada |
| Vanguard VOO | Holdings | web dinámica, sin descarga estable | — | no comprobado/no estable | descartada |
| [S&P Dow Jones Indices: comunicados](https://press.spglobal.com/) | Anuncios oficiales de cambios (p. ej. [Okta](https://press.spglobal.com/2026-10-30-Okta-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600)) | notas de prensa en prosa | el día del anuncio (antes de la fecha efectiva) | no (texto libre, sin API) | fuente **manual** de referencia para dudas |

Fiabilidad observada: la lista de Wikipedia y el archivo de SPY coinciden exactamente (503 símbolos; solo difieren el efectivo y
el marcador de SPY). Wikipedia tiene dos defectos conocidos que el diseño absorbe: (1) la fecha de la tabla puede ser la del
**anuncio** y no la efectiva (Okta: tabla 2026-10-30, efectivo 2026-11-04 según S&P DJI); (2) la tabla de cambios usa a veces
un ticker distinto del de la lista (EchoStar: `SATS` en la tabla, `ECHO` en la lista y en SPY). SPY puede ir un día por detrás.
Condiciones de uso: Wikipedia permite el uso automatizado razonable con User-Agent identificable (una descarga por comprobación,
caché de 6 h); SPDR publica el archivo para descarga pública (una descarga al día). No se usa ninguna clave ni se contacta con Alpaca.

## Decisión

- **Fuente primaria:** Wikipedia (lista actual + tabla de cambios, esta para la fecha efectiva).
- **Verificación cruzada:** holdings de SPDR SPY. Un cambio solo se aplica si ambas fuentes lo confirman.
- **Manual:** comunicados de S&P DJI para resolver discrepancias y fechas dudosas.

## Qué se corrigió en el fichero

- 19 bajas cerradas con su fecha real (18 con la fecha de la tabla; KVUE con 2026-11-04 por S&P DJI) más los 3
  cambios de ticker (BK 2026-05-21, EQR 2026-08-18, PSKY 2026-10-06, con su nota de la empresa/SEC) y ILMN (baja 2024-06-24 + alta
  2026-09-21): 24 cambios en `universos/sp500_membresias_cambios.jsonl`.
- Resultado: **503 vigentes**, idénticos a la lista actual de Wikipedia y a SPY (salvo `-` y `2602335D` de SPY).
- Cada intervalo tocado lleva campos opcionales de procedencia (`fuente_baja`, `fuente_alta`, `motivo_baja`,
  `fecha_baja_aproximada`); `universos.cargar_membresias_sp500` los ignora, la lectura no cambia.
- Convención de los cambios de ticker (la del fichero, ya usada con FB -> META): el símbolo antiguo se cierra en la fecha del cambio y
  el nuevo conserva el intervalo original.
- No se tocó la historia anterior a 2026 aunque se detectó una posible discrepancia: BWA y TFX figuran con baja 2025-04-24 y la tabla
  de Wikipedia dice 2025-03-24. Pendiente de decidir (cambiaría resultados de backtests de ese periodo).

## Cómo funciona la actualización

```
python -m ypyquant.data.membresias_sp500 actualizar [--simular] [--sin-cache]
python -m ypyquant.data.membresias_sp500 estado
```

1. Descarga (una petición por fuente, caché 6 h en `universos/.cache_fuentes/`, User-Agent genérico).
2. `reconciliar`: cambios propuestos = diferencia entre la lista de Wikipedia y los vigentes del fichero. Confirmación con SPY
   (alta: debe figurar; baja: ya no debe figurar). Fecha efectiva de la tabla de cambios (ignora fechas futuras); si falta, hoy
   con `fecha_aproximada` y aviso. Todo pasa a **pendiente** si la lista primaria tiene fuera de 480-520 símbolos o hay más de 15
   cambios a la vez, o si SPY no está disponible.
3. Si hay cambios confirmados y no es `--simular`: copia `sp500_membresias.json.anterior`, escritura atómica
   (`escribir_json_atomico`), una línea por cambio en `sp500_membresias_cambios.jsonl` (fuente, verificado por, fecha de
   comprobación) y estado en `sp500_membresias_estado.json`. Los pendientes se guardan en `sp500_membresias_pendientes.json`.
4. Desde la API: el planificador del paper interno (cada 30 min) lanza la comprobación **como mucho una vez al día, tras las
   16:30 de Nueva York**, en un hilo aparte. Tras un error de red reintenta a las 2 h. Desactivar con `YPYQUANT_MEMBRESIAS_AUTO=0`.
5. `GET /api/universos/membresias/estado` (solo lee ficheros) y la tarjeta «Membresías del S&P 500» de la página Paper interno
   muestran última comprobación, nº de vigentes, cambios aplicados, pendientes y antigüedad.
6. Avisos de calidad: la antigüedad cuenta la última comprobación correcta (sin cambios en el índice el fichero no «envejece»);
   nº de vigentes esperado 503 ± 10.

Los ficheros de estado, pendientes, copia anterior y caché están en `.gitignore`; el JSON de membresías y el historial se versionan.

## Cómo revisar los pendientes

Un pendiente significa que Wikipedia y SPY no coinciden (normalmente por 1 día de desfase: se resuelve solo en la siguiente
comprobación). Si persiste varios días: comprobar el comunicado de S&P DJI, y si Wikipedia es la equivocada esperar a que se
corrija; si lo es SPY, no hay acción. No existe un comando para forzar un cambio: edita el JSON a mano (con su procedencia) y
añade la línea al `.jsonl`. Una fecha de tabla que sea la del anuncio y no la efectiva (caso Okta) se corrige a mano.

## Devolver los cambios a TFM.db (opcional)

La fuente de verdad pasa a ser este JSON. Para llevar los cambios a `TFM.db` hay que escribir `tickers.index_memberships` (lista
de `{index: "SP500", date_added, date_removed}` por símbolo) a partir del JSON; el sentido inverso es
`scripts/exportar_membresias_sp500.py`, que **sobrescribiría** este arreglo: no lo ejecutes sin antes volcar los cambios a TFM.db.

## Impacto en backtests

Las huellas de los experimentos guardados no cambian (dependen de la configuración, no del contenido del fichero; comprobado con
`tests/test_membresias_sp500.py` y `tests/test_huellas_guardadas.py` contra `data/` real). Los resultados al relanzar solo pueden
variar en periodos posteriores a 2024-06-24 y, sobre todo, en 2026 (bajas y altas corregidas, ILMN fuera 2024-2026).
