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
- Revisión de fechas históricas (2026-11-22, rama `datos/membresias-fechas`): ver la sección «Fechas históricas contrastadas».

## Fechas históricas contrastadas

Se recorrió la tabla de cambios completa de Wikipedia (414 filas) con `AdaptadorWikipediaCambios` y se comparó con el fichero. El
fichero usa como fecha el primer día de pertenencia (comunicados de S&P DJI: «effective prior to the open of trading on D»,
cambio al cierre anterior); Wikipedia a veces pone la fecha del anuncio u otra. Las diferencias de 1-6 días (unas 150 filas) son esa
convención y no se tocan. Discrepancias de 7 o más días entre el mismo intervalo del fichero y la tabla:

| Símbolo | Fichero | Wikipedia | Resultado | Fuente |
|---|---|---|---|---|
| BWA (baja) | 2025-04-24 | 2025-03-24 | **corregido a 2025-03-24** | [S&P DJI 2025-03-07](https://press.spglobal.com/2025-03-07-DoorDash,-TKO-Group-Holdings,-Williams-Sonoma-and-Expand-Energy-Set-to-Join-S-P-500-Others-to-Join-S-P-100,-S-P-MidCap-400-and-S-P-SmallCap-600): «effective prior to the open of trading on Monday, March 24» |
| TFX (baja) | 2025-04-24 | 2025-03-24 | **corregido a 2025-03-24** | ídem |
| ALK (baja) | 2023-10-18 | 2023-12-18 | **corregido a 2023-12-18** | [S&P DJI 2023-12-01](https://press.spglobal.com/2023-12-01-Uber-Technologies,-Jabil-and-Builders-FirstSource-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600): «effective prior to the open of trading on Monday, December 18» |
| SEDG (baja) | 2023-10-18 | 2023-12-18 | **corregido a 2023-12-18** | ídem |
| SEE (baja) | 2023-10-18 | 2023-12-18 | **corregido a 2023-12-18** | ídem |
| SVU (baja) / PSX (alta) | 2012-05-01 | 2012-04-23 | fichero correcto (Wikipedia = anuncio) | [S&P DJI 2012-04-23](https://press.spglobal.com/2012-04-23-S-P-Indices-Announces-Changes-to-U-S-Indices): efectivo tras el cierre del 2012-04-30 |
| EP (baja) / KMI (alta) | 2012-05-25 | 2012-05-17 | fichero correcto (Wikipedia = anuncio) | [S&P DJI 2012-05-17](https://press.spglobal.com/2012-05-17-S-P-Indices-Announces-Change-to-U-S-Index): tras el cierre del 2012-05-24 |
| LRCX (alta) | 2012-06-29 | 2012-06-05 | **pendiente de verificar** | el comunicado del 2012-05-10 dice «TBA» |
| NWSA (alta) | 2013-08-01 | 2013-07-01 | **pendiente de verificar** | sin fuente localizada (escisión 2013-06-28) |
| GRN (baja) / CCL (alta) | 1998-12-22 | 1998-12-11 | pendiente de verificar | no se buscó (fuera de la cobertura fiable, anterior a 1999) |
| AN (baja) / CPWR (alta) | 1999-01-04 | 1998-12-11 | pendiente de verificar | ídem |
| CBE (baja) / ARG (alta) | 2009-09-09 | 2009-09-28 | pendiente de verificar | no se buscó |
| IQV (alta) | 2017-08-29 | 2017-11-15 | pendiente de verificar | la fila de Wikipedia es un cambio de nombre/ticker (QuintilesIMS), probablemente no un alta nueva |

Además hay ~45 filas de la tabla cuyo símbolo no existe en el fichero o cuyo intervalo es otro (reutilización de tickers, cambios de
nombre: p. ej. SATS/ECHO, MBC, TFCF, JCP, WYN, DPS): no son discrepancias de fecha y no se tocan. KVUE y OKTA (2026-10-30 en
Wikipedia, 2026-11-04 en el fichero) ya estaban corregidas con el comunicado de S&P DJI. Cada corrección lleva `fuente_baja` y
`fecha_baja_corregida_desde` en el intervalo y una línea `correccion_fecha_baja` en `sp500_membresias_cambios.jsonl`.

## Contraste con la fecha del dato de SPY

La cabecera del xlsx de SPY da la fecha de sus holdings. En cada comprobación con SPY disponible se guarda la observación
(fecha de holdings + símbolos) en `universos/sp500_membresias_observaciones_spy.json` (últimas 40, ignorado por git).
Para cada alta o baja propuesta por Wikipedia con fecha de tabla `fw` y dato de SPY de fecha `fs`:

1. SPY aún no lo refleja: **pendiente** (ya lo era); si `fw <= fs` el motivo indica que `fw` puede ser la fecha del anuncio. Se reintenta
   en cada comprobación.
2. SPY lo refleja y hay una observación anterior que no: fecha efectiva = **primera fecha de holdings de SPY que lo refleja**; si
   difiere de `fw`, aviso y `fecha_wikipedia` en el historial.
3. SPY lo refleja y no hay observación anterior (primera ejecución): si `fw <= fs` se acepta `fw` con aviso de que no se pudo
   contrastar; si `fw > fs` (SPY ya lo refleja antes de la fecha de Wikipedia) queda pendiente hasta que el dato de SPY alcance `fw`.

Caso Okta reproducido en tests: tabla 2026-10-30; SPY 11-02 y 11-03 sin Okta (pendiente por posible anuncio); SPY 11-04 con Okta
(se aplica con 2026-11-04, aviso por diferir de Wikipedia). Límite: si el índice rebalancea al cierre del día anterior y SPY
publica ya los holdings posteriores a ese cierre, la primera fecha de SPY puede ser un día hábil anterior a la efectiva; no se
dispone de series de SPY para calibrarlo, así que conviene mirar los avisos de las primeras semanas.

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
añade la línea al `.jsonl`. Una fecha de tabla que sea la del anuncio y no la efectiva (caso Okta) la contrasta la regla de SPY (sección anterior).

## Devolver los cambios a TFM.db (opcional)

La fuente de verdad pasa a ser este JSON. Para llevar los cambios a `TFM.db` hay que escribir `tickers.index_memberships` (lista
de `{index: "SP500", date_added, date_removed}` por símbolo) a partir del JSON; el sentido inverso es
`scripts/exportar_membresias_sp500.py`, que **sobrescribiría** este arreglo: no lo ejecutes sin antes volcar los cambios a TFM.db.

## Impacto en backtests

Las huellas de los experimentos guardados no cambian (dependen de la configuración, no del contenido del fichero; comprobado con
`tests/test_membresias_sp500.py` y `tests/test_huellas_guardadas.py` contra `data/` real). Los resultados al relanzar solo pueden
variar en periodos posteriores a 2024-06-24 y, sobre todo, en 2026 (bajas y altas corregidas, ILMN fuera 2024-2026).
Correcciones de fechas (rama `datos/membresias-fechas`): ALK, SEDG y SEE pasan a ser miembros hasta 2023-12-18 (antes 2023-10-18) y BWA y TFX
hasta 2025-03-24 (antes 2025-04-24). Los 8 experimentos guardados con `sp500_pit` (bayes_stein, bl_prior x2, canal_rsi_volatilidad,
cruce_medias, momentum, pesos_iguales, robusta_elipsoide) cubren 2015/2018-2024-12-31: ninguno toca 2025-03/04, pero todos incluyen
2023-10/12, así que sus resultados al relanzar podrían cambiar (las huellas no).
