/**
 * Utilidades puras del paper interno: series a base 100 con fecha común y textos de estado.
 * Responsabilidad:
 * - `normalizarBase100`: lleva varias curvas de equity (y el benchmark de la primera) a base 100 en la PRIMERA fecha
 *   que tienen todas en común, para que sesiones dadas de alta en días distintos sean comparables; descarta lo anterior.
 * - `textoPendientes`: «Al día» o «n día(s) pendiente(s)» de la columna de estado de la lista.
 * - `textoModo`: etiqueta legible del modo de ejecución.
 * - Tiempo de la sesión (puras, en español): `textoDuracion` («3 meses y 5 días»), `textoSesiones` («11 sesiones de mercado»),
 *   `textoPeriodo` (alta → última fecha), `textoFrecuencia` («rebalanceo mensual»), `textoProximoRebalanceo` y
 *   `resumenTiempo`, que los reúne para la tabla y la ficha (la fecha del próximo rebalanceo la calcula la API).
 * Se relaciona con: paginas/PaperInterno.tsx (gráfico de la selección), paginas/PaperInternoDetalle.tsx y
 * api/paper_interno.ts (tipos).
 */
import type { ModoEjecucionPaper, PuntoEquitySesion } from '../api/paper_interno'
import { fechaCorta } from './formato'

export interface PuntoBase100 {
  time: string
  value: number
}

export interface CurvaBase100 {
  id: string
  puntos: PuntoBase100[]
  benchmark: PuntoBase100[]
}

/**
 * Cada curva a base 100 en la primera fecha común a todas. Sin fecha común (o con menos de dos curvas con datos
 * en ella) cada curva se normaliza en su propio primer punto. El benchmark se normaliza igual con su valor de esa fecha.
 */
export function normalizarBase100(curvas: { id: string; equity: PuntoEquitySesion[] }[]): CurvaBase100[] {
  const conDatos = curvas.filter((c) => c.equity.length > 0)
  const comunes = conDatos.length
    ? conDatos.map((c) => new Set(c.equity.map((p) => p.fecha))).reduce((a, b) => new Set([...a].filter((f) => b.has(f))))
    : new Set<string>()
  const fechaBase = [...comunes].sort()[0]
  return conDatos.map((c) => {
    const desde = fechaBase ?? c.equity[0].fecha
    const puntos = c.equity.filter((p) => p.fecha >= desde)
    const base = puntos[0]?.equity ?? 1
    const baseBench = puntos.find((p) => p.benchmark != null)?.benchmark ?? null
    return {
      id: c.id,
      puntos: puntos.map((p) => ({ time: p.fecha, value: (p.equity / base) * 100 })),
      benchmark: baseBench ? puntos.filter((p) => p.benchmark != null).map((p) => ({ time: p.fecha, value: ((p.benchmark as number) / baseBench) * 100 })) : [],
    }
  })
}

export function textoPendientes(diasPendientes: number, alDia: boolean): string {
  if (alDia || diasPendientes <= 0) return 'Al día'
  return `${diasPendientes} ${diasPendientes === 1 ? 'día pendiente' : 'días pendientes'}`
}

export function textoModo(modo: ModoEjecucionPaper): string {
  return modo === 'cierre' ? 'Al cierre (optimista)' : 'Apertura siguiente'
}

// ---- Cuánto tiempo lleva la sesión y cuándo se rebalancea ------------------------------------------

const MS_DIA = 86_400_000

/** Milisegundos UTC de una fecha `YYYY-MM-DD` (sin zona horaria: evita saltos por cambio de hora). */
const utc = (fecha: string): number => {
  const [a, m, d] = fecha.slice(0, 10).split('-').map(Number)
  return Date.UTC(a, m - 1, d)
}

const plural = (n: number, singular: string, pluralTexto: string) => `${n} ${n === 1 ? singular : pluralTexto}`

/** Suma `meses` a una fecha UTC recortando el día al último del mes (31 ene + 1 mes = 28 feb). */
function sumarMeses(ms: number, meses: number): number {
  const f = new Date(ms)
  const destino = Date.UTC(f.getUTCFullYear(), f.getUTCMonth() + meses, 1)
  const ultimoDia = new Date(Date.UTC(new Date(destino).getUTCFullYear(), new Date(destino).getUTCMonth() + 1, 0)).getUTCDate()
  return destino + (Math.min(f.getUTCDate(), ultimoDia) - 1) * MS_DIA
}

/**
 * Tiempo de calendario entre `desde` y `hasta` (ambos días incluidos) en lenguaje natural: «5 días», «1 semana y 5 días»,
 * «3 meses y 5 días», «1 año y 2 meses» (con años se omiten los días).
 */
export function textoDuracion(desde: string, hasta: string): string {
  const inicio = utc(desde)
  const fin = utc(hasta) + MS_DIA // exclusivo: el día de `hasta` también cuenta
  let meses = 0
  while (sumarMeses(inicio, meses + 1) <= fin) meses++
  const dias = Math.round((fin - sumarMeses(inicio, meses)) / MS_DIA)
  if (meses >= 12) {
    const anios = Math.floor(meses / 12)
    const resto = meses % 12
    return resto ? `${plural(anios, 'año', 'años')} y ${plural(resto, 'mes', 'meses')}` : plural(anios, 'año', 'años')
  }
  if (meses > 0) return dias ? `${plural(meses, 'mes', 'meses')} y ${plural(dias, 'día', 'días')}` : plural(meses, 'mes', 'meses')
  if (dias < 7) return plural(dias, 'día', 'días')
  const semanas = Math.floor(dias / 7)
  const sueltos = dias % 7
  return sueltos ? `${plural(semanas, 'semana', 'semanas')} y ${plural(sueltos, 'día', 'días')}` : plural(semanas, 'semana', 'semanas')
}

/** «11 sesiones de mercado»: días de trading ya procesados (no días naturales). */
export function textoSesiones(n: number): string {
  return n > 0 ? plural(n, 'sesión de mercado', 'sesiones de mercado') : 'sin sesiones procesadas'
}

/** Periodo cubierto: fecha de alta → última fecha procesada («1 mar 2021 → 12 mar 2021»). */
export function textoPeriodo(fechaAlta: string, ultimaFecha: string | null): string {
  return `${fechaCorta(fechaAlta)} → ${ultimaFecha ? fechaCorta(ultimaFecha) : 'sin días procesados'}`
}

const FRECUENCIAS: Record<string, string> = { daily: 'diario', weekly: 'semanal', monthly: 'mensual', quarterly: 'trimestral' }

/** «rebalanceo mensual» (la frecuencia de la receta de la sesión). */
export function textoFrecuencia(frecuencia: string): string {
  return `rebalanceo ${FRECUENCIAS[frecuencia] ?? frecuencia}`
}

/** Fecha del próximo rebalanceo y distancia a `hoy`; una fecha pasada significa «en cuanto se procese la próxima sesión». */
export function textoProximoRebalanceo(proximo: string | null, hoy: string = hoyIso()): string {
  if (!proximo) return '—'
  const dias = Math.round((utc(proximo) - utc(hoy)) / MS_DIA)
  if (dias < 0) return 'en la próxima sesión que se procese'
  const relativo = dias === 0 ? 'hoy' : dias === 1 ? 'mañana' : `en ${dias} días`
  return `${fechaCorta(proximo)} (${relativo})`
}

/** Hoy en formato `YYYY-MM-DD` según el reloj del navegador. */
export function hoyIso(): string {
  const f = new Date()
  return `${f.getFullYear()}-${String(f.getMonth() + 1).padStart(2, '0')}-${String(f.getDate()).padStart(2, '0')}`
}

export interface DatosTiempoSesion {
  fecha_alta: string
  ultima_fecha: string | null
  dias: number
  dias_pendientes: number
  al_dia: boolean
  frecuencia_rebalanceo: string
  proximo_rebalanceo: string | null
}

export interface TextosTiempoSesion {
  periodo: string
  /** «11 sesiones de mercado · 2 semanas» */
  duracion: string
  pendientes: string
  frecuencia: string
  proximo: string
}

/** Todos los textos de tiempo de una sesión a la vez (los usan la tabla y la ficha). */
export function resumenTiempo(s: DatosTiempoSesion, hoy: string = hoyIso()): TextosTiempoSesion {
  return {
    periodo: textoPeriodo(s.fecha_alta, s.ultima_fecha),
    duracion: s.ultima_fecha && s.dias > 0 ? `${textoSesiones(s.dias)} · ${textoDuracion(s.fecha_alta, s.ultima_fecha)}` : textoSesiones(0),
    pendientes: textoPendientes(s.dias_pendientes, s.al_dia),
    frecuencia: textoFrecuencia(s.frecuencia_rebalanceo),
    proximo: textoProximoRebalanceo(s.proximo_rebalanceo, hoy),
  }
}

// ---- Comprobación automática (temporizador de la API) ---------------------------------------------

export interface UltimaComprobacionPaper {
  hora: string
  resultado: 'al_dia' | 'avanzado' | 'avance_en_curso' | 'sin_datos' | 'sin_novedades' | 'error'
  dias_procesados: number
  error: string | null
}

/** Qué hizo la última comprobación automática, en una frase (sin la hora). */
export function textoResultadoComprobacion(u: UltimaComprobacionPaper): string {
  switch (u.resultado) {
    case 'al_dia':
      return 'no había días pendientes'
    case 'avanzado':
      return `${plural(u.dias_procesados, 'día procesado', 'días procesados')}`
    case 'avance_en_curso':
      return 'ya había un avance en curso (no se solapan)'
    case 'sin_datos':
      return 'había días pendientes pero el proveedor aún no tiene sus precios'
    case 'sin_novedades':
      return 'sin precios nuevos: no se reintenta hasta la siguiente sesión cerrada'
    default:
      return `error: ${u.error ?? 'desconocido'}`
  }
}
