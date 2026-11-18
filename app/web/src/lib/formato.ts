/**
 * Formateo de números y fechas en español (es-ES) para toda la interfaz.
 * Responsabilidad: un único sitio para porcentajes, ratios, dinero y fechas; los valores nulos se
 * muestran como «—». Las fechas usan siempre el locale `es` (no el del dispositivo).
 * Se relaciona con: componentes/* y paginas/* (todos formatean con estas funciones).
 */

const nulo = '—'
const esfinito = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v)

export function porcentaje(v: number | null | undefined, decimales = 1): string {
  if (!esfinito(v)) return nulo
  return `${(v * 100).toLocaleString('es-ES', { minimumFractionDigits: decimales, maximumFractionDigits: decimales })} %`
}

export function ratio(v: number | null | undefined, decimales = 2): string {
  if (!esfinito(v)) return nulo
  return v.toLocaleString('es-ES', { minimumFractionDigits: decimales, maximumFractionDigits: decimales })
}

export function dinero(v: number | null | undefined, decimales = 0): string {
  if (!esfinito(v)) return nulo
  return v.toLocaleString('es-ES', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimales,
    maximumFractionDigits: decimales,
  })
}

export function entero(v: number | null | undefined): string {
  return esfinito(v) ? v.toLocaleString('es-ES', { maximumFractionDigits: 0 }) : nulo
}

/** 'YYYY-MM-DD' o ISO -> '3 nov 2026' (locale es). */
export function fechaCorta(texto: string | null | undefined): string {
  if (!texto) return nulo
  const fecha = new Date(texto.length === 10 ? `${texto}T00:00:00` : texto)
  if (Number.isNaN(fecha.getTime())) return texto
  return fecha.toLocaleDateString('es-ES', { day: 'numeric', month: 'short', year: 'numeric' })
}

export function fechaHora(texto: string | null | undefined): string {
  if (!texto) return nulo
  const fecha = new Date(texto)
  if (Number.isNaN(fecha.getTime())) return texto
  return fecha.toLocaleString('es-ES', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

/** Clase de color según el signo (verde/rojo/neutro). */
export function claseSigno(v: number | null | undefined): string {
  if (!esfinito(v) || v === 0) return ''
  return v > 0 ? 'text-pos' : 'text-neg'
}

export function resumenParametros(parametros: Record<string, unknown>): string {
  const partes = Object.entries(parametros).map(([k, v]) => `${k}=${typeof v === 'number' ? v : JSON.stringify(v)}`)
  return partes.length ? partes.join(', ') : 'sin parámetros'
}

export function resumenUniverso(universo: string | string[]): string {
  return typeof universo === 'string' ? universo : `${universo.length} tickers`
}
