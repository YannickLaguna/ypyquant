/**
 * Utilidades puras de barridos y validación (sin React ni red; todo testeado en barridos.test.ts).
 * Responsabilidad:
 * - Construir la rejilla: rango inicio/fin/paso -> lista de valores, texto -> valores tipados y
 *   recuento de combinaciones (máx. 500, igual que la API).
 * - Mapa de calor: matriz de celdas para 1 o 2 ejes con los demás ejes fijados, meseta calculada
 *   en cliente para métricas distintas de la objetivo, elección del mejor y de la alternativa por meseta.
 * - Semáforos orientativos de DSR, PBO y p-valor de la permutación (umbrales de docs/validacion.md).
 * - Histogramas, regresión IS/OOS y años del periodo de la permutación.
 * Se relaciona con: componentes/MapaCalor.tsx, componentes/GraficosValidacion.tsx,
 * componentes/barrido/* y paginas/BarridoDetalle.tsx (consumidores), api/tipos.ts (contratos).
 */
import type { FilaSuperficie, ResultadoPermutacion } from '../api/tipos'

/** Máximo de combinaciones que acepta `POST /api/barridos`. */
export const MAX_COMBINACIONES = 500
/** Por encima de este nº de backtests la API avisa de que la permutación será lenta. */
export const UMBRAL_BACKTESTS_LENTO = 20_000

// ---- Rejilla ------------------------------------------------------------------------------

/** Redondea para evitar ruido de coma flotante (0.1 + 0.2 -> 0.3) al generar rangos. */
const limpiarDecimales = (v: number) => Number(v.toPrecision(12))

/**
 * Valores `inicio, inicio+paso, ...` hasta `fin` inclusive. Devuelve un error legible si el rango no
 * es válido o genera más de `MAX_COMBINACIONES` valores.
 */
export function rangoAValores(inicio: number, fin: number, paso: number): { valores: number[]; error?: string } {
  if (![inicio, fin, paso].every(Number.isFinite)) return { valores: [], error: 'Inicio, fin y paso deben ser números' }
  if (paso === 0) return { valores: [], error: 'El paso no puede ser 0' }
  if ((fin - inicio) * paso < 0) return { valores: [], error: 'El paso tiene el signo contrario al rango' }
  const n = Math.floor((fin - inicio) / paso + 1e-9) + 1
  if (n > MAX_COMBINACIONES) return { valores: [], error: `El rango genera ${n} valores (máx. ${MAX_COMBINACIONES})` }
  return { valores: Array.from({ length: n }, (_, i) => limpiarDecimales(inicio + i * paso)) }
}

export type TipoValor = 'integer' | 'number' | 'string' | 'boolean'

/**
 * Texto `1, 2, 3` -> valores tipados según el tipo del parámetro. Los enteros deben serlo; los
 * booleanos aceptan true/false; se descartan duplicados conservando el orden.
 */
export function parsearValores(texto: string, tipo: TipoValor): { valores: unknown[]; error?: string } {
  const trozos = texto.split(/[\s,;]+/).map((t) => t.trim()).filter(Boolean)
  const valores: unknown[] = []
  for (const trozo of trozos) {
    let valor: unknown
    if (tipo === 'integer' || tipo === 'number') {
      const n = Number(trozo.replace(',', '.'))
      if (Number.isNaN(n)) return { valores: [], error: `«${trozo}» no es un número` }
      if (tipo === 'integer' && !Number.isInteger(n)) return { valores: [], error: `«${trozo}» debe ser un entero` }
      valor = n
    } else if (tipo === 'boolean') {
      if (!['true', 'false'].includes(trozo.toLowerCase())) return { valores: [], error: `«${trozo}» no es true/false` }
      valor = trozo.toLowerCase() === 'true'
    } else {
      valor = trozo
    }
    if (!valores.includes(valor)) valores.push(valor)
  }
  return { valores }
}

/** Producto cartesiano de los ejes (0 si no hay ninguno o alguno está vacío). */
export function contarCombinaciones(rejilla: Record<string, unknown[]>): number {
  const ejes = Object.values(rejilla)
  if (ejes.length === 0) return 0
  return ejes.reduce((total, valores) => total * valores.length, 1)
}

// ---- Mapa de calor ------------------------------------------------------------------------

export interface CeldaMapa {
  /** Fila de la superficie en esa celda (null si aún no hay resultado o no existe). */
  fila: FilaSuperficie | null
  valor: number | null
}

export interface MatrizMapa {
  ejeX: string
  ejeY: string | null
  etiquetasX: string[]
  etiquetasY: string[]
  /** celdas[iy][ix]; con 1 solo eje hay una única fila. */
  celdas: CeldaMapa[][]
  minimo: number | null
  maximo: number | null
}

export type ModoMapa = 'valor' | 'meseta'

const esFinito = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
export const textoValor = (v: unknown): string => String(v)

/** Valor de una métrica en una fila (la objetivo viene en `objetivo`). */
export function valorMetrica(fila: FilaSuperficie, metrica: string, metricaObjetivo: string): number | null {
  const v = metrica === metricaObjetivo && esFinito(fila.objetivo) ? fila.objetivo : fila.metricas[metrica]
  return esFinito(v) ? v : null
}

/**
 * Meseta de cada fila para una métrica cualquiera: media de la métrica en los vecinos a ±1 posición
 * en un eje con el resto igual (la propia celda no cuenta). Igual que la definición de la API, que
 * solo la calcula para la métrica objetivo.
 */
export function calcularMeseta(superficie: FilaSuperficie[], metrica: string, metricaObjetivo: string): Map<string, number | null> {
  const clave = (pos: Record<string, number>) => Object.keys(pos).sort().map((k) => `${k}:${pos[k]}`).join('|')
  const indice = new Map(superficie.map((f) => [clave(f.posiciones), f]))
  const resultado = new Map<string, number | null>()
  for (const fila of superficie) {
    const vecinos: number[] = []
    for (const eje of Object.keys(fila.posiciones)) {
      for (const delta of [-1, 1]) {
        const vecino = indice.get(clave({ ...fila.posiciones, [eje]: fila.posiciones[eje] + delta }))
        const v = vecino ? valorMetrica(vecino, metrica, metricaObjetivo) : null
        if (v !== null) vecinos.push(v)
      }
    }
    resultado.set(fila.id, vecinos.length ? vecinos.reduce((a, b) => a + b, 0) / vecinos.length : null)
  }
  return resultado
}

/**
 * Matriz del mapa de calor para `ejeX` (columnas) y `ejeY` (filas, opcional). `fijos` da la POSICIÓN
 * en la rejilla de cada eje no mostrado (por defecto 0). Con `modo: 'meseta'` el color es la meseta
 * (la del servidor para la métrica objetivo; calculada aquí para el resto de métricas).
 */
export function matrizMapaCalor(
  superficie: FilaSuperficie[],
  ejes: Record<string, unknown[]>,
  ejeX: string,
  ejeY: string | null,
  fijos: Record<string, number>,
  metrica: string,
  metricaObjetivo: string,
  modo: ModoMapa = 'valor',
): MatrizMapa {
  const mesetaPropia = modo === 'meseta' && metrica !== metricaObjetivo ? calcularMeseta(superficie, metrica, metricaObjetivo) : null
  const otros = Object.keys(ejes).filter((e) => e !== ejeX && e !== ejeY)
  const coincideFijos = (f: FilaSuperficie) => otros.every((e) => f.posiciones[e] === (fijos[e] ?? 0))
  const filas = superficie.filter(coincideFijos)

  const valorDe = (f: FilaSuperficie): number | null => {
    if (modo === 'valor') return valorMetrica(f, metrica, metricaObjetivo)
    const m = mesetaPropia ? mesetaPropia.get(f.id) : f.meseta
    return esFinito(m) ? m : null
  }

  const etiquetasX = (ejes[ejeX] ?? []).map(textoValor)
  const etiquetasY = ejeY ? (ejes[ejeY] ?? []).map(textoValor) : ['']
  const celdas: CeldaMapa[][] = etiquetasY.map(() => etiquetasX.map(() => ({ fila: null, valor: null })))
  for (const f of filas) {
    const ix = f.posiciones[ejeX]
    const iy = ejeY ? f.posiciones[ejeY] : 0
    if (celdas[iy]?.[ix] !== undefined) celdas[iy][ix] = { fila: f, valor: valorDe(f) }
  }
  const finitos = celdas.flat().map((c) => c.valor).filter(esFinito)
  return {
    ejeX,
    ejeY,
    etiquetasX,
    etiquetasY,
    celdas,
    minimo: finitos.length ? Math.min(...finitos) : null,
    maximo: finitos.length ? Math.max(...finitos) : null,
  }
}

/** La mejor combinación por métrica objetivo y la mejor por meseta (alternativa más robusta). */
export function elegirMejores(superficie: FilaSuperficie[]): { mejor: FilaSuperficie | null; porMeseta: FilaSuperficie | null } {
  let mejor: FilaSuperficie | null = null
  let porMeseta: FilaSuperficie | null = null
  for (const f of superficie) {
    if (esFinito(f.objetivo) && (mejor === null || f.objetivo > (mejor.objetivo as number))) mejor = f
    if (esFinito(f.meseta) && (porMeseta === null || f.meseta > (porMeseta.meseta as number))) porMeseta = f
  }
  return { mejor, porMeseta }
}

/** Color de una celda: interpola rojo (mínimo) -> verde (máximo) con luminosidad baja para texto claro. */
export function colorEscala(valor: number | null, minimo: number | null, maximo: number | null): string {
  if (valor === null || minimo === null || maximo === null) return 'transparent'
  const t = maximo === minimo ? 0.5 : (valor - minimo) / (maximo - minimo)
  const tono = Math.round(8 + 130 * Math.min(1, Math.max(0, t)))
  return `hsl(${tono} 55% 30%)`
}

// ---- Semáforos ----------------------------------------------------------------------------

export type Semaforo = 'verde' | 'ambar' | 'rojo' | 'nulo'

/** DSR (probabilidad de que el Sharpe supere el de la suerte): > 0,95 verde; > 0,5 ámbar; resto rojo. */
export function semaforoDsr(dsr: number | null | undefined): Semaforo {
  if (!esFinito(dsr)) return 'nulo'
  return dsr > 0.95 ? 'verde' : dsr > 0.5 ? 'ambar' : 'rojo'
}

/** PBO: < 0,1 verde; 0,1–0,2 ámbar; > 0,2 rojo. */
export function semaforoPbo(pbo: number | null | undefined): Semaforo {
  if (!esFinito(pbo)) return 'nulo'
  return pbo < 0.1 ? 'verde' : pbo <= 0.2 ? 'ambar' : 'rojo'
}

/**
 * p-valor de la permutación. In-sample: < 1 % verde. Walk-forward: < 5 % verde con un periodo de 1 año
 * y < 1 % con ≥ 2 años (más datos, exigencia mayor). Ámbar hasta 5 %, rojo por encima.
 */
export function semaforoPValor(p: number | null | undefined, modo: 'in_sample' | 'walk_forward', anios: number | null): Semaforo {
  if (!esFinito(p)) return 'nulo'
  const umbralVerde = modo === 'walk_forward' && (anios === null || anios < 2) ? 0.05 : 0.01
  return p < umbralVerde ? 'verde' : p < 0.05 ? 'ambar' : 'rojo'
}

/** Costes que se comen más de la mitad del edge: se destacan en ámbar. */
export const UMBRAL_COSTES = 0.5
export const costesAltos = (v: number | null | undefined): boolean => esFinito(v) && v > UMBRAL_COSTES

/** Años del periodo sobre el que se permuta (en walk-forward: desde el fin del primer entrenamiento). */
export function aniosPeriodoPermutacion(
  resultado: Pick<ResultadoPermutacion, 'modo' | 'inicio_permutacion' | 'ventana_entreno_dias'>,
  inicioBarrido: string | undefined,
  finDesarrollo: string | undefined,
): number | null {
  if (!finDesarrollo) return null
  let inicio: Date | null = null
  if (resultado.inicio_permutacion) inicio = new Date(resultado.inicio_permutacion)
  else if (inicioBarrido) {
    inicio = new Date(inicioBarrido)
    if (resultado.modo === 'walk_forward' && resultado.ventana_entreno_dias) inicio.setDate(inicio.getDate() + resultado.ventana_entreno_dias)
  }
  const fin = new Date(finDesarrollo)
  if (!inicio || Number.isNaN(inicio.getTime()) || Number.isNaN(fin.getTime())) return null
  return (fin.getTime() - inicio.getTime()) / (365.25 * 86_400_000)
}

// ---- Estadística para gráficos -------------------------------------------------------------

export interface Intervalo {
  desde: number
  hasta: number
  cuenta: number
}

/** Histograma de `n` intervalos iguales entre el mínimo y el máximo de los datos finitos. */
export function histograma(datos: number[], n = 20, extra: number[] = []): Intervalo[] {
  const v = datos.filter(esFinito)
  if (v.length === 0) return []
  const todos = [...v, ...extra.filter(esFinito)]
  const minimo = Math.min(...todos)
  const maximo = Math.max(...todos)
  const ancho = (maximo - minimo) / n || 1
  const intervalos: Intervalo[] = Array.from({ length: n }, (_, i) => ({ desde: minimo + i * ancho, hasta: minimo + (i + 1) * ancho, cuenta: 0 }))
  for (const x of v) intervalos[Math.min(n - 1, Math.floor((x - minimo) / ancho))].cuenta++
  return intervalos
}

/** Regresión lineal OOS ~ IS por mínimos cuadrados (pendiente e intercepto); null con < 2 puntos o x constante. */
export function regresionLineal(pares: [number, number][]): { pendiente: number; intercepto: number } | null {
  const p = pares.filter(([x, y]) => esFinito(x) && esFinito(y))
  if (p.length < 2) return null
  const mx = p.reduce((s, [x]) => s + x, 0) / p.length
  const my = p.reduce((s, [, y]) => s + y, 0) / p.length
  const sxx = p.reduce((s, [x]) => s + (x - mx) ** 2, 0)
  if (sxx === 0) return null
  const pendiente = p.reduce((s, [x, y]) => s + (x - mx) * (y - my), 0) / sxx
  return { pendiente, intercepto: my - pendiente * mx }
}
