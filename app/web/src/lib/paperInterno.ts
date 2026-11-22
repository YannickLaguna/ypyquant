/**
 * Utilidades puras del paper interno: series a base 100 con fecha común y textos de estado.
 * Responsabilidad:
 * - `normalizarBase100`: lleva varias curvas de equity (y el benchmark de la primera) a base 100 en la PRIMERA fecha
 *   que tienen todas en común, para que sesiones dadas de alta en días distintos sean comparables; descarta lo anterior.
 * - `textoPendientes`: «Al día» o «n día(s) pendiente(s)» de la columna de estado de la lista.
 * - `textoModo`: etiqueta legible del modo de ejecución.
 * Se relaciona con: paginas/PaperInterno.tsx (gráfico de la selección), paginas/PaperInternoDetalle.tsx y
 * api/paper_interno.ts (tipos).
 */
import type { ModoEjecucionPaper, PuntoEquitySesion } from '../api/paper_interno'

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
