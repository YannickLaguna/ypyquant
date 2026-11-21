/**
 * Utilidades de presentación de las estrategias de señales (`EstrategiaSenales`).
 * Responsabilidad: traducir el motivo de cada operación (`senal`, `stop_perdida`...) a una etiqueta legible,
 * nombrar el pseudo-ticker `CASH` como «Efectivo» y resumir cuántas entradas y salidas hubo y por qué.
 * Se relaciona con: paginas/ExperimentoDetalle.tsx (tabla de operaciones), componentes/Graficos.tsx (pesos)
 * y api/tipos.ts (`OperacionFila.motivo`).
 */
import type { OperacionFila } from '../api/tipos'

const MOTIVOS: Record<string, string> = {
  senal: 'Señal',
  stop_perdida: 'Stop de pérdida',
  stop_trailing: 'Stop trailing',
  take_profit: 'Take profit',
  tiempo: 'Tiempo máximo',
  fuera_universo: 'Fuera del universo',
  regla_propia: 'Regla propia',
  ajuste: 'Ajuste',
}

export function etiquetaMotivo(motivo: string | null | undefined): string {
  if (!motivo) return ''
  return MOTIVOS[motivo] ?? motivo
}

/** Nombre de una capa del gráfico de pesos: el pseudo-ticker `CASH` (efectivo no invertido) se muestra como «Efectivo». */
export function nombreCapa(ticker: string): string {
  return ticker === 'CASH' ? 'Efectivo' : ticker
}

export interface ResumenMotivos {
  entradas: number
  salidas: number
  /** Salidas por motivo (sin los ajustes). */
  porMotivo: Record<string, number>
  /** Compras o ventas parciales que solo recolocan pesos (deriva, costes). */
  ajustes: number
}

/** Resumen de entradas y salidas por motivo; null si ninguna operación trae motivo (estrategia sin señales). */
export function resumenMotivos(operaciones: OperacionFila[]): ResumenMotivos | null {
  if (!operaciones.some((o) => o.motivo)) return null
  const resumen: ResumenMotivos = { entradas: 0, salidas: 0, porMotivo: {}, ajustes: 0 }
  for (const o of operaciones) {
    if (!o.motivo) continue
    if (o.motivo === 'ajuste') resumen.ajustes++
    else if (o.lado === 'compra') resumen.entradas++
    else {
      resumen.salidas++
      resumen.porMotivo[o.motivo] = (resumen.porMotivo[o.motivo] ?? 0) + 1
    }
  }
  return resumen
}
