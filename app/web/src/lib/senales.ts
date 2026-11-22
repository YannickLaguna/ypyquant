/**
 * Utilidades de presentación de las estrategias de señales (`EstrategiaSenales`).
 * Responsabilidad: traducir el motivo de cada operación (`entrada_larga`, `stop_loss`, `senal_contraria`...; también el
 * vocabulario anterior `senal`, `stop_perdida`...) a una etiqueta legible, nombrar el pseudo-ticker `CASH` como «Efectivo» y
 * resumir cuántas entradas y salidas hubo y por qué. Con cortos una VENTA puede abrir y una COMPRA cerrar: se clasifica por
 * motivo, no por lado.
 * Se relaciona con: paginas/ExperimentoDetalle.tsx (tabla de operaciones), componentes/TablaMotivosSalida.tsx,
 * componentes/Graficos.tsx (pesos) y api/tipos.ts (`OperacionFila.motivo`, `MotivoSalida`).
 */
import type { OperacionFila } from '../api/tipos'
import { porcentaje } from './formato'

const MOTIVOS: Record<string, string> = {
  entrada_larga: 'Entrada larga',
  entrada_corta: 'Entrada corta',
  stop_loss: 'Stop de pérdida',
  take_profit: 'Take profit',
  trailing: 'Trailing stop',
  tiempo: 'Tiempo máximo',
  senal_salida: 'Señal de salida',
  senal_contraria: 'Señal contraria',
  salida_universo: 'Fuera del universo',
  regla_propia: 'Regla propia',
  llamada_margen: 'Llamada de margen',
  ajuste: 'Ajuste',
  // Vocabulario anterior (experimentos guardados antes de los cortos)
  senal: 'Señal',
  stop_perdida: 'Stop de pérdida',
  stop_trailing: 'Trailing stop',
  fuera_universo: 'Fuera del universo',
}

const ENTRADAS = new Set(['entrada_larga', 'entrada_corta'])

export function etiquetaMotivo(motivo: string | null | undefined): string {
  if (!motivo) return ''
  return MOTIVOS[motivo] ?? motivo
}

/** Tasa de acierto (fracción 0-1) como porcentaje; «—» si no hay dato. */
export function tasaDeAcierto(fraccion: number | null | undefined): string {
  return fraccion == null ? '—' : porcentaje(fraccion)
}

/**
 * Nombre de una capa del gráfico de pesos. `CASH` es el efectivo no invertido («Efectivo»); con peso NEGATIVO es el préstamo
 * de margen del apalancamiento («Préstamo (margen)»). Un activo con peso negativo es una posición corta («AAPL (corto)»).
 * `peso` es el de la capa (en una fecha o su mínimo en el periodo); sin él se usa el nombre por defecto.
 */
export function nombreCapa(ticker: string, peso?: number): string {
  const negativo = peso !== undefined && peso < -0.0005
  if (ticker === 'CASH') return negativo ? 'Préstamo (margen)' : 'Efectivo'
  return negativo ? `${ticker} (corto)` : ticker
}

/** Exposición bruta (suma de |w|) y neta (suma de w) de unos pesos, sin el `CASH`; null si no hay pesos. */
export function exposiciones(pesos: { ticker: string; peso: number }[]): { bruta: number; neta: number } | null {
  const posiciones = pesos.filter((p) => p.ticker !== 'CASH')
  if (posiciones.length === 0) return null
  return {
    bruta: posiciones.reduce((s, p) => s + Math.abs(p.peso), 0),
    neta: posiciones.reduce((s, p) => s + p.peso, 0),
  }
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
    // `senal` (anterior a los cortos) era la entrada si compraba y la salida si vendía
    const esEntrada = ENTRADAS.has(o.motivo) || (o.motivo === 'senal' && o.lado === 'compra')
    if (o.motivo === 'ajuste') resumen.ajustes++
    else if (esEntrada) resumen.entradas++
    else {
      resumen.salidas++
      resumen.porMotivo[o.motivo] = (resumen.porMotivo[o.motivo] ?? 0) + 1
    }
  }
  return resumen
}
