/**
 * Tabla «Salidas por motivo» de la ficha de un experimento de una estrategia de señales.
 * Responsabilidad: mostrar, por cada motivo de salida (stop de pérdida, take profit, trailing, tiempo, señal...) y lado de la
 * posición, cuántas salidas hubo, su PnL total y medio, la tasa de acierto y la duración media; así se ve qué regla gana o pierde
 * dinero. No pinta nada si el experimento no trae el desglose (estrategias sin señales).
 * Se relaciona con: paginas/ExperimentoDetalle.tsx (la usa), componentes/TablaOrdenable.tsx, lib/senales.ts (etiquetas),
 * lib/formato.ts y api/tipos.ts (`MotivoSalida`; lo calcula `desglose_por_motivo` de ypyquant.estrategias.evaluacion).
 */
import type { MotivoSalida } from '../api/tipos'
import { claseSigno, dinero, entero, ratio } from '../lib/formato'
import { etiquetaMotivo, tasaDeAcierto } from '../lib/senales'
import TablaOrdenable, { type Columna } from './TablaOrdenable'

const COLUMNAS: Columna<MotivoSalida>[] = [
  { clave: 'motivo', titulo: 'Motivo', valor: (f) => etiquetaMotivo(f.motivo), render: (f) => etiquetaMotivo(f.motivo) },
  { clave: 'lado', titulo: 'Posición', valor: (f) => f.lado, render: (f) => (f.lado === 'largo' ? 'Largo' : 'Corto') },
  { clave: 'n', titulo: 'Nº', alinear: 'der', valor: (f) => f.n, render: (f) => entero(f.n) },
  { clave: 'pnl_total', titulo: 'PnL total', alinear: 'der', valor: (f) => f.pnl_total, render: (f) => <span className={claseSigno(f.pnl_total)}>{dinero(f.pnl_total, 0)}</span> },
  { clave: 'pnl_medio', titulo: 'PnL medio', alinear: 'der', valor: (f) => f.pnl_medio, render: (f) => <span className={claseSigno(f.pnl_medio)}>{dinero(f.pnl_medio, 0)}</span> },
  { clave: 'tasa_acierto', titulo: 'Acierto', alinear: 'der', valor: (f) => f.tasa_acierto, render: (f) => tasaDeAcierto(f.tasa_acierto) },
  { clave: 'duracion', titulo: 'Duración media (días)', alinear: 'der', valor: (f) => f.duracion_media_dias, render: (f) => (f.duracion_media_dias == null ? '—' : ratio(f.duracion_media_dias, 1)) },
]

export default function TablaMotivosSalida({ filas }: { filas: MotivoSalida[] }) {
  if (filas.length === 0) return null
  return (
    <section aria-labelledby="h-motivos" className="space-y-2">
      <h2 id="h-motivos">Salidas por motivo</h2>
      <p className="text-xs text-suave">
        PnL de la operación de cierre (neto de su comisión, no de la de apertura). Los ajustes parciales por deriva no cuentan como salidas.
      </p>
      <TablaOrdenable titulo="Salidas por motivo" columnas={COLUMNAS} filas={filas} clave={(f) => `${f.motivo}-${f.lado}`} ordenInicial={{ clave: 'pnl_total', descendente: false }} />
    </section>
  )
}
