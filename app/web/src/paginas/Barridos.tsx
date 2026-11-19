/**
 * Lista de barridos: etiqueta, estrategia, ejes, nº de combinaciones, métrica, estado/progreso y
 * resultados disponibles (validación, walk-forward, permutación, reservado).
 * Responsabilidad: tabla ordenable con estados vacío/carga/error y sondeo mientras haya barridos en curso.
 * Se relaciona con: api/hooks.ts (useBarridos), componentes/TablaOrdenable.tsx, paginas/BarridoDetalle.tsx
 * (destino de cada fila) y paginas/EstrategiaDetalle.tsx (donde se lanzan).
 */
import { Link } from 'react-router-dom'
import { useBarridos } from '../api/hooks'
import type { ResumenBarrido } from '../api/tipos'
import { CajaError, Cargando, Insignia, Vacio } from '../componentes/Estado'
import TablaOrdenable, { type Columna } from '../componentes/TablaOrdenable'
import { fechaHora } from '../lib/formato'

const NOMBRES_RESULTADO: Record<string, string> = {
  validacion: 'validación',
  walk_forward: 'walk-forward',
  permutacion_in_sample: 'permutación IS',
  permutacion_walk_forward: 'permutación WF',
  reservado: 'reservado',
}

export const COLUMNAS_BARRIDOS: Columna<ResumenBarrido>[] = [
  {
    clave: 'etiqueta',
    titulo: 'Etiqueta',
    valor: (b) => b.etiqueta || b.id,
    render: (b) => (
      <Link to={`/barridos/${b.id}`} className="text-acento hover:underline">
        {b.etiqueta || b.id}
        {b.etiqueta && <span className="num ml-2 text-xs text-suave">{b.id}</span>}
      </Link>
    ),
  },
  { clave: 'estrategia', titulo: 'Estrategia', valor: (b) => b.estrategia, render: (b) => <span className="num">{b.estrategia}</span> },
  { clave: 'ejes', titulo: 'Ejes', render: (b) => <span className="num">{b.ejes.join(' × ') || '—'}</span> },
  { clave: 'n', titulo: 'Combinaciones', alinear: 'der', valor: (b) => b.n_combinaciones, render: (b) => b.n_combinaciones },
  { clave: 'metrica', titulo: 'Métrica', valor: (b) => b.metrica_objetivo, render: (b) => b.metrica_objetivo },
  {
    clave: 'estado',
    titulo: 'Estado',
    valor: (b) => b.estado,
    render: (b) => (
      <span className="inline-flex items-center gap-2">
        <Insignia estado={b.estado} />
        <span className="num text-xs text-suave">{b.progreso.hechos}/{b.progreso.total}{b.progreso.errores > 0 ? ` · ${b.progreso.errores} con error` : ''}</span>
      </span>
    ),
  },
  {
    clave: 'resultados',
    titulo: 'Resultados',
    valor: (b) => b.resultados.length,
    render: (b) => (
      <span className="whitespace-normal text-xs text-suave">
        {b.resultados.length ? b.resultados.map((r) => NOMBRES_RESULTADO[r] ?? r).join(', ') : 'ninguno'}
        {b.fecha_fin_desarrollo && ' · con reservado'}
      </span>
    ),
  },
  { clave: 'creado', titulo: 'Creado', valor: (b) => b.creado, render: (b) => fechaHora(b.creado) },
]

export default function Barridos() {
  const { data, isPending, error } = useBarridos()
  if (isPending) return <Cargando texto="Cargando barridos…" />
  if (error) return <CajaError error={error} />
  const barridos = data.barridos
  return (
    <div className="space-y-4">
      <header>
        <h1>Barridos</h1>
        <p className="text-suave">Rejillas de experimentos con validación estadística. Se lanzan desde la página de una <Link to="/estrategias" className="text-acento hover:underline">estrategia</Link>.</p>
      </header>
      {barridos.length === 0 ? (
        <Vacio titulo="Aún no hay barridos">Abre una estrategia, marca los parámetros que quieras barrer y pulsa «Lanzar barrido».</Vacio>
      ) : (
        <>
          <p className="text-xs text-suave">{barridos.length} {barridos.length === 1 ? 'barrido' : 'barridos'}</p>
          <TablaOrdenable titulo="Barridos" columnas={COLUMNAS_BARRIDOS} filas={barridos} clave={(b) => b.id} vacio="Sin barridos" />
        </>
      )}
    </div>
  )
}
