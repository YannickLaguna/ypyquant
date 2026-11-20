/**
 * Detalle de un marco: entorno, edición, lanzador de comparativas y comparativas ya lanzadas.
 * Responsabilidad: mostrar el entorno común, permitir editarlo (los experimentos ya lanzados no cambian: son
 * recetas congeladas), borrarlo, lanzar una comparativa con varias estrategias y reabrir las anteriores en Comparar.
 * Se relaciona con: componentes/FormularioMarco.tsx, componentes/LanzadorComparativa.tsx, api/hooks.ts y
 * paginas/Comparar.tsx (destino de «Abrir en Comparar»).
 */
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useBorrarMarco, useGuardarMarco, useMarco } from '../api/hooks'
import { CajaError, Cargando } from '../componentes/Estado'
import FormularioMarco from '../componentes/FormularioMarco'
import LanzadorComparativa, { enlaceComparar } from '../componentes/LanzadorComparativa'
import { dinero, fechaCorta, resumenUniverso } from '../lib/formato'

export default function MarcoDetalle() {
  const { id } = useParams()
  const navegar = useNavigate()
  const { data, isPending, error } = useMarco(id)
  const guardar = useGuardarMarco()
  const borrar = useBorrarMarco()
  const [editando, setEditando] = useState(false)

  if (isPending) return <Cargando />
  if (error) return <CajaError error={error} />
  const { marco: m, comparativas } = data

  const eliminar = () => {
    if (window.confirm(`¿Borrar el marco ${m.nombre}? Las comparativas ya lanzadas se conservan.`)) borrar.mutate(m.id, { onSuccess: () => navegar('/marcos') })
  }

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-suave"><Link to="/marcos" className="hover:text-acento">Marcos</Link> /</p>
          <h1>{m.nombre}</h1>
          <p className="text-suave">
            {m.universo_etiqueta ?? resumenUniverso(m.universo)} · {fechaCorta(m.fecha_inicio)} → {fechaCorta(m.fecha_fin)} · rebalanceo {m.frecuencia_rebalanceo ?? 'propio de cada estrategia'} ·
            comisión {m.comision} · slippage {m.slippage} · benchmark {m.benchmark} · capital {dinero(m.capital_inicial)} · semilla {m.semilla}
          </p>
          <p className="num text-xs text-suave">comunes: {Object.entries(m.parametros_comunes).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ') || 'ninguno'}</p>
          {m.notas && <p className="mt-1">{m.notas}</p>}
          {m.frecuencia_rebalanceo === null && (
            <p role="note" className="mt-2 rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">Sin frecuencia fija: cada estrategia rebalancea con la suya y la comparación no será ceteris paribus.</p>
          )}
        </div>
        <div className="flex gap-2">
          <button type="button" className="boton" onClick={() => setEditando((v) => !v)} aria-expanded={editando}>{editando ? 'Cancelar edición' : 'Editar'}</button>
          <button type="button" className="boton boton-peligro" onClick={eliminar} disabled={borrar.isPending}>Borrar</button>
        </div>
      </header>

      {editando && (
        <section aria-labelledby="h-editar-marco" className="space-y-2">
          <h2 id="h-editar-marco">Editar marco</h2>
          <p className="text-suave">Los experimentos ya lanzados no cambian; las próximas comparativas usarán el entorno nuevo.</p>
          {guardar.error && <CajaError error={guardar.error} titulo="No se pudo guardar el marco" />}
          <FormularioMarco marco={m} guardando={guardar.isPending} alGuardar={(marco) => guardar.mutate({ id: m.id, marco }, { onSuccess: () => setEditando(false) })} />
        </section>
      )}

      <LanzadorComparativa marco={m} />

      <section aria-labelledby="h-previas" className="space-y-2">
        <h2 id="h-previas">Comparativas lanzadas</h2>
        {comparativas.length === 0 ? (
          <p className="text-suave">Ninguna todavía.</p>
        ) : (
          <ul className="tarjeta divide-y divide-borde/60">
            {comparativas.map((c) => (
              <li key={c.id} className="flex flex-wrap items-center gap-3 px-3 py-2">
                <span className="num">{c.id}</span>
                <span className="text-suave">{fechaCorta(c.creada.slice(0, 10))}</span>
                <span className="text-xs">{c.experimentos.map((e) => e.estrategia).join(', ')}</span>
                <Link to={enlaceComparar(c)} className="ml-auto text-acento hover:underline">Abrir en Comparar</Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
