/**
 * Detalle de una configuración guardada: receta, origen (con las métricas del momento de guardar), versiones,
 * backtests de la misma receta y las acciones lanzar backtest y crear nueva versión.
 * Responsabilidad: mostrar una versión concreta (`?version=`), lanzar un backtest en un rango de fechas (reutiliza el
 * experimento si ya existe: misma huella), editar los parámetros como NUEVA versión (la anterior no cambia) y archivar.
 * Si la configuración apunta a un marco, muestra el enlace y usa sus fechas por defecto.
 * Se relaciona con: api/hooks.ts, componentes/FormularioSchema.tsx (nueva versión), paginas/Marcos.tsx y
 * paginas/ExperimentoDetalle.tsx (destino del backtest).
 */
import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  useArchivarConfiguracion,
  useBacktestConfiguracion,
  useConfiguracion,
  useEstrategia,
  useNuevaVersion,
} from '../api/hooks'
import type { Configuracion } from '../api/tipos'
import { CajaError, Cargando, Insignia } from '../componentes/Estado'
import FormularioSchema, { validarValores, type Valores } from '../componentes/FormularioSchema'
import { dinero, fechaCorta, porcentaje, ratio, resumenParametros, resumenUniverso } from '../lib/formato'

export default function ConfiguracionDetalle() {
  const { id } = useParams()
  const navegar = useNavigate()
  const [params, setParams] = useSearchParams()
  const version = params.get('version') ? Number(params.get('version')) : undefined
  const { data, isPending, error } = useConfiguracion(id, version)
  const archivar = useArchivarConfiguracion()

  if (isPending) return <Cargando />
  if (error) return <CajaError error={error} />
  const { configuracion: c, versiones, marco, backtests } = data
  const origen = c.origen

  const eliminar = () => {
    if (window.confirm(`¿Archivar la configuración ${c.nombre}? Se conserva y puede verse con «Ver archivadas».`))
      archivar.mutate(c.id, { onSuccess: () => navegar('/configuraciones') })
  }

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-suave"><Link to="/configuraciones" className="hover:text-acento">Configuraciones</Link> /</p>
          <h1 className="flex flex-wrap items-center gap-2">
            <span>{c.nombre}</span>
            <span className="num text-sm text-suave">v{c.version}</span>
            {c.archivada && <span className="text-xs text-suave">(archivada)</span>}
          </h1>
          <p className="text-suave">
            <span className="num">{c.estrategia}</span> · {c.universo_etiqueta ?? resumenUniverso(c.universo)} · rebalanceo {c.frecuencia_rebalanceo} ·
            comisión {porcentaje(c.comision, 2)} · slippage {porcentaje(c.slippage, 2)} · benchmark {c.benchmark} · capital {dinero(c.capital_inicial)}
          </p>
          <p className="num text-xs text-suave">{resumenParametros(c.parametros)}</p>
          {c.notas && <p className="mt-1">{c.notas}</p>}
        </div>
        <button type="button" className="boton boton-peligro" onClick={eliminar} disabled={archivar.isPending}>Archivar</button>
      </header>

      {marco && (
        <p role="note" className="rounded-lg border border-borde p-3 text-sm text-suave">
          Pertenece al marco <Link to={`/marcos/${marco.id}`} className="text-acento underline">{marco.nombre}</Link> ({marco.fecha_inicio} → {marco.fecha_fin}).
        </p>
      )}

      <section aria-labelledby="h-origen" className="tarjeta space-y-2 p-4">
        <h2 id="h-origen">Origen</h2>
        <p>
          {origen.tipo}
          {origen.criterio ? ` · criterio ${origen.criterio}` : ''}
          {origen.id_experimento && <> · experimento <Link to={`/experimentos/${origen.id_experimento}`} className="num text-acento hover:underline">{origen.id_experimento}</Link></>}
          {origen.id_barrido && <> · barrido <Link to={`/barridos/${origen.id_barrido}`} className="num text-acento hover:underline">{origen.id_barrido}</Link></>}
          {origen.fechas && ` · ${fechaCorta(origen.fechas[0])} → ${fechaCorta(origen.fechas[1])}`}
        </p>
        {origen.aviso && <p role="note" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">{origen.aviso}</p>}
        {Object.keys(origen.metricas).length > 0 && (
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-4">
            {(['sharpe', 'cagr', 'max_drawdown', 'meseta'] as const).map((k) =>
              origen.metricas[k] != null ? (
                <div key={k}>
                  <dt className="text-xs text-suave">{k} al guardar</dt>
                  <dd className="num">{k === 'cagr' || k === 'max_drawdown' ? porcentaje(origen.metricas[k]) : ratio(origen.metricas[k])}</dd>
                </div>
              ) : null,
            )}
          </dl>
        )}
      </section>

      <div className="grid gap-5 xl:grid-cols-2">
        <LanzarBacktest c={c} fechasMarco={marco ? [marco.fecha_inicio, marco.fecha_fin] : null} />
        <NuevaVersion c={c} alCrear={(n) => setParams({ version: String(n) })} />
      </div>

      <section aria-labelledby="h-versiones" className="space-y-2">
        <h2 id="h-versiones">Versiones</h2>
        <ul className="tarjeta divide-y divide-borde/60">
          {[...versiones].reverse().map((v) => (
            <li key={v.version} className="flex flex-wrap items-center gap-3 px-3 py-2">
              <Link to={`/configuraciones/${c.id}?version=${v.version}`} className={`num ${v.version === c.version ? 'font-semibold' : 'text-acento hover:underline'}`}>v{v.version}</Link>
              <span className="text-suave">{fechaCorta(v.creada.slice(0, 10))}</span>
              {v.deriva_de && <span className="text-xs text-suave">deriva de v{v.deriva_de}</span>}
              {v.notas && <span className="text-xs">{v.notas}</span>}
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="h-backtests" className="space-y-2">
        <h2 id="h-backtests">Backtests de esta receta</h2>
        {backtests.length === 0 ? (
          <p className="text-suave">Ninguno todavía. Lánzalo arriba.</p>
        ) : (
          <div className="tarjeta overflow-x-auto">
            <table className="w-full text-sm" aria-label="Backtests de la receta">
              <thead className="border-b border-borde"><tr>{['Experimento', 'Periodo', 'Estado', 'Sharpe', 'CAGR'].map((t) => <th key={t} scope="col" className="th">{t}</th>)}</tr></thead>
              <tbody>
                {backtests.map((e) => (
                  <tr key={e.id} className="border-b border-borde/60 last:border-0">
                    <td className="td"><Link to={`/experimentos/${e.id}`} className="num text-acento hover:underline">{e.id}</Link></td>
                    <td className="td">{fechaCorta(e.fecha_inicio)} → {fechaCorta(e.fecha_fin)}</td>
                    <td className="td"><Insignia estado={e.estado} /></td>
                    <td className="td num">{ratio(e.metricas.sharpe)}</td>
                    <td className="td num">{porcentaje(e.metricas.cagr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}

function LanzarBacktest({ c, fechasMarco }: { c: Configuracion; fechasMarco: string[] | null }) {
  const navegar = useNavigate()
  const lanzar = useBacktestConfiguracion(c.id)
  const [inicio, setInicio] = useState(fechasMarco?.[0] ?? '')
  const [fin, setFin] = useState(fechasMarco?.[1] ?? '')
  const enviar = (e: React.FormEvent) => {
    e.preventDefault()
    lanzar.mutate(
      { fecha_inicio: inicio || undefined, fecha_fin: fin || undefined, version: c.version },
      { onSuccess: (r) => navegar(`/experimentos/${r.id_experimento}`) },
    )
  }
  return (
    <form onSubmit={enviar} className="tarjeta space-y-3 p-4" aria-labelledby="h-backtest">
      <h2 id="h-backtest">Lanzar backtest</h2>
      <p className="text-suave">Si dejas las fechas vacías se usan las del marco o los últimos 5 años. Si ya existe un experimento igual se reutiliza.</p>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="flex flex-col gap-1"><label htmlFor="bt-inicio" className="font-medium">Inicio</label><input id="bt-inicio" type="date" className="campo" value={inicio} onChange={(e) => setInicio(e.target.value)} /></div>
        <div className="flex flex-col gap-1"><label htmlFor="bt-fin" className="font-medium">Fin</label><input id="bt-fin" type="date" className="campo" value={fin} onChange={(e) => setFin(e.target.value)} /></div>
      </div>
      {lanzar.error && <CajaError error={lanzar.error} titulo="No se pudo lanzar" />}
      <button type="submit" className="boton boton-primario" disabled={lanzar.isPending}>{lanzar.isPending ? 'Lanzando…' : 'Lanzar backtest'}</button>
    </form>
  )
}

function NuevaVersion({ c, alCrear }: { c: Configuracion; alCrear: (version: number) => void }) {
  const { data: estrategia } = useEstrategia(c.estrategia)
  const crear = useNuevaVersion(c.id)
  const [parametros, setParametros] = useState<Valores>(c.parametros)
  const [notas, setNotas] = useState('')
  useEffect(() => setParametros(c.parametros), [c])
  if (!estrategia) return null
  const invalido = Object.keys(validarValores(estrategia.parametros_schema, parametros)).length > 0
  const enviar = (e: React.FormEvent) => {
    e.preventDefault()
    crear.mutate({ parametros, ...(notas ? { notas } : {}) }, { onSuccess: (n) => alCrear(n.version) })
  }
  return (
    <form onSubmit={enviar} className="tarjeta space-y-3 p-4" aria-labelledby="h-version">
      <h2 id="h-version">Nueva versión</h2>
      <p className="text-suave">Editar crea la versión {c.version + 1}; la actual no se modifica.</p>
      <FormularioSchema esquema={estrategia.parametros_schema} valores={parametros} onChange={setParametros} idBase="version" />
      <div className="flex flex-col gap-1"><label htmlFor="notas-version" className="font-medium">Notas de la versión</label><textarea id="notas-version" className="campo" rows={2} value={notas} onChange={(e) => setNotas(e.target.value)} /></div>
      {crear.error && <CajaError error={crear.error} titulo="No se pudo crear la versión" />}
      <button type="submit" className="boton boton-primario" disabled={invalido || crear.isPending}>{crear.isPending ? 'Guardando…' : 'Crear nueva versión'}</button>
    </form>
  )
}
