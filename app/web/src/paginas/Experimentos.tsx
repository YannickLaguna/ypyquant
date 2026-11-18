/**
 * Página Experimentos: tabla ordenable y filtrable de backtests (estrategia, periodo, Sharpe, CAGR,
 * máx. drawdown, turnover, estado) con selección múltiple para comparar.
 * Responsabilidad: filtrar por texto/estrategia/estado en cliente, refrescarse sola mientras haya
 * experimentos en curso y ofrecer «Comparar» con los marcados.
 * Se relaciona con: api/hooks.ts (useExperimentos), componentes/TablaOrdenable.tsx, paginas/Comparar.tsx
 * y paginas/ExperimentoDetalle.tsx.
 */
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useExperimentos } from '../api/hooks'
import type { EstadoExperimento, ResumenExperimento } from '../api/tipos'
import { CajaError, Cargando, Insignia, Vacio } from '../componentes/Estado'
import TablaOrdenable, { type Columna } from '../componentes/TablaOrdenable'
import { claseSigno, fechaCorta, porcentaje, ratio, resumenParametros, resumenUniverso } from '../lib/formato'

export function filtrarExperimentos(lista: ResumenExperimento[], texto: string, estrategia: string, estado: string) {
  const t = texto.trim().toLowerCase()
  return lista.filter(
    (e) =>
      (!estrategia || e.estrategia === estrategia) &&
      (!estado || e.estado === estado) &&
      (!t || `${e.id} ${e.etiqueta} ${resumenParametros(e.parametros)} ${JSON.stringify(e.universo)}`.toLowerCase().includes(t)),
  )
}

export default function Experimentos() {
  const { data, isPending, error } = useExperimentos()
  const navegar = useNavigate()
  const [texto, setTexto] = useState('')
  const [estrategia, setEstrategia] = useState('')
  const [estado, setEstado] = useState('')
  const [marcados, setMarcados] = useState<Set<string>>(new Set())

  const lista = data?.experimentos ?? []
  const estrategias = useMemo(() => [...new Set(lista.map((e) => e.estrategia))].sort(), [lista])
  const filtrados = useMemo(() => filtrarExperimentos(lista, texto, estrategia, estado), [lista, texto, estrategia, estado])

  const alternar = (id: string) =>
    setMarcados((previos) => {
      const siguiente = new Set(previos)
      if (siguiente.has(id)) siguiente.delete(id)
      else siguiente.add(id)
      return siguiente
    })

  const columnas: Columna<ResumenExperimento>[] = [
    {
      clave: 'sel',
      titulo: '',
      render: (e) => (
        <input
          type="checkbox"
          className="h-4 w-4 accent-[rgb(var(--acento))]"
          aria-label={`Seleccionar ${e.id} para comparar`}
          checked={marcados.has(e.id)}
          disabled={e.estado !== 'hecho'}
          onChange={() => alternar(e.id)}
        />
      ),
    },
    {
      clave: 'estrategia',
      titulo: 'Estrategia',
      valor: (e) => e.estrategia,
      render: (e) => (
        <Link to={`/experimentos/${e.id}`} className="font-medium text-acento hover:underline">
          {e.estrategia}
          <span className="block max-w-[16rem] truncate text-xs font-normal text-suave" title={resumenParametros(e.parametros)}>
            {e.etiqueta || resumenParametros(e.parametros)}
          </span>
        </Link>
      ),
    },
    { clave: 'periodo', titulo: 'Periodo', valor: (e) => e.fecha_inicio, render: (e) => `${fechaCorta(e.fecha_inicio)} → ${fechaCorta(e.fecha_fin)}` },
    { clave: 'universo', titulo: 'Universo', render: (e) => resumenUniverso(e.universo) },
    { clave: 'sharpe', titulo: 'Sharpe', alinear: 'der', valor: (e) => e.metricas.sharpe, render: (e) => ratio(e.metricas.sharpe) },
    { clave: 'cagr', titulo: 'CAGR', alinear: 'der', valor: (e) => e.metricas.cagr, render: (e) => <span className={claseSigno(e.metricas.cagr)}>{porcentaje(e.metricas.cagr)}</span> },
    { clave: 'mdd', titulo: 'Máx. DD', alinear: 'der', valor: (e) => e.metricas.max_drawdown, render: (e) => <span className="text-neg">{porcentaje(e.metricas.max_drawdown)}</span> },
    { clave: 'turnover', titulo: 'Turnover', alinear: 'der', valor: (e) => e.metricas.turnover_anual, render: (e) => porcentaje(e.metricas.turnover_anual, 0) },
    { clave: 'estado', titulo: 'Estado', valor: (e) => e.estado, render: (e) => <Insignia estado={e.estado as EstadoExperimento} /> },
    { clave: 'creado', titulo: 'Creado', valor: (e) => e.creado, render: (e) => <span className="text-suave">{fechaCorta(e.creado)}</span> },
  ]

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1>Experimentos</h1>
          <p className="text-suave">Backtests reproducibles: misma configuración, mismo id.</p>
        </div>
        <button
          type="button"
          className="boton boton-primario"
          disabled={marcados.size < 2}
          onClick={() => navegar(`/comparar?ids=${[...marcados].join(',')}`)}
        >
          Comparar ({marcados.size})
        </button>
      </header>

      <div className="flex flex-wrap gap-2" role="search">
        <label className="sr-only" htmlFor="f-texto">Buscar</label>
        <input id="f-texto" className="campo max-w-xs" placeholder="Buscar por id, etiqueta, parámetros…" value={texto} onChange={(e) => setTexto(e.target.value)} />
        <label className="sr-only" htmlFor="f-estrategia">Estrategia</label>
        <select id="f-estrategia" className="campo w-auto" value={estrategia} onChange={(e) => setEstrategia(e.target.value)}>
          <option value="">Todas las estrategias</option>
          {estrategias.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <label className="sr-only" htmlFor="f-estado">Estado</label>
        <select id="f-estado" className="campo w-auto" value={estado} onChange={(e) => setEstado(e.target.value)}>
          <option value="">Todos los estados</option>
          {['hecho', 'en_curso', 'pendiente', 'error'].map((s) => <option key={s} value={s}>{s.replace('_', ' ')}</option>)}
        </select>
      </div>

      {isPending ? (
        <Cargando texto="Cargando experimentos…" />
      ) : error ? (
        <CajaError error={error} />
      ) : lista.length === 0 ? (
        <Vacio titulo="Aún no hay experimentos">
          Elige una <Link to="/estrategias" className="text-acento underline">estrategia</Link> y lanza el primero.
        </Vacio>
      ) : (
        <TablaOrdenable
          titulo="Experimentos"
          columnas={columnas}
          filas={filtrados}
          clave={(e) => e.id}
          ordenInicial={{ clave: 'creado', descendente: true }}
          vacio="Ningún experimento coincide con los filtros"
        />
      )}
    </div>
  )
}
