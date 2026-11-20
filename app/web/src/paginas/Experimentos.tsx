/**
 * Página Experimentos: tabla ordenable y filtrable de backtests (estrategia, periodo, Sharpe, CAGR,
 * máx. drawdown, turnover, estado) con selección múltiple para comparar o borrar.
 * Responsabilidad: filtrar por texto/estrategia/estado en cliente, refrescarse sola mientras haya
 * experimentos en curso, seleccionar (clic, rango con Mayús en el orden visible, «todos los filtrados»),
 * «Comparar» con los marcados que están terminados y «Borrar» los marcados tras confirmar (los que siguen
 * en marcha no se borran; se informa del resultado).
 * Se relaciona con: api/hooks.ts (useExperimentos, useBorrarExperimentos), componentes/TablaOrdenable.tsx
 * (alOrdenar: orden visible para los rangos), paginas/Comparar.tsx y paginas/ExperimentoDetalle.tsx.
 */
import { useCallback, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useBorrarExperimentos, useExperimentos } from '../api/hooks'
import type { EstadoExperimento, ResumenExperimento } from '../api/tipos'
import { CajaError, Cargando, Insignia, InsigniaTfm, Vacio } from '../componentes/Estado'
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
  const borrado = useBorrarExperimentos()
  // Orden en que la tabla muestra las filas (para los rangos con Mayús) y última casilla pulsada (ancla).
  const ordenVisible = useRef<ResumenExperimento[]>([])
  const ancla = useRef<string | null>(null)
  const alOrdenar = useCallback((filas: ResumenExperimento[]) => {
    ordenVisible.current = filas
  }, [])

  const lista = data?.experimentos ?? []
  const estrategias = useMemo(() => [...new Set(lista.map((e) => e.estrategia))].sort(), [lista])
  const filtrados = useMemo(() => filtrarExperimentos(lista, texto, estrategia, estado), [lista, texto, estrategia, estado])

  // Seleccionados que siguen existiendo, en el orden visible (los ocultos por un filtro van al final).
  const seleccion = useMemo(() => {
    const existentes = lista.filter((e) => marcados.has(e.id))
    const visibles = ordenVisible.current.filter((e) => marcados.has(e.id))
    const idsVisibles = new Set(visibles.map((e) => e.id))
    return [...visibles, ...existentes.filter((e) => !idsVisibles.has(e.id))]
  }, [lista, marcados])
  const paraComparar = seleccion.filter((e) => e.estado === 'hecho')

  /** Clic en una casilla; con Mayús aplica el nuevo estado a todo el rango desde el ancla (orden visible). */
  const marcar = (id: string, conMayus: boolean) => {
    const visibles = ordenVisible.current.map((e) => e.id)
    const desde = ancla.current ? visibles.indexOf(ancla.current) : -1
    const hasta = visibles.indexOf(id)
    const rango = conMayus && desde >= 0 && hasta >= 0
      ? visibles.slice(Math.min(desde, hasta), Math.max(desde, hasta) + 1)
      : [id]
    const activar = !marcados.has(id)
    setMarcados((previos) => {
      const siguiente = new Set(previos)
      for (const x of rango) {
        if (activar) siguiente.add(x)
        else siguiente.delete(x)
      }
      return siguiente
    })
    ancla.current = id
  }

  const seleccionarFiltrados = () => setMarcados((previos) => new Set([...previos, ...filtrados.map((e) => e.id)]))
  const quitarSeleccion = () => {
    setMarcados(new Set())
    ancla.current = null
  }

  const borrarSeleccion = () => {
    const n = seleccion.length
    const aviso = `¿Borrar ${n} ${n === 1 ? 'experimento' : 'experimentos'}? Se eliminan sus resultados y no se puede deshacer. ` +
      'Los que sigan en marcha no se borran.'
    if (!window.confirm(aviso)) return
    borrado.mutate(seleccion.map((e) => e.id), { onSuccess: quitarSeleccion })
  }

  const columnas: Columna<ResumenExperimento>[] = [
    {
      clave: 'sel',
      titulo: '',
      render: (e) => (
        <input
          type="checkbox"
          className="h-4 w-4 accent-[rgb(var(--acento))]"
          aria-label={`Seleccionar ${e.id}`}
          title="Mayús + clic: seleccionar un rango"
          checked={marcados.has(e.id)}
          onChange={(evento) => marcar(e.id, (evento.nativeEvent as MouseEvent).shiftKey === true)}
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
          <InsigniaTfm estrategia={e.estrategia} />
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
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className="boton boton-primario"
            disabled={paraComparar.length < 2}
            title="Solo cuentan los experimentos terminados"
            onClick={() => navegar(`/comparar?ids=${paraComparar.map((e) => e.id).join(',')}`)}
          >
            Comparar ({paraComparar.length})
          </button>
          <button type="button" className="boton text-neg" disabled={seleccion.length === 0 || borrado.isPending} onClick={borrarSeleccion}>
            {borrado.isPending ? 'Borrando…' : `Borrar (${seleccion.length})`}
          </button>
        </div>
      </header>

      {borrado.data && (
        <p role="status" aria-label="Resultado del borrado" className="text-sm text-suave">
          Borrados: {borrado.data.borrados.length}.
          {borrado.data.en_marcha.length > 0 && ` En marcha (no se borran): ${borrado.data.en_marcha.length}.`}
          {borrado.data.no_encontrados.length > 0 && ` No encontrados: ${borrado.data.no_encontrados.length}.`}
        </p>
      )}
      {borrado.error && <CajaError error={borrado.error} />}

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
        <button type="button" className="boton" disabled={filtrados.length === 0} onClick={seleccionarFiltrados}>
          Seleccionar los {filtrados.length} filtrados
        </button>
        {marcados.size > 0 && (
          <button type="button" className="boton" onClick={quitarSeleccion}>
            Quitar selección
          </button>
        )}
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
          alOrdenar={alOrdenar}
          vacio="Ningún experimento coincide con los filtros"
        />
      )}
    </div>
  )
}
