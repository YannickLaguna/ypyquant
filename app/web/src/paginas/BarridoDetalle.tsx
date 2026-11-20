/**
 * Detalle de un barrido (`/barridos/:id`): progreso, mapa de calor, tabla de combinaciones y los
 * paneles de validación (validar, permutación, walk-forward, periodo reservado).
 * Responsabilidad: coordinar la vista; sondea la API mientras haya experimentos o tareas en curso
 * (el sondeo vive en `useBarrido`). El mapa de calor admite 1 o 2 ejes visibles (con selectores de ejes
 * y valores fijos para el resto), métrica seleccionable y conmutador valor/meseta; marca la mejor
 * combinación (★) y la alternativa por meseta (◆) y abre el experimento al pulsar una celda.
 * Se relaciona con: api/hooks.ts, lib/barridos.ts (matriz y mejores), componentes/MapaCalor.tsx,
 * componentes/PanelesValidacion.tsx, componentes/PanelesRobustez.tsx, componentes/TablaOrdenable.tsx
 * y paginas/ExperimentoDetalle.tsx (destino de las celdas). La tabla de combinaciones permite marcar
 * varias y abrir paginas/Comparar.tsx con `/comparar?ids=a,b`.
 */
import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useBarrido, useBorrarBarrido } from '../api/hooks'
import type { DetalleBarrido, FilaSuperficie } from '../api/tipos'
import { CajaError, Cargando, Insignia, Vacio } from '../componentes/Estado'
import MapaCalor from '../componentes/MapaCalor'
import { PanelPermutacion, PanelReservado } from '../componentes/PanelesRobustez'
import { PanelValidar, PanelWalkForward } from '../componentes/PanelesValidacion'
import TablaOrdenable, { type Columna } from '../componentes/TablaOrdenable'
import { costesAltos, elegirMejores, matrizMapaCalor, textoValor, type ModoMapa } from '../lib/barridos'
import { fechaCorta, porcentaje, ratio, resumenUniverso } from '../lib/formato'

export default function BarridoDetalle() {
  const { id } = useParams()
  const navegar = useNavigate()
  const { data, isPending, error } = useBarrido(id)
  const borrar = useBorrarBarrido()

  if (isPending) return <Cargando texto="Cargando barrido…" />
  if (error) return <CajaError error={error} />

  const { progreso: p } = data
  const pct = p.total > 0 ? Math.round(((p.hechos + p.errores) / p.total) * 100) : 0
  const base = data.config.base
  const eliminar = () => {
    const confirmado = window.confirm(`¿Borrar el barrido ${data.id}?\n\nSolo se borra el barrido y sus validaciones; los experimentos se conservan.`)
    if (confirmado) borrar.mutate({ id: data.id, conExperimentos: false }, { onSuccess: () => navegar('/barridos') })
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-suave"><Link to="/barridos" className="hover:text-acento">Barridos</Link> /</p>
          <h1 className="flex flex-wrap items-center gap-2">
            <span className="break-all">{data.config.etiqueta || data.id}</span>
            <Insignia estado={data.estado} />
          </h1>
          <p className="num text-xs text-suave">{data.config.etiqueta ? `${data.id} · ` : ''}{base.estrategia} · {base.universo ? resumenUniverso(base.universo) : ''} · objetivo: {data.config.metrica_objetivo}</p>
          <p className="text-suave">
            {base.fecha_inicio && fechaCorta(base.fecha_inicio)} → {base.fecha_fin && fechaCorta(base.fecha_fin)}
            {data.config.fecha_fin_desarrollo && <> · desarrollo hasta {fechaCorta(data.config.fecha_fin_desarrollo)} (el resto está reservado)</>}
          </p>
        </div>
        <button type="button" className="boton boton-peligro" onClick={eliminar} disabled={borrar.isPending}>Borrar barrido</button>
      </header>
      {borrar.error && <CajaError error={borrar.error} titulo="No se pudo borrar" />}

      <section aria-labelledby="h-progreso" className="tarjeta space-y-2 p-3">
        <h2 id="h-progreso">Progreso</h2>
        <div className="h-2 overflow-hidden rounded bg-elevado" role="progressbar" aria-valuemin={0} aria-valuemax={p.total} aria-valuenow={p.hechos + p.errores} aria-label="Progreso del barrido">
          <div className={`h-full transition-all ${p.errores > 0 ? 'bg-aviso' : 'bg-acento'}`} style={{ width: `${pct}%` }} />
        </div>
        <p className="num text-xs text-suave" aria-live="polite">
          {p.hechos} de {p.total} hechos · {p.en_curso} en curso · {p.pendientes} pendientes{p.errores > 0 && <span className="text-neg"> · {p.errores} con error</span>}
        </p>
        {p.errores > 0 && (
          <p className="text-xs text-aviso">Algunos experimentos fallaron: sus combinaciones no aparecen en el mapa. La validación necesita al menos 2 combinaciones bien terminadas.</p>
        )}
      </section>

      {data.superficie.length === 0 ? (
        <Vacio titulo={data.estado === 'hecho' || data.estado === 'error' ? 'Sin resultados' : 'Esperando los primeros resultados…'}>
          El mapa de calor y la tabla aparecen según terminen los experimentos.
        </Vacio>
      ) : (
        <>
          <SeccionMapa barrido={data} />
          <SeccionTabla superficie={data.superficie} metricaObjetivo={data.config.metrica_objetivo} />
        </>
      )}

      <PanelValidar barrido={data} />
      <PanelPermutacion barrido={data} />
      <PanelWalkForward barrido={data} />
      {data.config.fecha_fin_desarrollo && <PanelReservado barrido={data} />}
    </div>
  )
}

// ---- Mapa de calor --------------------------------------------------------------------------

function SeccionMapa({ barrido }: { barrido: DetalleBarrido }) {
  const navegar = useNavigate()
  const nombresEjes = Object.keys(barrido.ejes)
  const objetivo = barrido.config.metrica_objetivo
  const [metrica, setMetrica] = useState(objetivo)
  const [modo, setModo] = useState<ModoMapa>('valor')
  const [ejeX, setEjeX] = useState(nombresEjes[0] ?? '')
  const [ejeYElegido, setEjeY] = useState<string>(nombresEjes[1] ?? '')
  const [fijos, setFijos] = useState<Record<string, number>>({})

  // Si el eje Y coincide con el X (o no existe), se desactiva
  const ejeY = ejeYElegido && ejeYElegido !== ejeX && nombresEjes.includes(ejeYElegido) ? ejeYElegido : null
  const ocultos = nombresEjes.filter((e) => e !== ejeX && e !== ejeY)

  const metricasDisponibles = useMemo(() => {
    const claves = new Set<string>([objetivo])
    for (const f of barrido.superficie) for (const [k, v] of Object.entries(f.metricas)) if (typeof v === 'number') claves.add(k)
    return [...claves].sort((a, b) => (a === objetivo ? -1 : b === objetivo ? 1 : a.localeCompare(b)))
  }, [barrido.superficie, objetivo])

  const matriz = useMemo(
    () => matrizMapaCalor(barrido.superficie, barrido.ejes, ejeX, ejeY, fijos, metrica, objetivo, modo),
    [barrido.superficie, barrido.ejes, ejeX, ejeY, fijos, metrica, objetivo, modo],
  )
  const { mejor, porMeseta } = useMemo(() => elegirMejores(barrido.superficie), [barrido.superficie])
  const enMapa = (f: FilaSuperficie | null) => (f && ocultos.every((e) => f.posiciones[e] === (fijos[e] ?? 0)) ? f.id : null)

  return (
    <section aria-labelledby="h-mapa" className="space-y-3">
      <h2 id="h-mapa">Mapa de calor</h2>
      <div className="tarjeta flex flex-wrap items-end gap-3 p-3">
        <div className="flex flex-col gap-1">
          <label htmlFor="mp-metrica" className="text-xs text-suave">Métrica</label>
          <select id="mp-metrica" className="campo num" value={metrica} onChange={(e) => setMetrica(e.target.value)}>
            {metricasDisponibles.map((m) => <option key={m} value={m}>{m}{m === objetivo ? ' (objetivo)' : ''}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1" role="group" aria-label="Qué mostrar en el color">
          <span className="text-xs text-suave">Color</span>
          <div className="flex overflow-hidden rounded-md border border-borde">
            {(['valor', 'meseta'] as const).map((m) => (
              <button key={m} type="button" aria-pressed={modo === m} onClick={() => setModo(m)} className={`px-3 py-1.5 text-sm ${modo === m ? 'bg-acento text-fondo' : 'bg-elevado text-suave hover:text-texto'}`}>
                {m === 'valor' ? 'Valor' : 'Meseta'}
              </button>
            ))}
          </div>
        </div>
        {nombresEjes.length >= 2 && (
          <>
            <div className="flex flex-col gap-1">
              <label htmlFor="mp-x" className="text-xs text-suave">Eje X</label>
              <select id="mp-x" className="campo num" value={ejeX} onChange={(e) => setEjeX(e.target.value)}>
                {nombresEjes.map((e) => <option key={e} value={e}>{e}</option>)}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label htmlFor="mp-y" className="text-xs text-suave">Eje Y</label>
              <select id="mp-y" className="campo num" value={ejeY ?? ''} onChange={(e) => setEjeY(e.target.value)}>
                <option value="">(ninguno)</option>
                {nombresEjes.filter((e) => e !== ejeX).map((e) => <option key={e} value={e}>{e}</option>)}
              </select>
            </div>
          </>
        )}
        {ocultos.map((e) => (
          <div key={e} className="flex flex-col gap-1">
            <label htmlFor={`mp-fijo-${e}`} className="text-xs text-suave">{e} fijo</label>
            <select id={`mp-fijo-${e}`} className="campo num" value={fijos[e] ?? 0} onChange={(ev) => setFijos((f) => ({ ...f, [e]: Number(ev.target.value) }))}>
              {barrido.ejes[e].map((v, i) => <option key={i} value={i}>{textoValor(v)}</option>)}
            </select>
          </div>
        ))}
      </div>
      <p className="text-xs text-suave">
        {modo === 'valor'
          ? 'Cada celda es la métrica de una combinación.'
          : 'Meseta: media de la métrica en las combinaciones vecinas (±1 en un eje). Un pico aislado puntúa bajo; una zona amplia de buenos resultados, alto.'}
        {' '}Pulsa una celda para abrir su experimento.
      </p>
      <MapaCalor
        titulo={`${metrica}${modo === 'meseta' ? ' (meseta)' : ''}`}
        matriz={matriz}
        idMejor={enMapa(mejor)}
        idMeseta={enMapa(porMeseta)}
        onAbrir={(idExp) => navegar(`/experimentos/${idExp}`)}
        formato={(v) => (metrica.includes('drawdown') || ['cagr', 'retorno_total', 'alfa', 'tasa_acierto'].includes(metrica) ? porcentaje(v, 1) : ratio(v, 2))}
      />
      <ul className="space-y-0.5 text-xs text-suave">
        {mejor && <li><span aria-hidden>★</span> Mejor por {objetivo}: <Link className="num text-acento hover:underline" to={`/experimentos/${mejor.id}`}>{mejor.etiqueta}</Link> ({ratio(mejor.objetivo, 3)}){ocultos.length > 0 && enMapa(mejor) === null && ' — fuera de la porción mostrada'}</li>}
        {porMeseta && <li><span aria-hidden>◆</span> Alternativa por meseta: <Link className="num text-acento hover:underline" to={`/experimentos/${porMeseta.id}`}>{porMeseta.etiqueta}</Link> (meseta {ratio(porMeseta.meseta, 3)}){porMeseta.id === mejor?.id && ' — coincide con la mejor'}{ocultos.length > 0 && enMapa(porMeseta) === null && ' — fuera de la porción mostrada'}</li>}
      </ul>
    </section>
  )
}

// ---- Tabla de combinaciones -----------------------------------------------------------------

export function SeccionTabla({ superficie, metricaObjetivo }: { superficie: FilaSuperficie[]; metricaObjetivo: string }) {
  const metrica = (nombre: string, formato: (v: number | null | undefined) => string, titulo: string): Columna<FilaSuperficie> => ({
    clave: nombre,
    titulo,
    alinear: 'der',
    valor: (f) => f.metricas[nombre],
    render: (f) => formato(f.metricas[nombre]),
  })
  const navegar = useNavigate()
  // Ids marcados para comparar, en el orden en que se marcaron
  const [marcados, setMarcados] = useState<string[]>([])
  const alternar = (id: string) => setMarcados((m) => (m.includes(id) ? m.filter((i) => i !== id) : [...m, id]))
  const columnas: Columna<FilaSuperficie>[] = [
    {
      clave: 'sel',
      titulo: '',
      render: (f) => (
        <input type="checkbox" className="accent-[rgb(var(--acento))]" checked={marcados.includes(f.id)} onChange={() => alternar(f.id)} aria-label={`Seleccionar ${f.id} para comparar`} />
      ),
    },
    { clave: 'etiqueta', titulo: 'Combinación', valor: (f) => f.etiqueta, render: (f) => <Link to={`/experimentos/${f.id}`} className="num text-acento hover:underline">{f.etiqueta}</Link> },
    { clave: 'objetivo', titulo: `Objetivo (${metricaObjetivo})`, alinear: 'der', valor: (f) => f.objetivo, render: (f) => ratio(f.objetivo, 3) },
    { clave: 'meseta', titulo: 'Meseta', alinear: 'der', valor: (f) => f.meseta, render: (f) => ratio(f.meseta, 3) },
    metrica('sharpe', ratio, 'Sharpe'),
    metrica('cagr', porcentaje, 'CAGR'),
    metrica('max_drawdown', porcentaje, 'Máx. DD'),
    metrica('ulcer', ratio, 'Ulcer'),
    metrica('martin', ratio, 'Martin'),
    {
      clave: 'costes_sobre_ganancia_bruta',
      titulo: 'Costes / gan. bruta',
      alinear: 'der',
      valor: (f) => f.metricas.costes_sobre_ganancia_bruta,
      render: (f) => {
        const v = f.metricas.costes_sobre_ganancia_bruta
        return costesAltos(v) ? <span className="font-semibold text-aviso" title="Los costes se comen más de la mitad del edge">{ratio(v, 2)} ▲</span> : ratio(v, 2)
      },
    },
  ]
  return (
    <section aria-labelledby="h-tabla" className="space-y-2">
      <h2 id="h-tabla">Combinaciones ({superficie.length})</h2>
      <TablaOrdenable titulo="Combinaciones del barrido" columnas={columnas} filas={superficie} clave={(f) => f.id} ordenInicial={{ clave: 'objetivo', descendente: true }} vacio="Sin combinaciones terminadas" />
      <button type="button" className="boton" disabled={marcados.length < 2} onClick={() => navegar(`/comparar?ids=${marcados.join(',')}`)}>
        Comparar seleccionados ({marcados.length})
      </button>
      <p className="text-xs text-suave">En ámbar, costes sobre ganancia bruta &gt; 0,5: las comisiones se comen más de la mitad del edge.</p>
    </section>
  )
}
