/**
 * Paneles de robustez de un barrido: test de permutación y periodo reservado.
 * Responsabilidad:
 * - `PanelPermutacion`: formulario (modo in-sample / walk-forward, nº de permutaciones, longitud de
 *   bloque, procesos y, en walk-forward, ventana y paso), nº de backtests estimado y aviso, barra de
 *   progreso con cancelación y resultado (histograma con la línea del valor real, p-valor con
 *   semáforo y la nota de que la medida no es un objetivo).
 * - `PanelReservado` (solo si el barrido tiene `fecha_fin_desarrollo`): evalúa UNA combinación en el
 *   periodo reservado, con el contador de usos previos y el aviso de sesgo de selección muy visibles;
 *   pide confirmación antes de volver a evaluar si ya se usó.
 * Se relaciona con: api/hooks.ts, api/tipos.ts, lib/barridos.ts (semáforos, umbrales),
 * componentes/GraficosValidacion.tsx, componentes/Estado.tsx y paginas/BarridoDetalle.tsx (los monta).
 */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useCancelarPermutacion, useEvaluarReservado, useLanzarPermutacion } from '../api/hooks'
import type { DetalleBarrido, ModoPermutacion, ResultadoPermutacion, ResultadoReservado } from '../api/tipos'
import { aniosPeriodoPermutacion, elegirMejores, semaforoPValor, UMBRAL_BACKTESTS_LENTO } from '../lib/barridos'
import { claseSigno, entero, fechaCorta, fechaHora, porcentaje, ratio } from '../lib/formato'
import { CajaError } from './Estado'
import { Histograma, InsigniaSemaforo } from './GraficosValidacion'
import { barridoSinTerminar } from './PanelesValidacion'

// ---- Permutación ----------------------------------------------------------------------------

export function PanelPermutacion({ barrido }: { barrido: DetalleBarrido }) {
  const lanzar = useLanzarPermutacion(barrido.id)
  const cancelar = useCancelarPermutacion(barrido.id)
  const [modo, setModo] = useState<ModoPermutacion>('in_sample')
  const [n, setN] = useState(100)
  const [bloque, setBloque] = useState(1)
  const [procesos, setProcesos] = useState(1)
  const [ventana, setVentana] = useState(756)
  const [paso, setPaso] = useState(90)

  const tarea = barrido.tareas?.[`permutacion_${modo}`]
  const enCurso = tarea?.estado === 'pendiente' || tarea?.estado === 'en_curso'
  const resultado: ResultadoPermutacion | null = modo === 'in_sample' ? barrido.permutacion_in_sample : barrido.permutacion_walk_forward
  const nBacktests = barrido.progreso.total * n
  const valido = n >= 1 && n <= 1000 && bloque >= 1 && procesos >= 1 && (modo === 'in_sample' || (ventana > 0 && paso > 0))
  const lanzarTarea = () =>
    lanzar.mutate({
      modo,
      n_permutaciones: n,
      longitud_bloque: bloque,
      procesos,
      ...(modo === 'walk_forward' ? { ventana_entreno_dias: ventana, paso_dias: paso } : {}),
    })

  return (
    <section aria-labelledby="h-perm" className="space-y-3">
      <h2 id="h-perm">Test de permutación</h2>
      <p className="text-xs text-suave">
        Reordena los retornos y repite todo el proceso de selección para ver qué tan bueno sería el mejor resultado por puro azar. Es la prueba más exigente y la más lenta.
      </p>
      <form
        className="tarjeta space-y-3 p-3"
        onSubmit={(e) => {
          e.preventDefault()
          if (valido && !enCurso) lanzarTarea()
        }}
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="flex flex-col gap-1">
            <label htmlFor="pm-modo" className="font-medium">Modo</label>
            <select id="pm-modo" className="campo" value={modo} onChange={(e) => setModo(e.target.value as ModoPermutacion)}>
              <option value="in_sample">In-sample (mejor métrica)</option>
              <option value="walk_forward">Walk-forward</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="pm-n" className="font-medium">Permutaciones (1–1000)</label>
            <input id="pm-n" type="number" min={1} max={1000} className="campo num" value={n} onChange={(e) => setN(Number(e.target.value))} aria-invalid={n < 1 || n > 1000} />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="pm-bloque" className="font-medium">Longitud de bloque</label>
            <input id="pm-bloque" type="number" min={1} className="campo num" value={bloque} onChange={(e) => setBloque(Number(e.target.value))} />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="pm-procesos" className="font-medium">Procesos</label>
            <input id="pm-procesos" type="number" min={1} className="campo num" value={procesos} onChange={(e) => setProcesos(Number(e.target.value))} />
          </div>
          {modo === 'walk_forward' && (
            <>
              <div className="flex flex-col gap-1">
                <label htmlFor="pm-ventana" className="font-medium">Ventana de entreno (días)</label>
                <input id="pm-ventana" type="number" min={1} className="campo num" value={ventana} onChange={(e) => setVentana(Number(e.target.value))} />
              </div>
              <div className="flex flex-col gap-1">
                <label htmlFor="pm-paso" className="font-medium">Paso (días)</label>
                <input id="pm-paso" type="number" min={1} className="campo num" value={paso} onChange={(e) => setPaso(Number(e.target.value))} />
              </div>
            </>
          )}
        </div>
        <p className="text-xs text-suave">
          Coste estimado: <span className="num text-texto">{entero(nBacktests)}</span> backtests ({barrido.progreso.total} combinaciones × {n} permutaciones).
          {nBacktests > UMBRAL_BACKTESTS_LENTO && <span className="text-aviso"> Son más de {entero(UMBRAL_BACKTESTS_LENTO)}: tardará mucho; considera menos permutaciones o más procesos.</span>}
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <button type="submit" className="boton boton-primario" disabled={!valido || enCurso || lanzar.isPending || barridoSinTerminar(barrido)}>
            {lanzar.isPending ? 'Lanzando…' : resultado ? 'Lanzar de nuevo' : 'Lanzar permutación'}
          </button>
          {enCurso && (
            <button type="button" className="boton boton-peligro" onClick={() => cancelar.mutate({ modo })} disabled={cancelar.isPending}>
              Cancelar
            </button>
          )}
        </div>
      </form>
      {lanzar.data?.aviso && <p role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">{lanzar.data.aviso}</p>}
      {lanzar.error && <CajaError error={lanzar.error} titulo="No se pudo lanzar la permutación" />}
      {cancelar.error && <CajaError error={cancelar.error} titulo="No se pudo cancelar" />}

      {tarea && (enCurso || tarea.estado === 'error' || tarea.estado === 'cancelado') && <ProgresoTarea tarea={tarea} />}
      {resultado && <ResultadoPermutacionVista r={resultado} barrido={barrido} />}
    </section>
  )
}

function ProgresoTarea({ tarea }: { tarea: NonNullable<DetalleBarrido['tareas'][string]> }) {
  const hechas = tarea.progreso?.completadas ?? 0
  const total = tarea.progreso?.total ?? 0
  const pct = total > 0 ? Math.round((hechas / total) * 100) : 0
  if (tarea.estado === 'error') return <CajaError error={new Error(tarea.error ?? 'La tarea falló')} titulo="La permutación falló" />
  if (tarea.estado === 'cancelado') return <p role="status" className="text-suave">Permutación cancelada ({hechas} de {total} completadas).</p>
  return (
    <div role="status" aria-live="polite" className="tarjeta space-y-1 p-3">
      <p className="text-xs text-suave">
        {tarea.estado === 'pendiente' ? 'En cola…' : 'Calculando…'} {hechas} de {total} permutaciones ({pct} %)
      </p>
      <div className="h-2 overflow-hidden rounded bg-elevado" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={hechas} aria-label="Progreso de la permutación">
        <div className="h-full bg-acento transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function ResultadoPermutacionVista({ r, barrido }: { r: ResultadoPermutacion; barrido: DetalleBarrido }) {
  const anios = aniosPeriodoPermutacion(r, barrido.config.base.fecha_inicio, barrido.config.fecha_fin_desarrollo ?? barrido.config.base.fecha_fin)
  const semaforo = semaforoPValor(r.p_valor, r.modo, anios)
  const umbral = r.modo === 'in_sample' ? '1 %' : anios !== null && anios >= 2 ? '1 % (periodo ≥ 2 años)' : '5 % (periodo de ~1 año)'
  return (
    <div className="space-y-3">
      <p className="text-xs text-suave">
        {fechaHora(r.fecha)} · {r.n_validas} permutaciones válidas de {r.n_permutaciones} · {entero(r.n_backtests)} backtests · {ratio(r.duracion_s, 1)} s
      </p>
      {r.avisos.length > 0 && (
        <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
          {r.avisos.map((a) => <p key={a}>{a}</p>)}
        </div>
      )}
      <div className="grid gap-3 md:grid-cols-[minmax(0,16rem)_minmax(0,1fr)]">
        <div className="tarjeta space-y-2 p-3">
          <p className="text-xs uppercase tracking-wide text-suave">p-valor</p>
          <p className="flex flex-wrap items-center gap-2">
            <span className="num text-2xl font-semibold">{r.p_valor === null ? '—' : porcentaje(r.p_valor, 2)}</span>
            <InsigniaSemaforo semaforo={semaforo} />
          </p>
          <p className="text-xs text-suave">Verde si p &lt; {umbral}. Fracción de permutaciones que igualan o superan el valor real.</p>
          <dl className="num space-y-0.5 text-xs">
            <div className="flex justify-between"><dt className="text-suave">Valor real</dt><dd>{ratio(r.valor_real, 3)}</dd></div>
            <div className="flex justify-between"><dt className="text-suave">Media permutadas</dt><dd>{ratio(r.media_permutadas, 3)}</dd></div>
            <div className="flex justify-between"><dt className="text-suave">P95 permutadas</dt><dd>{ratio(r.p95_permutadas, 3)}</dd></div>
          </dl>
        </div>
        <Histograma
          titulo={`Distribución de la permutada (${r.estadistico})`}
          datos={r.distribucion}
          referencia={r.valor_real}
          etiquetaReferencia="valor real"
          etiquetaX={`${r.modo === 'in_sample' ? 'Mejor' : 'Sharpe walk-forward'} ${r.metrica_objetivo} bajo reordenación aleatoria`}
        />
      </div>
      <p className="rounded-lg border border-borde bg-elevado p-3 text-xs text-suave">
        La medida no es un objetivo: si iteras la estrategia hasta que pase, deja de valer.
      </p>
    </div>
  )
}

// ---- Periodo reservado ----------------------------------------------------------------------

export function PanelReservado({ barrido }: { barrido: DetalleBarrido }) {
  const evaluar = useEvaluarReservado(barrido.id)
  const periodo = barrido.periodo_reservado
  const terminadas = barrido.combinaciones.filter((c) => c.estado === 'hecho')
  const porDefecto = barrido.validacion?.alternativa_meseta?.id ?? elegirMejores(barrido.superficie).porMeseta?.id ?? terminadas[0]?.id ?? ''
  const [seleccion, setSeleccion] = useState('')
  useEffect(() => {
    if (!seleccion && porDefecto) setSeleccion(porDefecto)
  }, [porDefecto, seleccion])

  const usosPrevios = barrido.reservado_usos?.length ?? 0
  const ultimo: ResultadoReservado | null = evaluar.data ?? barrido.reservado?.[barrido.reservado.length - 1] ?? null
  const nUsos = Math.max(ultimo?.n_usos ?? 0, usosPrevios)

  const evaluarAhora = () => {
    if (usosPrevios > 0) {
      const ok = window.confirm(
        `Este periodo reservado ya se ha usado ${usosPrevios} ${usosPrevios === 1 ? 'vez' : 'veces'}. Cada nueva mirada lo contamina por sesgo de selección y deja de ser fuera de muestra.\n\n¿Evaluar igualmente?`,
      )
      if (!ok) return
    }
    evaluar.mutate({ experimento: seleccion })
  }

  return (
    <section aria-labelledby="h-reservado" className="space-y-3">
      <h2 id="h-reservado">Periodo reservado</h2>
      <p className="text-xs text-suave">
        Tramo {periodo ? `${fechaCorta(periodo.inicio)} → ${fechaCorta(periodo.fin)}` : 'posterior a la fecha de fin de desarrollo'} que ninguna validación ha tocado. Se evalúa UNA combinación y cada mirada queda contada.
      </p>
      <div
        role="status"
        className={`rounded-lg border-2 p-4 ${nUsos > 0 ? 'border-aviso bg-aviso/10' : 'border-borde bg-elevado'}`}
      >
        <p className="text-xs uppercase tracking-wide text-suave">Usos previos de este periodo reservado</p>
        <p className={`num text-3xl font-bold ${nUsos > 0 ? 'text-aviso' : 'text-pos'}`}>{nUsos}</p>
        <p className="mt-1 text-sm">
          {nUsos === 0
            ? 'Intacto: la primera evaluación es una verdadera prueba fuera de muestra.'
            : 'Ya se ha mirado: deja de ser fuera de muestra por sesgo de selección. El contador no lo impide, lo hace visible.'}
        </p>
      </div>
      <form
        className="tarjeta flex flex-wrap items-end gap-3 p-3"
        onSubmit={(e) => {
          e.preventDefault()
          if (seleccion) evaluarAhora()
        }}
      >
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <label htmlFor="rs-comb" className="font-medium">Combinación a evaluar</label>
          <select id="rs-comb" className="campo num" value={seleccion} onChange={(e) => setSeleccion(e.target.value)}>
            {terminadas.map((c) => (
              <option key={c.id} value={c.id}>{c.etiqueta}{c.id === porDefecto ? ' (alternativa por meseta)' : ''}</option>
            ))}
          </select>
        </div>
        <button type="submit" className="boton boton-primario" disabled={!seleccion || evaluar.isPending || barridoSinTerminar(barrido)}>
          {evaluar.isPending ? 'Evaluando…' : 'Evaluar en el reservado'}
        </button>
      </form>
      {evaluar.error && <CajaError error={evaluar.error} titulo="No se pudo evaluar el reservado" />}
      {ultimo && <ResultadoReservadoVista r={ultimo} />}
    </section>
  )
}

function ResultadoReservadoVista({ r }: { r: ResultadoReservado }) {
  const m = r.metricas
  return (
    <div className="space-y-3">
      {r.aviso && (
        <div role="alert" className="rounded-lg border-2 border-aviso bg-aviso/10 p-4">
          <p className="font-semibold text-aviso">Sesgo de selección</p>
          <p className="mt-1">{r.aviso}</p>
        </div>
      )}
      {r.avisos.length > 0 && (
        <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
          {r.avisos.map((a) => <p key={a}>{a}</p>)}
        </div>
      )}
      <p className="text-xs text-suave">
        {fechaHora(r.fecha)} · <span className="num text-texto">{r.combinacion.etiqueta}</span> en {fechaCorta(r.periodo.inicio)} → {fechaCorta(r.periodo.fin)} · uso nº {r.n_usos} ·{' '}
        <Link className="text-acento underline" to={`/experimentos/${r.experimento_reservado}`}>abrir experimento</Link>
      </p>
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Dato etiqueta="Sharpe" valor={ratio(m.sharpe)} />
        <Dato etiqueta="CAGR" valor={porcentaje(m.cagr)} clase={claseSigno(m.cagr)} />
        <Dato etiqueta="Retorno total" valor={porcentaje(m.retorno_total)} clase={claseSigno(m.retorno_total)} />
        <Dato etiqueta="Máx. drawdown" valor={porcentaje(m.max_drawdown)} clase="text-neg" />
        <Dato etiqueta="Sortino" valor={ratio(m.sortino)} />
        <Dato etiqueta="Calmar" valor={ratio(m.calmar)} />
        <Dato etiqueta="Volatilidad" valor={porcentaje(m.volatilidad)} />
        <Dato etiqueta="Costes / ganancia bruta" valor={ratio(m.costes_sobre_ganancia_bruta)} />
      </dl>
    </div>
  )
}

function Dato({ etiqueta, valor, clase = '' }: { etiqueta: string; valor: string; clase?: string }) {
  return (
    <div className="tarjeta px-3 py-2">
      <dt className="text-xs uppercase tracking-wide text-suave">{etiqueta}</dt>
      <dd className={`num mt-0.5 text-lg font-semibold ${clase}`}>{valor}</dd>
    </div>
  )
}
