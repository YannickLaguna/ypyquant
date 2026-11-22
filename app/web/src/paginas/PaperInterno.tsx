/**
 * Página Paper interno: las sesiones de simulación forward (receta congelada de un experimento, precios reales, sin
 * broker) en una tabla ordenable, con el botón «Avanzar ahora» y el gráfico de la selección.
 * Responsabilidad:
 * - Tabla: nombre (enlace al detalle), estrategia, modo de ejecución, periodo (alta → última fecha procesada), duración
 *   («11 sesiones de mercado · 2 semanas»), retorno total (%), Sharpe, drawdown máximo (%), costes (USD), estado, días
 *   pendientes y rebalanceo (frecuencia de la receta y fecha del próximo). Las cabeceras llevan su unidad.
 * - Comprobación automática: el temporizador de la API (`GET /planificador`) avanza las sesiones cada N minutos mientras
 *   la API vive; aquí se ve cuándo es la próxima comprobación y qué hizo la última.
 * - «Avanzar ahora» lanza `POST /avanzar` y sondea su estado mientras está en curso (botón deshabilitado, aviso de
 *   progreso); al terminar muestra qué hizo cada sesión (días procesados, avisos, errores) y la tabla se refresca.
 * - Selección múltiple -> equity a base 100 en la fecha común (lib/paperInterno.ts) con el benchmark de la primera,
 *   con el mismo gráfico que Comparar. Las sesiones se crean desde la ficha de un experimento terminado.
 * Se relaciona con: api/paper_interno.ts (hooks), componentes/Graficos.tsx, TablaOrdenable.tsx, Estado.tsx,
 * InsigniaSesionPaper.tsx, lib/paperInterno.ts (textos de tiempo) y paginas/PaperInternoDetalle.tsx. NO es PaperTrading.tsx (Alpaca).
 */
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  useAvanzarPaper,
  useEquitiesSesionesPaper,
  useEstadoAvancePaper,
  usePlanificadorPaper,
  useSesionesPaper,
  type ResumenSesionPaper,
} from '../api/paper_interno'
import { CajaError, Cargando, Vacio } from '../componentes/Estado'
import { GraficoSeries, type SerieGrafico } from '../componentes/Graficos'
import InsigniaSesionPaper from '../componentes/InsigniaSesionPaper'
import TablaOrdenable, { type Columna } from '../componentes/TablaOrdenable'
import { claseSigno, dinero, entero, fechaHora, porcentaje, ratio } from '../lib/formato'
import { normalizarBase100, resumenTiempo, textoModo, textoResultadoComprobacion } from '../lib/paperInterno'
import { colorToken, PALETA_SERIES } from '../lib/tema'

export default function PaperInterno() {
  const { data, isPending, error } = useSesionesPaper()
  const avance = useEstadoAvancePaper()
  const planificador = usePlanificadorPaper().data
  const avanzar = useAvanzarPaper()
  const [marcadas, setMarcadas] = useState<Set<string>>(new Set())
  const enCurso = avance.data?.en_curso ?? false

  const sesiones = useMemo(() => data?.sesiones ?? [], [data])
  const ids = useMemo(() => sesiones.filter((s) => marcadas.has(s.id)).map((s) => s.id), [sesiones, marcadas])
  const equities = useEquitiesSesionesPaper(ids)
  const cargandoCurvas = equities.some((q) => q.isPending)

  const series = useMemo<SerieGrafico[]>(() => {
    const curvas = normalizarBase100(ids.map((id, i) => ({ id, equity: equities[i]?.data?.equity ?? [] })))
    const lineas: SerieGrafico[] = curvas.map((c) => ({
      id: c.id,
      nombre: sesiones.find((s) => s.id === c.id)?.nombre ?? c.id,
      color: PALETA_SERIES[ids.indexOf(c.id) % PALETA_SERIES.length],
      datos: c.puntos,
    }))
    const conBenchmark = curvas.find((c) => c.benchmark.length > 0)
    if (conBenchmark) lineas.push({ id: '__bench', nombre: 'Benchmark', color: colorToken('suave'), discontinua: true, datos: conBenchmark.benchmark })
    return lineas
  }, [ids, equities, sesiones])

  const alternar = (id: string) =>
    setMarcadas((previas) => {
      const nuevas = new Set(previas)
      if (nuevas.has(id)) nuevas.delete(id)
      else nuevas.add(id)
      return nuevas
    })

  const columnas: Columna<ResumenSesionPaper>[] = [
    {
      clave: 'sel',
      titulo: '',
      render: (s) => (
        <input
          type="checkbox"
          className="h-4 w-4 accent-[rgb(var(--acento))]"
          aria-label={`Seleccionar ${s.nombre}`}
          checked={marcadas.has(s.id)}
          onChange={() => alternar(s.id)}
        />
      ),
    },
    { clave: 'nombre', titulo: 'Sesión', valor: (s) => s.nombre, render: (s) => <Link to={`/paper-interno/${s.id}`} className="text-acento hover:underline">{s.nombre}</Link> },
    { clave: 'estrategia', titulo: 'Estrategia', valor: (s) => s.estrategia, render: (s) => s.estrategia },
    { clave: 'modo', titulo: 'Modo', valor: (s) => s.modo_ejecucion, render: (s) => textoModo(s.modo_ejecucion) },
    { clave: 'periodo', titulo: 'Periodo (alta → última fecha)', valor: (s) => s.fecha_alta, render: (s) => resumenTiempo(s).periodo },
    { clave: 'duracion', titulo: 'Duración', alinear: 'der', valor: (s) => s.dias, render: (s) => resumenTiempo(s).duracion },
    { clave: 'retorno', titulo: 'Retorno total (%)', alinear: 'der', valor: (s) => s.retorno_total, render: (s) => <span className={claseSigno(s.retorno_total)}>{porcentaje(s.retorno_total, 2)}</span> },
    { clave: 'sharpe', titulo: 'Sharpe', alinear: 'der', valor: (s) => s.sharpe, render: (s) => ratio(s.sharpe) },
    { clave: 'dd', titulo: 'Drawdown máx. (%)', alinear: 'der', valor: (s) => s.max_drawdown, render: (s) => <span className="text-neg">{porcentaje(s.max_drawdown, 1)}</span> },
    { clave: 'costes', titulo: 'Costes (USD)', alinear: 'der', valor: (s) => s.costes, render: (s) => dinero(s.costes, 0) },
    { clave: 'estado', titulo: 'Estado', valor: (s) => s.estado, render: (s) => <InsigniaSesionPaper estado={s.estado} /> },
    {
      clave: 'pendientes',
      titulo: 'Pendientes (días hábiles)',
      valor: (s) => s.dias_pendientes,
      render: (s) => <span className={s.al_dia ? 'text-pos' : 'text-aviso'}>{resumenTiempo(s).pendientes}</span>,
    },
    {
      clave: 'rebalanceo',
      titulo: 'Rebalanceo (frecuencia · próximo)',
      valor: (s) => s.proximo_rebalanceo,
      render: (s) => {
        const t = resumenTiempo(s)
        return <>{t.frecuencia}<span className="block text-xs text-suave">próximo: {t.proximo}</span></>
      },
    },
  ]

  const resultados = avance.data?.resultados ?? []
  const errores = resultados.filter((r) => r.error)

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1>Paper interno</h1>
          <p className="text-suave">
            Simulación forward de recetas congeladas con precios reales y sin broker. Crea una sesión desde la ficha de un
            experimento terminado (botón «Paper interno»).
          </p>
        </div>
        <button
          type="button"
          className="boton boton-primario"
          onClick={() => avanzar.mutate()}
          disabled={enCurso || avanzar.isPending}
          title="Procesa los días cerrados pendientes de todas las sesiones activas"
        >
          {enCurso ? 'Avanzando…' : 'Avanzar ahora'}
        </button>
      </header>

      {planificador && planificador.intervalo_min !== undefined && (
        <p role="note" className="rounded-lg border border-borde p-3 text-sm text-suave">
          {planificador.activo ? (
            <>
              Comprobación automática cada {entero(planificador.intervalo_min)} min mientras la API esté en marcha
              {planificador.proxima_comprobacion && <> · próxima: <span className="num">{fechaHora(planificador.proxima_comprobacion)}</span></>}
              {planificador.ultima && (
                <> · última ({fechaHora(planificador.ultima.hora)}): {textoResultadoComprobacion(planificador.ultima)}</>
              )}
              .
            </>
          ) : (
            <>La comprobación automática está desactivada (YPYQUANT_PAPER_INTERVALO_MIN=0): usa «Avanzar ahora» o el catch-up al arrancar la API.</>
          )}
        </p>
      )}
      {avanzar.error && <CajaError error={avanzar.error} titulo="No se pudo avanzar" />}
      {enCurso && <Cargando texto="Avanzando las sesiones: descargando precios y procesando días…" />}
      {!enCurso && avance.data?.fin && (
        <div role="status" className="rounded-lg border border-borde p-3 text-sm text-suave">
          Último avance ({fechaHora(avance.data.fin)}): {entero(resultados.reduce((n, r) => n + r.dias_procesados, 0))} días procesados en{' '}
          {entero(resultados.length)} {resultados.length === 1 ? 'sesión' : 'sesiones'}.
          {resultados.flatMap((r) => r.avisos.map((a) => <span key={`${r.id_sesion}-${a}`} className="block text-aviso">{r.id_sesion}: {a}</span>))}
        </div>
      )}
      {(errores.length > 0 || avance.data?.error) && (
        <div role="alert" className="rounded-lg border border-neg/50 bg-neg/10 p-3">
          <p className="font-medium text-neg">El avance terminó con errores</p>
          {avance.data?.error && <p>{avance.data.error}</p>}
          {errores.map((r) => <p key={r.id_sesion}><span className="num">{r.id_sesion}</span>: {r.error}</p>)}
        </div>
      )}

      {isPending ? (
        <Cargando texto="Cargando sesiones…" />
      ) : error ? (
        <CajaError error={error} />
      ) : sesiones.length === 0 ? (
        <Vacio titulo="Todavía no hay sesiones de paper interno">
          Abre un experimento terminado en <Link to="/experimentos" className="text-acento underline">Experimentos</Link> y pulsa «Paper interno»
          para congelar su receta y seguirla día a día.
        </Vacio>
      ) : (
        <TablaOrdenable titulo="Sesiones de paper interno" columnas={columnas} filas={sesiones} clave={(s) => s.id} ordenInicial={{ clave: 'nombre', descendente: false }} />
      )}

      {ids.length > 0 && (
        <section aria-labelledby="h-curvas" className="space-y-2">
          <h2 id="h-curvas">Equity de la selección (base 100 en la primera fecha común)</h2>
          {cargandoCurvas ? (
            <Cargando texto="Cargando curvas…" />
          ) : series.every((s) => s.datos.length === 0) ? (
            <Vacio titulo="Las sesiones elegidas todavía no tienen días procesados">Pulsa «Avanzar ahora» para procesar los días cerrados.</Vacio>
          ) : (
            <GraficoSeries titulo="Equity normalizada (base 100)" series={series} formatoValor={(v) => v.toFixed(1)} altura={340} />
          )}
          <p className="text-xs text-suave">Las curvas se llevan a base 100 en la primera fecha que comparten todas las sesiones elegidas.</p>
        </section>
      )}
    </div>
  )
}
