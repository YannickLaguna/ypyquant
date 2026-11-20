/**
 * Página Paper trading: KPIs de la cuenta de Alpaca (paper), equity, posiciones, órdenes y control del
 * bot (arrancar/parar una estrategia del registro) con log en vivo por polling.
 * Responsabilidad: si la API responde 503 (sin claves) lo explica con instrucciones en vez de fallar;
 * cada bloque gestiona sus propios estados de carga/vacío/error.
 * Se relaciona con: api/hooks.ts (usePaper*), componentes/FormularioSchema.tsx (parámetros del bot),
 * componentes/Graficos.tsx y app/api/rutas_paper.py.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useArrancarBot, useCuentaPaper, useEstadoPaper, useEstrategias, useFillsPaper, useHistorialPaper, useLogBot, useOrdenesPaper, usePararBot, usePosicionesPaper } from '../api/hooks'
import type { FillPaper, FrecuenciaRebalanceo, OrdenPaper, PosicionPaper } from '../api/tipos'
import { CajaError, Cargando, Vacio } from '../componentes/Estado'
import FormularioSchema, { validarValores, valoresPorDefecto, type Valores } from '../componentes/FormularioSchema'
import { GraficoSeries } from '../componentes/Graficos'
import { Kpi } from '../componentes/Piezas'
import TablaOrdenable, { type Columna } from '../componentes/TablaOrdenable'
import { claseSigno, dinero, fechaHora, porcentaje, ratio } from '../lib/formato'
import { colorToken } from '../lib/tema'

const PERIODOS = ['1W', '1M', '3M', '1A']
// Filas visibles en las tablas de órdenes y fills antes de «Ver las N»
const FILAS_PAPER = 20
const FRECUENCIAS: [FrecuenciaRebalanceo | '', string][] = [['', 'La que sugiere la estrategia'], ['daily', 'Diaria'], ['weekly', 'Semanal'], ['monthly', 'Mensual'], ['quarterly', 'Trimestral']]

export default function PaperTrading() {
  const estado = useEstadoPaper()
  const cuenta = useCuentaPaper()
  const sinClaves = estado.data ? !estado.data.alpaca_configurado : false

  return (
    <div className="space-y-5">
      <header>
        <h1>Paper trading</h1>
        <p className="text-suave">Cuenta paper de Alpaca (sin dinero real). El bot rebalancea hacia los pesos de una estrategia del registro.</p>
      </header>

      {sinClaves && (
        <div role="alert" className="rounded-lg border border-aviso/50 bg-aviso/10 p-4">
          <p className="font-medium text-aviso">Alpaca no está configurado</p>
          <p className="mt-1">
            Define <code>ALPACA_API_KEY</code> y <code>ALPACA_SECRET_KEY</code> (de la cuenta <em>paper</em>) en el <code>.env</code> de la raíz
            del repo y reinicia la API. Mientras tanto solo puedes ver el log del bot.
          </p>
        </div>
      )}

      {!sinClaves && <BloqueCuenta cuenta={cuenta} />}
      {!sinClaves && <BloqueEquity />}
      {!sinClaves && (
        <div className="grid gap-5 xl:grid-cols-2">
          <BloquePosiciones />
          <BloqueOrdenes />
        </div>
      )}
      {!sinClaves && <BloqueFills />}
      <ControlBot sinClaves={sinClaves} />
    </div>
  )
}

function BloqueCuenta({ cuenta }: { cuenta: ReturnType<typeof useCuentaPaper> }) {
  if (cuenta.isPending) return <Cargando texto="Cargando cuenta…" />
  if (cuenta.error) return <CajaError error={cuenta.error} titulo="No se pudo leer la cuenta" />
  const c = cuenta.data!
  return (
    <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-5">
      <Kpi etiqueta="Equity" valor={dinero(c.equity, 2)} />
      <Kpi etiqueta="Efectivo" valor={dinero(c.efectivo, 2)} />
      <Kpi etiqueta="Poder de compra" valor={dinero(c.poder_compra, 0)} />
      <Kpi etiqueta="PnL del día" valor={dinero(c.pnl_dia, 2)} clase={claseSigno(c.pnl_dia)} />
      <Kpi etiqueta="PnL día %" valor={porcentaje(c.pnl_dia_pct, 2)} clase={claseSigno(c.pnl_dia_pct)} />
    </dl>
  )
}

function BloqueEquity() {
  const [periodo, setPeriodo] = useState('1M')
  const { data, isPending, error } = useHistorialPaper(periodo)
  const series = useMemo(
    () =>
      data
        ? [{
            id: 'equity', nombre: 'Equity', color: colorToken('acento'),
            datos: data.fechas.map((f, i) => ({ time: f.slice(0, 10), value: data.equity[i] as number })).filter((p) => Number.isFinite(p.value)),
          }]
        : [],
    [data],
  )
  return (
    <section aria-labelledby="h-eq" className="space-y-2">
      <div className="flex items-center justify-between">
        <h2 id="h-eq">Equity</h2>
        <div role="group" aria-label="Periodo" className="flex gap-1">
          {PERIODOS.map((p) => (
            <button key={p} type="button" className={`boton px-2 py-0.5 text-xs ${p === periodo ? 'border-acento text-acento' : ''}`} aria-pressed={p === periodo} onClick={() => setPeriodo(p)}>{p}</button>
          ))}
        </div>
      </div>
      {isPending ? <Cargando /> : error ? <CajaError error={error} /> : series[0].datos.length === 0 ? <Vacio titulo="Sin historial de equity en este periodo" /> : (
        <GraficoSeries titulo="Equity de la cuenta (USD)" series={series} altura={260} formatoValor={(v) => dinero(v, 0)} />
      )}
    </section>
  )
}

function BloquePosiciones() {
  const { data, isPending, error } = usePosicionesPaper()
  const columnas: Columna<PosicionPaper>[] = [
    { clave: 't', titulo: 'Ticker', valor: (p) => p.ticker, render: (p) => <span className="font-medium">{p.ticker}</span> },
    { clave: 'q', titulo: 'Cant.', alinear: 'der', valor: (p) => p.cantidad, render: (p) => ratio(p.cantidad, 0) },
    { clave: 'pm', titulo: 'P. medio', alinear: 'der', valor: (p) => p.precio_medio, render: (p) => dinero(p.precio_medio, 2) },
    { clave: 'pa', titulo: 'Actual', alinear: 'der', valor: (p) => p.precio_actual, render: (p) => dinero(p.precio_actual, 2) },
    { clave: 'v', titulo: 'Valor', alinear: 'der', valor: (p) => p.valor_mercado, render: (p) => dinero(p.valor_mercado, 0) },
    { clave: 'pnl', titulo: 'PnL', alinear: 'der', valor: (p) => p.pnl, render: (p) => <span className={claseSigno(p.pnl)}>{dinero(p.pnl, 0)} ({porcentaje(p.pnl_pct, 1)})</span> },
  ]
  return (
    <section aria-labelledby="h-pos" className="space-y-2">
      <h2 id="h-pos">Posiciones</h2>
      {isPending ? <Cargando /> : error ? <CajaError error={error} /> : (
        <TablaOrdenable titulo="Posiciones" columnas={columnas} filas={data!.posiciones} clave={(p) => p.ticker} vacio="Sin posiciones abiertas" />
      )}
    </section>
  )
}

function BloqueOrdenes() {
  const { data, isPending, error } = useOrdenesPaper()
  const columnas: Columna<OrdenPaper>[] = [
    { clave: 'f', titulo: 'Creada', valor: (o) => o.creada, render: (o) => fechaHora(o.creada) },
    { clave: 't', titulo: 'Ticker', valor: (o) => o.ticker, render: (o) => o.ticker },
    { clave: 'l', titulo: 'Lado', valor: (o) => o.lado, render: (o) => <span className={o.lado === 'buy' ? 'text-pos' : 'text-neg'}>{o.lado}</span> },
    { clave: 'q', titulo: 'Cant.', alinear: 'der', valor: (o) => o.cantidad, render: (o) => ratio(o.cantidad, 0) },
    { clave: 'p', titulo: 'Precio', alinear: 'der', valor: (o) => o.precio_ejecucion, render: (o) => dinero(o.precio_ejecucion, 2) },
    { clave: 'e', titulo: 'Estado', valor: (o) => o.estado, render: (o) => o.estado },
  ]
  return (
    <section aria-labelledby="h-ord" className="space-y-2">
      <h2 id="h-ord">Órdenes recientes</h2>
      {isPending ? <Cargando /> : error ? <CajaError error={error} /> : (
        <TablaOrdenable titulo="Órdenes" columnas={columnas} filas={data!.ordenes} clave={(o) => o.id} ordenInicial={{ clave: 'f', descendente: true }} vacio="Sin órdenes" filasVisibles={FILAS_PAPER} />
      )}
    </section>
  )
}

function BloqueFills() {
  const { data, isPending, error } = useFillsPaper()
  const columnas: Columna<FillPaper>[] = [
    { clave: 'f', titulo: 'Fecha', valor: (o) => o.fecha, render: (o) => fechaHora(o.fecha) },
    { clave: 't', titulo: 'Ticker', valor: (o) => o.ticker, render: (o) => o.ticker },
    { clave: 'l', titulo: 'Lado', valor: (o) => o.lado, render: (o) => <span className={o.lado === 'buy' ? 'text-pos' : 'text-neg'}>{o.lado}</span> },
    { clave: 'q', titulo: 'Cant.', alinear: 'der', valor: (o) => o.cantidad, render: (o) => ratio(o.cantidad, 0) },
    { clave: 'p', titulo: 'Precio', alinear: 'der', valor: (o) => o.precio, render: (o) => dinero(o.precio, 2) },
    { clave: 'c', titulo: 'Comisión', alinear: 'der', valor: (o) => o.comision, render: (o) => dinero(o.comision, 2) },
  ]
  return (
    <section aria-labelledby="h-fills" className="space-y-2">
      <h2 id="h-fills">Fills del bot</h2>
      {isPending ? <Cargando /> : error ? <CajaError error={error} /> : (
        <TablaOrdenable titulo="Fills" columnas={columnas} filas={data!.fills} clave={(o) => `${o.id_orden}-${o.fecha}-${o.ticker}`} ordenInicial={{ clave: 'f', descendente: true }} vacio="Sin fills todavía" filasVisibles={FILAS_PAPER} />
      )}
    </section>
  )
}

function ControlBot({ sinClaves }: { sinClaves: boolean }) {
  const estado = useEstadoPaper()
  const { data: catalogo } = useEstrategias()
  const arrancar = useArrancarBot()
  const parar = usePararBot()
  const enMarcha = estado.data?.bot.en_marcha ?? false
  const log = useLogBot(enMarcha)
  const [nombre, setNombre] = useState('')
  const [parametros, setParametros] = useState<Valores>({})
  const [tickers, setTickers] = useState('AAPL, MSFT, NVDA, AMZN, GOOGL')
  const [intervalo, setIntervalo] = useState(3600)
  const [diasHistorico, setDiasHistorico] = useState('')
  const [frecuencia, setFrecuencia] = useState<FrecuenciaRebalanceo | ''>('')
  const visor = useRef<HTMLPreElement>(null)

  const estrategia = catalogo?.estrategias.find((e) => e.nombre === (nombre || catalogo.estrategias[0]?.nombre))
  useEffect(() => {
    if (estrategia) setParametros(valoresPorDefecto(estrategia.parametros_schema))
  }, [estrategia?.nombre]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    visor.current?.scrollTo({ top: visor.current.scrollHeight })
  }, [log.data?.lineas.length])

  const errores = estrategia ? validarValores(estrategia.parametros_schema, parametros) : {}
  const lanzar = (e: React.FormEvent) => {
    e.preventDefault()
    if (!estrategia || Object.keys(errores).length) return
    arrancar.mutate({ estrategia: estrategia.nombre, parametros, tickers: tickers.split(/[\s,;]+/).filter(Boolean), intervalo_s: intervalo,
      // Opcionales: vacío = el bot decide según la estrategia
      ...(diasHistorico ? { dias_historico: Number(diasHistorico) } : {}),
      ...(frecuencia ? { frecuencia_rebalanceo: frecuencia } : {}),
    })
  }
  const bot = estado.data?.bot

  return (
    <section aria-labelledby="h-bot" className="space-y-3">
      <h2 id="h-bot">Control del bot</h2>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
        <form onSubmit={lanzar} className="tarjeta space-y-3 p-4">
          <p role="status" className="flex items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${enMarcha ? 'bg-pos' : 'bg-suave'}`} aria-hidden />
            {enMarcha ? `En marcha (${bot?.estrategia}, pid ${bot?.pid})` : bot?.codigo_salida != null ? `Detenido (código ${bot.codigo_salida})` : 'Detenido'}
          </p>
          <div>
            <label htmlFor="bot-estrategia" className="font-medium">Estrategia</label>
            <select id="bot-estrategia" className="campo mt-1" value={estrategia?.nombre ?? ''} disabled={enMarcha} onChange={(e) => setNombre(e.target.value)}>
              {(catalogo?.estrategias ?? []).map((e) => <option key={e.nombre} value={e.nombre}>{e.nombre}</option>)}
            </select>
          </div>
          {estrategia && <FormularioSchema esquema={estrategia.parametros_schema} valores={parametros} onChange={setParametros} idBase="bot" />}
          <div>
            <label htmlFor="bot-tickers" className="font-medium">Tickers</label>
            <input id="bot-tickers" className="campo num mt-1" value={tickers} disabled={enMarcha} onChange={(e) => setTickers(e.target.value)} />
          </div>
          <div>
            <label htmlFor="bot-intervalo" className="font-medium">Segundos entre iteraciones</label>
            <input id="bot-intervalo" type="number" min={5} className="campo num mt-1" value={intervalo} disabled={enMarcha} onChange={(e) => setIntervalo(Number(e.target.value))} />
          </div>
          <div>
            <label htmlFor="bot-dias" className="font-medium">Días de historia (opcional)</label>
            <input id="bot-dias" type="number" min={1} placeholder="Según la estrategia" className="campo num mt-1" value={diasHistorico} disabled={enMarcha} onChange={(e) => setDiasHistorico(e.target.value)} />
          </div>
          <div>
            <label htmlFor="bot-frecuencia" className="font-medium">Frecuencia de rebalanceo (opcional)</label>
            <select id="bot-frecuencia" className="campo mt-1" value={frecuencia} disabled={enMarcha} onChange={(e) => setFrecuencia(e.target.value as FrecuenciaRebalanceo | '')}>
              {FRECUENCIAS.map(([valor, texto]) => <option key={valor} value={valor}>{texto}</option>)}
            </select>
          </div>
          {(arrancar.error || parar.error) && <CajaError error={arrancar.error ?? parar.error} titulo="El bot no respondió" />}
          <div className="flex gap-2">
            <button type="submit" className="boton boton-primario" disabled={enMarcha || sinClaves || arrancar.isPending || !estrategia}>Arrancar</button>
            <button type="button" className="boton boton-peligro" disabled={!enMarcha || parar.isPending} onClick={() => parar.mutate()}>Parar</button>
          </div>
        </form>
        <div className="tarjeta flex min-h-[16rem] min-w-0 flex-col p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-suave">Log</p>
          <pre ref={visor} role="log" aria-live="off" tabIndex={0} className="num h-64 flex-1 overflow-auto whitespace-pre-wrap break-words text-xs">
            {log.data?.lineas.length ? log.data.lineas.join('\n') : 'Sin salida todavía.'}
          </pre>
        </div>
      </div>
    </section>
  )
}
