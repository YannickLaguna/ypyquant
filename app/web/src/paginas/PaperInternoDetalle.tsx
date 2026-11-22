/**
 * Detalle de una sesión de paper interno: curva real frente al backtest espejo, métricas forward, posiciones actuales,
 * pesos objetivo, avisos, operaciones y los controles pausar / reanudar / detener.
 * Responsabilidad:
 * - «Real vs backtest espejo»: el espejo es el backtest de la MISMA receta del alta a la última fecha, ejecutado al cierre
 *   (se calcula en segundo plano y la página lo sondea); en modo `cierre` ambas curvas deben coincidir, en
 *   `apertura_siguiente` la diferencia es el coste de ejecutar a la apertura siguiente.
 * - Avisa de que el modo `cierre` es optimista y muestra la decisión pendiente de ejecutar (modo apertura siguiente).
 * - «Tiempo en paper interno»: periodo (alta → última fecha procesada), duración legible («11 sesiones de mercado · 2
 *   semanas»), días pendientes, frecuencia de rebalanceo y fecha del próximo (y del último) rebalanceo.
 * - Con universo `sp500_pit` avisa de membresías antiguas o con un número de miembros anómalo (`avisos_universo`).
 * - Detener pide confirmación (no se puede reanudar). Los avisos (re-bases, liquidaciones, errores de estrategia) salen
 *   de `avisos` de la API.
 * Se relaciona con: api/paper_interno.ts (hooks), componentes/Graficos.tsx, TablaOrdenable.tsx, Piezas.tsx (Kpi),
 * Estado.tsx, InsigniaSesionPaper.tsx, lib/paperInterno.ts (textos de tiempo) y paginas/PaperInterno.tsx (lista).
 */
import { useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  useControlSesionPaper,
  useDetalleSesionPaper,
  useEquitySesionPaper,
  useEspejoSesionPaper,
  useOperacionesSesionPaper,
  type OperacionSesionPaper,
  type PosicionSesionPaper,
} from '../api/paper_interno'
import { CajaError, Cargando, Vacio } from '../componentes/Estado'
import { GraficoSeries, type SerieGrafico } from '../componentes/Graficos'
import InsigniaSesionPaper from '../componentes/InsigniaSesionPaper'
import { Kpi } from '../componentes/Piezas'
import TablaOrdenable, { type Columna } from '../componentes/TablaOrdenable'
import { claseSigno, dinero, entero, fechaCorta, porcentaje, ratio, resumenParametros } from '../lib/formato'
import { resumenTiempo, textoModo } from '../lib/paperInterno'
import { colorToken } from '../lib/tema'

const FILAS_OPERACIONES = 20

export default function PaperInternoDetalle() {
  const { id } = useParams()
  const { data, isPending, error } = useDetalleSesionPaper(id)
  const equity = useEquitySesionPaper(id)
  const espejo = useEspejoSesionPaper(id)
  const operaciones = useOperacionesSesionPaper(id)
  const control = useControlSesionPaper(id ?? '')

  const series = useMemo<SerieGrafico[]>(() => {
    const real = equity.data?.equity ?? []
    const lineas: SerieGrafico[] = [{ id: 'real', nombre: 'Paper (real)', color: colorToken('acento'), datos: real.map((p) => ({ time: p.fecha, value: p.equity })) }]
    if (espejo.data && espejo.data.espejo.length > 0)
      lineas.push({ id: 'espejo', nombre: 'Backtest espejo (al cierre)', color: colorToken('suave'), discontinua: true, datos: espejo.data.espejo.map((p) => ({ time: p.fecha, value: p.equity })) })
    return lineas
  }, [equity.data, espejo.data])

  if (isPending) return <Cargando texto="Cargando sesión…" />
  if (error) return <CajaError error={error} />

  const m = data.metricas ?? {}
  const num = (k: string) => (typeof m[k] === 'number' ? (m[k] as number) : null)
  const receta = data.sesion.receta
  const activa = data.estado === 'activa'
  const tiempo = resumenTiempo(data)
  const detener = () => {
    if (window.confirm(`¿Detener la sesión ${data.nombre}? No se podrá reanudar.`)) control.mutate('detener')
  }

  const columnasPosiciones: Columna<PosicionSesionPaper>[] = [
    { clave: 'ticker', titulo: 'Ticker', valor: (p) => p.ticker, render: (p) => p.ticker },
    { clave: 'cantidad', titulo: 'Cantidad', alinear: 'der', valor: (p) => p.cantidad, render: (p) => ratio(p.cantidad, 2) },
    { clave: 'medio', titulo: 'Precio medio', alinear: 'der', valor: (p) => p.precio_medio, render: (p) => dinero(p.precio_medio, 2) },
    { clave: 'cierre', titulo: 'Último cierre', alinear: 'der', valor: (p) => p.ultimo_cierre, render: (p) => dinero(p.ultimo_cierre, 2) },
    { clave: 'valor', titulo: 'Valor', alinear: 'der', valor: (p) => p.valor, render: (p) => dinero(p.valor, 0) },
    { clave: 'peso', titulo: 'Peso', alinear: 'der', valor: (p) => p.peso, render: (p) => porcentaje(p.peso, 1) },
  ]
  const columnasOperaciones: Columna<OperacionSesionPaper>[] = [
    { clave: 'fecha', titulo: 'Fecha', valor: (o) => o.fecha, render: (o) => o.fecha },
    { clave: 'ticker', titulo: 'Ticker', valor: (o) => o.ticker, render: (o) => o.ticker },
    { clave: 'lado', titulo: 'Lado', valor: (o) => o.lado, render: (o) => <span className={o.lado === 'compra' ? 'text-pos' : 'text-neg'}>{o.lado}</span> },
    { clave: 'cantidad', titulo: 'Cantidad', alinear: 'der', valor: (o) => o.cantidad, render: (o) => ratio(o.cantidad, 2) },
    { clave: 'precio', titulo: 'Precio', alinear: 'der', valor: (o) => o.precio, render: (o) => dinero(o.precio, 2) },
    { clave: 'comision', titulo: 'Comisión', alinear: 'der', valor: (o) => o.comision, render: (o) => dinero(o.comision, 2) },
    { clave: 'slippage', titulo: 'Slippage', alinear: 'der', valor: (o) => o.slippage, render: (o) => dinero(o.slippage, 2) },
    { clave: 'pnl', titulo: 'PnL', alinear: 'der', valor: (o) => o.pnl, render: (o) => <span className={claseSigno(o.pnl)}>{dinero(o.pnl, 2)}</span> },
  ]

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-suave"><Link to="/paper-interno" className="hover:text-acento">Paper interno</Link> /</p>
          <h1 className="flex flex-wrap items-center gap-2">
            <span className="break-all">{data.nombre}</span>
            <InsigniaSesionPaper estado={data.estado} />
          </h1>
          <p className="text-suave">
            {receta.estrategia} · {textoModo(data.modo_ejecucion)} · alta {fechaCorta(data.fecha_alta)} · capital {dinero(data.capital_inicial, 0)} ·{' '}
            {typeof receta.universo === 'string' ? receta.universo : `${receta.universo.length} tickers`} · rebalanceo {receta.frecuencia_rebalanceo}
          </p>
          <p className="num text-xs text-suave">
            {resumenParametros(receta.parametros)} · comisión {porcentaje(receta.comision, 2)} · slippage {porcentaje(receta.slippage, 2)}
          </p>
          {data.id_experimento && (
            <p className="text-xs text-suave">Origen: <Link to={`/experimentos/${data.id_experimento}`} className="num text-acento hover:underline">{data.id_experimento}</Link></p>
          )}
        </div>
        <div className="flex gap-2">
          {activa && <button type="button" className="boton" onClick={() => control.mutate('pausar')} disabled={control.isPending}>Pausar</button>}
          {data.estado === 'pausada' && <button type="button" className="boton" onClick={() => control.mutate('reanudar')} disabled={control.isPending}>Reanudar</button>}
          {data.estado !== 'detenida' && <button type="button" className="boton boton-peligro" onClick={detener} disabled={control.isPending}>Detener</button>}
        </div>
      </header>

      {control.error && <CajaError error={control.error} titulo="No se pudo cambiar el estado" />}

      {data.ejecucion_optimista && (
        <p role="note" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
          Ejecución optimista: en modo «al cierre» la sesión decide y ejecuta con el mismo precio de cierre, igual que el backtest. Sirve para comprobar
          la paridad; para ver cómo se comportaría de verdad usa «apertura siguiente».
        </p>
      )}
      <section aria-labelledby="h-tiempo" className="space-y-2">
        <h2 id="h-tiempo">Tiempo en paper interno</h2>
        <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-5">
          <Kpi etiqueta="Periodo (alta → última fecha)" valor={tiempo.periodo} />
          <Kpi etiqueta="Duración" valor={tiempo.duracion} ayuda="Sesiones de mercado procesadas y tiempo de calendario desde el alta" />
          <Kpi etiqueta="Pendientes (días hábiles)" valor={tiempo.pendientes} clase={data.al_dia ? 'text-pos' : 'text-aviso'} ayuda="Días cerrados aún sin procesar; se avanzan solos mientras la API esté en marcha" />
          <Kpi etiqueta="Frecuencia" valor={tiempo.frecuencia} ayuda={data.ultimo_rebalanceo ? `Último rebalanceo: ${fechaCorta(data.ultimo_rebalanceo)}` : 'Aún no ha rebalanceado'} />
          <Kpi etiqueta="Próximo rebalanceo" valor={tiempo.proximo} ayuda="Estimado: no tiene en cuenta los festivos bursátiles" />
        </dl>
      </section>
      {data.avisos_universo.length > 0 && (
        <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-sm text-aviso">
          <p className="font-medium">Universo {typeof receta.universo === 'string' ? receta.universo : ''}: calidad de las membresías</p>
          <ul className="mt-1 list-disc pl-5">{data.avisos_universo.map((a) => <li key={a}>{a}</li>)}</ul>
        </div>
      )}
      {data.pesos_pendientes && (
        <div role="status" className="rounded-lg border border-borde p-3 text-sm text-suave">
          Decisión del {fechaCorta(data.fecha_decision_pendiente)} pendiente de ejecutar a la apertura siguiente:{' '}
          <span className="num">{Object.entries(data.pesos_pendientes).map(([t, p]) => `${t} ${porcentaje(p, 1)}`).join(' · ')}</span>
        </div>
      )}
      {data.avisos.length > 0 && (
        <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
          <p className="font-medium">Avisos de la sesión ({entero(data.avisos.length)})</p>
          <ul className="mt-1 list-disc pl-5 text-sm">
            {data.avisos.slice(-8).map((a) => <li key={`${a.fecha}-${a.mensaje}`}>{a.mensaje}</li>)}
          </ul>
        </div>
      )}

      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-7">
        <Kpi etiqueta="Equity (USD)" valor={dinero(data.equity, 0)} />
        <Kpi etiqueta="Retorno total (%)" valor={porcentaje(data.retorno_total, 2)} clase={claseSigno(data.retorno_total)} />
        <Kpi etiqueta="Sharpe" valor={ratio(num('sharpe'))} ayuda="Con pocas sesiones no es fiable" />
        <Kpi etiqueta="Máx. drawdown (%)" valor={porcentaje(num('max_drawdown'))} clase="text-neg" />
        <Kpi etiqueta="Volatilidad (%)" valor={porcentaje(num('volatilidad'))} />
        <Kpi etiqueta="Costes (USD)" valor={dinero(data.costes_detalle.total, 0)} ayuda={`Comisiones ${dinero(data.costes_detalle.comisiones, 0)} + slippage ${dinero(data.costes_detalle.slippage, 0)}`} />
        <Kpi etiqueta="Sesiones de mercado" valor={entero(data.dias)} ayuda="Días de trading procesados" />
      </dl>

      <section aria-labelledby="h-espejo" className="space-y-2">
        <h2 id="h-espejo">Real vs backtest espejo</h2>
        {equity.isPending ? (
          <Cargando />
        ) : (equity.data?.equity.length ?? 0) === 0 ? (
          <Vacio titulo="La sesión aún no tiene días procesados">Pulsa «Avanzar ahora» en la lista de sesiones para procesar los días cerrados.</Vacio>
        ) : (
          <>
            <GraficoSeries titulo="Equity real y backtest espejo (USD)" series={series} formatoValor={(v) => dinero(v, 0)} altura={340} />
            {espejo.data && (espejo.data.estado === 'pendiente' || espejo.data.estado === 'en_curso') && <Cargando texto="Calculando el backtest espejo…" />}
            {espejo.data?.estado === 'error' && <p className="text-sm text-neg">El backtest espejo falló.</p>}
            {espejo.data?.estado === 'sin_datos' && <p className="text-sm text-suave">El espejo aparece cuando hay al menos dos días procesados.</p>}
          </>
        )}
      </section>

      <div className="grid gap-5 lg:grid-cols-2">
        <section aria-labelledby="h-pos" className="min-w-0 space-y-2">
          <h2 id="h-pos">Posiciones actuales</h2>
          <TablaOrdenable titulo="Posiciones actuales" columnas={columnasPosiciones} filas={data.posiciones} clave={(p) => p.ticker} vacio={`Todo en efectivo (${dinero(data.efectivo, 0)})`} ordenInicial={{ clave: 'valor', descendente: true }} />
          <p className="text-xs text-suave">Efectivo: <span className="num">{dinero(data.efectivo, 0)}</span></p>
        </section>
        <section aria-labelledby="h-obj" className="min-w-0 space-y-2">
          <h2 id="h-obj">Pesos objetivo de la última decisión</h2>
          <div className="tarjeta">
            <table className="w-full text-sm" aria-label="Pesos objetivo">
              <tbody>
                {data.pesos_objetivo.length === 0 && <tr><td className="p-4 text-center text-suave">Todavía no hay decisiones</td></tr>}
                {[...data.pesos_objetivo].sort((a, b) => b.peso - a.peso).map((p) => (
                  <tr key={p.ticker} className="border-b border-borde/60 last:border-0">
                    <th scope="row" className="td text-left font-normal">{p.ticker}</th>
                    <td className="td num text-right">{porcentaje(p.peso, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <section aria-labelledby="h-ops" className="space-y-2">
        <h2 id="h-ops">Operaciones ({entero(data.n_operaciones)})</h2>
        <TablaOrdenable titulo="Operaciones" columnas={columnasOperaciones} filas={operaciones.data?.operaciones ?? []} clave={(o) => `${o.fecha}-${o.ticker}-${o.lado}-${o.cantidad}`} vacio="Sin operaciones" filasVisibles={FILAS_OPERACIONES} ordenInicial={{ clave: 'fecha', descendente: true }} />
      </section>
    </div>
  )
}
