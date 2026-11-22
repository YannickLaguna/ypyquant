/**
 * Página Paper trading (Alpaca multi-perfil): una pestaña por cuenta paper (hasta 3 activas a la vez).
 * Responsabilidad: elegir el perfil (`?perfil=` en la URL) y mostrar de esa cuenta el estado de conexión, el equity, las
 * posiciones, las órdenes, los fills del bot, las instantáneas diarias y el control del bot (vinculado a un experimento).
 * Un perfil sin claves se muestra «sin configurar» con las variables que hay que poner en el `.env` (nunca se piden
 * claves en pantalla); un fallo de conexión se explica sin ocultar el resto. `?experimento=ID` (botón «Llevar a Alpaca»
 * de la ficha del experimento) preselecciona el experimento del bot.
 * Se relaciona con: api/alpaca.ts (hooks por perfil), componentes/{SelectorPerfil,PerfilSinConfigurar,BotPerfil}.tsx,
 * componentes/Graficos.tsx y app/api/rutas_paper.py + rutas_alpaca.py.
 */
import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  type PerfilAlpaca,
  useCuentaPerfil,
  useFillsPerfil,
  useHistorialPerfil,
  useInstantaneasPerfil,
  useOrdenesPerfil,
  usePerfilesAlpaca,
  usePosicionesPerfil,
} from '../api/alpaca'
import type { FillPaper, OrdenPaper, PosicionPaper } from '../api/tipos'
import BotPerfil from '../componentes/BotPerfil'
import { CajaError, Cargando, Vacio } from '../componentes/Estado'
import { GraficoSeries } from '../componentes/Graficos'
import PerfilSinConfigurar from '../componentes/PerfilSinConfigurar'
import { Kpi } from '../componentes/Piezas'
import SelectorPerfil from '../componentes/SelectorPerfil'
import TablaOrdenable, { type Columna } from '../componentes/TablaOrdenable'
import { claseSigno, dinero, fechaHora, porcentaje, ratio } from '../lib/formato'
import { colorToken } from '../lib/tema'

const PERIODOS = ['1W', '1M', '3M', '1A']
// Filas visibles en las tablas de órdenes y fills antes de «Ver las N»
const FILAS_PAPER = 20

/** Perfil a mostrar: el de la URL si existe; si no, el primero configurado y sin bot, el primero configurado o el primero. */
export function perfilInicial(perfiles: PerfilAlpaca[], pedido: string | null): string {
  if (pedido && perfiles.some((p) => p.nombre === pedido)) return pedido
  const configurados = perfiles.filter((p) => p.configurado)
  return (configurados.find((p) => !p.bot.en_marcha) ?? configurados[0] ?? perfiles[0])?.nombre ?? ''
}

export default function PaperTrading() {
  const perfiles = usePerfilesAlpaca()
  const [parametros, setParametros] = useSearchParams()
  const lista = perfiles.data?.perfiles ?? []
  const seleccionado = perfilInicial(lista, parametros.get('perfil'))
  const perfil = lista.find((p) => p.nombre === seleccionado)

  const elegir = (nombre: string) => {
    const siguiente = new URLSearchParams(parametros)
    siguiente.set('perfil', nombre)
    setParametros(siguiente, { replace: true })
  }

  return (
    <div className="space-y-5">
      <header>
        <h1>Paper trading</h1>
        <p className="text-suave">
          Cuentas paper de Alpaca (sin dinero real), una por perfil. Cada bot opera un experimento: se congela su receta al arrancar.
        </p>
      </header>

      {perfiles.isPending && <Cargando texto="Buscando cuentas…" />}
      {perfiles.error && <CajaError error={perfiles.error} titulo="No se pudieron listar las cuentas" />}
      {perfiles.data?.aviso && (
        <p role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">{perfiles.data.aviso}</p>
      )}

      {lista.length > 0 && <SelectorPerfil perfiles={lista} seleccionado={seleccionado} alSeleccionar={elegir} />}

      {perfil && !perfil.configurado && <PerfilSinConfigurar perfil={perfil} />}
      {perfil?.configurado && <DetallePerfil perfil={perfil} experimentoInicial={parametros.get('experimento') ?? undefined} />}
    </div>
  )
}

function DetallePerfil({ perfil, experimentoInicial }: { perfil: PerfilAlpaca; experimentoInicial?: string }) {
  const nombre = perfil.nombre
  return (
    <div className="space-y-5" role="tabpanel" aria-labelledby={`perfil-${nombre}`}>
      {perfil.conexion.conectada === false && (
        <div role="alert" className="rounded-lg border border-neg/50 bg-neg/10 p-4">
          <p className="font-medium text-neg">Sin conexión con Alpaca ({nombre})</p>
          <p className="mt-1 break-words">{perfil.conexion.error ?? 'La cuenta no respondió.'}</p>
          <p className="mt-1 text-xs text-suave">Comprueba que las claves de este perfil sean de una cuenta paper y estén bien copiadas en el .env.</p>
        </div>
      )}
      <BloqueCuenta perfil={nombre} />
      <BloqueEquity perfil={nombre} />
      <div className="grid gap-5 xl:grid-cols-2">
        <BloquePosiciones perfil={nombre} />
        <BloqueOrdenes perfil={nombre} />
      </div>
      <BloqueFills perfil={nombre} />
      <BloqueInstantaneas perfil={nombre} />
      <BotPerfil perfil={perfil} experimentoInicial={experimentoInicial} />
    </div>
  )
}

function BloqueCuenta({ perfil }: { perfil: string }) {
  const cuenta = useCuentaPerfil(perfil)
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

function BloqueEquity({ perfil }: { perfil: string }) {
  const [periodo, setPeriodo] = useState('1M')
  const { data, isPending, error } = useHistorialPerfil(perfil, periodo)
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

function BloquePosiciones({ perfil }: { perfil: string }) {
  const { data, isPending, error } = usePosicionesPerfil(perfil)
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

function BloqueOrdenes({ perfil }: { perfil: string }) {
  const { data, isPending, error } = useOrdenesPerfil(perfil)
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

function BloqueFills({ perfil }: { perfil: string }) {
  const { data, isPending, error } = useFillsPerfil(perfil)
  const columnas: Columna<FillPaper & { precio_referencia?: number }>[] = [
    { clave: 'f', titulo: 'Fecha', valor: (o) => o.fecha, render: (o) => fechaHora(o.fecha) },
    { clave: 't', titulo: 'Ticker', valor: (o) => o.ticker, render: (o) => o.ticker },
    { clave: 'l', titulo: 'Lado', valor: (o) => o.lado, render: (o) => <span className={o.lado === 'compra' || o.lado === 'buy' ? 'text-pos' : 'text-neg'}>{o.lado}</span> },
    { clave: 'q', titulo: 'Cant.', alinear: 'der', valor: (o) => o.cantidad, render: (o) => ratio(o.cantidad, 0) },
    { clave: 'p', titulo: 'Precio', alinear: 'der', valor: (o) => o.precio, render: (o) => dinero(o.precio, 2) },
    { clave: 'r', titulo: 'P. referencia', alinear: 'der', valor: (o) => o.precio_referencia ?? null, render: (o) => (o.precio_referencia != null ? dinero(o.precio_referencia, 2) : '—') },
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

function BloqueInstantaneas({ perfil }: { perfil: string }) {
  const { data, isPending, error } = useInstantaneasPerfil(perfil)
  const nPosiciones = (texto: string) => Object.keys(JSON.parse(texto || '{}')).length
  const columnas: Columna<{ fecha: string; equity: number; efectivo: number; posiciones: string }>[] = [
    { clave: 'f', titulo: 'Fecha', valor: (i) => i.fecha, render: (i) => i.fecha },
    { clave: 'e', titulo: 'Equity', alinear: 'der', valor: (i) => i.equity, render: (i) => dinero(i.equity, 2) },
    { clave: 'c', titulo: 'Efectivo', alinear: 'der', valor: (i) => i.efectivo, render: (i) => dinero(i.efectivo, 2) },
    { clave: 'n', titulo: 'Posiciones', alinear: 'der', valor: (i) => nPosiciones(i.posiciones), render: (i) => nPosiciones(i.posiciones) },
  ]
  return (
    <section aria-labelledby="h-inst" className="space-y-2">
      <h2 id="h-inst">Instantáneas diarias</h2>
      <p className="text-xs text-suave">Una foto de la cuenta por día que guarda el bot (equity, efectivo y posiciones): sirve para medir el descuadre de cartera.</p>
      {isPending ? <Cargando /> : error ? <CajaError error={error} /> : (
        <TablaOrdenable titulo="Instantáneas" columnas={columnas} filas={data!.instantaneas} clave={(i) => i.fecha} ordenInicial={{ clave: 'f', descendente: true }} vacio="Aún no hay instantáneas: las guarda el bot al ejecutarse" filasVisibles={FILAS_PAPER} />
      )}
    </section>
  )
}
