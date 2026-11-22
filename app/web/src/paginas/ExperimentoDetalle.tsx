/**
 * Detalle de experimento: KPIs, equity vs benchmark (con marcas de rebalanceo), drawdown, tabla de
 * métricas, pesos en el tiempo y operaciones.
 * Responsabilidad: mostrar los estados pendiente/en curso (sondeo automático), error (con traza) y
 * hecho; permitir borrar el experimento; mostrar la procedencia (commit, versiones, hash de datos) en
 * un bloque plegable y las métricas de coste/riesgo (Ulcer, Martin, costes sobre ganancia bruta).
 * «Crear marco desde este experimento» abre Marcos con su entorno precargado (universo, fechas, rebalanceo, costes,
 * parámetros comunes) para lanzar otras estrategias en las mismas condiciones (ver lib/marcos.ts, marcoDesdeExperimento).
 * Los experimentos importados del TFM (estrategia `tfm_*`) llevan la insignia «TFM · importado» y una
 * nota de solo lectura; esta página no ofrece relanzar ningún experimento.
 * El botón «Paper interno» abre `CrearSesionPaper` (sesión forward con la receta congelada del experimento terminado).
 * Se relaciona con: api/hooks.ts (useExperimento, useBorrarExperimento), componentes/Graficos.tsx,
 * componentes/TablaOrdenable.tsx, componentes/Piezas.tsx y componentes/CrearSesionPaper.tsx.
 */
import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useBorrarExperimento, useExperimento } from '../api/hooks'
import type { OperacionFila } from '../api/tipos'
import CrearSesionPaper from '../componentes/CrearSesionPaper'
import { CajaError, Cargando, esImportadoTfm, Insignia, InsigniaTfm } from '../componentes/Estado'
import { GraficoPesos, GraficoSeries, type MarcaGrafico, type SerieGrafico } from '../componentes/Graficos'
import { Kpi } from '../componentes/Piezas'
import ProcedenciaExperimento from '../componentes/ProcedenciaExperimento'
import TablaOrdenable, { type Columna } from '../componentes/TablaOrdenable'
import { claseSigno, dinero, entero, fechaCorta, porcentaje, ratio, resumenParametros, resumenUniverso } from '../lib/formato'
import { etiquetaMotivo, resumenMotivos } from '../lib/senales'
import { colorToken } from '../lib/tema'
import { costesAltos } from '../lib/barridos'

// Filas de operaciones visibles antes de «Ver las N» (se ordena todo y luego se recorta; ver TablaOrdenable)
const FILAS_OPERACIONES = 20

const FILAS_METRICAS: [string, string, (v: number | null | undefined) => string][] = [
  ['retorno_total', 'Retorno total', porcentaje],
  ['cagr', 'CAGR', porcentaje],
  ['volatilidad', 'Volatilidad anual', porcentaje],
  ['sharpe', 'Sharpe', ratio],
  ['sortino', 'Sortino', ratio],
  ['calmar', 'Calmar', ratio],
  ['ulcer', 'Ulcer (índice, %)', ratio],
  ['martin', 'Martin (CAGR % / Ulcer %)', ratio],
  ['max_drawdown', 'Máx. drawdown', porcentaje],
  ['var_95', 'VaR 95 % (diario)', (v) => porcentaje(v, 2)],
  ['cvar_95', 'CVaR 95 % (diario)', (v) => porcentaje(v, 2)],
  ['alfa', 'Alfa vs benchmark', porcentaje],
  ['beta', 'Beta', ratio],
  ['tracking_error', 'Tracking error', porcentaje],
  ['information_ratio', 'Information ratio', ratio],
  ['benchmark_cagr', 'CAGR benchmark', porcentaje],
  ['benchmark_sharpe', 'Sharpe benchmark', ratio],
  ['sharpe_exceso_rf', 'Sharpe en exceso de rf', ratio],
  ['sharpe_media_desv', 'Sharpe √12·media/desv. (rf = 0)', ratio],
  ['tfm_sharpe_serie_promediada', 'TFM · Sharpe serie promediada', ratio],
  ['tfm_sharpe_media_replicas', 'TFM · Sharpe medio de réplicas', ratio],
  ['turnover_medio', 'Turnover por rebalanceo', porcentaje],
  ['turnover_anual', 'Turnover anual', porcentaje],
  ['n_rebalanceos', 'Rebalanceos', entero],
  ['n_operaciones', 'Operaciones', entero],
  ['comisiones_totales', 'Comisiones totales', (v) => dinero(v, 0)],
  ['costes_sobre_ganancia_bruta', 'Costes / ganancia bruta', ratio],
  ['capital_final', 'Capital final', (v) => dinero(v, 0)],
]

export default function ExperimentoDetalle() {
  const { id } = useParams()
  const navegar = useNavigate()
  const { data, isPending, error } = useExperimento(id)
  const borrar = useBorrarExperimento()
  const [verMarcas, setVerMarcas] = useState(false)
  const [creandoPaper, setCreandoPaper] = useState(false)

  const series = useMemo<SerieGrafico[]>(() => {
    if (!data) return []
    const lista: SerieGrafico[] = [{ id: 'equity', nombre: 'Estrategia', color: colorToken('acento'), datos: data.equity.map((p) => ({ time: p.fecha, value: p.equity })) }]
    if (data.equity.some((p) => p.benchmark != null))
      lista.push({ id: 'benchmark', nombre: data.benchmark, color: colorToken('suave'), discontinua: true, datos: data.equity.filter((p) => p.benchmark != null).map((p) => ({ time: p.fecha, value: p.benchmark as number })) })
    return lista
  }, [data])

  const drawdown = useMemo<SerieGrafico[]>(
    () => (data ? [{ id: 'dd', nombre: 'Drawdown', tipo: 'area', color: colorToken('neg'), datos: data.equity.map((p) => ({ time: p.fecha, value: (p.drawdown ?? 0) * 100 })) }] : []),
    [data],
  )

  // Una marca por fecha de rebalanceo: compras (▲) y ventas (▼) de ese día
  const marcas = useMemo<MarcaGrafico[] | undefined>(() => {
    if (!data || !verMarcas) return undefined
    const porFecha = new Map<string, { c: number; v: number }>()
    for (const o of data.operaciones) {
      const x = porFecha.get(o.fecha) ?? { c: 0, v: 0 }
      if (o.lado === 'compra') x.c++
      else x.v++
      porFecha.set(o.fecha, x)
    }
    return [...porFecha.entries()].map(([time, x]) => ({
      time,
      texto: `${x.c}c/${x.v}v`,
      posicion: x.v > x.c ? 'arriba' : 'abajo',
      color: colorToken(x.v > x.c ? 'neg' : 'pos'),
    }))
  }, [data, verMarcas])

  if (isPending) return <Cargando texto="Cargando experimento…" />
  if (error) return <CajaError error={error} />

  const m = data.metricas_completas ?? {}
  const num = (k: string) => (typeof m[k] === 'number' ? (m[k] as number) : null)
  const avisos = Array.isArray(m.avisos) ? (m.avisos as string[]) : []
  const terminado = data.estado === 'hecho'

  const columnasOperaciones: Columna<OperacionFila>[] = [
    { clave: 'fecha', titulo: 'Fecha', valor: (o) => o.fecha, render: (o) => o.fecha },
    { clave: 'ticker', titulo: 'Ticker', valor: (o) => o.ticker, render: (o) => o.ticker },
    { clave: 'lado', titulo: 'Lado', valor: (o) => o.lado, render: (o) => <span className={o.lado === 'compra' ? 'text-pos' : 'text-neg'}>{o.lado}</span> },
    { clave: 'cantidad', titulo: 'Cantidad', alinear: 'der', valor: (o) => o.cantidad, render: (o) => ratio(o.cantidad, 2) },
    { clave: 'precio', titulo: 'Precio', alinear: 'der', valor: (o) => o.precio, render: (o) => dinero(o.precio, 2) },
    { clave: 'comision', titulo: 'Comisión', alinear: 'der', valor: (o) => o.comision, render: (o) => dinero(o.comision, 2) },
    { clave: 'pnl', titulo: 'PnL', alinear: 'der', valor: (o) => o.pnl, render: (o) => <span className={claseSigno(o.pnl)}>{dinero(o.pnl, 2)}</span> },
  ]
  // Estrategias de señales: cada operación trae su motivo (señal, stop, tiempo...) y se resume arriba de la tabla
  const resumenSenales = resumenMotivos(data.operaciones)
  if (resumenSenales)
    columnasOperaciones.splice(3, 0, { clave: 'motivo', titulo: 'Motivo', valor: (o) => etiquetaMotivo(o.motivo), render: (o) => etiquetaMotivo(o.motivo) })

  const eliminar = () => {
    if (window.confirm(`¿Borrar el experimento ${data.id}?`)) borrar.mutate(data.id, { onSuccess: () => navegar('/experimentos') })
  }

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-suave"><Link to="/experimentos" className="hover:text-acento">Experimentos</Link> /</p>
          <h1 className="flex flex-wrap items-center gap-2">
            <span className="num break-all">{data.id}</span>
            <Insignia estado={data.estado} />
            <InsigniaTfm estrategia={data.estrategia} />
          </h1>
          <p className="text-suave">
            {fechaCorta(data.fecha_inicio)} → {fechaCorta(data.fecha_fin)} · {resumenUniverso(data.universo)} · rebalanceo {data.frecuencia_rebalanceo} · benchmark {data.benchmark}
          </p>
          <p className="num text-xs text-suave">{resumenParametros(data.parametros)}{data.etiqueta ? ` · ${data.etiqueta}` : ''}</p>
        </div>
        <div className="flex gap-2">
          {data.estado === 'hecho' && !esImportadoTfm(data.estrategia) && (
            <Link to={`/marcos?desde=${encodeURIComponent(data.id)}`} className="boton" title="Lanza otras estrategias con el mismo universo, fechas, rebalanceo, costes y parámetros comunes">
              Crear marco desde este experimento
            </Link>
          )}
          {data.estado === 'hecho' && !esImportadoTfm(data.estrategia) && (
            <button type="button" className="boton" onClick={() => setCreandoPaper((v) => !v)} aria-expanded={creandoPaper} title="Sigue la receta de este experimento día a día con precios reales (simulación forward, sin broker)">
              Paper interno
            </button>
          )}
          <button type="button" className="boton boton-peligro" onClick={eliminar} disabled={borrar.isPending}>Borrar</button>
        </div>
      </header>

      {creandoPaper && <CrearSesionPaper idExperimento={data.id} nombreSugerido={`${data.estrategia} ${resumenParametros(data.parametros)}`.slice(0, 80)} alCerrar={() => setCreandoPaper(false)} />}

      {esImportadoTfm(data.estrategia) && (
        <p role="note" className="rounded-lg border border-borde p-3 text-sm text-suave">
          Experimento importado del TFM original: son los resultados originales, no un backtest de esta interfaz, así que
          no se puede relanzar ni tiene código de estrategia. Úsalo como línea base en <Link to="/comparar" className="text-acento underline">Comparar</Link>.
        </p>
      )}

      {(data.estado === 'pendiente' || data.estado === 'en_curso') && <Cargando texto={data.estado === 'pendiente' ? 'En cola…' : 'Calculando el backtest…'} />}

      {data.estado === 'error' && (
        <div role="alert" className="rounded-lg border border-neg/50 bg-neg/10 p-4">
          <p className="font-medium text-neg">El experimento falló</p>
          <p className="mt-1 break-words">{data.error}</p>
          {data.traza && (
            <details className="mt-2">
              <summary className="cursor-pointer text-suave">Traza</summary>
              <pre className="mt-2 max-h-80 overflow-auto text-xs">{data.traza}</pre>
            </details>
          )}
        </div>
      )}

      {terminado && (
        <>
          {avisos.length > 0 && (
            <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
              {avisos.map((a) => <p key={a}>{a}</p>)}
            </div>
          )}

          {costesAltos(num('costes_sobre_ganancia_bruta')) && (
            <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
              Los costes se comen {porcentaje(num('costes_sobre_ganancia_bruta'), 0)} de la ganancia bruta (&gt; 50 %): el edge es frágil frente a las comisiones.
            </div>
          )}

          <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8">
            <Kpi etiqueta="Sharpe" valor={ratio(num('sharpe'))} />
            <Kpi etiqueta="CAGR" valor={porcentaje(num('cagr'))} clase={claseSigno(num('cagr'))} />
            <Kpi etiqueta="Retorno" valor={porcentaje(num('retorno_total'))} clase={claseSigno(num('retorno_total'))} />
            <Kpi etiqueta="Máx. DD" valor={porcentaje(num('max_drawdown'))} clase="text-neg" />
            <Kpi etiqueta="Volatilidad" valor={porcentaje(num('volatilidad'))} />
            <Kpi etiqueta="Sortino" valor={ratio(num('sortino'))} />
            <Kpi etiqueta="Calmar" valor={ratio(num('calmar'))} />
            <Kpi etiqueta="Turnover/año" valor={porcentaje(num('turnover_anual'), 0)} />
          </dl>

          <div className="space-y-2">
            <div className="flex justify-end">
              <label className="flex items-center gap-2 text-xs text-suave">
                <input type="checkbox" className="accent-[rgb(var(--acento))]" checked={verMarcas} onChange={(e) => setVerMarcas(e.target.checked)} />
                Marcar rebalanceos (compras/ventas por día)
              </label>
            </div>
            <GraficoSeries titulo="Equity vs benchmark (USD)" series={series} marcas={marcas} formatoValor={(v) => dinero(v, 0)} />
          </div>
          <GraficoSeries titulo="Drawdown (%)" series={drawdown} altura={180} formatoValor={(v) => `${v.toFixed(1)} %`} />

          <div className="grid gap-5 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
            <section aria-labelledby="h-met" className="space-y-2">
              <h2 id="h-met">Métricas</h2>
              <div className="tarjeta">
                <table className="w-full text-sm" aria-label="Métricas del experimento">
                  <tbody>
                    {FILAS_METRICAS.filter(([k]) => num(k) !== null).map(([k, etiqueta, fmt]) => (
                      <tr key={k} className="border-b border-borde/60 last:border-0">
                        <th scope="row" className="td text-left font-normal text-suave">{etiqueta}</th>
                        <td className="td num text-right">{fmt(num(k))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
            <section aria-labelledby="h-pesos" className="min-w-0 space-y-2">
              <h2 id="h-pesos">Pesos</h2>
              <GraficoPesos pesos={data.pesos} />
            </section>
          </div>

          <ProcedenciaExperimento procedencia={data.procedencia} />

          <section aria-labelledby="h-ops" className="space-y-2">
            <h2 id="h-ops">Operaciones ({entero(data.operaciones.length)})</h2>
            {resumenSenales && (
              <p className="text-xs text-suave" data-testid="resumen-senales">
                {entero(resumenSenales.entradas)} entradas · {entero(resumenSenales.salidas)} salidas
                {Object.entries(resumenSenales.porMotivo).map(([m, n]) => ` · ${etiquetaMotivo(m).toLowerCase()}: ${entero(n)}`).join('')}
                {resumenSenales.ajustes > 0 && ` · ${entero(resumenSenales.ajustes)} ajustes por deriva`}
              </p>
            )}
            <TablaOrdenable titulo="Operaciones" columnas={columnasOperaciones} filas={data.operaciones} clave={(o) => `${o.fecha}-${o.ticker}-${o.lado}-${o.cantidad}`} vacio="Sin operaciones" filasVisibles={FILAS_OPERACIONES} />
          </section>
        </>
      )}
    </div>
  )
}
