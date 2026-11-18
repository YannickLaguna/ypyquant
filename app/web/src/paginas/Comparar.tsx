/**
 * Página Comparar: selección múltiple de experimentos terminados, equity normalizada (base 100)
 * superpuesta y tabla de métricas lado a lado con el mejor valor de cada fila resaltado.
 * Responsabilidad: la selección vive en la URL (`?ids=a,b`) para poder compartir/recargar la vista.
 * Se relaciona con: api/hooks.ts (useExperimentos, useComparacion), componentes/Graficos.tsx y
 * paginas/Experimentos.tsx (botón Comparar).
 */
import { useMemo } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useComparacion, useExperimentos } from '../api/hooks'
import { CajaError, Cargando, Vacio } from '../componentes/Estado'
import { GraficoSeries, type SerieGrafico } from '../componentes/Graficos'
import { dinero, entero, porcentaje, ratio, resumenParametros } from '../lib/formato'
import { colorToken, PALETA_SERIES } from '../lib/tema'

/** [clave, etiqueta, formato, mejor = 'max' | 'min'] */
const FILAS: [string, string, (v: number | null | undefined) => string, 'max' | 'min'][] = [
  ['sharpe', 'Sharpe', ratio, 'max'],
  ['sortino', 'Sortino', ratio, 'max'],
  ['calmar', 'Calmar', ratio, 'max'],
  ['cagr', 'CAGR', porcentaje, 'max'],
  ['retorno_total', 'Retorno total', porcentaje, 'max'],
  ['volatilidad', 'Volatilidad', porcentaje, 'min'],
  ['max_drawdown', 'Máx. drawdown', porcentaje, 'max'],
  ['alfa', 'Alfa', porcentaje, 'max'],
  ['beta', 'Beta', ratio, 'min'],
  ['turnover_anual', 'Turnover anual', (v) => porcentaje(v, 0), 'min'],
  ['n_operaciones', 'Operaciones', entero, 'min'],
  ['comisiones_totales', 'Comisiones', (v) => dinero(v, 0), 'min'],
]

export default function Comparar() {
  const [params, setParams] = useSearchParams()
  const ids = useMemo(() => (params.get('ids') ?? '').split(',').filter(Boolean), [params])
  const { data: lista, isPending: cargandoLista, error: errorLista } = useExperimentos({ estado: 'hecho' })
  const { data: comparacion, isFetching, error } = useComparacion(ids)

  const alternar = (id: string) => {
    const siguiente = ids.includes(id) ? ids.filter((i) => i !== id) : [...ids, id]
    setParams(siguiente.length ? { ids: siguiente.join(',') } : {}, { replace: true })
  }

  const series = useMemo<SerieGrafico[]>(() => {
    if (!comparacion) return []
    const lineas: SerieGrafico[] = comparacion.ids.map((id, i) => ({
      id,
      nombre: id,
      color: PALETA_SERIES[i % PALETA_SERIES.length],
      datos: comparacion.fechas.map((f, k) => ({ time: f, value: comparacion.series[id][k] as number })),
    }))
    if (comparacion.benchmark.some((v) => v != null))
      lineas.push({
        id: '__bench', nombre: `Benchmark (${comparacion.resumenes[comparacion.ids[0]]?.benchmark ?? ''})`, color: colorToken('suave'), discontinua: true,
        datos: comparacion.fechas.map((f, k) => ({ time: f, value: comparacion.benchmark[k] as number })),
      })
    return lineas
  }, [comparacion])

  return (
    <div className="space-y-5">
      <header>
        <h1>Comparar</h1>
        <p className="text-suave">Equity normalizada a base 100 sobre las fechas comunes de los experimentos elegidos.</p>
      </header>

      <section aria-labelledby="h-sel" className="space-y-2">
        <h2 id="h-sel">Experimentos terminados</h2>
        {cargandoLista ? (
          <Cargando />
        ) : errorLista ? (
          <CajaError error={errorLista} />
        ) : (lista?.experimentos.length ?? 0) === 0 ? (
          <Vacio titulo="Todavía no hay experimentos terminados">
            Lanza alguno desde <Link to="/estrategias" className="text-acento underline">Estrategias</Link>.
          </Vacio>
        ) : (
          <ul className="tarjeta grid max-h-64 gap-x-4 overflow-auto p-2 sm:grid-cols-2 xl:grid-cols-3">
            {lista!.experimentos.map((e) => (
              <li key={e.id}>
                <label className="flex cursor-pointer items-start gap-2 rounded px-2 py-1 hover:bg-elevado">
                  <input type="checkbox" className="mt-1 accent-[rgb(var(--acento))]" checked={ids.includes(e.id)} onChange={() => alternar(e.id)} />
                  <span className="min-w-0">
                    <span className="num block truncate">{e.id}</span>
                    <span className="block truncate text-xs text-suave">{e.etiqueta || resumenParametros(e.parametros)}</span>
                  </span>
                </label>
              </li>
            ))}
          </ul>
        )}
      </section>

      {ids.length === 0 ? (
        <Vacio titulo="Elige dos o más experimentos para compararlos" />
      ) : error ? (
        <CajaError error={error} />
      ) : !comparacion ? (
        isFetching && <Cargando texto="Comparando…" />
      ) : (
        <>
          <GraficoSeries titulo="Equity normalizada (base 100)" series={series} formatoValor={(v) => v.toFixed(1)} altura={360} />
          <div className="tarjeta overflow-x-auto">
            <table className="w-full text-sm" aria-label="Métricas lado a lado">
              <thead className="border-b border-borde">
                <tr>
                  <th scope="col" className="th">Métrica</th>
                  {comparacion.ids.map((id, i) => (
                    <th key={id} scope="col" className="th text-right">
                      <span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: PALETA_SERIES[i % PALETA_SERIES.length] }} aria-hidden />
                      <Link to={`/experimentos/${id}`} className="normal-case hover:text-acento">{id}</Link>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {FILAS.map(([clave, etiqueta, fmt, mejor]) => {
                  const valores = comparacion.ids.map((id) => comparacion.metricas[id]?.[clave] ?? null)
                  const validos = valores.filter((v): v is number => typeof v === 'number')
                  const objetivo = validos.length > 1 ? (mejor === 'max' ? Math.max(...validos) : Math.min(...validos)) : null
                  return (
                    <tr key={clave} className="border-b border-borde/60 last:border-0">
                      <th scope="row" className="td text-left font-normal text-suave">{etiqueta}</th>
                      {valores.map((v, i) => (
                        <td key={comparacion.ids[i]} className={`td num text-right ${v !== null && v === objetivo ? 'font-semibold text-acento' : ''}`}>{fmt(v)}</td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
