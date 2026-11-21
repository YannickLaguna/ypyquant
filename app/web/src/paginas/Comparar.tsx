/**
 * Página Comparar: selección múltiple de experimentos terminados, equity normalizada (base 100)
 * superpuesta y tabla de métricas lado a lado con el mejor valor de cada fila resaltado.
 * Responsabilidad: la selección vive en la URL (`?ids=a,b`) para poder compartir/recargar la vista. Avisa «no es
 * ceteris paribus» cuando los experimentos difieren del entorno (universo, fechas, frecuencia, costes...) o de los
 * parámetros comunes (`ceteris_paribus` de `GET /api/comparar`); con `?comparativa=<id>` (viene de un marco) muestra
 * además qué comunes se aplicaron a cada estrategia.
 * Indicador conmutable (`?indicador=sharpe`, por defecto `equity`): escalares (sharpe, sortino, cagr, volatilidad, drawdown
 * máx., alfa, turnover, nº de operaciones...) como barras ordenadas con «sin dato» para quien no tiene la métrica, y
 * temporales (equity, drawdown, Sharpe y volatilidad móviles, turnover por rebalanceo) como líneas; el color de cada
 * experimento es el mismo en barras, líneas y tabla. Las series temporales las calcula `GET /api/comparar/series`.
 * Se relaciona con: api/hooks.ts (useExperimentos, useComparacion, useSeriesComparacion), componentes/Graficos.tsx,
 * componentes/GraficoBarras.tsx, componentes/SelectorIndicador.tsx, lib/indicadores.ts, lib/marcos.ts (texto del
 * aviso), paginas/MarcoDetalle.tsx (enlace con `comparativa`) y paginas/Experimentos.tsx.
 */
import { useMemo } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useComparacion, useComparativa, useExperimentos, useSeriesComparacion } from '../api/hooks'
import { CajaError, Cargando, InsigniaTfm, Vacio } from '../componentes/Estado'
import { GraficoSeries, type SerieGrafico } from '../componentes/Graficos'
import GraficoBarras from '../componentes/GraficoBarras'
import SelectorIndicador from '../componentes/SelectorIndicador'
import { resumenParametros } from '../lib/formato'
import { barrasOrdenadas, escalares, INDICADOR_POR_DEFECTO, indicadorDeUrl } from '../lib/indicadores'
import { resumenDiferencias } from '../lib/marcos'
import { colorToken, PALETA_SERIES } from '../lib/tema'

/** Filas de la tabla: los indicadores escalares (misma lista y orden que las barras). */
const FILAS = escalares()

export default function Comparar() {
  const [params, setParams] = useSearchParams()
  const ids = useMemo(() => (params.get('ids') ?? '').split(',').filter(Boolean), [params])
  const { data: lista, isPending: cargandoLista, error: errorLista } = useExperimentos({ estado: 'hecho' })
  const { data: comparacion, isFetching, error } = useComparacion(ids)
  const { data: comparativa } = useComparativa(params.get('comparativa'))
  const indicador = indicadorDeUrl(params.get('indicador'))
  const temporal = indicador.tipo === 'temporal' && indicador.clave !== 'equity'
  const { data: temporales, isFetching: cargandoSeries, error: errorSeries } = useSeriesComparacion(ids, indicador.clave, temporal)
  const avisoCeteris = resumenDiferencias(comparacion?.ceteris_paribus?.diferencias)
  const avisosMarco = comparativa ? comparativa.experimentos.flatMap((e) => e.informe.avisos) : []

  const alternar = (id: string) => {
    const siguiente = ids.includes(id) ? ids.filter((i) => i !== id) : [...ids, id]
    setParams(
      (previos) => {
        const nuevos = new URLSearchParams(previos) // conserva `indicador` y `comparativa`
        if (siguiente.length) nuevos.set('ids', siguiente.join(','))
        else nuevos.delete('ids')
        return nuevos
      },
      { replace: true },
    )
  }
  const elegirIndicador = (clave: string) =>
    setParams(
      (previos) => {
        const nuevos = new URLSearchParams(previos)
        if (clave === INDICADOR_POR_DEFECTO) nuevos.delete('indicador')
        else nuevos.set('indicador', clave)
        return nuevos
      },
      { replace: true },
    )

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

  /** Líneas de un indicador temporal distinto de la equity: cada experimento con su color de siempre. */
  const seriesTemporales = useMemo<SerieGrafico[]>(() => {
    if (!temporales || !comparacion) return []
    return comparacion.ids.flatMap((id, i) => {
      const s = temporales.series[id]
      if (!s || s.fechas.length === 0) return []
      const datos = s.fechas.flatMap((f, k) => (s.valores[k] == null ? [] : [{ time: f, value: s.valores[k] as number }]))
      return datos.length ? [{ id, nombre: id, color: PALETA_SERIES[i % PALETA_SERIES.length], datos }] : []
    })
  }, [temporales, comparacion])
  const barras = useMemo(
    () => (comparacion && indicador.tipo === 'escalar' ? barrasOrdenadas(comparacion.ids, comparacion.metricas, indicador.clave, indicador.mejor) : []),
    [comparacion, indicador],
  )

  return (
    <div className="space-y-5">
      <header>
        <h1>Comparar</h1>
        <p className="text-suave">Compara los experimentos elegidos por indicador; las series usan las fechas comunes de todos ellos.</p>
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
                    <span className="num block truncate">{e.id}<InsigniaTfm estrategia={e.estrategia} /></span>
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
          {avisoCeteris && (
            <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
              <p>{avisoCeteris}</p>
              <ul className="mt-1 list-disc pl-5 text-xs">
                {comparacion.ceteris_paribus!.diferencias.map((d) => (
                  <li key={d.campo}><span className="num">{d.campo}</span>: {Object.entries(d.valores).map(([id, v]) => `${id} = ${JSON.stringify(v)}`).join(' · ')}</li>
                ))}
              </ul>
            </div>
          )}
          {avisosMarco.length > 0 && (
            <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
              <p className="font-medium">Avisos del marco {comparativa!.id_marco}</p>
              {avisosMarco.map((a) => <p key={a}>{a}</p>)}
            </div>
          )}
          <SelectorIndicador valor={indicador.clave} onChange={elegirIndicador} />
          <p className="text-xs text-suave">{indicador.ayuda}</p>
          {indicador.tipo === 'escalar' ? (
            <GraficoBarras titulo={`${indicador.etiqueta} por experimento`} barras={barras} formato={indicador.formato} />
          ) : !temporal ? (
            <GraficoSeries titulo="Equity normalizada (base 100)" series={series} formatoValor={(v) => v.toFixed(1)} altura={360} />
          ) : errorSeries ? (
            <CajaError error={errorSeries} />
          ) : !temporales ? (
            cargandoSeries && <Cargando texto="Calculando la serie…" />
          ) : (
            <>
              {seriesTemporales.length > 0 && (
                <GraficoSeries titulo={temporales.etiqueta} series={seriesTemporales} formatoValor={indicador.formato} altura={360} />
              )}
              {temporales.sin_dato.length > 0 && (
                <p role="status" className="rounded-lg border border-borde p-3 text-sm text-suave">
                  Sin dato para {indicador.etiqueta}: {temporales.sin_dato.join(', ')}.
                  {indicador.clave === 'turnover_rebalanceo' ? ' No guardan pesos (p. ej. importados del TFM).' : ' La serie es demasiado corta o no hay volatilidad.'}
                </p>
              )}
            </>
          )}
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
                {FILAS.map(({ clave, etiqueta, formato: fmt, mejor }) => {
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
