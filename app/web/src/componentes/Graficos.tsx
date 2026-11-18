/**
 * Gráficos: líneas/áreas con lightweight-charts (TradingView) y área apilada de pesos en SVG.
 * Responsabilidad:
 * - `GraficoSeries`: varias series temporales (equity, benchmark, drawdown, comparación) con leyenda
 *   accesible que muestra los valores bajo el cursor, marcas opcionales (rebalanceos/operaciones),
 *   autoajuste al contenedor y recoloreado al cambiar de tema.
 * - `GraficoPesos`: pesos objetivo en el tiempo como área apilada escalonada (los 11 mayores + «otros»)
 *   con lectura de la fecha bajo el cursor.
 * Se relaciona con: lib/tema.ts (colores y evento `tema-cambiado`), paginas/ExperimentoDetalle.tsx,
 * Comparar.tsx y PaperTrading.tsx. En los tests se sustituye por un mock (jsdom no tiene canvas).
 */
import {
  AreaSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { PesoFila } from '../api/tipos'
import { porcentaje } from '../lib/formato'
import { colorToken, PALETA_SERIES } from '../lib/tema'

export interface PuntoSerie {
  time: string
  value: number
}

export interface SerieGrafico {
  id: string
  nombre: string
  datos: PuntoSerie[]
  color: string
  tipo?: 'linea' | 'area'
  discontinua?: boolean
}

export interface MarcaGrafico {
  time: string
  texto: string
  posicion: 'arriba' | 'abajo'
  color: string
}

interface PropsSeries {
  series: SerieGrafico[]
  altura?: number
  /** Marcas sobre la primera serie. */
  marcas?: MarcaGrafico[]
  formatoValor?: (v: number) => string
  titulo: string
}

const ordenadas = (datos: PuntoSerie[]) => {
  const vistos = new Map<string, number>()
  for (const d of datos) if (Number.isFinite(d.value)) vistos.set(d.time, d.value)
  return [...vistos.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([time, value]) => ({ time: time as Time, value }))
}

export function GraficoSeries({ series, altura = 320, marcas, formatoValor, titulo }: PropsSeries) {
  const contenedor = useRef<HTMLDivElement>(null)
  const grafico = useRef<IChartApi | null>(null)
  const [lectura, setLectura] = useState<{ fecha: string; valores: Record<string, number> } | null>(null)
  const formato = formatoValor ?? ((v: number) => v.toLocaleString('es-ES', { maximumFractionDigits: 2 }))

  // Opciones dependientes del tema
  const colores = () => ({
    layout: { background: { type: ColorType.Solid, color: colorToken('panel') }, textColor: colorToken('suave'), fontFamily: 'Inter, system-ui, sans-serif' },
    grid: { vertLines: { color: colorToken('borde', 0.5) }, horzLines: { color: colorToken('borde', 0.5) } },
    rightPriceScale: { borderColor: colorToken('borde') },
    timeScale: { borderColor: colorToken('borde') },
  })

  useEffect(() => {
    if (!contenedor.current) return
    const chart = createChart(contenedor.current, { autoSize: true, height: altura, ...colores(), crosshair: { mode: 0 } })
    grafico.current = chart
    const apis: [string, ISeriesApi<'Line' | 'Area'>][] = []
    series.forEach((s, i) => {
      const comun = { color: s.color, lineWidth: 2 as const, lineStyle: s.discontinua ? 2 : 0, priceLineVisible: false, lastValueVisible: false }
      const api =
        s.tipo === 'area'
          ? chart.addSeries(AreaSeries, { lineColor: s.color, topColor: s.color + '55', bottomColor: s.color + '08', lineWidth: 2, priceLineVisible: false, lastValueVisible: false })
          : chart.addSeries(LineSeries, comun)
      api.setData(ordenadas(s.datos))
      apis.push([s.id, api])
      if (i === 0 && marcas?.length) {
        const ms: SeriesMarker<Time>[] = [...marcas]
          .sort((a, b) => a.time.localeCompare(b.time))
          .map((m) => ({ time: m.time as Time, position: m.posicion === 'arriba' ? 'aboveBar' : 'belowBar', shape: m.posicion === 'arriba' ? 'arrowDown' : 'arrowUp', color: m.color, text: m.texto }))
        createSeriesMarkers(api, ms)
      }
    })
    chart.timeScale().fitContent()
    chart.subscribeCrosshairMove((p) => {
      if (!p.time || !p.point) return setLectura(null)
      const valores: Record<string, number> = {}
      for (const [id, api] of apis) {
        const d = p.seriesData.get(api) as { value?: number } | undefined
        if (d?.value !== undefined) valores[id] = d.value
      }
      setLectura({ fecha: String(p.time), valores })
    })
    const alCambiarTema = () => chart.applyOptions(colores())
    window.addEventListener('tema-cambiado', alCambiarTema)
    return () => {
      window.removeEventListener('tema-cambiado', alCambiarTema)
      chart.remove()
      grafico.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, marcas, altura])

  return (
    <figure className="tarjeta p-3" aria-label={titulo}>
      <figcaption className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
        <span className="font-semibold uppercase tracking-wide text-suave">{titulo}</span>
        {series.map((s) => (
          <span key={s.id} className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-4 rounded-sm" style={{ background: s.color }} aria-hidden />
            {s.nombre}
            {lectura?.valores[s.id] !== undefined && <span className="num text-texto">{formato(lectura.valores[s.id])}</span>}
          </span>
        ))}
        {lectura && <span className="num ml-auto text-suave">{lectura.fecha}</span>}
      </figcaption>
      <div ref={contenedor} style={{ height: altura }} role="img" aria-label={`Gráfico: ${titulo}`} />
    </figure>
  )
}

// ---- Pesos en el tiempo --------------------------------------------------------------------

export function GraficoPesos({ pesos }: { pesos: PesoFila[] }) {
  const [indice, setIndice] = useState<number | null>(null)

  const modelo = useMemo(() => {
    const fechas = [...new Set(pesos.map((p) => p.fecha))].sort()
    const media = new Map<string, number>()
    for (const p of pesos) media.set(p.ticker, (media.get(p.ticker) ?? 0) + p.peso)
    const ranking = [...media.entries()].sort((a, b) => b[1] - a[1]).map(([t]) => t)
    const visibles = ranking.slice(0, 11)
    const hayOtros = ranking.length > visibles.length
    const capas = hayOtros ? [...visibles, 'otros'] : visibles
    const tabla = fechas.map((f) => {
      const fila: Record<string, number> = Object.fromEntries(capas.map((c) => [c, 0]))
      for (const p of pesos.filter((x) => x.fecha === f)) fila[visibles.includes(p.ticker) ? p.ticker : 'otros'] += p.peso
      return fila
    })
    return { fechas, capas, tabla }
  }, [pesos])

  if (modelo.fechas.length === 0) return null
  const { fechas, capas, tabla } = modelo
  const W = 1000
  const H = 260
  const t0 = new Date(fechas[0]).getTime()
  const t1 = new Date(fechas[fechas.length - 1]).getTime()
  const x = (f: string) => (t1 === t0 ? 0 : ((new Date(f).getTime() - t0) / (t1 - t0)) * W)
  const y = (v: number) => H - v * H

  // Área escalonada (el peso objetivo se mantiene hasta el siguiente rebalanceo)
  const poligonos = capas.map((capa, k) => {
    const base = (fila: Record<string, number>) => capas.slice(0, k).reduce((s, c) => s + fila[c], 0)
    const arriba: string[] = []
    const abajo: string[] = []
    fechas.forEach((f, i) => {
      const xi = x(f)
      const xs = i + 1 < fechas.length ? x(fechas[i + 1]) : W
      const b = base(tabla[i])
      const t = b + tabla[i][capa]
      arriba.push(`${xi},${y(t)}`, `${xs},${y(t)}`)
      abajo.unshift(`${xi},${y(b)}`)
      abajo.unshift(`${xs},${y(b)}`)
    })
    return { capa, puntos: [...arriba, ...abajo].join(' '), color: PALETA_SERIES[k % PALETA_SERIES.length] }
  })

  const alMover = (e: React.MouseEvent<SVGSVGElement>) => {
    const caja = e.currentTarget.getBoundingClientRect()
    const t = t0 + ((e.clientX - caja.left) / caja.width) * (t1 - t0)
    let mejor = 0
    fechas.forEach((f, i) => {
      if (new Date(f).getTime() <= t) mejor = i
    })
    setIndice(mejor)
  }
  const sel = indice ?? fechas.length - 1

  return (
    <figure className="tarjeta p-3" aria-label="Pesos objetivo en el tiempo">
      <figcaption className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <span className="font-semibold uppercase tracking-wide text-suave">Pesos en el tiempo</span>
        {poligonos.map((p) => (
          <span key={p.capa} className="flex items-center gap-1">
            <span className="inline-block h-2 w-3 rounded-sm" style={{ background: p.color }} aria-hidden />
            {p.capa}
          </span>
        ))}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-56 w-full" onMouseMove={alMover}
        onMouseLeave={() => setIndice(null)} role="img" aria-label="Área apilada de pesos por activo">
        {poligonos.map((p) => (
          <polygon key={p.capa} points={p.puntos} fill={p.color} fillOpacity={0.85} stroke="rgb(var(--panel))" strokeWidth={0.6} vectorEffect="non-scaling-stroke" />
        ))}
        <line x1={x(fechas[sel])} x2={x(fechas[sel])} y1={0} y2={H} stroke="rgb(var(--texto))" strokeWidth={1} vectorEffect="non-scaling-stroke" strokeDasharray="4 3" />
      </svg>
      <p className="num mt-2 text-xs text-suave" aria-live="polite">
        <span className="text-texto">{fechas[sel]}</span>
        {' — '}
        {capas.filter((c) => tabla[sel][c] > 0.0005).map((c) => `${c} ${porcentaje(tabla[sel][c], 1)}`).join(' · ')}
      </p>
    </figure>
  )
}
