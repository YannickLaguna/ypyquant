/**
 * Piezas visuales de la validación en SVG (sin dependencias): semáforo, histograma con línea de
 * referencia, dispersión IS/OOS con recta de regresión y tarjeta de resultado con semáforo.
 * Responsabilidad: mostrar DSR/PBO/permutación de forma legible; el color nunca es el único canal
 * (cada semáforo lleva texto: «favorable», «dudoso», «desfavorable»).
 * Se relaciona con: lib/barridos.ts (Semaforo, histograma, regresionLineal), componentes/barrido/*
 * (paneles que los usan), lib/tema.ts (colores del tema).
 */
import { useMemo } from 'react'
import { histograma, regresionLineal, type Semaforo } from '../lib/barridos'
import { ratio } from '../lib/formato'

const ESTILO_SEMAFORO: Record<Semaforo, { clase: string; texto: string }> = {
  verde: { clase: 'border-pos/60 bg-pos/10 text-pos', texto: 'favorable' },
  ambar: { clase: 'border-aviso/60 bg-aviso/10 text-aviso', texto: 'dudoso' },
  rojo: { clase: 'border-neg/60 bg-neg/10 text-neg', texto: 'desfavorable' },
  nulo: { clase: 'border-borde text-suave', texto: 'sin dato' },
}

/** Insignia de semáforo con punto de color y texto. */
export function InsigniaSemaforo({ semaforo }: { semaforo: Semaforo }) {
  const { clase, texto } = ESTILO_SEMAFORO[semaforo]
  return (
    <span className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-xs font-medium ${clase}`}>
      <span aria-hidden>●</span>
      {texto}
    </span>
  )
}

/** Tarjeta de un resultado: etiqueta, valor grande, semáforo y explicación breve. */
export function TarjetaResultado({ etiqueta, valor, semaforo, ayuda }: { etiqueta: string; valor: string; semaforo: Semaforo; ayuda: string }) {
  return (
    <div className="tarjeta space-y-1 p-3">
      <p className="text-xs uppercase tracking-wide text-suave">{etiqueta}</p>
      <p className="flex flex-wrap items-center gap-2">
        <span className="num text-2xl font-semibold">{valor}</span>
        <InsigniaSemaforo semaforo={semaforo} />
      </p>
      <p className="text-xs text-suave">{ayuda}</p>
    </div>
  )
}

const W = 600
const H = 220
const M = { i: 40, d: 12, s: 10, b: 28 }

interface PropsHistograma {
  datos: number[]
  titulo: string
  intervalos?: number
  /** Valor de referencia (λ = 0, valor real de la permutación...). */
  referencia?: number | null
  etiquetaReferencia?: string
  etiquetaX?: string
}

/** Histograma de barras con una línea vertical opcional (el valor real frente a las permutadas). */
export function Histograma({ datos, titulo, intervalos = 20, referencia, etiquetaReferencia, etiquetaX }: PropsHistograma) {
  const modelo = useMemo(() => {
    const extra = referencia != null && Number.isFinite(referencia) ? [referencia] : []
    const bins = histograma(datos, intervalos, extra)
    return { bins, minimo: bins[0]?.desde ?? 0, maximo: bins[bins.length - 1]?.hasta ?? 1, tope: Math.max(1, ...bins.map((b) => b.cuenta)) }
  }, [datos, intervalos, referencia])
  if (modelo.bins.length === 0) return <p className="text-suave">Sin datos para el histograma.</p>
  const { bins, minimo, maximo, tope } = modelo
  const x = (v: number) => M.i + ((v - minimo) / (maximo - minimo || 1)) * (W - M.i - M.d)
  const y = (c: number) => H - M.b - (c / tope) * (H - M.b - M.s)
  return (
    <figure className="tarjeta p-3" aria-label={titulo}>
      <figcaption className="mb-1 text-xs font-semibold uppercase tracking-wide text-suave">{titulo}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`${titulo}: ${datos.length} valores${referencia != null ? `, referencia en ${ratio(referencia, 3)}` : ''}`}>
        <line x1={M.i} x2={W - M.d} y1={H - M.b} y2={H - M.b} stroke="rgb(var(--borde))" />
        {bins.map((b, i) => (
          <rect key={i} x={x(b.desde) + 0.5} y={y(b.cuenta)} width={Math.max(0, x(b.hasta) - x(b.desde) - 1)} height={H - M.b - y(b.cuenta)} fill="rgb(var(--acento))" fillOpacity={0.7}>
            <title>{`${ratio(b.desde, 3)} a ${ratio(b.hasta, 3)}: ${b.cuenta}`}</title>
          </rect>
        ))}
        {[minimo, (minimo + maximo) / 2, maximo].map((v, i) => (
          <text key={i} x={x(v)} y={H - 10} textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'} fontSize={11} fill="rgb(var(--suave))" className="num">{ratio(v, 2)}</text>
        ))}
        <text x={M.i - 6} y={M.s + 8} textAnchor="end" fontSize={11} fill="rgb(var(--suave))" className="num">{tope}</text>
        {referencia != null && Number.isFinite(referencia) && (
          <g>
            <line x1={x(referencia)} x2={x(referencia)} y1={M.s} y2={H - M.b} stroke="rgb(var(--aviso))" strokeWidth={2} strokeDasharray="5 3" />
            <text x={Math.min(x(referencia) + 4, W - 90)} y={M.s + 10} fontSize={11} fill="rgb(var(--aviso))">{etiquetaReferencia ?? 'referencia'}</text>
          </g>
        )}
      </svg>
      {etiquetaX && <p className="text-center text-xs text-suave">{etiquetaX}</p>}
    </figure>
  )
}

/** Dispersión IS (x) frente a OOS (y) de la configuración elegida en cada partición, con su recta. */
export function DispersionIsOos({ pares, pendienteApi }: { pares: [number, number][]; pendienteApi: number | null }) {
  const modelo = useMemo(() => {
    const validos = pares.filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b))
    if (validos.length === 0) return null
    const xs = validos.map((p) => p[0])
    const ys = validos.map((p) => p[1])
    const margen = (a: number[]) => { const lo = Math.min(...a); const hi = Math.max(...a); const m = (hi - lo) * 0.05 || 0.01; return [lo - m, hi + m] as const }
    return { validos, rx: margen(xs), ry: margen(ys), recta: regresionLineal(validos) }
  }, [pares])
  if (!modelo) return <p className="text-suave">Sin pares IS/OOS (T demasiado pequeño).</p>
  const { validos, rx, ry, recta } = modelo
  const x = (v: number) => M.i + ((v - rx[0]) / (rx[1] - rx[0])) * (W - M.i - M.d)
  const y = (v: number) => H - M.b - ((v - ry[0]) / (ry[1] - ry[0])) * (H - M.b - M.s)
  const pendiente = pendienteApi ?? recta?.pendiente ?? null
  return (
    <figure className="tarjeta p-3" aria-label="Dispersión IS frente a OOS">
      <figcaption className="mb-1 flex flex-wrap items-center gap-x-3 text-xs">
        <span className="font-semibold uppercase tracking-wide text-suave">Rendimiento IS vs OOS</span>
        <span className="text-suave">Pendiente OOS~IS: <span className="num text-texto">{ratio(pendiente, 2)}</span> {pendiente !== null && pendiente <= 0 && <span className="text-neg">(≤ 0: lo que gana en muestra no se repite fuera)</span>}</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Dispersión de ${validos.length} pares IS/OOS con pendiente ${ratio(pendiente, 2)}`}>
        <line x1={M.i} x2={W - M.d} y1={H - M.b} y2={H - M.b} stroke="rgb(var(--borde))" />
        <line x1={M.i} x2={M.i} y1={M.s} y2={H - M.b} stroke="rgb(var(--borde))" />
        {ry[0] < 0 && ry[1] > 0 && <line x1={M.i} x2={W - M.d} y1={y(0)} y2={y(0)} stroke="rgb(var(--borde))" strokeDasharray="3 3" />}
        {validos.map(([a, b], i) => (
          <circle key={i} cx={x(a)} cy={y(b)} r={2.5} fill="rgb(var(--acento))" fillOpacity={0.55}>
            <title>{`IS ${ratio(a, 3)} · OOS ${ratio(b, 3)}`}</title>
          </circle>
        ))}
        {recta && <line x1={x(rx[0])} y1={y(recta.intercepto + recta.pendiente * rx[0])} x2={x(rx[1])} y2={y(recta.intercepto + recta.pendiente * rx[1])} stroke="rgb(var(--aviso))" strokeWidth={2} />}
        <text x={W - M.d} y={H - 10} textAnchor="end" fontSize={11} fill="rgb(var(--suave))">IS →</text>
        <text x={M.i + 4} y={M.s + 8} fontSize={11} fill="rgb(var(--suave))">OOS ↑</text>
      </svg>
    </figure>
  )
}
