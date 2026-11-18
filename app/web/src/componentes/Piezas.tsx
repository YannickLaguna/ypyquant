/**
 * Piezas pequeñas: tarjeta KPI y bloque de código con resaltado (solo lectura).
 * Responsabilidad: `Kpi` (etiqueta + valor monoespaciado coloreado) y `Codigo` (highlight.js, solo Python).
 * Se relaciona con: paginas/EstrategiaDetalle.tsx, ExperimentoDetalle.tsx y PaperTrading.tsx.
 */
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import { useMemo } from 'react'

hljs.registerLanguage('python', python)

export function Kpi({ etiqueta, valor, clase = '', ayuda }: { etiqueta: string; valor: string; clase?: string; ayuda?: string }) {
  return (
    <div className="tarjeta px-3 py-2" title={ayuda}>
      <dt className="text-xs uppercase tracking-wide text-suave">{etiqueta}</dt>
      <dd className={`num mt-0.5 text-lg font-semibold ${clase}`}>{valor}</dd>
    </div>
  )
}

export function Codigo({ fuente }: { fuente: string }) {
  const html = useMemo(() => hljs.highlight(fuente, { language: 'python' }).value, [fuente])
  return (
    <pre className="tarjeta max-h-[28rem] overflow-auto p-3 text-xs leading-relaxed" tabIndex={0} aria-label="Código fuente (solo lectura)">
      <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
    </pre>
  )
}
