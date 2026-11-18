/**
 * Tabla genérica ordenable (clic o teclado en la cabecera, `aria-sort`) y densa, estilo herramienta quant.
 * Responsabilidad: ordenar por una columna con valor numérico/texto, mensaje de vacío y scroll horizontal.
 * Se relaciona con: paginas/Experimentos.tsx, Comparar.tsx, ExperimentoDetalle.tsx, PaperTrading.tsx.
 */
import { useMemo, useState, type ReactNode } from 'react'

export interface Columna<T> {
  clave: string
  titulo: string
  /** Valor para ordenar (si falta, la columna no es ordenable). */
  valor?: (fila: T) => number | string | null | undefined
  render: (fila: T) => ReactNode
  alinear?: 'izq' | 'der'
}

interface Props<T> {
  columnas: Columna<T>[]
  filas: T[]
  clave: (fila: T) => string
  ordenInicial?: { clave: string; descendente: boolean }
  vacio?: string
  titulo: string
}

export default function TablaOrdenable<T>({ columnas, filas, clave, ordenInicial, vacio = 'Sin datos', titulo }: Props<T>) {
  const [orden, setOrden] = useState(ordenInicial ?? null)

  const ordenadas = useMemo(() => {
    const columna = columnas.find((c) => c.clave === orden?.clave)
    if (!columna?.valor || !orden) return filas
    const obtener = columna.valor
    return [...filas].sort((a, b) => {
      const va = obtener(a)
      const vb = obtener(b)
      if (va == null && vb == null) return 0
      if (va == null) return 1 // los nulos siempre al final
      if (vb == null) return -1
      const cmp = typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb), 'es')
      return orden.descendente ? -cmp : cmp
    })
  }, [filas, columnas, orden])

  const alternar = (c: Columna<T>) => {
    if (!c.valor) return
    setOrden((o) => (o?.clave === c.clave ? { clave: c.clave, descendente: !o.descendente } : { clave: c.clave, descendente: true }))
  }

  return (
    <div className="tarjeta overflow-x-auto">
      <table className="w-full border-collapse text-sm" aria-label={titulo}>
        <thead className="border-b border-borde">
          <tr>
            {columnas.map((c) => {
              const activa = orden?.clave === c.clave
              return (
                <th
                  key={c.clave}
                  scope="col"
                  className={`th ${c.alinear === 'der' ? 'text-right' : ''}`}
                  aria-sort={activa ? (orden!.descendente ? 'descending' : 'ascending') : c.valor ? 'none' : undefined}
                >
                  {c.valor ? (
                    <button type="button" onClick={() => alternar(c)} className="inline-flex items-center gap-1 uppercase hover:text-texto">
                      {c.titulo}
                      <span aria-hidden className="w-3 text-acento">{activa ? (orden!.descendente ? '▼' : '▲') : ''}</span>
                    </button>
                  ) : (
                    c.titulo
                  )}
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {ordenadas.length === 0 && (
            <tr>
              <td colSpan={columnas.length} className="p-6 text-center text-suave">{vacio}</td>
            </tr>
          )}
          {ordenadas.map((fila) => (
            <tr key={clave(fila)} className="border-b border-borde/60 last:border-0 hover:bg-elevado/60">
              {columnas.map((c) => (
                <td key={c.clave} className={`td ${c.alinear === 'der' ? 'text-right num' : ''}`}>{c.render(fila)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
