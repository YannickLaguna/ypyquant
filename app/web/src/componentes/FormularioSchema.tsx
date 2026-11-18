/**
 * Formulario autogenerado desde el JSON Schema de los parámetros de una estrategia.
 * Responsabilidad: por cada propiedad del esquema (integer, number, boolean, string, enum) pinta el
 * control adecuado con etiqueta, descripción y límites (minimum/maximum); `valoresPorDefecto` y
 * `validarValores` permiten inicializar y comprobar antes de enviar.
 * Se relaciona con: paginas/EstrategiaDetalle.tsx y paginas/PaperTrading.tsx (lo usan), api/tipos.ts.
 */
import type { EsquemaParametros, EsquemaPropiedad } from '../api/tipos'

export type Valores = Record<string, unknown>

/** Resuelve `anyOf` (p. ej. `[{type: integer}, {type: null}]` de los opcionales) al primer tipo no nulo. */
function resolver(propiedad: EsquemaPropiedad): EsquemaPropiedad {
  if (propiedad.anyOf) {
    const util = propiedad.anyOf.find((p) => (p.type as string) !== 'null')
    return { ...util, ...propiedad, type: util?.type ?? propiedad.type }
  }
  return propiedad
}

export function valoresPorDefecto(esquema: EsquemaParametros): Valores {
  const valores: Valores = {}
  for (const [nombre, propiedad] of Object.entries(esquema.properties ?? {})) {
    if (propiedad.default !== undefined) valores[nombre] = propiedad.default
  }
  return valores
}

function limites(p: EsquemaPropiedad): { min?: number; max?: number } {
  return { min: p.minimum ?? p.exclusiveMinimum, max: p.maximum ?? p.exclusiveMaximum }
}

/** Errores por campo (vacío si todo es válido). */
export function validarValores(esquema: EsquemaParametros, valores: Valores): Record<string, string> {
  const errores: Record<string, string> = {}
  for (const [nombre, bruta] of Object.entries(esquema.properties ?? {})) {
    const propiedad = resolver(bruta)
    const valor = valores[nombre]
    if (propiedad.type === 'integer' || propiedad.type === 'number') {
      if (typeof valor !== 'number' || Number.isNaN(valor)) {
        errores[nombre] = 'Introduce un número'
        continue
      }
      if (propiedad.type === 'integer' && !Number.isInteger(valor)) errores[nombre] = 'Debe ser un entero'
      else if (propiedad.minimum !== undefined && valor < propiedad.minimum) errores[nombre] = `Mínimo ${propiedad.minimum}`
      else if (propiedad.maximum !== undefined && valor > propiedad.maximum) errores[nombre] = `Máximo ${propiedad.maximum}`
      else if (propiedad.exclusiveMinimum !== undefined && valor <= propiedad.exclusiveMinimum)
        errores[nombre] = `Debe ser mayor que ${propiedad.exclusiveMinimum}`
    } else if (propiedad.type === 'string' && (valor === undefined || valor === '') && !propiedad.enum) {
      errores[nombre] = 'Obligatorio'
    }
  }
  return errores
}

interface Props {
  esquema: EsquemaParametros
  valores: Valores
  onChange: (valores: Valores) => void
  idBase?: string
}

export default function FormularioSchema({ esquema, valores, onChange, idBase = 'param' }: Props) {
  const propiedades = Object.entries(esquema.properties ?? {})
  const errores = validarValores(esquema, valores)

  if (propiedades.length === 0) {
    return <p className="text-suave">Esta estrategia no tiene parámetros.</p>
  }

  const cambiar = (nombre: string, valor: unknown) => onChange({ ...valores, [nombre]: valor })

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {propiedades.map(([nombre, bruta]) => {
        const p = resolver(bruta)
        const id = `${idBase}-${nombre}`
        const idAyuda = `${id}-ayuda`
        const error = errores[nombre]
        const etiqueta = p.title ?? nombre
        const { min, max } = limites(p)
        const valor = valores[nombre]
        const ayuda = [p.description, min !== undefined || max !== undefined ? `Rango: ${min ?? '−∞'} a ${max ?? '∞'}` : '']
          .filter(Boolean)
          .join(' · ')

        let control
        if (p.enum) {
          control = (
            <select id={id} className="campo" value={String(valor ?? '')} aria-describedby={idAyuda}
              onChange={(e) => cambiar(nombre, p.type === 'integer' || p.type === 'number' ? Number(e.target.value) : e.target.value)}>
              {p.enum.map((o) => (
                <option key={String(o)} value={String(o)}>{String(o)}</option>
              ))}
            </select>
          )
        } else if (p.type === 'boolean') {
          control = (
            <input id={id} type="checkbox" className="h-4 w-4 accent-[rgb(var(--acento))]" checked={Boolean(valor)}
              aria-describedby={idAyuda} onChange={(e) => cambiar(nombre, e.target.checked)} />
          )
        } else if (p.type === 'integer' || p.type === 'number') {
          control = (
            <input id={id} type="number" className="campo num" value={typeof valor === 'number' && !Number.isNaN(valor) ? valor : ''}
              min={p.minimum} max={p.maximum} step={p.type === 'integer' ? 1 : 'any'} aria-invalid={!!error}
              aria-describedby={idAyuda}
              onChange={(e) => cambiar(nombre, e.target.value === '' ? NaN : Number(e.target.value))} />
          )
        } else {
          control = (
            <input id={id} type="text" className="campo" value={String(valor ?? '')} aria-invalid={!!error}
              aria-describedby={idAyuda} onChange={(e) => cambiar(nombre, e.target.value)} />
          )
        }

        return (
          <div key={nombre} className="flex flex-col gap-1">
            <label htmlFor={id} className="font-medium">{etiqueta}</label>
            {control}
            <p id={idAyuda} className={`text-xs ${error ? 'text-neg' : 'text-suave'}`}>{error ?? ayuda}</p>
          </div>
        )
      })}
    </div>
  )
}
