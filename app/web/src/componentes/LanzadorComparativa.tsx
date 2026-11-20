/**
 * Lanzador de comparativas: elige varias estrategias, ajusta sus parámetros PROPIOS y las lanza sobre un marco.
 * Responsabilidad: por cada estrategia marcada muestra su formulario (`FormularioSchema`) sin los parámetros
 * que fija el marco (esos los pone el marco); valida en cliente, llama a `POST /api/marcos/{id}/comparativa` y
 * muestra el informe: qué comunes se aplicaron a cada estrategia, cuáles no y los avisos (frecuencia propia,
 * parámetro propio pisado), con el enlace para abrir Comparar con todos los experimentos de golpe.
 * Se relaciona con: lib/marcos.ts (esquemaSinComunes), componentes/FormularioSchema.tsx, api/hooks.ts
 * (useEstrategias, useLanzarComparativa) y paginas/Comparar.tsx (destino).
 */
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useEstrategias, useLanzarComparativa } from '../api/hooks'
import type { Comparativa, Estrategia, Marco } from '../api/tipos'
import { esquemaSinComunes } from '../lib/marcos'
import { CajaError, Cargando } from './Estado'
import FormularioSchema, { validarValores, valoresPorDefecto, type Valores } from './FormularioSchema'

/** Enlace a Comparar con los experimentos de la comparativa (y su id para mostrar los avisos del marco). */
export const enlaceComparar = (c: Comparativa) => `/comparar?ids=${c.experimentos.map((e) => e.id_experimento).join(',')}&comparativa=${c.id}`

export default function LanzadorComparativa({ marco }: { marco: Marco }) {
  const { data, isPending, error } = useEstrategias()
  const lanzar = useLanzarComparativa(marco.id)
  const [elegidas, setElegidas] = useState<Record<string, Valores>>({})
  const [resultado, setResultado] = useState<Comparativa | null>(null)

  if (isPending) return <Cargando />
  if (error) return <CajaError error={error} />

  const esquemaPropio = (e: Estrategia) => esquemaSinComunes(e.parametros_schema, marco.parametros_comunes)
  const alternar = (e: Estrategia) => {
    const siguiente = { ...elegidas }
    if (e.nombre in siguiente) delete siguiente[e.nombre]
    else siguiente[e.nombre] = valoresPorDefecto(esquemaPropio(e))
    setElegidas(siguiente)
  }
  const invalido = Object.entries(elegidas).some(([nombre, valores]) => {
    const estrategia = data.estrategias.find((e) => e.nombre === nombre)
    return !estrategia || Object.keys(validarValores(esquemaPropio(estrategia), valores)).length > 0
  })
  const lista = Object.keys(elegidas)

  const enviar = (e: React.FormEvent) => {
    e.preventDefault()
    if (lista.length === 0 || invalido) return
    lanzar.mutate({ estrategias: lista.map((nombre) => ({ estrategia: nombre, parametros: elegidas[nombre] })) }, { onSuccess: setResultado })
  }

  return (
    <form onSubmit={enviar} className="space-y-4" aria-labelledby="h-comparativa">
      <h2 id="h-comparativa">Lanzar comparativa</h2>
      <p className="text-suave">
        Elige las estrategias. Cada una usa el entorno del marco; aquí solo eliges sus parámetros propios. Los experimentos ya
        calculados con el mismo entorno se reutilizan.
      </p>
      <ul className="tarjeta grid gap-x-4 p-2 sm:grid-cols-2 xl:grid-cols-3">
        {data.estrategias.map((e) => (
          <li key={e.nombre}>
            <label className="flex cursor-pointer items-start gap-2 rounded px-2 py-1 hover:bg-elevado">
              <input type="checkbox" className="mt-1 accent-[rgb(var(--acento))]" checked={e.nombre in elegidas} onChange={() => alternar(e)} />
              <span className="min-w-0"><span className="num block truncate">{e.nombre}</span><span className="block truncate text-xs text-suave">{e.descripcion}</span></span>
            </label>
          </li>
        ))}
      </ul>

      {lista.map((nombre) => {
        const estrategia = data.estrategias.find((e) => e.nombre === nombre)!
        const esquema = esquemaPropio(estrategia)
        const fijados = Object.keys(estrategia.parametros_schema.properties ?? {}).filter((p) => p in marco.parametros_comunes)
        return (
          <fieldset key={nombre} className="tarjeta space-y-3 p-4">
            <legend className="num px-1 font-medium">{nombre}</legend>
            {fijados.length > 0 && <p className="text-xs text-suave">Fijados por el marco: {fijados.map((p) => `${p}=${JSON.stringify(marco.parametros_comunes[p])}`).join(', ')}.</p>}
            <FormularioSchema esquema={esquema} valores={elegidas[nombre]} onChange={(v) => setElegidas({ ...elegidas, [nombre]: v })} idBase={`cmp-${nombre}`} />
          </fieldset>
        )
      })}

      {lanzar.error && <CajaError error={lanzar.error} titulo="No se pudo lanzar la comparativa" />}
      <button type="submit" className="boton boton-primario" disabled={lista.length === 0 || invalido || lanzar.isPending}>
        {lanzar.isPending ? 'Lanzando…' : `Lanzar comparativa (${lista.length})`}
      </button>

      {resultado && <ResultadoComparativa comparativa={resultado} />}
    </form>
  )
}

/** Informe de una comparativa lanzada: por estrategia, qué comunes se aplicaron y qué avisos hay. */
export function ResultadoComparativa({ comparativa }: { comparativa: Comparativa }) {
  const avisos = comparativa.experimentos.flatMap((e) => e.informe.avisos)
  return (
    <section aria-labelledby="h-resultado" className="tarjeta space-y-3 p-4">
      <h3 id="h-resultado">Comparativa lanzada</h3>
      {avisos.length > 0 && (
        <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
          {avisos.map((a) => <p key={a}>{a}</p>)}
        </div>
      )}
      <ul className="space-y-1">
        {comparativa.experimentos.map((e) => (
          <li key={e.id_experimento}>
            <Link to={`/experimentos/${e.id_experimento}`} className="num text-acento hover:underline">{e.id_experimento}</Link>
            {e.reutilizado && <span className="ml-2 text-xs text-suave">(reutilizado)</span>}
            <span className="ml-2 text-xs text-suave">
              comunes aplicados: {Object.keys(e.informe.aplicados).join(', ') || 'ninguno'}
              {e.informe.no_aplicados.length > 0 && ` · no aplicados: ${e.informe.no_aplicados.join(', ')}`}
            </span>
          </li>
        ))}
      </ul>
      <Link to={enlaceComparar(comparativa)} className="boton boton-primario inline-block">Abrir en Comparar</Link>
    </section>
  )
}
