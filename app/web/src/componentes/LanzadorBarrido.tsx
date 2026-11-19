/**
 * Formulario para lanzar un barrido desde la página de una estrategia.
 * Responsabilidad: por cada parámetro del esquema permite «barrerlo» con una lista de valores (para los
 * numéricos, con ayuda de rango inicio/fin/paso); pide la fecha de fin de desarrollo (opcional), la
 * métrica objetivo y una etiqueta; muestra el nº de combinaciones (máx. 500) antes de lanzar y, al
 * crear el barrido (`POST /api/barridos`), navega a su detalle. Los parámetros que no se barren toman el
 * valor del formulario de experimento de la página.
 * Se relaciona con: paginas/EstrategiaDetalle.tsx (lo monta y le pasa la configuración base),
 * componentes/FormularioSchema.tsx (resolver tipos), lib/barridos.ts (rejilla), api/hooks.ts (useCrearBarrido).
 */
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCrearBarrido } from '../api/hooks'
import { METRICAS_OBJETIVO, type ConfiguracionExperimento, type EsquemaParametros } from '../api/tipos'
import { contarCombinaciones, MAX_COMBINACIONES, parsearValores, rangoAValores, type TipoValor } from '../lib/barridos'
import { CajaError } from './Estado'
import { resolver } from './FormularioSchema'

interface EstadoEje {
  activo: boolean
  texto: string
  ini: string
  fin: string
  paso: string
}

interface Props {
  esquema: EsquemaParametros
  /** Configuración del experimento base (con los parámetros actuales del formulario). */
  construirBase: () => ConfiguracionExperimento
  parametrosValidos: boolean
}

const EJE_VACIO: EstadoEje = { activo: false, texto: '', ini: '', fin: '', paso: '' }

export default function LanzadorBarrido({ esquema, construirBase, parametrosValidos }: Props) {
  const navegar = useNavigate()
  const crear = useCrearBarrido()
  const [ejes, setEjes] = useState<Record<string, EstadoEje>>({})
  const [finDesarrollo, setFinDesarrollo] = useState('')
  const [metrica, setMetrica] = useState<string>('sharpe')
  const [etiqueta, setEtiqueta] = useState('')
  const [errorRango, setErrorRango] = useState<Record<string, string>>({})

  const propiedades = Object.entries(esquema.properties ?? {}).map(([nombre, bruta]) => ({ nombre, propiedad: resolver(bruta) }))
  const cambiar = (nombre: string, parche: Partial<EstadoEje>) => setEjes((e) => ({ ...e, [nombre]: { ...EJE_VACIO, ...e[nombre], ...parche } }))

  // Rejilla y errores por eje
  const rejilla: Record<string, unknown[]> = {}
  const errores: Record<string, string> = {}
  for (const { nombre, propiedad } of propiedades) {
    const eje = ejes[nombre]
    if (!eje?.activo) continue
    const { valores, error } = parsearValores(eje.texto, (propiedad.type ?? 'string') as TipoValor)
    if (error) errores[nombre] = error
    else if (valores.length === 0) errores[nombre] = 'Introduce al menos un valor'
    else rejilla[nombre] = valores
  }
  const total = contarCombinaciones(rejilla)

  const base = construirBase()
  let errorFecha = ''
  if (finDesarrollo) {
    if (finDesarrollo <= base.fecha_inicio) errorFecha = 'Debe ser posterior al inicio'
    else if (finDesarrollo >= base.fecha_fin) errorFecha = 'Debe ser anterior al fin del periodo (si no, no queda nada reservado)'
  }

  const hayErrores = Object.keys(errores).length > 0 || !!errorFecha
  const puedeLanzar = parametrosValidos && !hayErrores && total >= 1 && total <= MAX_COMBINACIONES && !crear.isPending

  const enviar = (e: React.FormEvent) => {
    e.preventDefault()
    if (!puedeLanzar) return
    crear.mutate(
      { base, rejilla, fecha_fin_desarrollo: finDesarrollo || null, metrica_objetivo: metrica, etiqueta: etiqueta.trim() },
      { onSuccess: (r) => navegar(`/barridos/${r.id}`) },
    )
  }

  const generarRango = (nombre: string, tipo: string | undefined) => {
    const eje = ejes[nombre] ?? EJE_VACIO
    const { valores, error } = rangoAValores(Number(eje.ini), Number(eje.fin), Number(eje.paso))
    if (error) return setErrorRango((r) => ({ ...r, [nombre]: error }))
    if (tipo === 'integer' && valores.some((v) => !Number.isInteger(v))) return setErrorRango((r) => ({ ...r, [nombre]: 'El parámetro es entero: usa inicio y paso enteros' }))
    setErrorRango((r) => ({ ...r, [nombre]: '' }))
    cambiar(nombre, { texto: valores.join(', ') })
  }

  return (
    <form onSubmit={enviar} className="tarjeta space-y-4 p-4" aria-labelledby="h-barrido">
      <div>
        <h2 id="h-barrido">Lanzar barrido (rejilla de parámetros)</h2>
        <p className="mt-1 text-xs text-suave">
          Ejecuta todas las combinaciones y permite validarlas (DSR, PBO, permutación). Usa los ajustes del formulario de arriba; los parámetros sin marcar conservan su valor actual.
        </p>
      </div>

      <div className="space-y-2">
        {propiedades.length === 0 && <p className="text-suave">Esta estrategia no tiene parámetros que barrer.</p>}
        {propiedades.map(({ nombre, propiedad }) => {
          const eje = ejes[nombre] ?? EJE_VACIO
          const numerico = propiedad.type === 'integer' || propiedad.type === 'number'
          return (
            <fieldset key={nombre} className="rounded-md border border-borde p-2">
              <legend className="px-1">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    className="accent-[rgb(var(--acento))]"
                    checked={eje.activo}
                    onChange={(e) => cambiar(nombre, { activo: e.target.checked, texto: eje.texto || (e.target.checked && propiedad.default !== undefined ? String(propiedad.default) : '') })}
                  />
                  <span className="num font-medium">{nombre}</span>
                  <span className="text-xs text-suave">{propiedad.type}{propiedad.description ? ` · ${propiedad.description}` : ''}</span>
                </label>
              </legend>
              {eje.activo && (
                <div className="space-y-2">
                  <div>
                    <label htmlFor={`valores-${nombre}`} className="text-xs text-suave">Valores separados por comas</label>
                    <input id={`valores-${nombre}`} className="campo num" value={eje.texto} onChange={(e) => cambiar(nombre, { texto: e.target.value })} placeholder="3, 5, 8" aria-invalid={!!errores[nombre]} />
                    {errores[nombre] && <p className="mt-1 text-xs text-neg">{errores[nombre]}</p>}
                  </div>
                  {numerico && (
                    <div className="flex flex-wrap items-end gap-2">
                      {(['ini', 'fin', 'paso'] as const).map((campo) => (
                        <div key={campo} className="flex flex-col gap-1">
                          <label htmlFor={`${campo}-${nombre}`} className="text-xs text-suave">{campo === 'ini' ? 'Inicio' : campo === 'fin' ? 'Fin' : 'Paso'}</label>
                          <input id={`${campo}-${nombre}`} type="number" step="any" className="campo num w-24" value={eje[campo]} onChange={(e) => cambiar(nombre, { [campo]: e.target.value })} />
                        </div>
                      ))}
                      <button type="button" className="boton" onClick={() => generarRango(nombre, propiedad.type)} disabled={!eje.ini || !eje.fin || !eje.paso}>
                        Generar rango
                      </button>
                      {errorRango[nombre] && <p className="w-full text-xs text-neg">{errorRango[nombre]}</p>}
                    </div>
                  )}
                </div>
              )}
            </fieldset>
          )
        })}
      </div>

      <div className="grid gap-3 border-t border-borde pt-3 sm:grid-cols-2">
        <div className="flex flex-col gap-1">
          <label htmlFor="b-metrica" className="font-medium">Métrica objetivo</label>
          <select id="b-metrica" className="campo" value={metrica} onChange={(e) => setMetrica(e.target.value)}>
            {METRICAS_OBJETIVO.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          <p className="text-xs text-suave">Mayor es mejor. Guía el mapa de calor y la elección.</p>
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="b-etiqueta" className="font-medium">Etiqueta (opcional)</label>
          <input id="b-etiqueta" className="campo" value={etiqueta} onChange={(e) => setEtiqueta(e.target.value)} placeholder="p. ej. momentum 2×2" />
        </div>
        <div className="flex flex-col gap-1 sm:col-span-2">
          <label htmlFor="b-fin-dev" className="font-medium">Fin de desarrollo (opcional)</label>
          <input id="b-fin-dev" type="date" className="campo w-48" value={finDesarrollo} onChange={(e) => setFinDesarrollo(e.target.value)} aria-invalid={!!errorFecha} aria-describedby="ayuda-fin-dev" />
          <p id="ayuda-fin-dev" className="text-xs text-suave">Lo posterior a esta fecha queda reservado: ninguna validación lo toca.</p>
          {errorFecha && <p className="text-xs text-neg">{errorFecha}</p>}
        </div>
      </div>

      <p className="num" aria-live="polite">
        Combinaciones: <strong className={total > MAX_COMBINACIONES ? 'text-neg' : ''}>{total}</strong>
        <span className="text-suave"> (máx. {MAX_COMBINACIONES})</span>
        {total > MAX_COMBINACIONES && <span className="text-neg"> · demasiadas: reduce los valores</span>}
      </p>
      {crear.error && <CajaError error={crear.error} titulo="No se pudo lanzar el barrido" />}
      <button type="submit" className="boton boton-primario" disabled={!puedeLanzar}>
        {crear.isPending ? 'Lanzando…' : `Lanzar barrido de ${total} ${total === 1 ? 'combinación' : 'combinaciones'}`}
      </button>
    </form>
  )
}
