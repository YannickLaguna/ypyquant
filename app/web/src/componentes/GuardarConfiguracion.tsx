/**
 * Panel «Guardar como configuración»: pide nombre y notas y crea la configuración desde un experimento o desde
 * una combinación de un barrido (criterio `mejor` | `meseta` | `elegida`).
 * Responsabilidad: validar el nombre, mandar la fuente correcta a `POST /api/configuraciones`, mostrar el error de
 * la API (p. ej. 409 id duplicado) y navegar a la configuración creada. Desde un barrido avisa del riesgo de
 * sobreajuste (winner's curse). No pide marco: una configuración guardada así no pertenece a ningún marco.
 * Se relaciona con: paginas/ExperimentoDetalle.tsx y paginas/BarridoDetalle.tsx (lo abren), api/hooks.ts
 * (useCrearConfiguracion) y paginas/ConfiguracionDetalle.tsx (destino).
 */
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCrearConfiguracion } from '../api/hooks'
import type { PeticionConfiguracion } from '../api/tipos'
import { CajaError } from './Estado'

export type FuenteConfiguracion =
  | { tipo: 'experimento'; id: string }
  | { tipo: 'barrido'; id: string; criterio: 'mejor' | 'meseta' | 'elegida'; idExperimento?: string }

interface Props {
  fuente: FuenteConfiguracion
  nombreSugerido: string
  alCerrar: () => void
}

export default function GuardarConfiguracion({ fuente, nombreSugerido, alCerrar }: Props) {
  const navegar = useNavigate()
  const crear = useCrearConfiguracion()
  const [nombre, setNombre] = useState(nombreSugerido)
  const [notas, setNotas] = useState('')
  const [errorNombre, setErrorNombre] = useState<string | null>(null)

  const guardar = (e: React.FormEvent) => {
    e.preventDefault()
    if (!nombre.trim()) {
      setErrorNombre('El nombre es obligatorio')
      return
    }
    setErrorNombre(null)
    const peticion: PeticionConfiguracion =
      fuente.tipo === 'experimento'
        ? { nombre: nombre.trim(), notas, desde_experimento: fuente.id }
        : { nombre: nombre.trim(), notas, desde_barrido: fuente.id, criterio: fuente.criterio, id_experimento: fuente.idExperimento }
    crear.mutate(peticion, { onSuccess: (c) => navegar(`/configuraciones/${c.id}`) })
  }

  return (
    <form onSubmit={guardar} className="tarjeta space-y-3 p-4" aria-labelledby="h-guardar-config" role="dialog" aria-label="Guardar como configuración">
      <h2 id="h-guardar-config">Guardar como configuración</h2>
      <p className="text-suave">
        Congela la receta (estrategia, parámetros, universo, frecuencia y costes) sin fechas; se puede relanzar y versionar.
        Se guardan las métricas actuales para medir luego cuánto se degradan.
      </p>
      {fuente.tipo === 'barrido' && (
        <p role="note" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
          Viene de un barrido: es el máximo (o la meseta) de varias combinaciones y puede estar sobreajustada. Revisa DSR/PBO, la meseta
          y el periodo reservado antes de fiarte de ella.
        </p>
      )}
      <div className="flex flex-col gap-1">
        <label htmlFor="nombre-config" className="font-medium">Nombre</label>
        <input id="nombre-config" className="campo" value={nombre} onChange={(e) => setNombre(e.target.value)} aria-invalid={!!errorNombre} aria-describedby={errorNombre ? 'error-nombre-config' : undefined} />
        {errorNombre && <p id="error-nombre-config" className="text-xs text-neg">{errorNombre}</p>}
      </div>
      <div className="flex flex-col gap-1">
        <label htmlFor="notas-config" className="font-medium">Notas</label>
        <textarea id="notas-config" className="campo" rows={2} value={notas} onChange={(e) => setNotas(e.target.value)} />
      </div>
      {crear.error && <CajaError error={crear.error} titulo="No se pudo guardar" />}
      <div className="flex gap-2">
        <button type="submit" className="boton boton-primario" disabled={crear.isPending}>{crear.isPending ? 'Guardando…' : 'Guardar configuración'}</button>
        <button type="button" className="boton" onClick={alCerrar}>Cancelar</button>
      </div>
    </form>
  )
}
