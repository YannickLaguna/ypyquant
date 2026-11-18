/**
 * Estados de interfaz reutilizables: carga, vacío y error, más la insignia de estado de experimento.
 * Responsabilidad: que cada página trate de forma uniforme «cargando / no hay nada / falló» con
 * roles ARIA (status/alert) y mensajes accionables.
 * Se relaciona con: todas las páginas (paginas/*) y api/cliente.ts (ErrorApi).
 */
import type { ReactNode } from 'react'
import { ErrorApi } from '../api/cliente'
import type { EstadoExperimento } from '../api/tipos'

export function Cargando({ texto = 'Cargando…' }: { texto?: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-2 p-6 text-suave">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-borde border-t-acento" aria-hidden />
      {texto}
    </div>
  )
}

export function Vacio({ titulo, children }: { titulo: string; children?: ReactNode }) {
  return (
    <div className="tarjeta flex flex-col items-center gap-2 p-8 text-center">
      <p className="font-medium">{titulo}</p>
      {children && <div className="max-w-md text-suave">{children}</div>}
    </div>
  )
}

export function CajaError({ error, titulo = 'No se pudo cargar' }: { error: unknown; titulo?: string }) {
  const mensaje = error instanceof Error ? error.message : String(error)
  // 503 = servicio no configurado (p. ej. sin claves de Alpaca): es un aviso, no un fallo
  const esAviso = error instanceof ErrorApi && error.estado === 503
  return (
    <div
      role="alert"
      className={`rounded-lg border p-4 ${esAviso ? 'border-aviso/50 bg-aviso/10' : 'border-neg/50 bg-neg/10'}`}
    >
      <p className={`font-medium ${esAviso ? 'text-aviso' : 'text-neg'}`}>{esAviso ? 'Servicio no configurado' : titulo}</p>
      <p className="mt-1 break-words text-texto">{mensaje}</p>
    </div>
  )
}

const ESTILOS: Record<EstadoExperimento, string> = {
  hecho: 'border-pos/50 text-pos',
  en_curso: 'border-acento/60 text-acento',
  pendiente: 'border-borde text-suave',
  error: 'border-neg/60 text-neg',
}
const ETIQUETAS: Record<EstadoExperimento, string> = {
  hecho: 'hecho',
  en_curso: 'en curso',
  pendiente: 'pendiente',
  error: 'error',
}

export function Insignia({ estado }: { estado: EstadoExperimento }) {
  return (
    <span className={`inline-block rounded border px-1.5 py-0.5 text-xs font-medium ${ESTILOS[estado]}`}>
      {ETIQUETAS[estado]}
    </span>
  )
}
