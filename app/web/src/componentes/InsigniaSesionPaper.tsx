/**
 * Insignia del estado de control de una sesión de paper interno (activa, pausada o detenida).
 * Responsabilidad: un único sitio con los colores y las etiquetas de los tres estados.
 * Se relaciona con: paginas/PaperInterno.tsx (lista) y paginas/PaperInternoDetalle.tsx (cabecera);
 * los estados vienen de api/paper_interno.ts.
 */
import type { EstadoSesionPaper } from '../api/paper_interno'

const ESTILOS: Record<EstadoSesionPaper, string> = {
  activa: 'border-pos/50 text-pos',
  pausada: 'border-aviso/60 text-aviso',
  detenida: 'border-borde text-suave',
}

export default function InsigniaSesionPaper({ estado }: { estado: EstadoSesionPaper }) {
  return <span className={`inline-block rounded border px-1.5 py-0.5 text-xs font-medium ${ESTILOS[estado]}`}>{estado}</span>
}
