/**
 * Tipos y hook del estado de las membresías del S&P 500 (`GET /api/universos/membresias/estado`).
 * Responsabilidad: describir la respuesta (última comprobación, vigentes, cambios aplicados, pendientes de revisar,
 * antigüedad y avisos) y leerla con react-query, releyéndola cada 5 minutos.
 * Se relaciona con: api/cliente.ts (fetch), componentes/TarjetaMembresias.tsx y paginas/PaperInterno.tsx;
 * el backend es app/api/rutas_universos.py (ypyquant.data.membresias_sp500).
 */
import { useQuery } from '@tanstack/react-query'
import { pedir } from './cliente'

export interface CambioMembresia {
  tipo: 'alta' | 'baja'
  simbolo: string
  fecha: string
  fecha_aproximada?: boolean
  fuente?: string
  verificado_por?: string | null
  comprobado?: string
  motivo_pendiente?: string
}

export interface EstadoMembresias {
  activo: boolean
  ultima_comprobacion: string | null
  resultado: 'al_dia' | 'actualizado' | 'pendiente' | 'error' | null
  error: string | null
  n_vigentes: number
  ultima_comprobacion_correcta: string | null
  antiguedad_dias: number | null
  cambios_aplicados_total: number
  cambios_aplicados_recientes: CambioMembresia[]
  pendientes_de_revisar: CambioMembresia[]
  avisos: string[]
  avisos_calidad: string[]
}

export const useEstadoMembresias = () =>
  useQuery({
    queryKey: ['universos', 'membresias', 'estado'],
    queryFn: () => pedir<EstadoMembresias>('/api/universos/membresias/estado'),
    refetchInterval: 5 * 60_000,
  })
