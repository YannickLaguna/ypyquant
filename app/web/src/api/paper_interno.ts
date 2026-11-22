/**
 * Tipos y hooks del paper interno (`/api/paper/interno`): sesiones forward de una receta congelada.
 * Responsabilidad: tipos de las respuestas de la API y un hook por recurso con claves de caché propias
 * (`['paper-interno', ...]`): lista de sesiones, detalle, equity (una o varias a la vez), operaciones, backtest
 * espejo, estado del avance con sondeo mientras está en curso (al terminar refresca lista y detalle) y las
 * mutaciones (crear sesión, avanzar, pausar/reanudar/detener). Vive aparte de `hooks.ts`/`tipos.ts` para no
 * mezclar el paper interno con el resto de recursos.
 * Se relaciona con: api/cliente.ts (fetch), paginas/PaperInterno.tsx, PaperInternoDetalle.tsx,
 * componentes/CrearSesionPaper.tsx y lib/paperInterno.ts (utilidades puras de las series).
 */
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'
import { pedir } from './cliente'

const RAIZ = '/api/paper/interno'
const MS_SONDEO = 1000

export type ModoEjecucionPaper = 'cierre' | 'apertura_siguiente'
export type EstadoSesionPaper = 'activa' | 'pausada' | 'detenida'
export type AccionSesionPaper = 'pausar' | 'reanudar' | 'detener'

/** Una fila de la lista de sesiones (`GET /sesiones`). */
export interface ResumenSesionPaper {
  id: string
  nombre: string
  estrategia: string
  modo_ejecucion: ModoEjecucionPaper
  estado: EstadoSesionPaper
  fecha_alta: string
  capital_inicial: number
  equity: number
  dias: number
  ultima_fecha: string | null
  retorno_total: number | null
  sharpe: number | null
  max_drawdown: number | null
  volatilidad: number | null
  costes: number
  dias_pendientes: number
  al_dia: boolean
  id_experimento: string | null
  id_configuracion: string | null
}

export interface PosicionSesionPaper {
  ticker: string
  cantidad: number
  precio_medio: number
  ultimo_cierre: number | null
  valor: number
  peso: number
}

export interface EventoSesionPaper {
  fecha: string
  tipo: string
  mensaje: string
}

/** Detalle de una sesión (`GET /sesiones/{id}`): el resumen más receta, métricas forward, posiciones y avisos. */
export interface DetalleSesionPaper extends ResumenSesionPaper {
  sesion: { receta: { estrategia: string; parametros: Record<string, unknown>; universo: string[] | string; benchmark: string; comision: number; slippage: number; frecuencia_rebalanceo: string } }
  metricas: Record<string, number | null> | null
  costes_detalle: { comisiones: number; slippage: number; total: number }
  efectivo: number
  posiciones: PosicionSesionPaper[]
  pesos_objetivo: { ticker: string; peso: number }[]
  pesos_pendientes: Record<string, number> | null
  fecha_decision_pendiente: string | null
  avisos: EventoSesionPaper[]
  n_operaciones: number
  ejecucion_optimista: boolean
}

export interface PuntoEquitySesion {
  fecha: string
  equity: number
  efectivo: number
  n_posiciones: number
  benchmark: number | null
  retorno: number | null
  drawdown: number | null
}

export interface OperacionSesionPaper {
  fecha: string
  ticker: string
  lado: 'compra' | 'venta'
  cantidad: number
  precio: number
  comision: number
  pnl: number | null
  slippage: number
}

export interface ResultadoAvanceSesion {
  id_sesion: string
  dias_procesados: number
  desde: string | null
  hasta: string | null
  avisos: string[]
  omitida: string | null
  error: string | null
}

export interface EstadoAvancePaper {
  en_curso: boolean
  inicio: string | null
  fin: string | null
  resultados: ResultadoAvanceSesion[]
  error: string | null
}

export interface EspejoSesionPaper {
  estado: 'sin_datos' | 'pendiente' | 'en_curso' | 'hecho' | 'error' | null
  id_experimento: string | null
  real: PuntoEquitySesion[]
  espejo: { fecha: string; equity: number }[]
}

export interface PeticionSesionPaper {
  id_experimento?: string
  id_configuracion?: string
  capital_inicial?: number
  modo_ejecucion: ModoEjecucionPaper
  comision?: number
  slippage?: number
  nombre?: string
  fecha_alta?: string
}

export const useSesionesPaper = () =>
  useQuery({ queryKey: ['paper-interno', 'sesiones'], queryFn: () => pedir<{ sesiones: ResumenSesionPaper[] }>(`${RAIZ}/sesiones`) })

export const useDetalleSesionPaper = (id: string | undefined) =>
  useQuery({
    queryKey: ['paper-interno', 'sesion', id],
    queryFn: () => pedir<DetalleSesionPaper>(`${RAIZ}/sesiones/${encodeURIComponent(id!)}`),
    enabled: !!id,
  })

export const useEquitySesionPaper = (id: string | undefined) =>
  useQuery({
    queryKey: ['paper-interno', 'equity', id],
    queryFn: () => pedir<{ equity: PuntoEquitySesion[] }>(`${RAIZ}/sesiones/${encodeURIComponent(id!)}/equity`),
    enabled: !!id,
  })

/** La equity de varias sesiones a la vez (selección múltiple de la lista), en el mismo orden que `ids`. */
export function useEquitiesSesionesPaper(ids: string[]) {
  return useQueries({
    queries: ids.map((id) => ({
      queryKey: ['paper-interno', 'equity', id],
      queryFn: () => pedir<{ equity: PuntoEquitySesion[] }>(`${RAIZ}/sesiones/${encodeURIComponent(id)}/equity`),
    })),
  })
}

export const useOperacionesSesionPaper = (id: string | undefined) =>
  useQuery({
    queryKey: ['paper-interno', 'operaciones', id],
    queryFn: () => pedir<{ operaciones: OperacionSesionPaper[] }>(`${RAIZ}/sesiones/${encodeURIComponent(id!)}/operaciones`),
    enabled: !!id,
  })

/** Curva real frente al backtest espejo; se sondea mientras el espejo se calcula en segundo plano. */
export const useEspejoSesionPaper = (id: string | undefined, activo = true) =>
  useQuery({
    queryKey: ['paper-interno', 'espejo', id],
    queryFn: () => pedir<EspejoSesionPaper>(`${RAIZ}/sesiones/${encodeURIComponent(id!)}/espejo`),
    enabled: !!id && activo,
    refetchInterval: (consulta) => (consulta.state.data?.estado === 'pendiente' || consulta.state.data?.estado === 'en_curso' ? MS_SONDEO : false),
  })

/**
 * Estado del avance: se sondea cada segundo mientras `en_curso`; cuando termina refresca la lista y el detalle de
 * las sesiones (las curvas y los «días pendientes» cambian).
 */
export function useEstadoAvancePaper() {
  const cliente = useQueryClient()
  const consulta = useQuery({
    queryKey: ['paper-interno', 'avance'],
    queryFn: () => pedir<EstadoAvancePaper>(`${RAIZ}/avanzar/estado`),
    refetchInterval: (c) => (c.state.data?.en_curso ? MS_SONDEO : false),
  })
  const estabaEnCurso = useRef(false)
  const enCurso = consulta.data?.en_curso ?? false
  useEffect(() => {
    if (estabaEnCurso.current && !enCurso) cliente.invalidateQueries({ queryKey: ['paper-interno'], predicate: (q) => q.queryKey[1] !== 'avance' })
    estabaEnCurso.current = enCurso
  }, [enCurso, cliente])
  return consulta
}

export function useAvanzarPaper() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: () => pedir<EstadoAvancePaper>(`${RAIZ}/avanzar`, { metodo: 'POST' }),
    onSuccess: (estado) => cliente.setQueryData(['paper-interno', 'avance'], estado),
  })
}

export function useCrearSesionPaper() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (peticion: PeticionSesionPaper) => pedir<DetalleSesionPaper>(`${RAIZ}/sesiones`, { metodo: 'POST', cuerpo: peticion }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['paper-interno', 'sesiones'] }),
  })
}

export function useControlSesionPaper(id: string) {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (accion: AccionSesionPaper) => pedir<DetalleSesionPaper>(`${RAIZ}/sesiones/${encodeURIComponent(id)}/${accion}`, { metodo: 'POST' }),
    onSuccess: (detalle) => {
      cliente.setQueryData(['paper-interno', 'sesion', id], detalle)
      cliente.invalidateQueries({ queryKey: ['paper-interno', 'sesiones'] })
    },
  })
}
