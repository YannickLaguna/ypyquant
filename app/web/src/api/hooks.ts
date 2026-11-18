/**
 * Hooks de datos (TanStack Query) sobre la API.
 * Responsabilidad: una función por recurso, con claves de caché consistentes, sondeo automático
 * mientras haya experimentos pendientes/en curso y mutaciones que invalidan la caché.
 * Se relaciona con: api/cliente.ts (fetch), api/tipos.ts y las páginas, que solo consumen estos hooks.
 */
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ErrorApi, pedir, querystring } from './cliente'
import type {
  Comparacion,
  ConfiguracionExperimento,
  CuentaPaper,
  DetalleExperimento,
  EstadoPaper,
  EstrategiaDetalle,
  HistorialPaper,
  LogBot,
  OrdenPaper,
  PosicionPaper,
  RespuestaEstrategias,
  ResumenExperimento,
  Universo,
} from './tipos'

const MS_SONDEO = 1500

export const useEstrategias = () =>
  useQuery({ queryKey: ['estrategias'], queryFn: () => pedir<RespuestaEstrategias>('/api/estrategias') })

export const useEstrategia = (nombre: string | undefined) =>
  useQuery({
    queryKey: ['estrategia', nombre],
    queryFn: () => pedir<EstrategiaDetalle>(`/api/estrategias/${encodeURIComponent(nombre!)}`),
    enabled: !!nombre,
  })

export const useUniversos = () =>
  useQuery({ queryKey: ['universos'], queryFn: () => pedir<{ universos: Universo[] }>('/api/universos') })

export function useExperimentos(filtros: { estrategia?: string; estado?: string; texto?: string } = {}) {
  return useQuery({
    queryKey: ['experimentos', filtros],
    queryFn: () => pedir<{ experimentos: ResumenExperimento[] }>(`/api/experimentos${querystring(filtros)}`),
    placeholderData: keepPreviousData,
    // Mientras haya algo sin terminar, se refresca solo
    refetchInterval: (consulta) =>
      consulta.state.data?.experimentos.some((e) => e.estado === 'pendiente' || e.estado === 'en_curso') ? MS_SONDEO : false,
  })
}

export const useExperimento = (id: string | undefined) =>
  useQuery({
    queryKey: ['experimento', id],
    queryFn: () => pedir<DetalleExperimento>(`/api/experimentos/${encodeURIComponent(id!)}`),
    enabled: !!id,
    refetchInterval: (consulta) =>
      consulta.state.data && (consulta.state.data.estado === 'pendiente' || consulta.state.data.estado === 'en_curso')
        ? MS_SONDEO
        : false,
  })

export const useComparacion = (ids: string[]) =>
  useQuery({
    queryKey: ['comparar', ids],
    queryFn: () => pedir<Comparacion>(`/api/comparar${querystring({ ids: ids.join(',') })}`),
    enabled: ids.length > 0,
  })

export function useLanzarExperimento() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (config: ConfiguracionExperimento) =>
      pedir<{ id: string; reutilizado: boolean }>('/api/experimentos', { metodo: 'POST', cuerpo: config }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['experimentos'] }),
  })
}

export function useLanzarBarrido() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (peticion: { config: ConfiguracionExperimento; rejilla: Record<string, unknown[]> }) =>
      pedir<{ ids: string[]; total: number }>('/api/experimentos/barrido', { metodo: 'POST', cuerpo: peticion }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['experimentos'] }),
  })
}

export function useBorrarExperimento() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => pedir<void>(`/api/experimentos/${encodeURIComponent(id)}`, { metodo: 'DELETE' }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['experimentos'] }),
  })
}

// ---- Paper trading -----------------------------------------------------------------------

/** Los 503 (sin claves) no se reintentan: el mensaje es definitivo hasta que se configuren. */
const sinReintentarEn503 = (intentos: number, error: unknown) =>
  !(error instanceof ErrorApi && [502, 503].includes(error.estado)) && intentos < 2

export const useEstadoPaper = () =>
  useQuery({ queryKey: ['paper', 'estado'], queryFn: () => pedir<EstadoPaper>('/api/paper/estado'), refetchInterval: 5000 })

export const useCuentaPaper = () =>
  useQuery({
    queryKey: ['paper', 'cuenta'],
    queryFn: () => pedir<CuentaPaper>('/api/paper/cuenta'),
    retry: sinReintentarEn503,
    refetchInterval: 30_000,
  })

export const usePosicionesPaper = () =>
  useQuery({
    queryKey: ['paper', 'posiciones'],
    queryFn: () => pedir<{ posiciones: PosicionPaper[] }>('/api/paper/posiciones'),
    retry: sinReintentarEn503,
    refetchInterval: 30_000,
  })

export const useOrdenesPaper = () =>
  useQuery({
    queryKey: ['paper', 'ordenes'],
    queryFn: () => pedir<{ ordenes: OrdenPaper[] }>('/api/paper/ordenes'),
    retry: sinReintentarEn503,
    refetchInterval: 30_000,
  })

export const useHistorialPaper = (periodo: string) =>
  useQuery({
    queryKey: ['paper', 'historial', periodo],
    queryFn: () => pedir<HistorialPaper>(`/api/paper/historial${querystring({ periodo })}`),
    retry: sinReintentarEn503,
  })

export function useLogBot(activo: boolean) {
  return useQuery({
    queryKey: ['paper', 'log'],
    queryFn: () => pedir<LogBot>('/api/paper/bot/log?desde=0'),
    refetchInterval: activo ? 2000 : false,
  })
}

export function useArrancarBot() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (peticion: { estrategia: string; parametros: Record<string, unknown>; tickers: string[]; intervalo_s: number }) =>
      pedir('/api/paper/bot/arrancar', { metodo: 'POST', cuerpo: peticion }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['paper'] }),
  })
}

export function usePararBot() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: () => pedir('/api/paper/bot/parar', { metodo: 'POST' }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['paper'] }),
  })
}
