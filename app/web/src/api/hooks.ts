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
  Comparativa,
  Configuracion,
  ConfiguracionBarrido,
  ConfiguracionExperimento,
  CuentaPaper,
  DetalleConfiguracion,
  DetalleBarrido,
  DetalleExperimento,
  EstadoPaper,
  EstrategiaDetalle,
  FillPaper,
  FrecuenciaRebalanceo,
  HistorialPaper,
  LogBot,
  Marco,
  ModoPermutacion,
  OrdenPaper,
  PeticionConfiguracion,
  PeticionMarco,
  PeticionPermutacion,
  PosicionPaper,
  RespuestaPermutacion,
  ResultadoBorrado,
  ResultadoReservado,
  ResultadoValidacion,
  ResultadoWalkForward,
  ResumenBarrido,
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

export function useBorrarExperimento() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => pedir<void>(`/api/experimentos/${encodeURIComponent(id)}`, { metodo: 'DELETE' }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['experimentos'] }),
  })
}

/** Borrado múltiple (selección de la tabla de experimentos); los que siguen en marcha no se tocan. */
export function useBorrarExperimentos() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (ids: string[]) => pedir<ResultadoBorrado>('/api/experimentos/borrar', { metodo: 'POST', cuerpo: { ids } }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['experimentos'] }),
  })
}

// ---- Barridos y validación ---------------------------------------------------------------

/** ¿Hay algo en marcha en el barrido? (experimentos sin terminar o tareas de permutación activas). */
export function barridoActivo(b: DetalleBarrido | undefined): boolean {
  if (!b) return false
  return (
    b.estado === 'pendiente' ||
    b.estado === 'en_curso' ||
    Object.values(b.tareas ?? {}).some((t) => t.estado === 'pendiente' || t.estado === 'en_curso')
  )
}

export const useBarridos = () =>
  useQuery({
    queryKey: ['barridos'],
    queryFn: () => pedir<{ barridos: ResumenBarrido[] }>('/api/barridos'),
    refetchInterval: (consulta) =>
      consulta.state.data?.barridos.some((b) => b.estado === 'pendiente' || b.estado === 'en_curso') ? MS_SONDEO : false,
  })

export const useBarrido = (id: string | undefined) =>
  useQuery({
    queryKey: ['barrido', id],
    queryFn: () => pedir<DetalleBarrido>(`/api/barridos/${encodeURIComponent(id!)}`),
    enabled: !!id,
    refetchInterval: (consulta) => (barridoActivo(consulta.state.data) ? MS_SONDEO : false),
  })

export function useCrearBarrido() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (config: ConfiguracionBarrido) =>
      pedir<{ id: string; ids: string[]; total: number; reutilizado: boolean }>('/api/barridos', { metodo: 'POST', cuerpo: config }),
    onSuccess: () => {
      cliente.invalidateQueries({ queryKey: ['barridos'] })
      cliente.invalidateQueries({ queryKey: ['experimentos'] })
    },
  })
}

export function useBorrarBarrido() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: ({ id, conExperimentos }: { id: string; conExperimentos: boolean }) =>
      pedir<void>(`/api/barridos/${encodeURIComponent(id)}${querystring({ con_experimentos: conExperimentos })}`, { metodo: 'DELETE' }),
    onSuccess: () => {
      cliente.invalidateQueries({ queryKey: ['barridos'] })
      cliente.invalidateQueries({ queryKey: ['experimentos'] })
    },
  })
}

/** Mutación POST sobre un barrido que, al acabar, refresca su detalle (donde se guardan los resultados). */
function useAccionBarrido<E, R>(id: string, accion: string) {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (cuerpo?: E) => pedir<R>(`/api/barridos/${encodeURIComponent(id)}/${accion}`, { metodo: 'POST', cuerpo: cuerpo ?? {} }),
    onSettled: () => cliente.invalidateQueries({ queryKey: ['barrido', id] }),
  })
}

export const useValidarBarrido = (id: string) => useAccionBarrido<undefined, ResultadoValidacion>(id, 'validar')

export const useWalkForwardBarrido = (id: string) =>
  useAccionBarrido<{ ventana_entreno_dias: number; paso_dias: number }, ResultadoWalkForward>(id, 'walk_forward')

export const useLanzarPermutacion = (id: string) => useAccionBarrido<PeticionPermutacion, RespuestaPermutacion>(id, 'permutacion')

export const useCancelarPermutacion = (id: string) =>
  useAccionBarrido<{ modo: ModoPermutacion }, { tarea: string; cancelando: boolean }>(id, 'permutacion/cancelar')

export const useEvaluarReservado = (id: string) => useAccionBarrido<{ experimento: string }, ResultadoReservado>(id, 'reservado')

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

export const useFillsPaper = () =>
  useQuery({
    queryKey: ['paper', 'fills'],
    queryFn: () => pedir<{ fills: FillPaper[] }>('/api/paper/fills'),
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
    mutationFn: (peticion: {
      estrategia: string
      parametros: Record<string, unknown>
      tickers: string[]
      intervalo_s: number
      /** Mínimo de días de historia; omitido = lo que necesite la estrategia. */
      dias_historico?: number
      /** Omitida = la frecuencia que sugiere la estrategia. */
      frecuencia_rebalanceo?: FrecuenciaRebalanceo
    }) =>
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

// ---- Configuraciones guardadas ---------------------------------------------------------------

export const useConfiguraciones = (filtros: { estrategia?: string; texto?: string; archivadas?: boolean } = {}) =>
  useQuery({
    queryKey: ['configuraciones', filtros],
    queryFn: () => pedir<{ configuraciones: Configuracion[] }>(`/api/configuraciones${querystring(filtros)}`),
    placeholderData: keepPreviousData,
  })

export const useConfiguracion = (id: string | undefined, version?: number) =>
  useQuery({
    queryKey: ['configuracion', id, version],
    queryFn: () => pedir<DetalleConfiguracion>(`/api/configuraciones/${encodeURIComponent(id!)}${querystring({ version })}`),
    enabled: !!id,
  })

export function useCrearConfiguracion() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (peticion: PeticionConfiguracion) => pedir<Configuracion>('/api/configuraciones', { metodo: 'POST', cuerpo: peticion }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['configuraciones'] }),
  })
}

export function useNuevaVersion(id: string) {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (cambios: Record<string, unknown>) =>
      pedir<Configuracion>(`/api/configuraciones/${encodeURIComponent(id)}/versiones`, { metodo: 'POST', cuerpo: { cambios } }),
    onSuccess: () => {
      cliente.invalidateQueries({ queryKey: ['configuraciones'] })
      cliente.invalidateQueries({ queryKey: ['configuracion', id] })
    },
  })
}

export function useBacktestConfiguracion(id: string) {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (cuerpo: { fecha_inicio?: string; fecha_fin?: string; version?: number; forzar?: boolean }) =>
      pedir<{ id_experimento: string; reutilizado: boolean }>(`/api/configuraciones/${encodeURIComponent(id)}/backtest`, { metodo: 'POST', cuerpo }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['experimentos'] }),
  })
}

export function useArchivarConfiguracion() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => pedir<void>(`/api/configuraciones/${encodeURIComponent(id)}`, { metodo: 'DELETE' }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['configuraciones'] }),
  })
}

// ---- Marcos de comparación y comparativas -------------------------------------------------------

export const useMarcos = () => useQuery({ queryKey: ['marcos'], queryFn: () => pedir<{ marcos: Marco[] }>('/api/marcos') })

export const useMarco = (id: string | undefined) =>
  useQuery({
    queryKey: ['marco', id],
    queryFn: () => pedir<{ marco: Marco; comparativas: Comparativa[] }>(`/api/marcos/${encodeURIComponent(id!)}`),
    enabled: !!id,
  })

/** Crea el marco (sin `id`) o lo reemplaza (con `id`). */
export function useGuardarMarco() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: ({ id, marco }: { id?: string; marco: PeticionMarco }) =>
      id
        ? pedir<Marco>(`/api/marcos/${encodeURIComponent(id)}`, { metodo: 'PUT', cuerpo: marco })
        : pedir<Marco>('/api/marcos', { metodo: 'POST', cuerpo: marco }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['marcos'] }).then(() => cliente.invalidateQueries({ queryKey: ['marco'] })),
  })
}

export function useBorrarMarco() {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => pedir<void>(`/api/marcos/${encodeURIComponent(id)}`, { metodo: 'DELETE' }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['marcos'] }),
  })
}

export function useLanzarComparativa(idMarco: string) {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (cuerpo: { estrategias: { estrategia: string; parametros: Record<string, unknown> }[]; forzar?: boolean }) =>
      pedir<Comparativa>(`/api/marcos/${encodeURIComponent(idMarco)}/comparativa`, { metodo: 'POST', cuerpo }),
    onSuccess: () => {
      cliente.invalidateQueries({ queryKey: ['experimentos'] })
      cliente.invalidateQueries({ queryKey: ['marco', idMarco] })
    },
  })
}

/** Cuerpo de `POST /api/configuraciones/{id}/comparativa`: estrategias con sus parámetros propios y fechas opcionales. */
export interface PeticionComparativaConfiguracion {
  estrategias: { estrategia: string; parametros: Record<string, unknown> }[]
  fecha_inicio?: string
  fecha_fin?: string
  version?: number
  forzar?: boolean
}

/** Lanza la misma configuración de entorno de una configuración guardada contra varias estrategias (crea o reutiliza el marco). */
export function useLanzarComparativaConfiguracion(idConfiguracion: string) {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (cuerpo: PeticionComparativaConfiguracion) =>
      pedir<Comparativa>(`/api/configuraciones/${encodeURIComponent(idConfiguracion)}/comparativa`, { metodo: 'POST', cuerpo }),
    onSuccess: () => {
      cliente.invalidateQueries({ queryKey: ['experimentos'] })
      cliente.invalidateQueries({ queryKey: ['marcos'] })
      cliente.invalidateQueries({ queryKey: ['configuracion', idConfiguracion] })
    },
  })
}

export const useComparativa = (id: string | null) =>
  useQuery({
    queryKey: ['comparativa', id],
    queryFn: () => pedir<Comparativa>(`/api/comparativas/${encodeURIComponent(id!)}`),
    enabled: !!id,
  })
