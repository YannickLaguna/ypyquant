/**
 * Datos de Alpaca multi-perfil: tipos y hooks (TanStack Query) de las cuentas paper por perfil y de su bot.
 * Responsabilidad: una función por recurso de `/api/paper/alpaca/*` y de las rutas `/api/paper/*?perfil=X`, con claves de
 * caché que incluyen el perfil (cambiar de pestaña no mezcla cuentas) y mutaciones del bot que invalidan lo de ese perfil.
 * Nunca maneja claves: de una cuenta solo llega «configurada sí/no» y los 4 últimos caracteres de la API key.
 * Vive aparte de api/hooks.ts y api/tipos.ts para no solapar con otras ramas (se puede fusionar allí sin cambios).
 * Se relaciona con: api/cliente.ts (fetch), app/api/rutas_alpaca.py y rutas_paper.py (el backend),
 * paginas/PaperTrading.tsx y componentes/{SelectorPerfil,PerfilSinConfigurar,BotPerfil}.tsx.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ErrorApi, pedir, querystring } from './cliente'
import type { CuentaPaper, FillPaper, HistorialPaper, LogBot, OrdenPaper, PosicionPaper } from './tipos'

export const PERFIL_DEFECTO = 'DEFECTO'

export type ModoBot = 'continuo' | 'diario'

export interface ConexionPerfil {
  configurada: boolean
  /** null = no se consultó (el listado se pidió con `conexion=false`). */
  conectada: boolean | null
  equity: number | null
  error: string | null
}

/** Estado del bot de un perfil (`GestorBots.estado`). */
export interface EstadoBotPerfil {
  perfil: string
  en_marcha: boolean
  pid: number | null
  codigo_salida: number | null
  adoptado: boolean
  modo: ModoBot | null
  intervalo_s: number | null
  experimento_id: string | null
  configuracion_id: string | null
  inicio: string | null
}

/** Resumen del vínculo cuenta-experimento (sin la receta completa). */
export interface VinculoPerfil {
  origen: { tipo: 'experimento' | 'configuracion'; id: string }
  estrategia: string
  huella_receta: string
  frecuencia_rebalanceo: string | null
  tickers: string[]
  fecha_alta: string
  equity_inicial: number | null
}

export interface PerfilAlpaca {
  nombre: string
  configurado: boolean
  /** Últimos 4 caracteres de la API key (nunca del secreto); null si no está configurado. */
  ultimos4: string | null
  /** Variables de entorno que faltan para configurarlo. */
  faltan: string[]
  /** Las dos variables del `.env` de este perfil (API key y secreto). */
  variables: string[]
  conexion: ConexionPerfil
  bot: EstadoBotPerfil
  vinculo: VinculoPerfil | null
}

export interface RespuestaPerfiles {
  perfiles: PerfilAlpaca[]
  limite_activos: number
  activos: string[]
  /** Aviso (p. ej. más cuentas configuradas que el límite blando). */
  aviso: string | null
}

export interface PeticionBotPerfil {
  experimento_id: string
  modo: ModoBot
  intervalo_s: number
}

export interface InstantaneaCuenta {
  fecha: string
  equity: number
  efectivo: number
  /** JSON de `{ticker: {cantidad, precio}}`. */
  posiciones: string
}

/** Los 502/503 (sin claves, Alpaca caída) no se reintentan: el mensaje es definitivo hasta que cambie la configuración. */
const sinReintentarEn503 = (intentos: number, error: unknown) =>
  !(error instanceof ErrorApi && [502, 503].includes(error.estado)) && intentos < 2

const consultaPerfil = (perfil: string) => querystring({ perfil })

export const usePerfilesAlpaca = () =>
  useQuery({
    queryKey: ['paper', 'alpaca', 'perfiles'],
    queryFn: () => pedir<RespuestaPerfiles>('/api/paper/alpaca/perfiles'),
    refetchInterval: 15_000,
  })

export const useCuentaPerfil = (perfil: string, activo = true) =>
  useQuery({
    queryKey: ['paper', 'alpaca', perfil, 'cuenta'],
    queryFn: () => pedir<CuentaPaper>(`/api/paper/cuenta${consultaPerfil(perfil)}`),
    enabled: activo,
    retry: sinReintentarEn503,
    refetchInterval: 30_000,
  })

export const usePosicionesPerfil = (perfil: string, activo = true) =>
  useQuery({
    queryKey: ['paper', 'alpaca', perfil, 'posiciones'],
    queryFn: () => pedir<{ posiciones: PosicionPaper[] }>(`/api/paper/posiciones${consultaPerfil(perfil)}`),
    enabled: activo,
    retry: sinReintentarEn503,
    refetchInterval: 30_000,
  })

export const useOrdenesPerfil = (perfil: string, activo = true) =>
  useQuery({
    queryKey: ['paper', 'alpaca', perfil, 'ordenes'],
    queryFn: () => pedir<{ ordenes: OrdenPaper[] }>(`/api/paper/ordenes${consultaPerfil(perfil)}`),
    enabled: activo,
    retry: sinReintentarEn503,
    refetchInterval: 30_000,
  })

export const useHistorialPerfil = (perfil: string, periodo: string, activo = true) =>
  useQuery({
    queryKey: ['paper', 'alpaca', perfil, 'historial', periodo],
    queryFn: () => pedir<HistorialPaper>(`/api/paper/historial${querystring({ perfil, periodo })}`),
    enabled: activo,
    retry: sinReintentarEn503,
  })

/** Fills del perfil (campos extra del bot: `perfil`, `experimento_id`, `tipo`, `precio_referencia`; todos opcionales). */
export type FillPerfil = FillPaper & { perfil?: string; experimento_id?: string; configuracion_id?: string; tipo?: string; precio_referencia?: number }

export const useFillsPerfil = (perfil: string) =>
  useQuery({
    queryKey: ['paper', 'alpaca', perfil, 'fills'],
    queryFn: () => pedir<{ fills: FillPerfil[] }>(`/api/paper/fills${consultaPerfil(perfil)}`),
    refetchInterval: 30_000,
  })

export const useInstantaneasPerfil = (perfil: string) =>
  useQuery({
    queryKey: ['paper', 'alpaca', perfil, 'instantaneas'],
    queryFn: () => pedir<{ instantaneas: InstantaneaCuenta[] }>(`/api/paper/alpaca/${encodeURIComponent(perfil)}/instantaneas`),
    refetchInterval: 60_000,
  })

export const useEstadoBotPerfil = (perfil: string) =>
  useQuery({
    queryKey: ['paper', 'alpaca', perfil, 'bot'],
    queryFn: () => pedir<EstadoBotPerfil & { vinculo: VinculoPerfil | null }>(`/api/paper/alpaca/bots/${encodeURIComponent(perfil)}`),
    refetchInterval: 5000,
  })

export const useLogBotPerfil = (perfil: string, activo: boolean) =>
  useQuery({
    queryKey: ['paper', 'alpaca', perfil, 'log'],
    queryFn: () => pedir<LogBot>(`/api/paper/alpaca/bots/${encodeURIComponent(perfil)}/log?desde=0`),
    refetchInterval: activo ? 2000 : false,
  })

export function useArrancarBotPerfil(perfil: string) {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: (peticion: PeticionBotPerfil) =>
      pedir<EstadoBotPerfil>(`/api/paper/alpaca/bots/${encodeURIComponent(perfil)}/arrancar`, { metodo: 'POST', cuerpo: peticion }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['paper', 'alpaca'] }),
  })
}

export function usePararBotPerfil(perfil: string) {
  const cliente = useQueryClient()
  return useMutation({
    mutationFn: () => pedir<EstadoBotPerfil>(`/api/paper/alpaca/bots/${encodeURIComponent(perfil)}/parar`, { metodo: 'POST' }),
    onSuccess: () => cliente.invalidateQueries({ queryKey: ['paper', 'alpaca'] }),
  })
}
