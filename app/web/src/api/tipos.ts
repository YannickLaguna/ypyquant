/**
 * Tipos de la API (reflejan los modelos de app/api y ypyquant.estrategias).
 * Responsabilidad: contrato TypeScript único entre el frontend y la API.
 * Se relaciona con: api/cliente.ts y api/hooks.ts (los usan), paginas/* y componentes/*.
 */

/** Subconjunto de JSON Schema que genera Pydantic para los parámetros de una estrategia. */
export interface EsquemaPropiedad {
  type?: 'integer' | 'number' | 'string' | 'boolean' | 'array' | 'object'
  title?: string
  description?: string
  default?: unknown
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
  enum?: (string | number)[]
  anyOf?: EsquemaPropiedad[]
}

export interface EsquemaParametros {
  title?: string
  type?: 'object'
  properties?: Record<string, EsquemaPropiedad>
  required?: string[]
}

export type Frecuencia = 'daily' | 'weekly' | 'monthly' | 'quarterly'

export interface Estrategia {
  nombre: string
  descripcion: string
  origen: 'incluida' | 'usuario'
  archivo: string | null
  frecuencia_rebalanceo: Frecuencia
  parametros_schema: EsquemaParametros
  parametros_defecto: Record<string, unknown>
}

export interface EstrategiaDetalle extends Estrategia {
  codigo_fuente: string
}

export interface RespuestaEstrategias {
  estrategias: Estrategia[]
  errores: Record<string, string>
}

export interface Universo {
  nombre: string
  n_tickers: number
}

export type EstadoExperimento = 'pendiente' | 'en_curso' | 'hecho' | 'error'

export interface MetricasResumen {
  retorno_total?: number | null
  cagr?: number | null
  volatilidad?: number | null
  sharpe?: number | null
  sortino?: number | null
  calmar?: number | null
  max_drawdown?: number | null
  turnover_anual?: number | null
  alfa?: number | null
  beta?: number | null
  n_operaciones?: number | null
  ulcer?: number | null
  martin?: number | null
  costes_sobre_ganancia_bruta?: number | null
}

/** Respuesta de `POST /api/experimentos/borrar`: qué pasó con cada id pedido. */
export interface ResultadoBorrado {
  borrados: string[]
  /** Pendientes o en curso: no se borran porque el ejecutor seguiría escribiendo en su carpeta. */
  en_marcha: string[]
  no_encontrados: string[]
}

export interface ResumenExperimento {
  id: string
  estrategia: string
  parametros: Record<string, unknown>
  universo: string | string[]
  fecha_inicio: string
  fecha_fin: string
  benchmark: string
  frecuencia_rebalanceo: Frecuencia
  etiqueta: string
  estado: EstadoExperimento
  error: string | null
  creado: string | null
  actualizado: string | null
  metricas: MetricasResumen
}

export interface PuntoEquity {
  fecha: string
  equity: number
  benchmark: number | null
  drawdown: number | null
  retorno: number | null
}

export interface PesoFila {
  fecha: string
  ticker: string
  peso: number
}

export interface OperacionFila {
  fecha: string
  ticker: string
  lado: 'compra' | 'venta'
  cantidad: number
  precio: number
  comision: number
  pnl: number | null
}

export interface ConfiguracionExperimento {
  estrategia: string
  parametros: Record<string, unknown>
  universo: string | string[]
  fecha_inicio: string
  fecha_fin: string
  capital_inicial: number
  comision: number
  slippage: number
  frecuencia_rebalanceo: Frecuencia | null
  benchmark: string
  semilla: number
  etiqueta?: string
}

/** Con qué se obtuvo un experimento (no entra en la huella). `null` en experimentos anteriores a v0.5.0. */
export interface Procedencia {
  commit: string | null
  arbol_sucio: boolean | null
  version_ypyquant?: string | null
  python?: string | null
  numpy?: string | null
  pandas?: string | null
  cvxpy?: string | null
  hash_precios?: string | null
  ejecutado?: string | null
  /** Solo en experimentos importados (p. ej. «importado del TFM»): datos de origen en vez de commit y precios. */
  origen?: string | null
  commit_tfm?: string | null
  archivos?: Record<string, { ruta: string; sha256: string }>
  importado?: string | null
}

export interface DetalleExperimento extends ResumenExperimento {
  procedencia?: Procedencia | null
  config: ConfiguracionExperimento
  traza: string | null
  metricas_completas: (Record<string, number | null | Record<string, unknown> | string[]> & { avisos?: string[] }) | null
  equity: PuntoEquity[]
  pesos: PesoFila[]
  operaciones: OperacionFila[]
}

export interface Comparacion {
  ids: string[]
  fechas: string[]
  series: Record<string, (number | null)[]>
  benchmark: (number | null)[]
  metricas: Record<string, Record<string, number | null>>
  resumenes: Record<string, ResumenExperimento>
}

export interface CuentaPaper {
  numero: string | null
  estado: string | null
  equity: number | null
  efectivo: number | null
  poder_compra: number | null
  valor_cartera: number | null
  equity_cierre_anterior: number | null
  pnl_dia: number | null
  pnl_dia_pct: number | null
}

export interface PosicionPaper {
  ticker: string
  cantidad: number | null
  lado: string | null
  precio_medio: number | null
  precio_actual: number | null
  valor_mercado: number | null
  pnl: number | null
  pnl_pct: number | null
}

export interface OrdenPaper {
  id: string
  ticker: string
  lado: string | null
  tipo: string | null
  cantidad: number | null
  cantidad_ejecutada: number | null
  precio_ejecucion: number | null
  estado: string | null
  creada: string | null
  ejecutada: string | null
}

/** Fill real registrado por el bot en `data/paper_trading/fills.jsonl` (`GET /api/paper/fills`). */
export interface FillPaper {
  ticker: string
  lado: string
  cantidad: number
  precio: number
  comision: number
  fecha: string
  id_orden: string
}

export type FrecuenciaRebalanceo = 'daily' | 'weekly' | 'monthly' | 'quarterly'

export interface HistorialPaper {
  fechas: string[]
  equity: (number | null)[]
  pnl: (number | null)[]
  base: number | null
}

export interface EstadoBot {
  en_marcha: boolean
  pid: number | null
  codigo_salida: number | null
  estrategia?: string
  parametros?: Record<string, unknown>
  tickers?: string[]
  intervalo_s?: number
  dias_historico?: number
  frecuencia_rebalanceo?: FrecuenciaRebalanceo
  inicio?: string
}

export interface EstadoPaper {
  alpaca_configurado: boolean
  bot: EstadoBot
}

export interface LogBot {
  lineas: string[]
  siguiente: number
}

// ---- Barridos y validación (contratos de docs/ui.md) ----------------------------------------

export type EstadoBarrido = 'pendiente' | 'en_curso' | 'hecho' | 'error'
export type ModoPermutacion = 'in_sample' | 'walk_forward'
export type EstadoTarea = 'pendiente' | 'en_curso' | 'hecho' | 'error' | 'cancelado'
export type MetricasCombinacion = Record<string, number | null | undefined>

/** Métricas objetivo admitidas por la API (mayor es mejor). */
export const METRICAS_OBJETIVO = [
  'sharpe', 'sortino', 'calmar', 'martin', 'cagr', 'retorno_total', 'max_drawdown', 'information_ratio', 'alfa', 'tasa_acierto',
] as const

export interface ConfiguracionBarrido {
  base: Partial<ConfiguracionExperimento> & { estrategia: string }
  rejilla: Record<string, unknown[]>
  fecha_fin_desarrollo: string | null
  metrica_objetivo: string
  etiqueta: string
}

export interface ProgresoBarrido {
  total: number
  hechos: number
  errores: number
  en_curso: number
  pendientes: number
}

export interface ResumenBarrido {
  id: string
  etiqueta: string
  estrategia: string
  universo: string | string[]
  ejes: string[]
  n_combinaciones: number
  metrica_objetivo: string
  fecha_fin_desarrollo: string | null
  estado: EstadoBarrido
  progreso: ProgresoBarrido
  resultados: string[]
  creado: string | null
}

export interface CombinacionBarrido {
  id: string
  etiqueta: string
  valores: Record<string, unknown>
  posiciones: Record<string, number>
  estado: EstadoExperimento
}

export interface FilaSuperficie {
  id: string
  etiqueta: string
  valores: Record<string, unknown>
  posiciones: Record<string, number>
  metricas: MetricasCombinacion
  objetivo: number | null
  meseta: number | null
  n_vecinos: number
}

export interface TareaBarrido {
  estado: EstadoTarea
  progreso?: { completadas: number; total: number }
  p_valor?: number | null
  error?: string | null
  parametros?: Record<string, unknown>
  actualizado?: string
}

export interface Dsr {
  dsr: number | null
  sr: number | null
  sr_umbral: number | null
  sr_anualizado: number | null
  n_pruebas: number
  T: number
  asimetria: number | null
  curtosis: number | null
}

export interface Pbo {
  pbo: number | null
  pendiente: number | null
  prob_perdida_oos: number | null
  n_combinaciones: number
  n_bloques: number
  n_configuraciones: number
  lambdas: number[]
  pares_is_oos: [number, number][]
}

export interface Rachas {
  rachas: number
  mu: number
  sigma: number
  z: number | null
  p_valor: number | null
  n_positivos: number
  n_negativos: number
}

export interface ElegidaBarrido {
  id: string
  etiqueta: string
  valores: Record<string, unknown>
  objetivo: number | null
  meseta: number | null
  n_vecinos: number
  metricas: MetricasCombinacion
}

export interface ResultadoValidacion {
  fecha: string
  metrica_objetivo: string
  n_combinaciones: number
  n_observaciones: number
  fecha_inicio: string
  fecha_fin: string
  relleno: string
  elegida: ElegidaBarrido
  dsr: Dsr
  pbo: Pbo | null
  rachas: Rachas | null
  superficie: FilaSuperficie[]
  alternativa_meseta: {
    id: string
    etiqueta: string
    valores: Record<string, unknown>
    meseta: number | null
    objetivo: number | null
    es_la_elegida: boolean
  } | null
  avisos: string[]
}

export interface ResultadoWalkForward {
  fecha: string
  ventana_entreno_dias: number
  paso_dias: number
  ventana_filas: number
  paso_filas: number
  n_combinaciones: number
  sharpe_periodo: number | null
  metricas: Record<string, number | null>
  elecciones: { fecha: string; id: string; etiqueta: string; metrica_entreno: number | null }[]
  curva: { fechas: string[]; equity: number[] }
}

export interface ResultadoPermutacion {
  fecha: string
  modo: ModoPermutacion
  estadistico: string
  metrica_objetivo: string
  n_permutaciones: number
  n_validas: number
  longitud_bloque: number
  semilla: number
  procesos: number
  ventana_entreno_dias: number | null
  paso_dias: number | null
  inicio_permutacion: string | null
  n_combinaciones: number
  n_backtests: number
  valor_real: number | null
  p_valor: number | null
  distribucion: number[]
  media_permutadas: number | null
  p95_permutadas: number | null
  duracion_s: number
  avisos: string[]
}

export interface UsoReservado {
  fecha: string
  barrido: string
  experimento: string
}

export interface ResultadoReservado {
  fecha: string
  experimento_reservado: string
  combinacion: { id: string; etiqueta: string; valores: Record<string, unknown>; posiciones?: Record<string, number> }
  periodo: { inicio: string; fin: string }
  metricas: MetricasCombinacion
  usos_previos: UsoReservado[]
  n_usos: number
  aviso: string | null
  avisos: string[]
}

export interface DetalleBarrido {
  id: string
  config: ConfiguracionBarrido
  estado: EstadoBarrido
  progreso: ProgresoBarrido
  creado: string | null
  actualizado: string | null
  ejes: Record<string, unknown[]>
  combinaciones: CombinacionBarrido[]
  superficie: FilaSuperficie[]
  tareas: Record<string, TareaBarrido>
  validacion: ResultadoValidacion | null
  walk_forward: ResultadoWalkForward | null
  permutacion_in_sample: ResultadoPermutacion | null
  permutacion_walk_forward: ResultadoPermutacion | null
  reservado: ResultadoReservado[]
  reservado_usos: UsoReservado[]
  periodo_reservado: { inicio: string; fin: string } | null
}

export interface PeticionPermutacion {
  modo: ModoPermutacion
  n_permutaciones: number
  longitud_bloque: number
  procesos: number
  ventana_entreno_dias?: number | null
  paso_dias?: number | null
  semilla?: number
}

export interface RespuestaPermutacion {
  tarea: string
  estado: EstadoTarea
  n_backtests: number
  aviso: string | null
}
