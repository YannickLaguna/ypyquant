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

export interface DetalleExperimento extends ResumenExperimento {
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
