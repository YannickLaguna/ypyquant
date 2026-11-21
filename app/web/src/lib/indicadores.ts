/**
 * Indicadores de la página Comparar y utilidades puras para dibujarlos (sin React ni red; test en indicadores.test.ts).
 * Responsabilidad: catálogo único de indicadores —escalares (una barra por experimento, salen de `metricas`) y
 * temporales (una línea por experimento: la equity base 100 viene de `/api/comparar` y el resto de
 * `/api/comparar/series`)—, validación del indicador de la URL (`?indicador=`), ordenación de barras con los
 * «sin dato» al final (un valor ausente NUNCA se trata como 0) y geometría de las barras con valores negativos.
 * Se relaciona con: paginas/Comparar.tsx (tabla y figura), componentes/SelectorIndicador.tsx, componentes/GraficoBarras.tsx
 * y, en Python, ypyquant.estrategias.series_comparacion (los mismos indicadores temporales).
 */
import { dinero, entero, porcentaje, ratio } from './formato'

export type Formato = (v: number | null | undefined) => string

export interface Indicador {
  clave: string
  etiqueta: string
  grupo: 'resumen' | 'tiempo'
  /** `escalar`: valor en `metricas[clave]` (una barra por experimento). `temporal`: serie (una línea por experimento). */
  tipo: 'escalar' | 'temporal'
  formato: Formato
  /** Qué valor es mejor (resalta la tabla y ordena las barras): `null` = mayor primero sin juicio de valor. */
  mejor: 'max' | 'min' | null
  /** Frase corta bajo la figura para saber qué se está viendo. */
  ayuda: string
}

const porcentaje0: Formato = (v) => porcentaje(v, 0)
const ratio1: Formato = (v) => ratio(v, 1)

/** Todos los indicadores, en el orden en que se muestran (la tabla de métricas usa los escalares en este orden). */
export const INDICADORES: Indicador[] = [
  { clave: 'sharpe', etiqueta: 'Sharpe', grupo: 'resumen', tipo: 'escalar', formato: ratio, mejor: 'max', ayuda: 'Retorno anualizado entre volatilidad anualizada (rf = 0).' },
  { clave: 'sortino', etiqueta: 'Sortino', grupo: 'resumen', tipo: 'escalar', formato: ratio, mejor: 'max', ayuda: 'Como el Sharpe, pero solo penaliza la volatilidad a la baja.' },
  { clave: 'calmar', etiqueta: 'Calmar', grupo: 'resumen', tipo: 'escalar', formato: ratio, mejor: 'max', ayuda: 'CAGR dividido por el máximo drawdown.' },
  { clave: 'cagr', etiqueta: 'CAGR', grupo: 'resumen', tipo: 'escalar', formato: porcentaje, mejor: 'max', ayuda: 'Retorno anual compuesto.' },
  { clave: 'retorno_total', etiqueta: 'Retorno total', grupo: 'resumen', tipo: 'escalar', formato: porcentaje, mejor: 'max', ayuda: 'Retorno acumulado en todo el periodo del experimento.' },
  { clave: 'volatilidad', etiqueta: 'Volatilidad', grupo: 'resumen', tipo: 'escalar', formato: porcentaje, mejor: 'min', ayuda: 'Desviación típica anualizada de los retornos.' },
  { clave: 'max_drawdown', etiqueta: 'Máx. drawdown', grupo: 'resumen', tipo: 'escalar', formato: porcentaje, mejor: 'max', ayuda: 'Peor caída desde un máximo (más cerca de 0 es mejor).' },
  { clave: 'alfa', etiqueta: 'Alfa', grupo: 'resumen', tipo: 'escalar', formato: porcentaje, mejor: 'max', ayuda: 'Retorno anualizado no explicado por el benchmark.' },
  { clave: 'beta', etiqueta: 'Beta', grupo: 'resumen', tipo: 'escalar', formato: ratio, mejor: 'min', ayuda: 'Sensibilidad al benchmark.' },
  { clave: 'information_ratio', etiqueta: 'Ratio de información', grupo: 'resumen', tipo: 'escalar', formato: ratio, mejor: 'max', ayuda: 'Retorno activo entre tracking error frente al benchmark.' },
  { clave: 'ulcer', etiqueta: 'Índice Ulcer', grupo: 'resumen', tipo: 'escalar', formato: ratio1, mejor: 'min', ayuda: 'Profundidad y duración de los drawdowns (menor es mejor).' },
  { clave: 'turnover_anual', etiqueta: 'Turnover anual', grupo: 'resumen', tipo: 'escalar', formato: porcentaje0, mejor: 'min', ayuda: 'Rotación anual de la cartera con la rotación real del motor.' },
  { clave: 'n_operaciones', etiqueta: 'Operaciones', grupo: 'resumen', tipo: 'escalar', formato: entero, mejor: 'min', ayuda: 'Número de operaciones ejecutadas (los importados del TFM no tienen).' },
  { clave: 'comisiones_totales', etiqueta: 'Comisiones', grupo: 'resumen', tipo: 'escalar', formato: (v) => dinero(v, 0), mejor: 'min', ayuda: 'Comisiones totales pagadas (los importados del TFM no tienen).' },
  { clave: 'equity', etiqueta: 'Equity base 100', grupo: 'tiempo', tipo: 'temporal', formato: (v) => ratio(v, 1), mejor: null, ayuda: 'Equity normalizada a 100 en la primera fecha común.' },
  { clave: 'drawdown', etiqueta: 'Drawdown', grupo: 'tiempo', tipo: 'temporal', formato: porcentaje, mejor: null, ayuda: 'Caída desde el máximo previo, sobre las fechas comunes.' },
  { clave: 'sharpe_movil', etiqueta: 'Sharpe móvil 12 m', grupo: 'tiempo', tipo: 'temporal', formato: ratio, mejor: null, ayuda: 'Sharpe de los últimos 12 meses (una ventana de 12 meses de observaciones del eje común).' },
  { clave: 'volatilidad_movil', etiqueta: 'Volatilidad móvil 12 m', grupo: 'tiempo', tipo: 'temporal', formato: porcentaje, mejor: null, ayuda: 'Volatilidad anualizada de los últimos 12 meses.' },
  { clave: 'turnover_rebalanceo', etiqueta: 'Turnover por rebalanceo', grupo: 'tiempo', tipo: 'temporal', formato: porcentaje, mejor: null, ayuda: '½ Σ|Δpeso| entre pesos objetivo consecutivos; sin dato si el experimento no guarda pesos.' },
]

export const INDICADOR_POR_DEFECTO = 'equity'

export const GRUPOS: { grupo: Indicador['grupo']; titulo: string }[] = [
  { grupo: 'resumen', titulo: 'Resumen (una barra por experimento)' },
  { grupo: 'tiempo', titulo: 'En el tiempo (una línea por experimento)' },
]

export const escalares = (): Indicador[] => INDICADORES.filter((i) => i.tipo === 'escalar')

/** Indicador de la URL (`?indicador=`); cualquier valor ausente o desconocido cae en la equity. */
export function indicadorDeUrl(valor: string | null | undefined): Indicador {
  return INDICADORES.find((i) => i.clave === valor) ?? INDICADORES.find((i) => i.clave === INDICADOR_POR_DEFECTO)!
}

export interface Barra {
  id: string
  /** Posición del experimento en la comparación: da el mismo color que su línea de equity. */
  indice: number
  /** null = sin dato (el experimento no tiene esa métrica). */
  valor: number | null
}

const esNumero = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/**
 * Una barra por experimento, ordenadas de mejor a peor según `mejor` (`max`: mayor primero; `min`: menor primero;
 * `null`: mayor primero) y con los «sin dato» al final en su orden original. Un valor ausente, null o no finito
 * es «sin dato», nunca 0.
 */
export function barrasOrdenadas(
  ids: string[],
  metricas: Record<string, Record<string, number | null | undefined>>,
  clave: string,
  mejor: 'max' | 'min' | null,
): Barra[] {
  const barras: Barra[] = ids.map((id, indice) => {
    const valor = metricas[id]?.[clave]
    return { id, indice, valor: esNumero(valor) ? valor : null }
  })
  const con = barras.filter((b) => b.valor !== null).sort((a, b) => (mejor === 'min' ? a.valor! - b.valor! : b.valor! - a.valor!))
  return [...con, ...barras.filter((b) => b.valor === null)]
}

/**
 * Posición y ancho (en %) de una barra sobre un eje que siempre incluye el 0: los valores negativos salen a la
 * izquierda del cero y los positivos a la derecha. `rango` es `[mínimo, máximo]` de todas las barras con dato.
 */
export function geometriaBarra(valor: number, rango: [number, number]): { izquierda: number; ancho: number; cero: number } {
  const minimo = Math.min(rango[0], 0)
  const maximo = Math.max(rango[1], 0)
  const total = maximo - minimo || 1
  const cero = (0 - minimo) / total * 100
  const ancho = (Math.abs(valor) / total) * 100
  return { izquierda: valor >= 0 ? cero : cero - ancho, ancho, cero }
}

/** `[mínimo, máximo]` de las barras con dato (null si ninguna lo tiene). */
export function rangoBarras(barras: Barra[]): [number, number] | null {
  const valores = barras.filter((b) => b.valor !== null).map((b) => b.valor!)
  return valores.length ? [Math.min(...valores), Math.max(...valores)] : null
}
