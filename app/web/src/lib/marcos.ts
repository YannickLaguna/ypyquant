/**
 * Utilidades puras de los marcos de comparación (sin React ni red; testeadas en marcos.test.ts).
 * Responsabilidad: catálogo de parámetros comunes sugeridos, conversión del estado del formulario a
 * `parametros_comunes`, formulario propio de cada estrategia sin los parámetros que fija el marco, y el
 * texto del aviso «no es ceteris paribus» a partir de `ceteris_paribus.diferencias` de `GET /api/comparar`.
 * Se relaciona con: componentes/FormularioMarco.tsx, componentes/LanzadorComparativa.tsx, paginas/Comparar.tsx
 * y api/tipos.ts (contratos); la lógica de fondo está en ypyquant.estrategias.marcos (Python).
 */
import type { DiferenciaCeterisParibus, EsquemaParametros } from '../api/tipos'

export interface ComunSugerido {
  nombre: string
  etiqueta: string
  tipo: 'entero' | 'decimal' | 'opcion'
  opciones?: string[]
  defecto: number | string
  ayuda: string
}

/** Parámetros que suelen compartir varias estrategias (se aplican solo a las que los declaran). */
export const COMUNES: ComunSugerido[] = [
  { nombre: 'ventana_dias', etiqueta: 'Ventana de estimación (días)', tipo: 'entero', defecto: 756, ayuda: 'Días de historia para estimar µ y Σ (756 ≈ 3 años)' },
  { nombre: 'peso_maximo', etiqueta: 'Peso máximo por activo', tipo: 'decimal', defecto: 0.3, ayuda: 'Tope de peso por activo' },
  { nombre: 'tasa_libre_riesgo', etiqueta: 'Tasa libre de riesgo (anual)', tipo: 'decimal', defecto: 0, ayuda: 'Para el max-Sharpe' },
  { nombre: 'estimador_sigma', etiqueta: 'Estimador de Σ', tipo: 'opcion', opciones: ['ledoit_wolf', 'muestral'], defecto: 'ledoit_wolf', ayuda: 'Solo los estimadores robustos lo declaran' },
  { nombre: 'frecuencia_estimacion', etiqueta: 'Frecuencia de los retornos', tipo: 'opcion', opciones: ['diaria', 'mensual'], defecto: 'diaria', ayuda: 'mensual = cierres a fin de mes y 12 periodos/año, como el TFM' },
]

/** Claves de `parametros_comunes` conocidas (espejo de `COMUNES_CONOCIDOS` en marcos.py). */
const CLAVES_COMUNES = COMUNES.map((c) => c.nombre)

/** Los parámetros de una configuración que pasan al marco derivado (los comunes conocidos que trae). */
export function comunesDeConfiguracion(parametros: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(parametros).filter(([nombre]) => CLAVES_COMUNES.includes(nombre)))
}

/** Los parámetros de una configuración que son PROPIOS de su estrategia (los comunes los pone el marco). */
export function parametrosPropios(parametros: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(parametros).filter(([nombre]) => !CLAVES_COMUNES.includes(nombre)))
}

export type EstadoComunes = Record<string, { activo: boolean; valor: number | string }>

/** `parametros_comunes` del marco: solo los activados. */
export function comunesActivos(estado: EstadoComunes): Record<string, number | string> {
  return Object.fromEntries(Object.entries(estado).filter(([, c]) => c.activo).map(([nombre, c]) => [nombre, c.valor]))
}

/** Copia del esquema de parámetros sin los que fija el marco (no se eligen por estrategia). No muta el original. */
export function esquemaSinComunes(esquema: EsquemaParametros, comunes: Record<string, unknown>): EsquemaParametros {
  const propiedades = Object.fromEntries(Object.entries(esquema.properties ?? {}).filter(([nombre]) => !(nombre in comunes)))
  return { ...esquema, properties: propiedades }
}

const ETIQUETAS_CAMPO: Record<string, string> = {
  universo: 'universo',
  fecha_inicio: 'fecha de inicio',
  fecha_fin: 'fecha de fin',
  frecuencia_rebalanceo: 'frecuencia de rebalanceo',
  comision: 'comisión',
  slippage: 'slippage',
  benchmark: 'benchmark',
  capital_inicial: 'capital inicial',
  semilla: 'semilla',
  dias_historia_previa: 'historia previa',
}

/** Texto del aviso de Comparar, o null si la comparación es ceteris paribus. */
export function resumenDiferencias(diferencias: DiferenciaCeterisParibus[] | undefined): string | null {
  if (!diferencias || diferencias.length === 0) return null
  const partes = diferencias.map((d) => (d.tipo === 'comun' ? `el parámetro común ${d.campo}` : (ETIQUETAS_CAMPO[d.campo] ?? d.campo)))
  const lista = partes.length === 1 ? partes[0] : `${partes.slice(0, -1).join(', ')} y ${partes[partes.length - 1]}`
  return `No es ceteris paribus: ${partes.length === 1 ? 'difiere' : 'difieren'} ${lista}.`
}
