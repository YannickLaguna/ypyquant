/**
 * Tema oscuro (por defecto) y claro.
 * Responsabilidad: leer/guardar la preferencia (localStorage con try/catch), aplicarla como
 * `data-theme` en <html> y avisar a los gráficos (evento `tema-cambiado`) para que recolores.
 * Se relaciona con: src/index.css (tokens), componentes/Layout.tsx (botón), componentes/Graficos.tsx.
 */

export type Tema = 'dark' | 'light'
const CLAVE = 'tema-sistema-trading'

export function temaGuardado(): Tema {
  try {
    return localStorage.getItem(CLAVE) === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

export function aplicarTema(tema: Tema): void {
  document.documentElement.setAttribute('data-theme', tema)
  try {
    localStorage.setItem(CLAVE, tema)
  } catch {
    /* sin almacenamiento: la preferencia dura solo la sesión */
  }
  window.dispatchEvent(new Event('tema-cambiado'))
}

/** Color de un token CSS (`--acento`...) como `rgb(r g b)`; con alfa opcional. */
export function colorToken(nombre: string, alfa?: number): string {
  const valor = getComputedStyle(document.documentElement).getPropertyValue(`--${nombre}`).trim()
  if (!valor) return '#888888'
  return alfa === undefined ? `rgb(${valor})` : `rgb(${valor} / ${alfa})`
}

/** Paleta categórica para series (distinguible en ambos temas). */
export const PALETA_SERIES = [
  '#2dd4bf', '#60a5fa', '#f59e0b', '#c084fc', '#f472b6', '#a3e635',
  '#fb923c', '#38bdf8', '#e879f9', '#facc15', '#4ade80', '#94a3b8',
]
