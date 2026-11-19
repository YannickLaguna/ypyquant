/**
 * Estructura de la aplicación: barra lateral (arriba en móvil) con la navegación y el selector de tema.
 * Responsabilidad: layout responsive, enlace «saltar al contenido», navegación con `aria-current` y
 * botón de tema oscuro/claro.
 * Se relaciona con: App.tsx (rutas), lib/tema.ts (persistencia del tema).
 */
import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { aplicarTema, temaGuardado, type Tema } from '../lib/tema'

const ENLACES = [
  { a: '/estrategias', texto: 'Estrategias' },
  { a: '/experimentos', texto: 'Experimentos' },
  { a: '/barridos', texto: 'Barridos' },
  { a: '/comparar', texto: 'Comparar' },
  { a: '/paper', texto: 'Paper trading' },
]

export default function Layout() {
  const [tema, setTema] = useState<Tema>(temaGuardado())
  const alternar = () => {
    const siguiente: Tema = tema === 'dark' ? 'light' : 'dark'
    aplicarTema(siguiente)
    setTema(siguiente)
  }
  return (
    <div className="flex min-h-full flex-col md:flex-row">
      <a
        href="#contenido"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-acento focus:px-3 focus:py-1 focus:text-fondo"
      >
        Saltar al contenido
      </a>
      <aside className="flex shrink-0 items-center gap-3 border-b border-borde bg-panel px-3 py-2 md:min-h-screen md:w-52 md:flex-col md:items-stretch md:border-b-0 md:border-r md:px-3 md:py-4">
        <div className="flex items-center gap-2 px-1 md:mb-4">
          <svg viewBox="0 0 32 32" className="h-6 w-6 shrink-0" aria-hidden>
            <rect width="32" height="32" rx="6" className="fill-elevado" />
            <path d="M4 22l7-8 5 5 9-11" fill="none" stroke="rgb(var(--acento))" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="whitespace-nowrap font-semibold tracking-tight">Sistema trading</span>
        </div>
        <nav aria-label="Principal" className="flex flex-1 gap-1 overflow-x-auto md:flex-col md:overflow-visible">
          {ENLACES.map((e) => (
            <NavLink
              key={e.a}
              to={e.a}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-md px-2.5 py-1.5 transition-colors ${
                  isActive ? 'bg-elevado font-medium text-acento' : 'text-suave hover:bg-elevado hover:text-texto'
                }`
              }
            >
              {e.texto}
            </NavLink>
          ))}
        </nav>
        <button
          type="button"
          className="boton shrink-0"
          onClick={alternar}
          aria-label={`Cambiar a tema ${tema === 'dark' ? 'claro' : 'oscuro'}`}
        >
          {tema === 'dark' ? 'Tema claro' : 'Tema oscuro'}
        </button>
      </aside>
      <main id="contenido" className="min-w-0 flex-1 p-4 md:p-6" tabIndex={-1}>
        <Outlet />
      </main>
    </div>
  )
}
