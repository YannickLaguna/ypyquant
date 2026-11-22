/**
 * Selector de perfil de Alpaca: una pestaña por cuenta paper (hasta 3 activas a la vez).
 * Responsabilidad: mostrar de un vistazo cada perfil con su estado (sin configurar, sin conexión, equity, bot en marcha
 * y a qué experimento está vinculado) y avisar cuál está seleccionado (`aria-selected`). No consulta nada: recibe la lista.
 * Se relaciona con: api/alpaca.ts (`PerfilAlpaca`), paginas/PaperTrading.tsx (lo usa) y su test SelectorPerfil.test.tsx.
 */
import type { PerfilAlpaca } from '../api/alpaca'
import { dinero } from '../lib/formato'

/** Texto corto del estado de un perfil para la pestaña (el primero que aplique). */
export function estadoDePerfil(perfil: PerfilAlpaca): { texto: string; clase: string } {
  if (!perfil.configurado) return { texto: 'sin configurar', clase: 'border-aviso/60 text-aviso' }
  if (perfil.conexion.conectada === false) return { texto: 'sin conexión', clase: 'border-neg/60 text-neg' }
  if (perfil.bot.en_marcha) return { texto: 'bot en marcha', clase: 'border-pos/60 text-pos' }
  return { texto: 'bot parado', clase: 'border-borde text-suave' }
}

export default function SelectorPerfil({ perfiles, seleccionado, alSeleccionar }: {
  perfiles: PerfilAlpaca[]
  seleccionado: string
  alSeleccionar: (nombre: string) => void
}) {
  return (
    <div role="tablist" aria-label="Cuentas de Alpaca" className="flex flex-wrap gap-2">
      {perfiles.map((perfil) => {
        const activo = perfil.nombre === seleccionado
        const estado = estadoDePerfil(perfil)
        return (
          <button
            key={perfil.nombre}
            type="button"
            role="tab"
            id={`perfil-${perfil.nombre}`}
            aria-selected={activo}
            className={`boton flex min-w-[10rem] flex-col items-start gap-1 text-left ${activo ? 'border-acento' : ''}`}
            onClick={() => alSeleccionar(perfil.nombre)}
          >
            <span className="flex w-full items-center justify-between gap-2">
              <span className="font-semibold">{perfil.nombre}</span>
              {perfil.ultimos4 && <span className="num text-xs text-suave" title="Últimos 4 caracteres de la API key">…{perfil.ultimos4}</span>}
            </span>
            <span className={`inline-block rounded border px-1.5 py-0.5 text-xs font-medium ${estado.clase}`}>{estado.texto}</span>
            {perfil.configurado && perfil.conexion.equity != null && <span className="num text-xs">{dinero(perfil.conexion.equity, 0)}</span>}
            {perfil.vinculo && <span className="num max-w-[14rem] truncate text-xs text-suave" title={perfil.vinculo.origen.id}>{perfil.vinculo.origen.id}</span>}
          </button>
        )
      })}
    </div>
  )
}
