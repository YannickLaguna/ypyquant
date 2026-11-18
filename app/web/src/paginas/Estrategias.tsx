/**
 * Página Estrategias: catálogo con descripción, origen y parámetros; cada tarjeta lleva al detalle.
 * Responsabilidad: listar las estrategias del registro (incluidas y de `estrategias_usuario/`) y mostrar
 * los errores de importación de archivos de usuario si los hay.
 * Se relaciona con: api/hooks.ts (useEstrategias), paginas/EstrategiaDetalle.tsx, componentes/Estado.tsx.
 */
import { Link } from 'react-router-dom'
import { useEstrategias } from '../api/hooks'
import { Cargando, CajaError, Vacio } from '../componentes/Estado'

export default function Estrategias() {
  const { data, isPending, error } = useEstrategias()
  if (isPending) return <Cargando texto="Cargando estrategias…" />
  if (error) return <CajaError error={error} />
  const errores = Object.entries(data.errores)

  return (
    <div className="space-y-4">
      <header>
        <h1>Estrategias</h1>
        <p className="text-suave">Se escriben en Python (carpeta <code>estrategias_usuario/</code>); aquí se lanzan y se comparan.</p>
      </header>
      {errores.length > 0 && (
        <div role="alert" className="rounded-lg border border-neg/50 bg-neg/10 p-3">
          <p className="font-medium text-neg">Archivos de usuario con errores</p>
          <ul className="mt-1 text-xs">
            {errores.map(([ruta, msg]) => (
              <li key={ruta}><code>{ruta}</code>: {msg}</li>
            ))}
          </ul>
        </div>
      )}
      {data.estrategias.length === 0 ? (
        <Vacio titulo="No hay estrategias registradas">Añade un archivo .py a <code>estrategias_usuario/</code>.</Vacio>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {data.estrategias.map((e) => {
            const parametros = Object.entries(e.parametros_defecto)
            return (
              <li key={e.nombre}>
                <Link to={`/estrategias/${e.nombre}`} className="tarjeta block h-full p-4 transition-colors hover:border-acento">
                  <div className="flex items-center justify-between gap-2">
                    <span className="num font-semibold">{e.nombre}</span>
                    <span className={`rounded border px-1.5 text-xs ${e.origen === 'usuario' ? 'border-acento/60 text-acento' : 'border-borde text-suave'}`}>
                      {e.origen}
                    </span>
                  </div>
                  <p className="mt-1 text-suave">{e.descripcion}</p>
                  <p className="num mt-3 text-xs text-suave">
                    {parametros.length ? parametros.map(([k, v]) => `${k}=${String(v)}`).join(' · ') : 'sin parámetros'}
                  </p>
                  <p className="mt-1 text-xs text-suave">Rebalanceo sugerido: {e.frecuencia_rebalanceo}</p>
                </Link>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
