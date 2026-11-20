/**
 * Página Configuraciones: tabla de las recetas guardadas (última versión de cada una).
 * Responsabilidad: listar, filtrar por texto y ver archivadas; cada fila enlaza con su detalle. Las
 * configuraciones se crean desde «Guardar como configuración» en un experimento o en un barrido.
 * Se relaciona con: api/hooks.ts (useConfiguraciones), paginas/ConfiguracionDetalle.tsx y paginas/Marcos.tsx
 * (un marco fija el entorno común para comparar estrategias).
 */
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useConfiguraciones } from '../api/hooks'
import { CajaError, Cargando, Vacio } from '../componentes/Estado'
import { fechaCorta, ratio, resumenParametros, resumenUniverso } from '../lib/formato'

export default function Configuraciones() {
  const [texto, setTexto] = useState('')
  const [archivadas, setArchivadas] = useState(false)
  const { data, isPending, error } = useConfiguraciones({ texto: texto || undefined, archivadas })

  return (
    <div className="space-y-5">
      <header>
        <h1>Configuraciones</h1>
        <p className="text-suave">
          Recetas congeladas y versionadas (estrategia, parámetros, universo, frecuencia y costes, sin fechas). Se crean desde un
          experimento o un barrido y se relanzan como backtest. Para comparar estrategias con el mismo entorno usa los{' '}
          <Link to="/marcos" className="text-acento underline">marcos</Link>.
        </p>
      </header>

      <div className="flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1">
          <label htmlFor="filtro-config" className="font-medium">Buscar</label>
          <input id="filtro-config" className="campo" value={texto} onChange={(e) => setTexto(e.target.value)} placeholder="nombre, notas o etiqueta" />
        </div>
        <label className="flex items-center gap-2">
          <input type="checkbox" className="accent-[rgb(var(--acento))]" checked={archivadas} onChange={(e) => setArchivadas(e.target.checked)} />
          Ver archivadas
        </label>
      </div>

      {isPending ? (
        <Cargando />
      ) : error ? (
        <CajaError error={error} />
      ) : data.configuraciones.length === 0 ? (
        <Vacio titulo="Todavía no hay configuraciones">
          Abre un experimento terminado y pulsa «Guardar como configuración», o elige la mejor combinación de un barrido.
        </Vacio>
      ) : (
        <div className="tarjeta overflow-x-auto">
          <table className="w-full text-sm" aria-label="Configuraciones guardadas">
            <thead className="border-b border-borde">
              <tr>
                {['Nombre', 'Estrategia', 'Versión', 'Parámetros', 'Universo', 'Sharpe al guardar', 'Origen', 'Creada'].map((t) => (
                  <th key={t} scope="col" className="th">{t}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.configuraciones.map((c) => (
                <tr key={c.id} className="border-b border-borde/60 last:border-0">
                  <td className="td">
                    <Link to={`/configuraciones/${c.id}`} className="text-acento hover:underline">{c.nombre}</Link>
                    {c.archivada && <span className="ml-2 text-xs text-suave">(archivada)</span>}
                  </td>
                  <td className="td num">{c.estrategia}</td>
                  <td className="td num">v{c.version}</td>
                  <td className="td num text-xs">{resumenParametros(c.parametros)}</td>
                  <td className="td">{c.universo_etiqueta ?? resumenUniverso(c.universo)}</td>
                  <td className="td num">{ratio(c.origen.metricas.sharpe)}</td>
                  <td className="td">{c.origen.tipo}{c.origen.criterio ? ` · ${c.origen.criterio}` : ''}</td>
                  <td className="td">{fechaCorta(c.creada.slice(0, 10))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
