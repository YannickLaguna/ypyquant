/**
 * Página Marcos: lista de marcos de comparación y alta de uno nuevo.
 * Responsabilidad: un marco fija una vez el entorno (universo, fechas, frecuencia, costes, benchmark, capital,
 * historia previa y parámetros comunes) para lanzar varias estrategias sobre él y comparar solo el efecto de la
 * estrategia (ceteris paribus). Es una página propia (no una sección de Configuraciones) porque tiene su flujo:
 * definir el entorno -> elegir estrategias -> comparar, y porque un marco agrupa muchas estrategias mientras una
 * configuración es una sola receta.
 * Con `?desde=<id de experimento>` abre el alta con el entorno de ese experimento precargado (lib/marcos.ts,
 * marcoDesdeExperimento); viene del botón «Crear marco desde este experimento» de ExperimentoDetalle.
 * Se relaciona con: componentes/FormularioMarco.tsx, paginas/MarcoDetalle.tsx, api/hooks.ts (useMarcos, useGuardarMarco, useExperimento).
 */
import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useExperimento, useGuardarMarco, useMarcos } from '../api/hooks'
import { CajaError, Cargando, Vacio } from '../componentes/Estado'
import FormularioMarco from '../componentes/FormularioMarco'
import { fechaCorta, resumenUniverso } from '../lib/formato'
import { marcoDesdeExperimento } from '../lib/marcos'

export default function Marcos() {
  const navegar = useNavigate()
  const { data, isPending, error } = useMarcos()
  const guardar = useGuardarMarco()
  const [parametrosUrl] = useSearchParams()
  const desde = parametrosUrl.get('desde') ?? undefined
  const { data: experimento, error: errorExperimento } = useExperimento(desde)
  const [nuevo, setNuevo] = useState(Boolean(desde))
  const inicial = experimento?.config ? marcoDesdeExperimento(experimento.config, experimento.id) : undefined
  // Con `?desde=` se espera al experimento para montar el formulario ya precargado.
  const esperandoExperimento = Boolean(desde) && !experimento && !errorExperimento

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1>Marcos de comparación</h1>
          <p className="text-suave">
            Fija el entorno una vez y lanza varias estrategias sobre él: mismo universo, fechas, rebalanceo, costes y estimadores, para comparar
            solo el efecto de la estrategia (ceteris paribus).
          </p>
        </div>
        <button type="button" className="boton boton-primario" onClick={() => setNuevo((v) => !v)} aria-expanded={nuevo}>{nuevo ? 'Cancelar' : 'Nuevo marco'}</button>
      </header>

      {nuevo && (
        <section aria-labelledby="h-nuevo-marco" className="space-y-2">
          <h2 id="h-nuevo-marco">Nuevo marco</h2>
          {guardar.error && <CajaError error={guardar.error} titulo="No se pudo guardar el marco" />}
          {desde && <p className="text-sm text-suave">Entorno copiado del experimento <Link to={`/experimentos/${desde}`} className="text-acento underline">{desde}</Link>; revísalo y crea el marco. Después, en su detalle, añade las estrategias a comparar.</p>}
          {errorExperimento && <CajaError error={errorExperimento} titulo="No se pudo leer el experimento de origen" />}
          {esperandoExperimento ? <Cargando texto="Leyendo el experimento…" /> : <FormularioMarco key={inicial ? desde : 'vacio'} inicial={inicial} guardando={guardar.isPending} textoBoton="Crear marco" alGuardar={(marco) => guardar.mutate({ marco }, { onSuccess: (m) => navegar(`/marcos/${m.id}`) })} />}
        </section>
      )}

      {isPending ? (
        <Cargando />
      ) : error ? (
        <CajaError error={error} />
      ) : data.marcos.length === 0 ? (
        <Vacio titulo="Todavía no hay marcos">Crea uno con «Nuevo marco»: te sugerimos el universo point-in-time sp500_pit y rebalanceo mensual.</Vacio>
      ) : (
        <div className="tarjeta overflow-x-auto">
          <table className="w-full text-sm" aria-label="Marcos">
            <thead className="border-b border-borde"><tr>{['Nombre', 'Universo', 'Periodo', 'Rebalanceo', 'Costes', 'Comunes'].map((t) => <th key={t} scope="col" className="th">{t}</th>)}</tr></thead>
            <tbody>
              {data.marcos.map((m) => (
                <tr key={m.id} className="border-b border-borde/60 last:border-0">
                  <td className="td"><Link to={`/marcos/${m.id}`} className="text-acento hover:underline">{m.nombre}</Link></td>
                  <td className="td">{m.universo_etiqueta ?? resumenUniverso(m.universo)}</td>
                  <td className="td">{fechaCorta(m.fecha_inicio)} → {fechaCorta(m.fecha_fin)}</td>
                  <td className="td">{m.frecuencia_rebalanceo ?? 'propia de cada estrategia'}</td>
                  <td className="td num">{m.comision} / {m.slippage}</td>
                  <td className="td num text-xs">{Object.keys(m.parametros_comunes).join(', ') || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
