/**
 * Bloque plegable con la procedencia de un experimento (con qué se obtuvo el resultado).
 * Responsabilidad: mostrar commit, si el árbol estaba sucio, versiones (ypyquant, Python, numpy,
 * pandas, cvxpy), hash SHA-256 de los precios y fecha de ejecución; avisa cuando el árbol estaba sucio
 * (el commit no basta para reproducir) y explica el valor `null` de los experimentos anteriores a v0.5.0.
 * Para los experimentos importados del TFM (`procedencia.origen`) muestra el origen, el commit del repo
 * `tfm`, los Parquet fuente con su SHA-256 y la fecha de importación.
 * Se relaciona con: paginas/ExperimentoDetalle.tsx (lo monta), api/tipos.ts (Procedencia) y
 * docs/ui.md (sección «Procedencia de un experimento»).
 */
import type { Procedencia } from '../api/tipos'
import { fechaHora } from '../lib/formato'

export default function ProcedenciaExperimento({ procedencia }: { procedencia: Procedencia | null | undefined }) {
  if (!procedencia) {
    return (
      <p className="text-xs text-suave">Sin procedencia registrada (experimento anterior a la v0.5.0).</p>
    )
  }
  const filas: [string, string | null | undefined][] = procedencia.origen ? [
    // Experimento importado: no hay commit ni precios propios; se muestra de dónde vienen los números
    ['Origen', procedencia.origen],
    ['Commit del repo de origen', procedencia.commit_tfm ?? 'no disponible'],
    ...Object.entries(procedencia.archivos ?? {}).map(([nombre, a]): [string, string] => [`${nombre} (SHA-256)`, `${a.sha256} · ${a.ruta}`]),
    ['Importado', procedencia.importado ? fechaHora(procedencia.importado) : null],
  ] : [
    ['Commit', procedencia.commit ?? 'no disponible (la librería no está en un repo git)'],
    ['Árbol de trabajo', procedencia.arbol_sucio === null ? 'desconocido' : procedencia.arbol_sucio ? 'sucio (cambios sin commitear)' : 'limpio'],
    ['ypyquant', procedencia.version_ypyquant],
    ['Python', procedencia.python],
    ['numpy', procedencia.numpy],
    ['pandas', procedencia.pandas],
    ['cvxpy', procedencia.cvxpy],
    ['Hash de los precios (SHA-256)', procedencia.hash_precios],
    ['Ejecutado', procedencia.ejecutado ? fechaHora(procedencia.ejecutado) : null],
  ]
  return (
    <details className="tarjeta p-3">
      <summary className="cursor-pointer font-medium">
        Procedencia
        <span className="num ml-2 text-xs text-suave">
          {procedencia.origen ?? (procedencia.commit ? procedencia.commit.slice(0, 8) : 'sin commit')}
          {procedencia.arbol_sucio ? ' · sucio' : ''}
        </span>
        {procedencia.arbol_sucio && <span className="ml-2 text-xs text-aviso">árbol sucio</span>}
      </summary>
      <dl className="mt-3 grid gap-x-4 gap-y-1 text-sm sm:grid-cols-[max-content_1fr]">
        {filas.filter(([, v]) => v).map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-suave">{k}</dt>
            <dd className="num break-all">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-suave">
        Dos ejecuciones con el mismo hash de precios vieron exactamente los mismos datos. La procedencia no entra en el identificador del experimento.
      </p>
    </details>
  )
}
