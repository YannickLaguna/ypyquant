/**
 * Tarjeta pequeña con el estado de las membresías del S&P 500 (universo `sp500_pit`, paper interno y bot).
 * Responsabilidad: mostrar cuántos miembros vigentes hay, cuándo se comprobó por última vez contra las fuentes
 * (Wikipedia + SPDR SPY), cuántos cambios se han aplicado, los pendientes de revisar (las fuentes discrepan) y los avisos
 * de calidad; todo con el color de aviso si hay algo que revisar. No muestra nada si la API no devuelve el estado.
 * Se relaciona con: api/universos.ts (hook y tipos) y paginas/PaperInterno.tsx (donde se coloca).
 */
import { useEstadoMembresias } from '../api/universos'
import { entero, fechaHora } from '../lib/formato'

const TEXTO_RESULTADO: Record<string, string> = {
  al_dia: 'al día', actualizado: 'actualizado', pendiente: 'con cambios pendientes de revisar', error: 'error al comprobar',
}

export default function TarjetaMembresias() {
  const e = useEstadoMembresias().data
  if (!e || e.n_vigentes === undefined) return null
  const avisos = [...(e.avisos_calidad ?? []), ...(e.error ? [`Error de la última comprobación: ${e.error}`] : [])]
  const hayQueRevisar = e.pendientes_de_revisar.length > 0 || avisos.length > 0
  return (
    <section aria-label="Membresías del S&P 500" className={`rounded-lg border p-3 text-sm ${hayQueRevisar ? 'border-aviso/60' : 'border-borde'}`}>
      <p className="text-suave">
        <span className="font-medium text-texto">Membresías del S&P 500:</span>{' '}
        <span className="num">{entero(e.n_vigentes)}</span> vigentes
        {e.ultima_comprobacion ? <> · comprobadas {fechaHora(e.ultima_comprobacion)} ({TEXTO_RESULTADO[e.resultado ?? ''] ?? e.resultado})</> : <> · sin comprobar todavía</>}
        {e.antiguedad_dias !== null && <> · última comprobación correcta hace <span className="num">{entero(e.antiguedad_dias)}</span> d</>}
        {' '}· <span className="num">{entero(e.cambios_aplicados_total)}</span> cambios aplicados
        {!e.activo && <> · comprobación automática desactivada (YPYQUANT_MEMBRESIAS_AUTO=0)</>}.
      </p>
      {e.pendientes_de_revisar.length > 0 && (
        <ul aria-label="Pendientes de revisar" className="mt-2 list-disc pl-5 text-aviso">
          {e.pendientes_de_revisar.map((c) => (
            <li key={`${c.tipo}-${c.simbolo}`}>{c.tipo} de {c.simbolo} ({c.fecha}): {c.motivo_pendiente}</li>
          ))}
        </ul>
      )}
      {avisos.map((a) => <p key={a} className="mt-2 text-aviso">{a}</p>)}
    </section>
  )
}
