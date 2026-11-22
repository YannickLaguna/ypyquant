/**
 * Control del bot de un perfil de Alpaca: vincular la cuenta a un experimento, arrancar/parar y ver el log (redactado).
 * Responsabilidad: elegir el experimento (hechos, no importados del TFM), el modo (`continuo`: bucle cada N segundos;
 * `diario`: una pasada y sale) y el intervalo; mostrar el estado del bot, el vínculo actual (estrategia, tickers, equity
 * inicial) y el log por polling. El experimento se congela al arrancar: una receta, una cuenta. Los errores 409
 * (perfil ocupado / límite blando de 3 bots) y 422 (universo no operable) salen con el mensaje del servidor.
 * Se relaciona con: api/alpaca.ts (hooks y tipos), api/hooks.ts (`useExperimentos`), paginas/PaperTrading.tsx, y
 * paginas/ExperimentoDetalle.tsx (el botón «Llevar a Alpaca» preselecciona el experimento con `?experimento=`).
 */
import { useEffect, useRef, useState } from 'react'
import {
  type ModoBot,
  type PerfilAlpaca,
  useArrancarBotPerfil,
  useEstadoBotPerfil,
  useLogBotPerfil,
  usePararBotPerfil,
} from '../api/alpaca'
import { useExperimentos } from '../api/hooks'
import { dinero, fechaHora, resumenParametros } from '../lib/formato'
import { CajaError, esImportadoTfm } from './Estado'

const MODOS: [ModoBot, string][] = [
  ['continuo', 'Continuo: un bucle que mira cada N segundos y rebalancea cuando toca'],
  ['diario', 'Diario: una sola pasada y sale (para el programador de tareas)'],
]

export default function BotPerfil({ perfil, experimentoInicial }: { perfil: PerfilAlpaca; experimentoInicial?: string }) {
  const estado = useEstadoBotPerfil(perfil.nombre)
  const { data: lista } = useExperimentos({ estado: 'hecho' })
  const arrancar = useArrancarBotPerfil(perfil.nombre)
  const parar = usePararBotPerfil(perfil.nombre)
  const bot = estado.data
  const enMarcha = bot?.en_marcha ?? perfil.bot.en_marcha
  const log = useLogBotPerfil(perfil.nombre, enMarcha)
  const vinculo = bot?.vinculo ?? perfil.vinculo
  const candidatos = (lista?.experimentos ?? []).filter((e) => !esImportadoTfm(e.estrategia))
  const [experimento, setExperimento] = useState(experimentoInicial ?? '')
  const [modo, setModo] = useState<ModoBot>('continuo')
  const [intervalo, setIntervalo] = useState(3600)
  const visor = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (visor.current) visor.current.scrollTop = visor.current.scrollHeight
  }, [log.data?.lineas.length])

  // El experimento elegido, el que ya opera la cuenta o, si no hay nada, el primero de la lista
  const elegido = experimento || vinculo?.origen.id || candidatos[0]?.id || ''
  const lanzar = (e: React.FormEvent) => {
    e.preventDefault()
    if (!elegido) return
    arrancar.mutate({ experimento_id: elegido, modo, intervalo_s: intervalo })
  }

  return (
    <section aria-labelledby="h-bot" className="space-y-3">
      <h2 id="h-bot">Bot de la cuenta {perfil.nombre}</h2>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
        <form onSubmit={lanzar} className="tarjeta space-y-3 p-4">
          <p role="status" className="flex items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${enMarcha ? 'bg-pos' : 'bg-suave'}`} aria-hidden />
            {enMarcha
              ? `En marcha (${bot?.modo ?? 'continuo'}, pid ${bot?.pid ?? '—'})`
              : bot?.codigo_salida != null ? `Detenido (código ${bot.codigo_salida})` : 'Detenido'}
          </p>
          {vinculo && (
            <dl className="rounded-md border border-borde p-2 text-xs" aria-label="Vínculo actual">
              <dt className="text-suave">Opera</dt>
              <dd className="num break-all">{vinculo.origen.id}</dd>
              <dd className="text-suave">
                {vinculo.estrategia} · {vinculo.tickers.length} tickers · rebalanceo {vinculo.frecuencia_rebalanceo ?? 'según estrategia'}
              </dd>
              <dd className="text-suave">
                Alta {fechaHora(vinculo.fecha_alta)} · equity inicial {dinero(vinculo.equity_inicial, 0)}
              </dd>
            </dl>
          )}
          <div>
            <label htmlFor="bot-experimento" className="font-medium">Experimento</label>
            <select id="bot-experimento" className="campo mt-1" value={elegido} disabled={enMarcha} onChange={(e) => setExperimento(e.target.value)}>
              {candidatos.length === 0 && <option value="">No hay experimentos terminados</option>}
              {elegido && !candidatos.some((c) => c.id === elegido) && <option value={elegido}>{elegido}</option>}
              {candidatos.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.estrategia} · {resumenParametros(e.parametros) || 'sin parámetros'} · {e.id}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-suave">Se congela su receta (estrategia, parámetros y entorno): la cuenta opera eso, aunque el experimento cambie después.</p>
          </div>
          <fieldset className="space-y-1" disabled={enMarcha}>
            <legend className="font-medium">Modo</legend>
            {MODOS.map(([valor, texto]) => (
              <label key={valor} className="flex items-start gap-2 text-sm">
                <input type="radio" name={`modo-${perfil.nombre}`} value={valor} checked={modo === valor} onChange={() => setModo(valor)} className="mt-1" />
                <span>{texto}</span>
              </label>
            ))}
          </fieldset>
          {modo === 'continuo' && (
            <div>
              <label htmlFor="bot-intervalo" className="font-medium">Segundos entre iteraciones</label>
              <input id="bot-intervalo" type="number" min={5} className="campo num mt-1" value={intervalo} disabled={enMarcha} onChange={(e) => setIntervalo(Number(e.target.value))} />
            </div>
          )}
          {(arrancar.error || parar.error) && <CajaError error={arrancar.error ?? parar.error} titulo="El bot no respondió" />}
          <div className="flex gap-2">
            <button type="submit" className="boton boton-primario" disabled={enMarcha || arrancar.isPending || !elegido}>Arrancar</button>
            <button type="button" className="boton boton-peligro" disabled={!enMarcha || parar.isPending} onClick={() => parar.mutate()}>Parar</button>
          </div>
        </form>
        <div className="tarjeta flex min-h-[16rem] min-w-0 flex-col p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-suave">Log (claves ocultas)</p>
          <pre ref={visor} role="log" aria-live="off" tabIndex={0} className="num h-64 flex-1 overflow-auto whitespace-pre-wrap break-words text-xs">
            {log.data?.lineas.length ? log.data.lineas.join('\n') : 'Sin salida todavía.'}
          </pre>
        </div>
      </div>
    </section>
  )
}
