/**
 * Paneles de validación estadística de un barrido: «Validar» (DSR, PBO, rachas) y «Walk-forward».
 * Responsabilidad:
 * - `PanelValidar`: botón que llama a `POST /validar` (síncrono) y muestra DSR y PBO con semáforo
 *   orientativo, histograma de λ, dispersión IS/OOS con su pendiente, prueba de rachas y avisos.
 * - `PanelWalkForward`: formulario ventana/paso (días naturales) -> curva de equity fuera de muestra,
 *   métricas y tabla de elecciones por fecha.
 * El resultado se lee siempre del detalle del barrido (la API lo persiste), de modo que sobrevive a recargas.
 * Se relaciona con: api/hooks.ts (mutaciones), api/tipos.ts, componentes/GraficosValidacion.tsx,
 * componentes/Graficos.tsx, componentes/TablaOrdenable.tsx y paginas/BarridoDetalle.tsx (lo monta).
 */
import { useMemo, useState } from 'react'
import { useValidarBarrido, useWalkForwardBarrido } from '../api/hooks'
import type { DetalleBarrido, ResultadoValidacion, ResultadoWalkForward } from '../api/tipos'
import { semaforoDsr, semaforoPbo } from '../lib/barridos'
import { claseSigno, fechaCorta, fechaHora, porcentaje, ratio } from '../lib/formato'
import { colorToken } from '../lib/tema'
import { CajaError } from './Estado'
import { GraficoSeries, type SerieGrafico } from './Graficos'
import { DispersionIsOos, Histograma, TarjetaResultado } from './GraficosValidacion'
import TablaOrdenable, { type Columna } from './TablaOrdenable'

/** Hay experimentos sin terminar: la validación devolvería 409. */
export const barridoSinTerminar = (b: DetalleBarrido) => b.progreso.pendientes + b.progreso.en_curso > 0 || b.estado === 'pendiente' || b.estado === 'en_curso'

export function PanelValidar({ barrido }: { barrido: DetalleBarrido }) {
  const validar = useValidarBarrido(barrido.id)
  const resultado: ResultadoValidacion | null = barrido.validacion
  return (
    <section aria-labelledby="h-validar" className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="h-validar">Validar (DSR, PBO, rachas)</h2>
        <button type="button" className="boton boton-primario" onClick={() => validar.mutate(undefined)} disabled={validar.isPending || barridoSinTerminar(barrido)}>
          {validar.isPending ? 'Validando…' : resultado ? 'Validar de nuevo' : 'Validar'}
        </button>
      </div>
      <p className="text-xs text-suave">
        Contrasta si el mejor resultado sobrevive a haber probado {barrido.progreso.total} combinaciones. Los semáforos son orientativos (docs/validacion.md): no sustituyen al juicio.
      </p>
      {barridoSinTerminar(barrido) && <p className="text-xs text-aviso">Espera a que terminen todos los experimentos del barrido.</p>}
      {validar.error && <CajaError error={validar.error} titulo="No se pudo validar" />}
      {resultado && <ResultadoValidar r={resultado} />}
    </section>
  )
}

function ResultadoValidar({ r }: { r: ResultadoValidacion }) {
  const pbo = r.pbo
  return (
    <div className="space-y-3">
      <p className="text-xs text-suave">
        {fechaHora(r.fecha)} · {r.n_observaciones} observaciones ({fechaCorta(r.fecha_inicio)} → {fechaCorta(r.fecha_fin)}) · elegida: <span className="num text-texto">{r.elegida.etiqueta}</span>
        {r.alternativa_meseta && !r.alternativa_meseta.es_la_elegida && <> · alternativa por meseta: <span className="num text-texto">{r.alternativa_meseta.etiqueta}</span></>}
      </p>
      {r.avisos.length > 0 && (
        <div role="status" className="rounded-lg border border-aviso/50 bg-aviso/10 p-3 text-aviso">
          {r.avisos.map((a) => <p key={a}>{a}</p>)}
        </div>
      )}
      <div className="grid gap-3 md:grid-cols-3">
        <TarjetaResultado
          etiqueta="DSR (Sharpe deflactado)"
          valor={ratio(r.dsr.dsr, 3)}
          semaforo={semaforoDsr(r.dsr.dsr)}
          ayuda={`Probabilidad de que el Sharpe de la elegida supere al que daría el azar tras ${r.dsr.n_pruebas} pruebas. Verde > 0,95. Sharpe anualizado ${ratio(r.dsr.sr_anualizado, 2)}, T = ${r.dsr.T}.`}
        />
        <TarjetaResultado
          etiqueta="PBO (prob. de sobreajuste)"
          valor={pbo ? ratio(pbo.pbo, 3) : '—'}
          semaforo={semaforoPbo(pbo?.pbo)}
          ayuda={pbo ? `Frecuencia con que la mejor en muestra queda bajo la mediana fuera de muestra (${pbo.n_bloques} bloques). Verde < 0,10; ámbar 0,10–0,20; rojo > 0,20.` : 'No calculable (pocas observaciones o combinaciones).'}
        />
        <div className="tarjeta space-y-1 p-3">
          <p className="text-xs uppercase tracking-wide text-suave">Prueba de rachas</p>
          {r.rachas ? (
            <>
              <p className="num text-2xl font-semibold">p = {ratio(r.rachas.p_valor, 3)}</p>
              <p className="text-xs text-suave">
                {r.rachas.rachas} rachas (esperadas {ratio(r.rachas.mu, 1)}), z = {ratio(r.rachas.z, 2)}. p &lt; 0,05 indica dependencia en el signo de los retornos de la elegida.
              </p>
            </>
          ) : (
            <p className="text-suave">No calculable.</p>
          )}
        </div>
      </div>
      {pbo && (
        <div className="grid gap-3 lg:grid-cols-2">
          <Histograma
            titulo="Distribución de λ (logit del rango OOS de la mejor IS)"
            datos={pbo.lambdas}
            referencia={0}
            etiquetaReferencia="λ = 0"
            etiquetaX="λ ≤ 0: la mejor en muestra queda bajo la mediana fuera de muestra"
          />
          <DispersionIsOos pares={pbo.pares_is_oos} pendienteApi={pbo.pendiente} />
        </div>
      )}
    </div>
  )
}

// ---- Walk-forward ---------------------------------------------------------------------------

export function PanelWalkForward({ barrido }: { barrido: DetalleBarrido }) {
  const walk = useWalkForwardBarrido(barrido.id)
  const [ventana, setVentana] = useState(756)
  const [paso, setPaso] = useState(90)
  const resultado: ResultadoWalkForward | null = barrido.walk_forward
  const valido = ventana > 0 && paso > 0

  const series = useMemo<SerieGrafico[]>(
    () =>
      resultado
        ? [{ id: 'oos', nombre: 'Equity OOS (base 1)', color: colorToken('acento'), datos: resultado.curva.fechas.map((f, i) => ({ time: f, value: resultado.curva.equity[i] })) }]
        : [],
    [resultado],
  )
  const columnas: Columna<ResultadoWalkForward['elecciones'][number]>[] = [
    { clave: 'fecha', titulo: 'Fecha', valor: (e) => e.fecha, render: (e) => e.fecha },
    { clave: 'etiqueta', titulo: 'Elegida', valor: (e) => e.etiqueta, render: (e) => <span className="num">{e.etiqueta}</span> },
    { clave: 'm', titulo: 'Métrica en entreno', alinear: 'der', valor: (e) => e.metrica_entreno, render: (e) => ratio(e.metrica_entreno, 3) },
  ]
  const m = resultado?.metricas ?? {}

  return (
    <section aria-labelledby="h-wf" className="space-y-3">
      <h2 id="h-wf">Walk-forward</h2>
      <p className="text-xs text-suave">
        Cada <span className="num">paso</span> días elige la mejor combinación con los últimos <span className="num">ventana</span> días y la aplica al tramo siguiente. Ventana y paso en días naturales. Ignora el coste de cambiar de combinación (optimista).
      </p>
      <form
        className="tarjeta flex flex-wrap items-end gap-3 p-3"
        onSubmit={(e) => {
          e.preventDefault()
          if (valido) walk.mutate({ ventana_entreno_dias: ventana, paso_dias: paso })
        }}
      >
        <div className="flex flex-col gap-1">
          <label htmlFor="wf-ventana" className="font-medium">Ventana de entreno (días)</label>
          <input id="wf-ventana" type="number" min={1} className="campo num w-40" value={ventana} onChange={(e) => setVentana(Number(e.target.value))} />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="wf-paso" className="font-medium">Paso (días)</label>
          <input id="wf-paso" type="number" min={1} className="campo num w-32" value={paso} onChange={(e) => setPaso(Number(e.target.value))} />
        </div>
        <button type="submit" className="boton boton-primario" disabled={!valido || walk.isPending || barridoSinTerminar(barrido)}>
          {walk.isPending ? 'Calculando…' : 'Ejecutar walk-forward'}
        </button>
      </form>
      {walk.error && <CajaError error={walk.error} titulo="No se pudo ejecutar el walk-forward" />}
      {resultado && (
        <div className="space-y-3">
          <p className="text-xs text-suave">
            {fechaHora(resultado.fecha)} · ventana {resultado.ventana_entreno_dias} d, paso {resultado.paso_dias} d · {resultado.elecciones.length} elecciones
          </p>
          <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metrica etiqueta="Sharpe anualizado" valor={ratio(m.sharpe_anualizado)} />
            <Metrica etiqueta="CAGR" valor={porcentaje(m.cagr)} clase={claseSigno(m.cagr)} />
            <Metrica etiqueta="Retorno total" valor={porcentaje(m.retorno_total)} clase={claseSigno(m.retorno_total)} />
            <Metrica etiqueta="Máx. drawdown" valor={porcentaje(m.max_drawdown)} clase="text-neg" />
          </dl>
          <GraficoSeries titulo="Equity walk-forward fuera de muestra (base 1)" series={series} altura={240} formatoValor={(v) => ratio(v, 3)} />
          <TablaOrdenable titulo="Elecciones por fecha" columnas={columnas} filas={resultado.elecciones} clave={(e) => e.fecha} ordenInicial={{ clave: 'fecha', descendente: false }} vacio="Sin elecciones" />
        </div>
      )}
    </section>
  )
}

function Metrica({ etiqueta, valor, clase = '' }: { etiqueta: string; valor: string; clase?: string }) {
  return (
    <div className="tarjeta px-3 py-2">
      <dt className="text-xs uppercase tracking-wide text-suave">{etiqueta}</dt>
      <dd className={`num mt-0.5 text-lg font-semibold ${clase}`}>{valor}</dd>
    </div>
  )
}
