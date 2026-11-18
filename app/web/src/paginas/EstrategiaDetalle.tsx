/**
 * Detalle de estrategia: código fuente (solo lectura, resaltado) y formulario para lanzar un experimento
 * o un barrido de parámetros.
 * Responsabilidad: formulario autogenerado desde el JSON Schema de los parámetros + ajustes generales del
 * experimento (universo, fechas, capital, costes, frecuencia, benchmark, semilla); valida en cliente y
 * envía a la API; tras lanzar navega al detalle del experimento (o a la lista si es un barrido).
 * Se relaciona con: componentes/FormularioSchema.tsx, api/hooks.ts, paginas/ExperimentoDetalle.tsx.
 */
import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useEstrategia, useLanzarBarrido, useLanzarExperimento, useUniversos } from '../api/hooks'
import type { ConfiguracionExperimento, Frecuencia } from '../api/tipos'
import { CajaError, Cargando } from '../componentes/Estado'
import FormularioSchema, { validarValores, valoresPorDefecto, type Valores } from '../componentes/FormularioSchema'
import { Codigo } from '../componentes/Piezas'

const UNIVERSO_INICIAL = 'AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, XOM, JNJ, PG, KO'

/** «sp500» o «sp500:30» (nombre de universo) frente a una lista de tickers separados por comas/espacios. */
export function interpretarUniverso(texto: string, nombresUniversos: string[]): string | string[] {
  const limpio = texto.trim()
  const nombre = limpio.split(':')[0].toLowerCase()
  if (/^[a-z0-9_-]+(:\d+)?$/i.test(limpio) && nombresUniversos.includes(nombre)) return limpio.toLowerCase()
  return limpio.split(/[\s,;]+/).filter(Boolean).map((t) => t.toUpperCase())
}

/** Líneas `parametro=1,2,3` -> `{parametro: [1,2,3]}`; devuelve el error de la primera línea inválida. */
export function interpretarRejilla(texto: string): { rejilla: Record<string, unknown[]>; error?: string } {
  const rejilla: Record<string, unknown[]> = {}
  for (const linea of texto.split('\n').map((l) => l.trim()).filter(Boolean)) {
    const [clave, resto] = linea.split('=')
    const valores = (resto ?? '').split(',').map((v) => v.trim()).filter(Boolean)
    if (!clave?.trim() || valores.length === 0) return { rejilla, error: `Línea inválida: «${linea}» (usa parametro=1,2,3)` }
    rejilla[clave.trim()] = valores.map((v) => (v !== '' && !Number.isNaN(Number(v)) ? Number(v) : v))
  }
  return { rejilla }
}

export default function EstrategiaDetalle() {
  const { nombre } = useParams()
  const navegar = useNavigate()
  const { data: estrategia, isPending, error } = useEstrategia(nombre)
  const { data: universos } = useUniversos()
  const lanzar = useLanzarExperimento()
  const barrido = useLanzarBarrido()

  const [parametros, setParametros] = useState<Valores>({})
  const [universo, setUniverso] = useState(UNIVERSO_INICIAL)
  const [inicio, setInicio] = useState('2018-01-01')
  const [fin, setFin] = useState('2024-12-31')
  const [capital, setCapital] = useState(100000)
  const [comision, setComision] = useState(0.001)
  const [slippage, setSlippage] = useState(0.001)
  const [frecuencia, setFrecuencia] = useState<Frecuencia | ''>('')
  const [benchmark, setBenchmark] = useState('SPY')
  const [semilla, setSemilla] = useState(42)
  const [textoRejilla, setTextoRejilla] = useState('')

  useEffect(() => {
    if (estrategia) setParametros(valoresPorDefecto(estrategia.parametros_schema))
  }, [estrategia])

  if (isPending) return <Cargando />
  if (error) return <CajaError error={error} />

  const erroresParametros = validarValores(estrategia.parametros_schema, parametros)
  const { rejilla, error: errorRejilla } = interpretarRejilla(textoRejilla)
  const hayBarrido = Object.keys(rejilla).length > 0
  const invalido = Object.keys(erroresParametros).length > 0 || !!errorRejilla

  const construirConfig = (): ConfiguracionExperimento => ({
    estrategia: estrategia.nombre,
    parametros,
    universo: interpretarUniverso(universo, (universos?.universos ?? []).map((u) => u.nombre)),
    fecha_inicio: inicio,
    fecha_fin: fin,
    capital_inicial: capital,
    comision,
    slippage,
    frecuencia_rebalanceo: frecuencia || null,
    benchmark: benchmark.trim().toUpperCase(),
    semilla,
  })

  const enviar = (e: React.FormEvent) => {
    e.preventDefault()
    if (invalido) return
    if (hayBarrido) {
      barrido.mutate({ config: construirConfig(), rejilla }, { onSuccess: () => navegar('/experimentos') })
    } else {
      lanzar.mutate(construirConfig(), { onSuccess: (r) => navegar(`/experimentos/${r.id}`) })
    }
  }
  const errorEnvio = lanzar.error ?? barrido.error

  return (
    <div className="space-y-5">
      <header>
        <p className="text-xs text-suave"><Link to="/estrategias" className="hover:text-acento">Estrategias</Link> /</p>
        <h1 className="num">{estrategia.nombre}</h1>
        <p className="text-suave">{estrategia.descripcion}</p>
        {estrategia.archivo && <p className="num mt-1 text-xs text-suave">{estrategia.archivo}</p>}
      </header>

      <div className="grid gap-5 xl:grid-cols-2">
        <section aria-labelledby="h-codigo" className="space-y-2">
          <h2 id="h-codigo">Código fuente</h2>
          <Codigo fuente={estrategia.codigo_fuente} />
        </section>

        <form onSubmit={enviar} className="tarjeta space-y-4 p-4" aria-labelledby="h-lanzar">
          <h2 id="h-lanzar">Lanzar experimento</h2>
          <FormularioSchema esquema={estrategia.parametros_schema} valores={parametros} onChange={setParametros} />

          <fieldset className="grid gap-3 border-t border-borde pt-3 sm:grid-cols-2">
            <legend className="sr-only">Ajustes del experimento</legend>
            <div className="sm:col-span-2">
              <label htmlFor="universo" className="font-medium">Universo</label>
              <textarea id="universo" className="campo num mt-1" rows={2} value={universo} onChange={(e) => setUniverso(e.target.value)} aria-describedby="ayuda-universo" />
              <p id="ayuda-universo" className="mt-1 text-xs text-suave">
                Tickers separados por comas o un universo ({(universos?.universos ?? []).map((u) => `${u.nombre}:N`).join(', ') || 'sp500:30'}).
              </p>
            </div>
            <Campo id="inicio" etiqueta="Inicio"><input id="inicio" type="date" className="campo" value={inicio} onChange={(e) => setInicio(e.target.value)} /></Campo>
            <Campo id="fin" etiqueta="Fin"><input id="fin" type="date" className="campo" value={fin} onChange={(e) => setFin(e.target.value)} /></Campo>
            <Campo id="capital" etiqueta="Capital inicial (USD)"><input id="capital" type="number" min={1} className="campo num" value={capital} onChange={(e) => setCapital(Number(e.target.value))} /></Campo>
            <Campo id="frecuencia" etiqueta="Rebalanceo">
              <select id="frecuencia" className="campo" value={frecuencia} onChange={(e) => setFrecuencia(e.target.value as Frecuencia | '')}>
                <option value="">Sugerido ({estrategia.frecuencia_rebalanceo})</option>
                <option value="daily">Diario</option><option value="weekly">Semanal</option>
                <option value="monthly">Mensual</option><option value="quarterly">Trimestral</option>
              </select>
            </Campo>
            <Campo id="comision" etiqueta="Comisión (fracción)"><input id="comision" type="number" min={0} step="any" className="campo num" value={comision} onChange={(e) => setComision(Number(e.target.value))} /></Campo>
            <Campo id="slippage" etiqueta="Slippage (fracción)"><input id="slippage" type="number" min={0} step="any" className="campo num" value={slippage} onChange={(e) => setSlippage(Number(e.target.value))} /></Campo>
            <Campo id="benchmark" etiqueta="Benchmark"><input id="benchmark" className="campo num" value={benchmark} onChange={(e) => setBenchmark(e.target.value)} /></Campo>
            <Campo id="semilla" etiqueta="Semilla"><input id="semilla" type="number" className="campo num" value={semilla} onChange={(e) => setSemilla(Number(e.target.value))} /></Campo>
          </fieldset>

          <details className="border-t border-borde pt-3">
            <summary className="cursor-pointer font-medium">Barrido de parámetros (opcional)</summary>
            <label htmlFor="rejilla" className="mt-2 block text-xs text-suave">Una línea por parámetro: <code>top_k=3,5,8</code>. Se ejecuta el producto cartesiano.</label>
            <textarea id="rejilla" className="campo num mt-1" rows={3} value={textoRejilla} onChange={(e) => setTextoRejilla(e.target.value)} placeholder="top_k=3,5,8" aria-invalid={!!errorRejilla} />
            {errorRejilla && <p className="mt-1 text-xs text-neg">{errorRejilla}</p>}
          </details>

          {errorEnvio && <CajaError error={errorEnvio} titulo="No se pudo lanzar" />}
          <button type="submit" className="boton boton-primario" disabled={invalido || lanzar.isPending || barrido.isPending}>
            {lanzar.isPending || barrido.isPending ? 'Lanzando…' : hayBarrido ? 'Lanzar barrido' : 'Lanzar experimento'}
          </button>
        </form>
      </div>
    </div>
  )
}

function Campo({ id, etiqueta, children }: { id: string; etiqueta: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="font-medium">{etiqueta}</label>
      {children}
    </div>
  )
}
