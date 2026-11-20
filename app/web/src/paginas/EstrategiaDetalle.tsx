/**
 * Detalle de estrategia: código fuente (solo lectura, resaltado) y formularios para lanzar un experimento
 * o un barrido de parámetros.
 * Responsabilidad: formulario autogenerado desde el JSON Schema de los parámetros + ajustes generales del
 * experimento (universo, fechas, capital, costes, frecuencia, benchmark, semilla); valida en cliente y
 * envía a la API; tras lanzar un experimento navega a su detalle. El barrido lo gestiona
 * componentes/LanzadorBarrido.tsx con la configuración base de este formulario.
 * Se relaciona con: componentes/FormularioSchema.tsx, componentes/LanzadorBarrido.tsx, api/hooks.ts,
 * paginas/ExperimentoDetalle.tsx y paginas/BarridoDetalle.tsx.
 */
import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useEstrategia, useLanzarExperimento, useUniversos } from '../api/hooks'
import type { ConfiguracionExperimento, Frecuencia } from '../api/tipos'
import { CajaError, Cargando } from '../componentes/Estado'
import FormularioSchema, { validarValores, valoresPorDefecto, type Valores } from '../componentes/FormularioSchema'
import LanzadorBarrido from '../componentes/LanzadorBarrido'
import { Codigo } from '../componentes/Piezas'

const UNIVERSO_INICIAL = 'AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, XOM, JNJ, PG, KO'

/**
 * «sp500» o «sp500:30» (nombre de universo) frente a una lista de tickers separados por comas/espacios.
 * Los universos dinámicos (`sp500_pit`) no admiten límite: se descarta un «:N» si se escribió.
 */
export function interpretarUniverso(texto: string, nombresUniversos: string[], dinamicos: string[] = []): string | string[] {
  const limpio = texto.trim()
  const nombre = limpio.split(':')[0].toLowerCase()
  if (/^[a-z0-9_-]+(:\d+)?$/i.test(limpio) && nombresUniversos.includes(nombre)) {
    return dinamicos.includes(nombre) ? nombre : limpio.toLowerCase()
  }
  return limpio.split(/[\s,;]+/).filter(Boolean).map((t) => t.toUpperCase())
}

export default function EstrategiaDetalle() {
  const { nombre } = useParams()
  const navegar = useNavigate()
  const { data: estrategia, isPending, error } = useEstrategia(nombre)
  const { data: universos } = useUniversos()
  const lanzar = useLanzarExperimento()

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

  useEffect(() => {
    if (!estrategia) return
    setParametros(valoresPorDefecto(estrategia.parametros_schema))
    // Las estrategias de sección cruzada proponen el universo point-in-time (sin sesgo de supervivencia)
    if (estrategia.universo_sugerido) setUniverso(estrategia.universo_sugerido)
  }, [estrategia])

  if (isPending) return <Cargando />
  if (error) return <CajaError error={error} />

  const listaUniversos = universos?.universos ?? []
  const nombresUniversos = listaUniversos.map((u) => u.nombre)
  const dinamicos = listaUniversos.filter((u) => u.dinamico).map((u) => u.nombre)
  const universoEscrito = universo.trim().split(':')[0].toLowerCase()
  const esDinamico = dinamicos.includes(universoEscrito)

  const erroresParametros = validarValores(estrategia.parametros_schema, parametros)
  const invalido = Object.keys(erroresParametros).length > 0

  const construirConfig = (): ConfiguracionExperimento => ({
    estrategia: estrategia.nombre,
    parametros,
    universo: interpretarUniverso(universo, nombresUniversos, dinamicos),
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
    lanzar.mutate(construirConfig(), { onSuccess: (r) => navegar(`/experimentos/${r.id}`) })
  }

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
              {listaUniversos.length > 0 && (
                <select
                  id="universo-predefinido" aria-label="Universo predefinido" className="campo mt-1"
                  value={nombresUniversos.includes(universoEscrito) && !universo.includes(':') ? universoEscrito : ''}
                  onChange={(e) => e.target.value && setUniverso(e.target.value)}
                >
                  <option value="">Tickers a mano…</option>
                  {listaUniversos.map((u) => (
                    <option key={u.nombre} value={u.nombre}>
                      {u.nombre} ({u.n_tickers} tickers{u.dinamico ? ', por fecha' : ', lista fija'})
                    </option>
                  ))}
                </select>
              )}
              <textarea id="universo" className="campo num mt-1" rows={2} value={universo} onChange={(e) => setUniverso(e.target.value)} aria-describedby="ayuda-universo" />
              <p id="ayuda-universo" className="mt-1 text-xs text-suave">
                Tickers separados por comas o un universo ({listaUniversos.map((u) => (u.dinamico ? u.nombre : `${u.nombre}:N`)).join(', ') || 'sp500:30'}).
              </p>
              {esDinamico && (
                <p className="mt-1 text-xs text-suave">
                  Universo point-in-time: en cada rebalanceo solo entran los miembros del índice en cada fecha (con 12 meses de historia),
                  lo que evita el sesgo de supervivencia. Los deslistados sin precios en la fuente no pueden elegirse; el experimento lo avisa.
                </p>
              )}
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

          {lanzar.error && <CajaError error={lanzar.error} titulo="No se pudo lanzar" />}
          <button type="submit" className="boton boton-primario" disabled={invalido || lanzar.isPending}>
            {lanzar.isPending ? 'Lanzando…' : 'Lanzar experimento'}
          </button>
        </form>
      </div>

      <LanzadorBarrido esquema={estrategia.parametros_schema} construirBase={construirConfig} parametrosValidos={!invalido} />
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
