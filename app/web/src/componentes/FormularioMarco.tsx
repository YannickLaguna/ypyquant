/**
 * Formulario de un marco de comparación: el entorno que se fija una vez para comparar estrategias ceteris paribus.
 * Responsabilidad: universo (con `sp500_pit` sugerido), fechas, frecuencia de rebalanceo, costes, benchmark, capital,
 * semilla, historia previa y los parámetros comunes (cada uno se activa o no; solo se aplican a las estrategias
 * que lo declaran). Valida en cliente y entrega un `PeticionMarco` por `alGuardar`. Sirve para crear y para editar.
 * Se relaciona con: lib/marcos.ts (COMUNES, comunesActivos), paginas/Marcos.tsx y paginas/MarcoDetalle.tsx,
 * paginas/EstrategiaDetalle.tsx (`interpretarUniverso`, mismo formato de universo) y api/hooks.ts (useUniversos).
 */
import { useState } from 'react'
import { useUniversos } from '../api/hooks'
import type { Frecuencia, Marco, PeticionMarco } from '../api/tipos'
import { COMUNES, comunesActivos, type EstadoComunes } from '../lib/marcos'
import { interpretarUniverso } from '../paginas/EstrategiaDetalle'

interface Props {
  /** Marco a editar; sin él se crea uno nuevo con valores sugeridos. */
  marco?: Marco
  alGuardar: (marco: PeticionMarco) => void
  guardando?: boolean
  textoBoton?: string
}

function comunesIniciales(marco?: Marco): EstadoComunes {
  const estado: EstadoComunes = {}
  for (const c of COMUNES) {
    const valor = marco?.parametros_comunes[c.nombre]
    estado[c.nombre] = { activo: valor !== undefined, valor: (valor as number | string | undefined) ?? c.defecto }
  }
  return estado
}

export default function FormularioMarco({ marco, alGuardar, guardando = false, textoBoton = 'Guardar marco' }: Props) {
  const { data: universos } = useUniversos()
  const [nombre, setNombre] = useState(marco?.nombre ?? '')
  const [notas, setNotas] = useState(marco?.notas ?? '')
  const [universo, setUniverso] = useState(marco ? (marco.universo_etiqueta ?? (Array.isArray(marco.universo) ? marco.universo.join(', ') : marco.universo)) : 'sp500_pit')
  const [inicio, setInicio] = useState(marco?.fecha_inicio ?? '2015-01-01')
  const [fin, setFin] = useState(marco?.fecha_fin ?? '2024-12-31')
  const [frecuencia, setFrecuencia] = useState<Frecuencia | ''>(marco ? (marco.frecuencia_rebalanceo ?? '') : 'monthly')
  const [comision, setComision] = useState(marco?.comision ?? 0.001)
  const [slippage, setSlippage] = useState(marco?.slippage ?? 0.001)
  const [benchmark, setBenchmark] = useState(marco?.benchmark ?? 'SPY')
  const [capital, setCapital] = useState(marco?.capital_inicial ?? 100000)
  const [semilla, setSemilla] = useState(marco?.semilla ?? 42)
  const [historia, setHistoria] = useState(marco?.dias_historia_previa ?? 1100)
  const [comunes, setComunes] = useState<EstadoComunes>(comunesIniciales(marco))
  const [errores, setErrores] = useState<string[]>([])

  const lista = universos?.universos ?? []
  const nombres = lista.map((u) => u.nombre)
  const dinamicos = lista.filter((u) => u.dinamico).map((u) => u.nombre)

  const enviar = (e: React.FormEvent) => {
    e.preventDefault()
    const problemas: string[] = []
    if (!nombre.trim()) problemas.push('El nombre es obligatorio')
    if (!inicio || !fin || inicio >= fin) problemas.push('La fecha de inicio debe ser anterior a la de fin')
    if (!universo.trim()) problemas.push('El universo no puede estar vacío')
    setErrores(problemas)
    if (problemas.length) return
    alGuardar({
      nombre: nombre.trim(), notas, universo: interpretarUniverso(universo, nombres, dinamicos),
      fecha_inicio: inicio, fecha_fin: fin, frecuencia_rebalanceo: frecuencia || null, comision, slippage,
      benchmark: benchmark.trim().toUpperCase(), capital_inicial: capital, semilla, dias_historia_previa: historia,
      parametros_comunes: comunesActivos(comunes),
    })
  }

  return (
    <form onSubmit={enviar} className="tarjeta space-y-4 p-4" aria-label="Marco de comparación">
      {errores.length > 0 && (
        <div role="alert" className="rounded-lg border border-neg/50 bg-neg/10 p-3 text-neg">{errores.map((m) => <p key={m}>{m}</p>)}</div>
      )}
      <div className="grid gap-3 sm:grid-cols-2">
        <Campo id="marco-nombre" etiqueta="Nombre"><input id="marco-nombre" className="campo" value={nombre} onChange={(e) => setNombre(e.target.value)} disabled={!!marco} /></Campo>
        <Campo id="marco-notas" etiqueta="Notas"><input id="marco-notas" className="campo" value={notas} onChange={(e) => setNotas(e.target.value)} /></Campo>
        <div className="sm:col-span-2">
          <label htmlFor="marco-universo" className="font-medium">Universo</label>
          {lista.length > 0 && (
            <select aria-label="Universo predefinido" className="campo mt-1" value={nombres.includes(universo.trim().toLowerCase()) ? universo.trim().toLowerCase() : ''} onChange={(e) => e.target.value && setUniverso(e.target.value)}>
              <option value="">Tickers a mano…</option>
              {lista.map((u) => <option key={u.nombre} value={u.nombre}>{u.nombre} ({u.n_tickers} tickers{u.dinamico ? ', por fecha' : ', lista fija'})</option>)}
            </select>
          )}
          <textarea id="marco-universo" className="campo num mt-1" rows={2} value={universo} onChange={(e) => setUniverso(e.target.value)} />
          <p className="mt-1 text-xs text-suave">
            Se sugiere <span className="num">sp500_pit</span> (miembros del índice en cada fecha, sin sesgo de supervivencia). Un universo con nombre de
            archivo se congela como lista de tickers al guardar el marco.
          </p>
        </div>
        <Campo id="marco-inicio" etiqueta="Inicio"><input id="marco-inicio" type="date" className="campo" value={inicio} onChange={(e) => setInicio(e.target.value)} /></Campo>
        <Campo id="marco-fin" etiqueta="Fin"><input id="marco-fin" type="date" className="campo" value={fin} onChange={(e) => setFin(e.target.value)} /></Campo>
        <Campo id="marco-frecuencia" etiqueta="Rebalanceo">
          <select id="marco-frecuencia" className="campo" value={frecuencia} onChange={(e) => setFrecuencia(e.target.value as Frecuencia | '')}>
            <option value="monthly">Mensual</option><option value="quarterly">Trimestral</option>
            <option value="weekly">Semanal</option><option value="daily">Diario</option>
            <option value="">La sugerida por cada estrategia (no ceteris paribus)</option>
          </select>
        </Campo>
        <Campo id="marco-capital" etiqueta="Capital inicial (USD)"><input id="marco-capital" type="number" min={1} className="campo num" value={capital} onChange={(e) => setCapital(Number(e.target.value))} /></Campo>
        <Campo id="marco-comision" etiqueta="Comisión (fracción)"><input id="marco-comision" type="number" min={0} step="any" className="campo num" value={comision} onChange={(e) => setComision(Number(e.target.value))} /></Campo>
        <Campo id="marco-slippage" etiqueta="Slippage (fracción)"><input id="marco-slippage" type="number" min={0} step="any" className="campo num" value={slippage} onChange={(e) => setSlippage(Number(e.target.value))} /></Campo>
        <Campo id="marco-benchmark" etiqueta="Benchmark"><input id="marco-benchmark" className="campo num" value={benchmark} onChange={(e) => setBenchmark(e.target.value)} /></Campo>
        <Campo id="marco-semilla" etiqueta="Semilla"><input id="marco-semilla" type="number" className="campo num" value={semilla} onChange={(e) => setSemilla(Number(e.target.value))} /></Campo>
        <Campo id="marco-historia" etiqueta="Historia previa (días naturales)"><input id="marco-historia" type="number" min={0} className="campo num" value={historia} onChange={(e) => setHistoria(Number(e.target.value))} /></Campo>
      </div>

      <fieldset className="space-y-2 border-t border-borde pt-3">
        <legend className="font-medium">Parámetros comunes</legend>
        <p className="text-xs text-suave">
          Activa los que quieras fijar para todas las estrategias. Cada uno se aplica solo a las estrategias que lo declaran (momentum, por
          ejemplo, no tiene ventana) y el resultado te dice cuáles se aplicaron y cuáles no. El marco prevalece sobre el valor propio.
        </p>
        {COMUNES.map((c) => (
          <div key={c.nombre} className="flex flex-wrap items-center gap-3">
            <label className="flex w-72 items-center gap-2">
              <input type="checkbox" className="accent-[rgb(var(--acento))]" checked={comunes[c.nombre].activo} aria-label={`Fijar ${c.nombre}`}
                onChange={(e) => setComunes({ ...comunes, [c.nombre]: { ...comunes[c.nombre], activo: e.target.checked } })} />
              <span><span className="num">{c.nombre}</span> <span className="text-xs text-suave">{c.etiqueta}</span></span>
            </label>
            {c.tipo === 'opcion' ? (
              <select aria-label={`Valor de ${c.nombre}`} className="campo w-40" disabled={!comunes[c.nombre].activo} value={comunes[c.nombre].valor}
                onChange={(e) => setComunes({ ...comunes, [c.nombre]: { ...comunes[c.nombre], valor: e.target.value } })}>
                {c.opciones!.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            ) : (
              <input aria-label={`Valor de ${c.nombre}`} type="number" step="any" className="campo num w-40" disabled={!comunes[c.nombre].activo} value={comunes[c.nombre].valor}
                onChange={(e) => setComunes({ ...comunes, [c.nombre]: { ...comunes[c.nombre], valor: Number(e.target.value) } })} />
            )}
            <span className="text-xs text-suave">{c.ayuda}</span>
          </div>
        ))}
      </fieldset>

      <button type="submit" className="boton boton-primario" disabled={guardando}>{guardando ? 'Guardando…' : textoBoton}</button>
    </form>
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
