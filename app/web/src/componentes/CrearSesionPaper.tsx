/**
 * Panel «Paper interno» de la ficha de un experimento: crea una sesión de simulación forward con la receta congelada.
 * Responsabilidad: pedir nombre, modo de ejecución (apertura siguiente por defecto; «al cierre» es optimista), capital
 * inicial, costes opcionales de la sesión (en %, vacío = los del experimento) y fecha de alta (vacío = hoy); validar los
 * números, mandar `POST /api/paper/interno/sesiones` con `id_experimento` y navegar al detalle de la sesión creada.
 * `construirPeticion` es pura y está probada aparte (CrearSesionPaper.test.tsx).
 * Se relaciona con: paginas/ExperimentoDetalle.tsx (lo abre), api/paper_interno.ts (useCrearSesionPaper) y
 * paginas/PaperInternoDetalle.tsx (destino).
 */
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCrearSesionPaper, type ModoEjecucionPaper, type PeticionSesionPaper } from '../api/paper_interno'
import { CajaError } from './Estado'

export interface CamposSesionPaper {
  nombre: string
  modo: ModoEjecucionPaper
  capital: string
  comisionPct: string
  slippagePct: string
  fechaAlta: string
}

const numero = (texto: string) => Number(texto.trim().replace(',', '.'))

/** Petición de la API desde los campos del formulario, o el primer error de validación (en español). */
export function construirPeticion(idExperimento: string, c: CamposSesionPaper): { peticion: PeticionSesionPaper } | { error: string } {
  const capital = c.capital.trim() ? numero(c.capital) : undefined
  if (capital !== undefined && !(capital > 0)) return { error: 'El capital inicial debe ser un número positivo' }
  const costes: Partial<PeticionSesionPaper> = {}
  for (const [clave, texto, etiqueta] of [['comision', c.comisionPct, 'La comisión'], ['slippage', c.slippagePct, 'El slippage']] as const) {
    if (!texto.trim()) continue
    const pct = numero(texto)
    if (!(pct >= 0 && pct <= 10)) return { error: `${etiqueta} debe estar entre 0 y 10 %` }
    costes[clave] = pct / 100
  }
  return {
    peticion: {
      id_experimento: idExperimento,
      modo_ejecucion: c.modo,
      ...(capital !== undefined && { capital_inicial: capital }),
      ...costes,
      ...(c.nombre.trim() && { nombre: c.nombre.trim() }),
      ...(c.fechaAlta && { fecha_alta: c.fechaAlta }),
    },
  }
}

interface Props {
  idExperimento: string
  nombreSugerido: string
  alCerrar: () => void
}

export default function CrearSesionPaper({ idExperimento, nombreSugerido, alCerrar }: Props) {
  const navegar = useNavigate()
  const crear = useCrearSesionPaper()
  const [campos, setCampos] = useState<CamposSesionPaper>({ nombre: nombreSugerido, modo: 'apertura_siguiente', capital: '100000', comisionPct: '', slippagePct: '', fechaAlta: '' })
  const [errorLocal, setErrorLocal] = useState<string | null>(null)
  const cambiar = (parcial: Partial<CamposSesionPaper>) => setCampos((c) => ({ ...c, ...parcial }))

  const enviar = (e: React.FormEvent) => {
    e.preventDefault()
    const resultado = construirPeticion(idExperimento, campos)
    if ('error' in resultado) {
      setErrorLocal(resultado.error)
      return
    }
    setErrorLocal(null)
    crear.mutate(resultado.peticion, { onSuccess: (sesion) => navegar(`/paper-interno/${sesion.id}`) })
  }

  return (
    <form onSubmit={enviar} className="tarjeta space-y-3 p-4" role="dialog" aria-label="Crear sesión de paper interno">
      <h2>Paper interno</h2>
      <p className="text-suave">
        Congela la receta de este experimento (estrategia, parámetros, universo y entorno) y la sigue día a día con precios reales, sin broker.
        La sesión empieza en la fecha de alta con todo el capital en efectivo.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="flex flex-col gap-1 sm:col-span-2">
          <label htmlFor="nombre-paper" className="font-medium">Nombre</label>
          <input id="nombre-paper" className="campo" value={campos.nombre} onChange={(e) => cambiar({ nombre: e.target.value })} />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="modo-paper" className="font-medium">Modo de ejecución</label>
          <select id="modo-paper" className="campo" value={campos.modo} onChange={(e) => cambiar({ modo: e.target.value as ModoEjecucionPaper })}>
            <option value="apertura_siguiente">Apertura siguiente (realista)</option>
            <option value="cierre">Al cierre (optimista, como el backtest)</option>
          </select>
          <p className="text-xs text-suave">Apertura siguiente decide con el cierre de D y ejecuta a la apertura de D+1, como una orden de mercado en Alpaca.</p>
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="capital-paper" className="font-medium">Capital inicial (USD)</label>
          <input id="capital-paper" className="campo" inputMode="decimal" value={campos.capital} onChange={(e) => cambiar({ capital: e.target.value })} />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="comision-paper" className="font-medium">Comisión (%)</label>
          <input id="comision-paper" className="campo" inputMode="decimal" placeholder="la del experimento" value={campos.comisionPct} onChange={(e) => cambiar({ comisionPct: e.target.value })} />
          <p className="text-xs text-suave">Alpaca no cobra comisión: pon 0 para costes tipo Alpaca.</p>
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="slippage-paper" className="font-medium">Slippage (%)</label>
          <input id="slippage-paper" className="campo" inputMode="decimal" placeholder="el del experimento" value={campos.slippagePct} onChange={(e) => cambiar({ slippagePct: e.target.value })} />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="alta-paper" className="font-medium">Fecha de alta</label>
          <input id="alta-paper" type="date" className="campo" value={campos.fechaAlta} onChange={(e) => cambiar({ fechaAlta: e.target.value })} />
          <p className="text-xs text-suave">Vacío = hoy. Una fecha pasada se simula al avanzar, pero no es forward real.</p>
        </div>
      </div>
      {errorLocal && <p role="alert" className="text-sm text-neg">{errorLocal}</p>}
      {crear.error && <CajaError error={crear.error} titulo="No se pudo crear la sesión" />}
      <div className="flex gap-2">
        <button type="submit" className="boton boton-primario" disabled={crear.isPending}>{crear.isPending ? 'Creando…' : 'Crear sesión'}</button>
        <button type="button" className="boton" onClick={alCerrar}>Cancelar</button>
      </div>
    </form>
  )
}
