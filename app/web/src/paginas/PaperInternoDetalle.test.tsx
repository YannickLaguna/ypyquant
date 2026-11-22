/**
 * Tests del detalle de una sesión de paper interno: cabecera, avisos, posiciones, estado vacío de las operaciones y
 * controles (pausar, detener con confirmación). Se relaciona con: paginas/PaperInternoDetalle.tsx y api/paper_interno.ts.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DetalleSesionPaper } from '../api/paper_interno'
import PaperInternoDetalle from './PaperInternoDetalle'

vi.mock('../componentes/Graficos', () => ({
  GraficoSeries: ({ titulo, series }: { titulo: string; series: { nombre: string }[] }) => <div role="img" aria-label={`${titulo}: ${series.map((s) => s.nombre).join(', ')}`} />,
}))

const detalle: DetalleSesionPaper = {
  id: 'momentum-a', nombre: 'Momentum A', estrategia: 'momentum', modo_ejecucion: 'cierre', estado: 'activa', fecha_alta: '2026-11-02',
  capital_inicial: 100000, equity: 101500, dias: 12, ultima_fecha: '2026-11-20', retorno_total: 0.015, sharpe: 1.2, max_drawdown: -0.02,
  volatilidad: 0.12, costes: 85, dias_pendientes: 2, al_dia: false, id_experimento: 'exp-1', id_configuracion: null,
  sesion: { receta: { estrategia: 'momentum', parametros: { top_k: 3 }, universo: ['AAA', 'BBB'], benchmark: 'SPY', comision: 0.001, slippage: 0.001, frecuencia_rebalanceo: 'monthly' } },
  metricas: { sharpe: 1.2, max_drawdown: -0.02, volatilidad: 0.12 }, costes_detalle: { comisiones: 60, slippage: 25, total: 85 }, efectivo: 1200,
  posiciones: [{ ticker: 'AAA', cantidad: 10.5, precio_medio: 100, ultimo_cierre: 105, valor: 1102.5, peso: 0.4 }],
  pesos_objetivo: [{ ticker: 'AAA', peso: 0.6 }, { ticker: 'BBB', peso: 0.4 }], pesos_pendientes: null, fecha_decision_pendiente: null,
  avisos: [{ fecha: '2026-11-10', tipo: 'rebase', mensaje: 'AAA: el histórico ajustado cambió (factor 0.980000)' }], n_operaciones: 0, ejecucion_optimista: true,
}

function simularApi(llamadas: string[]) {
  vi.stubGlobal('fetch', vi.fn(async (ruta: string, opciones?: RequestInit) => {
    llamadas.push(`${opciones?.method ?? 'GET'} ${ruta}`)
    let cuerpo: unknown = detalle
    if (ruta.endsWith('/equity')) cuerpo = { equity: [{ fecha: '2026-11-02', equity: 100000, efectivo: 0, n_posiciones: 1, benchmark: 400, retorno: null, drawdown: 0 }] }
    else if (ruta.endsWith('/operaciones')) cuerpo = { operaciones: [] }
    else if (ruta.endsWith('/espejo')) cuerpo = { estado: 'sin_datos', id_experimento: null, real: [], espejo: [] }
    else if (ruta.endsWith('/pausar')) cuerpo = { ...detalle, estado: 'pausada' }
    return { ok: true, status: 200, json: async () => cuerpo }
  }))
}

function montar() {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={['/paper-interno/momentum-a']}>
        <Routes><Route path="/paper-interno/:id" element={<PaperInternoDetalle />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('Detalle de sesión de paper interno', () => {
  it('muestra cabecera, aviso de ejecución optimista, pendientes, avisos y posiciones', async () => {
    simularApi([])
    montar()
    expect(await screen.findByRole('heading', { level: 1, name: /Momentum A/ })).toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent('Ejecución optimista')
    expect(screen.getByText('2 días pendientes')).toBeInTheDocument()
    expect(screen.getByText(/el histórico ajustado cambió/)).toBeInTheDocument()
    const posiciones = screen.getByRole('table', { name: 'Posiciones actuales' })
    expect(within(posiciones).getByText('AAA')).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Operaciones' })).toHaveTextContent('Sin operaciones')
    expect(screen.getByRole('link', { name: 'exp-1' })).toHaveAttribute('href', '/experimentos/exp-1')
  })

  it('pausar llama a la API; detener pide confirmación y no actúa si se rechaza', async () => {
    const llamadas: string[] = []
    simularApi(llamadas)
    const confirmar = vi.spyOn(window, 'confirm').mockReturnValue(false)
    montar()
    await userEvent.click(await screen.findByRole('button', { name: 'Pausar' }))
    expect(llamadas).toContain('POST /api/paper/interno/sesiones/momentum-a/pausar')
    await userEvent.click(await screen.findByRole('button', { name: 'Reanudar' }))  // ya aparece como pausada
    await userEvent.click(screen.getByRole('button', { name: 'Detener' }))
    expect(confirmar).toHaveBeenCalled()
    expect(llamadas.some((l) => l.endsWith('/detener'))).toBe(false)
    confirmar.mockRestore()
  })
})
