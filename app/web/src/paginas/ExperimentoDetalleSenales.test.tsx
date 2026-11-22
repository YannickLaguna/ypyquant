/**
 * Tests del detalle de un experimento de una estrategia de señales: columna «Motivo» y resumen de entradas y
 * salidas en la tabla de operaciones, y el efectivo (`CASH`) nombrado «Efectivo» en la leyenda de pesos.
 * Una estrategia sin señales no muestra ni columna ni resumen.
 * Se relaciona con: paginas/ExperimentoDetalle.tsx, lib/senales.ts y componentes/Graficos.tsx (GraficoPesos).
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DetalleExperimento, OperacionFila } from '../api/tipos'
import ExperimentoDetalle from './ExperimentoDetalle'

// Los gráficos de series usan canvas, que jsdom no implementa; el de pesos es SVG y se deja real
vi.mock('../componentes/Graficos', async (importar) => ({ ...(await importar<object>()), GraficoSeries: () => <div role="img" aria-label="gráfico" /> }))

const operacion = (lado: 'compra' | 'venta', motivo: string | undefined, fecha = '2020-02-03'): OperacionFila =>
  ({ fecha, ticker: 'AAA', lado, cantidad: 10, precio: 100, comision: 1, pnl: lado === 'venta' ? -50 : null, motivo })

function detalle(operaciones: OperacionFila[]): DetalleExperimento {
  return {
    id: 'cruce_medias-abc', estrategia: 'cruce_medias', parametros: {}, universo: ['AAA'], fecha_inicio: '2020-01-01', fecha_fin: '2020-12-31',
    benchmark: 'SPY', frecuencia_rebalanceo: 'daily', etiqueta: '', estado: 'hecho', error: null, creado: null, actualizado: null, metricas: {},
    config: { estrategia: 'cruce_medias', parametros: {}, universo: ['AAA'], fecha_inicio: '2020-01-01', fecha_fin: '2020-12-31', capital_inicial: 100000, comision: 0.001, slippage: 0.001, frecuencia_rebalanceo: 'daily', benchmark: 'SPY', semilla: 42 },
    traza: null, metricas_completas: {}, procedencia: null,
    equity: [{ fecha: '2020-01-02', equity: 100000, drawdown: 0, benchmark: 100000 }, { fecha: '2020-01-03', equity: 100100, drawdown: 0, benchmark: 100050 }] as never,
    pesos: [{ fecha: '2020-01-02', ticker: 'AAA', peso: 0.25 }, { fecha: '2020-01-02', ticker: 'CASH', peso: 0.75 }],
    operaciones,
  } as DetalleExperimento
}

function montar(cuerpo: DetalleExperimento) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => cuerpo }))
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={['/experimentos/cruce_medias-abc']}>
        <Routes><Route path="/experimentos/:id" element={<ExperimentoDetalle />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('ExperimentoDetalle de una estrategia de señales', () => {
  it('muestra la columna Motivo, el resumen de entradas y salidas y llama Efectivo a CASH', async () => {
    montar(detalle([operacion('compra', 'senal'), operacion('venta', 'stop_perdida', '2020-03-02'), operacion('compra', 'ajuste', '2020-03-03')]))
    expect(await screen.findByRole('columnheader', { name: /Motivo/ })).toBeInTheDocument()
    expect(screen.getByText('Stop de pérdida', { selector: 'td' })).toBeInTheDocument()
    expect(screen.getByTestId('resumen-senales').textContent).toContain('1 entradas · 1 salidas · stop de pérdida: 1 · 1 ajustes por deriva')
    expect(screen.getByText('Efectivo')).toBeInTheDocument()
    expect(screen.queryByText('CASH')).not.toBeInTheDocument()
  })

  it('muestra el desglose de salidas por motivo cuando las métricas lo traen', async () => {
    const cuerpo = detalle([operacion('compra', 'entrada_larga'), operacion('venta', 'take_profit', '2020-03-02')])
    cuerpo.metricas_completas = { motivos_salida: [{ motivo: 'take_profit', lado: 'largo', n: 1, pnl_total: 90, pnl_medio: 90, tasa_acierto: 1, duracion_media_dias: 28 }] as never }
    montar(cuerpo)
    expect(await screen.findByRole('table', { name: 'Salidas por motivo' })).toBeInTheDocument()
  })

  it('una estrategia sin motivos no añade la columna ni el resumen', async () => {
    montar(detalle([operacion('compra', undefined), operacion('venta', undefined, '2020-03-02')]))
    await screen.findByText(/Operaciones \(2\)/)
    expect(screen.queryByRole('columnheader', { name: /Motivo/ })).not.toBeInTheDocument()
    expect(screen.queryByTestId('resumen-senales')).not.toBeInTheDocument()
    expect(screen.queryByRole('table', { name: 'Salidas por motivo' })).not.toBeInTheDocument()
  })
})
