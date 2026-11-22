/**
 * Test del detalle de un experimento con avisos: el aviso agregado de «volatilidad objetivo inalcanzable» (se usó la
 * cartera de mínima varianza) que guarda `metricas.avisos` debe verse en el detalle.
 * Se relaciona con: paginas/ExperimentoDetalle.tsx (bloque de avisos) y ypyquant.estrategias.incluidas (los genera).
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DetalleExperimento } from '../api/tipos'
import ExperimentoDetalle from './ExperimentoDetalle'

vi.mock('../componentes/Graficos', async (importar) => ({ ...(await importar<object>()), GraficoSeries: () => <div role="img" aria-label="gráfico" /> }))

const AVISO = 'Volatilidad objetivo inalcanzable en 20 de 23 rebalanceos (primero 2020-02-03, último 2021-12-01): σ objetivo 0,1 % < σ mínima alcanzable 6,2 %; en esos rebalanceos se usó la cartera de mínima varianza.'

function detalle(avisos: string[]): DetalleExperimento {
  return {
    id: 'bayes_stein-abc', estrategia: 'bayes_stein', parametros: { volatilidad_objetivo: 0.001 }, universo: ['AAA'], fecha_inicio: '2020-01-01', fecha_fin: '2020-12-31',
    benchmark: 'SPY', frecuencia_rebalanceo: 'monthly', etiqueta: '', estado: 'hecho', error: null, creado: null, actualizado: null, metricas: {},
    config: { estrategia: 'bayes_stein', parametros: { volatilidad_objetivo: 0.001 }, universo: ['AAA'], fecha_inicio: '2020-01-01', fecha_fin: '2020-12-31', capital_inicial: 100000, comision: 0.001, slippage: 0.001, frecuencia_rebalanceo: 'monthly', benchmark: 'SPY', semilla: 42 },
    traza: null, metricas_completas: { avisos }, procedencia: null,
    equity: [{ fecha: '2020-01-02', equity: 100000, drawdown: 0, benchmark: 100000 }, { fecha: '2020-01-03', equity: 100100, drawdown: 0, benchmark: 100050 }] as never,
    pesos: [{ fecha: '2020-01-02', ticker: 'AAA', peso: 1 }], operaciones: [],
  } as DetalleExperimento
}

function montar(cuerpo: DetalleExperimento) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => cuerpo }))
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={['/experimentos/bayes_stein-abc']}>
        <Routes><Route path="/experimentos/:id" element={<ExperimentoDetalle />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('ExperimentoDetalle con avisos', () => {
  it('muestra el aviso de volatilidad objetivo inalcanzable (mínima varianza)', async () => {
    montar(detalle([AVISO]))
    expect(await screen.findByText(/Volatilidad objetivo inalcanzable en 20 de 23 rebalanceos/)).toBeInTheDocument()
    expect(screen.getByText(/cartera de mínima varianza/)).toBeInTheDocument()
  })

  it('con cortos y préstamo muestra exposición bruta/neta, las llamadas de margen y rotula los pesos', async () => {
    const cuerpo = detalle(['Llamada de margen 2020-03-12: equity 40.000 por debajo del mantenimiento 42.000 (exposición bruta 160.000); liquidación forzosa: se conserva el 60 % de las posiciones.'])
    cuerpo.metricas_completas = { ...cuerpo.metricas_completas, exposicion_bruta_maxima: 1.6, exposicion_neta_media: 1.0, llamadas_margen: 1 }
    cuerpo.pesos = [{ fecha: '2020-01-02', ticker: 'AAA', peso: 1.3 }, { fecha: '2020-01-02', ticker: 'BBB', peso: -0.3 }, { fecha: '2020-01-02', ticker: 'CASH', peso: -0.1 }]
    montar(cuerpo)
    expect(await screen.findByText(/Llamada de margen 2020-03-12/)).toBeInTheDocument()
    expect(screen.getByText('Exposición bruta máxima').closest('tr')).toHaveTextContent('160,0')
    expect(screen.getByText('Llamadas de margen').closest('tr')).toHaveTextContent('1')
    expect(screen.getByText('BBB (corto)')).toBeInTheDocument()
    expect(screen.getByText('Préstamo (margen)')).toBeInTheDocument()
  })
})
