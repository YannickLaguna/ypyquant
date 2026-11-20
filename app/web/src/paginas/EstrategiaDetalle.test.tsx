/**
 * Tests del selector de universo del detalle de estrategia (fase 0.0c): `sp500_pit` aparece como opción,
 * se propone por defecto cuando la estrategia lo sugiere (sección cruzada) y `interpretarUniverso` no
 * envía un límite `:N` a un universo dinámico.
 * Se relaciona con: paginas/EstrategiaDetalle.tsx, api/hooks.ts (useEstrategia, useUniversos) y api/tipos.ts.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { EstrategiaDetalle as TipoEstrategia, Universo } from '../api/tipos'
import EstrategiaDetalle, { interpretarUniverso } from './EstrategiaDetalle'

const UNIVERSOS: Universo[] = [
  { nombre: 'sp500', n_tickers: 505, dinamico: false },
  { nombre: 'sp500_pit', n_tickers: 1214, dinamico: true },
]

const estrategia = (sugerido: string | null): TipoEstrategia => ({
  nombre: 'momentum', descripcion: 'Top-k por retorno acumulado', origen: 'incluida', archivo: null,
  frecuencia_rebalanceo: 'monthly', universo_sugerido: sugerido, codigo_fuente: 'class Momentum: ...',
  parametros_schema: { type: 'object', properties: { top_k: { type: 'integer', title: 'Top K', default: 5, minimum: 1 } } },
  parametros_defecto: { top_k: 5 },
})

/** `fetch` simulado por ruta: estrategia, universos y el POST de lanzar (que se registra). */
function simularApi(sugerido: string | null) {
  const envios: unknown[] = []
  vi.stubGlobal('fetch', vi.fn(async (ruta: string, opciones?: { method?: string; body?: string }) => {
    if (opciones?.method === 'POST') {
      envios.push(JSON.parse(opciones.body ?? '{}'))
      return { ok: true, status: 202, json: async () => ({ id: 'momentum-abc', reutilizado: false }) }
    }
    const cuerpo = ruta.startsWith('/api/universos') ? { universos: UNIVERSOS } : estrategia(sugerido)
    return { ok: true, status: 200, json: async () => cuerpo }
  }))
  return envios
}

function montar() {
  const cliente = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={cliente}>
      <MemoryRouter initialEntries={['/estrategias/momentum']}>
        <Routes>
          <Route path="/estrategias/:nombre" element={<EstrategiaDetalle />} />
          <Route path="/experimentos/:id" element={<p>detalle del experimento</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('interpretarUniverso', () => {
  const nombres = ['sp500', 'sp500_pit']
  const dinamicos = ['sp500_pit']

  it('reconoce el universo dinámico y descarta un límite «:N» que no admite', () => {
    expect(interpretarUniverso('sp500_pit', nombres, dinamicos)).toBe('sp500_pit')
    expect(interpretarUniverso('SP500_PIT:30', nombres, dinamicos)).toBe('sp500_pit')
  })

  it('conserva el límite en los universos fijos y la lista manual de tickers', () => {
    expect(interpretarUniverso('sp500:30', nombres, dinamicos)).toBe('sp500:30')
    expect(interpretarUniverso('aapl, msft  jpm', nombres, dinamicos)).toEqual(['AAPL', 'MSFT', 'JPM'])
  })
})

describe('Selector de universo del detalle de estrategia', () => {
  beforeEach(() => vi.unstubAllGlobals())
  afterEach(() => vi.unstubAllGlobals())

  it('ofrece sp500_pit como opción y explica que es point-in-time', async () => {
    simularApi(null)
    montar()
    const selector = await screen.findByLabelText('Universo predefinido')
    expect(screen.getByRole('option', { name: /sp500_pit/ })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: /sp500 \(/ })).toBeInTheDocument()
    await userEvent.setup().selectOptions(selector, 'sp500_pit')
    expect((screen.getByLabelText('Universo') as HTMLTextAreaElement).value).toBe('sp500_pit')
    expect(screen.getByText(/miembros del índice en cada fecha/i)).toBeInTheDocument()
  })

  it('usa el universo sugerido por la estrategia como valor inicial y lo envía al lanzar', async () => {
    const envios = simularApi('sp500_pit')
    montar()
    await waitFor(() => expect((screen.getByLabelText('Universo') as HTMLTextAreaElement).value).toBe('sp500_pit'))
    await userEvent.setup().click(screen.getByRole('button', { name: 'Lanzar experimento' }))
    await waitFor(() => expect(envios).toHaveLength(1))
    expect(envios[0]).toMatchObject({ estrategia: 'momentum', universo: 'sp500_pit' })
  })

  it('sin sugerencia mantiene la lista de tickers a mano por defecto', async () => {
    simularApi(null)
    montar()
    await waitFor(() => expect((screen.getByLabelText('Universo') as HTMLTextAreaElement).value).toMatch(/^AAPL, MSFT/))
  })
})
