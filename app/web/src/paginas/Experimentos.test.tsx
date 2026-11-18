/**
 * Tests de la tabla de experimentos con `fetch` simulado: estados de carga/vacío/error, orden por
 * columna, filtros y selección para comparar.
 * Se relaciona con: paginas/Experimentos.tsx, api/hooks.ts y componentes/TablaOrdenable.tsx.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ResumenExperimento } from '../api/tipos'
import Experimentos, { filtrarExperimentos } from './Experimentos'

const base: ResumenExperimento = {
  id: 'x', estrategia: 'momentum', parametros: { top_k: 5 }, universo: ['A', 'B'], fecha_inicio: '2018-01-01',
  fecha_fin: '2024-12-31', benchmark: 'SPY', frecuencia_rebalanceo: 'monthly', etiqueta: '', estado: 'hecho', error: null,
  creado: '2026-11-01T10:00:00', actualizado: null, metricas: {},
}
const EXPERIMENTOS: ResumenExperimento[] = [
  { ...base, id: 'momentum-1', metricas: { sharpe: 0.5, cagr: 0.08, max_drawdown: -0.3, turnover_anual: 2.5 }, creado: '2026-11-01T10:00:00' },
  { ...base, id: 'bayes_stein-2', estrategia: 'bayes_stein', parametros: {}, metricas: { sharpe: 1.2, cagr: 0.12, max_drawdown: -0.2, turnover_anual: 0.6 }, creado: '2026-11-02T10:00:00' },
  { ...base, id: 'momentum-3', estado: 'error', error: 'boom', creado: '2026-11-03T10:00:00' },
]

function responder(cuerpo: unknown, estado = 200) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: estado < 400, status: estado, json: async () => cuerpo }))
}

function montar() {
  const cliente = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  let ruta = ''
  render(
    <QueryClientProvider client={cliente}>
      <MemoryRouter initialEntries={['/experimentos']}>
        <Routes>
          <Route path="/experimentos" element={<Experimentos />} />
          <Route path="/comparar" element={<Localizador alCargar={(r) => (ruta = r)} />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return { ruta: () => ruta }
}

function Localizador({ alCargar }: { alCargar: (r: string) => void }) {
  alCargar(window.location.search || 'comparar')
  return <p>vista comparar</p>
}

const idsEnOrden = () =>
  within(screen.getByRole('table', { name: 'Experimentos' })).getAllByRole('link').map((a) => a.getAttribute('href')?.split('/').pop())

beforeEach(() => vi.unstubAllGlobals())
afterEach(() => vi.unstubAllGlobals())

describe('Experimentos', () => {
  it('muestra la carga y luego las filas con métricas formateadas en español', async () => {
    responder({ experimentos: EXPERIMENTOS })
    montar()
    expect(screen.getByRole('status')).toHaveTextContent('Cargando')
    await screen.findByRole('table', { name: 'Experimentos' })
    expect(screen.getByText('0,50')).toBeInTheDocument() // Sharpe
    expect(screen.getByText('12,0 %')).toBeInTheDocument() // CAGR
    expect(screen.getByText('-30,0 %')).toBeInTheDocument() // Máx. DD
    expect(screen.getAllByText('error').length).toBeGreaterThan(0)
  })

  it('ordena por columna al pulsar la cabecera y alterna el sentido', async () => {
    responder({ experimentos: EXPERIMENTOS })
    montar()
    await screen.findByRole('table', { name: 'Experimentos' })
    expect(idsEnOrden()).toEqual(['momentum-3', 'bayes_stein-2', 'momentum-1']) // por creado desc
    const usuario = userEvent.setup()
    await usuario.click(screen.getByRole('button', { name: /Sharpe/ }))
    expect(idsEnOrden()).toEqual(['bayes_stein-2', 'momentum-1', 'momentum-3']) // desc; sin métrica (null) al final
    expect(screen.getByRole('columnheader', { name: /Sharpe/ })).toHaveAttribute('aria-sort', 'descending')
    await usuario.click(screen.getByRole('button', { name: /Sharpe/ }))
    expect(idsEnOrden()).toEqual(['momentum-1', 'bayes_stein-2', 'momentum-3'])
  })

  it('filtra por texto y por estado', async () => {
    responder({ experimentos: EXPERIMENTOS })
    montar()
    await screen.findByRole('table', { name: 'Experimentos' })
    const usuario = userEvent.setup()
    await usuario.type(screen.getByLabelText('Buscar'), 'bayes')
    expect(idsEnOrden()).toEqual(['bayes_stein-2'])
    await usuario.clear(screen.getByLabelText('Buscar'))
    await usuario.selectOptions(screen.getByLabelText('Estado'), 'error')
    expect(idsEnOrden()).toEqual(['momentum-3'])
    await usuario.type(screen.getByLabelText('Buscar'), 'nada-coincide')
    expect(screen.getByText('Ningún experimento coincide con los filtros')).toBeInTheDocument()
  })

  it('solo permite comparar experimentos terminados y navega con los ids elegidos', async () => {
    responder({ experimentos: EXPERIMENTOS })
    montar()
    await screen.findByRole('table', { name: 'Experimentos' })
    expect(screen.getByLabelText('Seleccionar momentum-3 para comparar')).toBeDisabled()
    const comparar = screen.getByRole('button', { name: /Comparar/ })
    expect(comparar).toBeDisabled()
    const usuario = userEvent.setup()
    await usuario.click(screen.getByLabelText('Seleccionar momentum-1 para comparar'))
    await usuario.click(screen.getByLabelText('Seleccionar bayes_stein-2 para comparar'))
    expect(comparar).toBeEnabled()
    await usuario.click(comparar)
    expect(await screen.findByText('vista comparar')).toBeInTheDocument()
  })

  it('muestra el estado vacío', async () => {
    responder({ experimentos: [] })
    montar()
    expect(await screen.findByText('Aún no hay experimentos')).toBeInTheDocument()
  })

  it('muestra el error de la API', async () => {
    responder({ detail: 'almacén no disponible' }, 500)
    montar()
    const alerta = await screen.findByRole('alert')
    expect(alerta).toHaveTextContent('almacén no disponible')
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())
  })
})

describe('filtrarExperimentos', () => {
  it('combina texto, estrategia y estado', () => {
    expect(filtrarExperimentos(EXPERIMENTOS, '', 'momentum', 'hecho').map((e) => e.id)).toEqual(['momentum-1'])
    expect(filtrarExperimentos(EXPERIMENTOS, 'top_k=5', '', '').length).toBe(2)
  })
})
