/**
 * Tests del panel «Paper interno» de la ficha de un experimento: validación de los campos, cuerpo de la petición y
 * navegación al detalle de la sesión creada. Se relaciona con: componentes/CrearSesionPaper.tsx.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import CrearSesionPaper, { construirPeticion, type CamposSesionPaper } from './CrearSesionPaper'

const campos = (cambios: Partial<CamposSesionPaper> = {}): CamposSesionPaper => ({
  nombre: '', modo: 'apertura_siguiente', capital: '100000', comisionPct: '', slippagePct: '', fechaAlta: '', ...cambios,
})

describe('construirPeticion', () => {
  it('manda lo mínimo con los valores por defecto', () => {
    expect(construirPeticion('exp-1', campos())).toEqual({ peticion: { id_experimento: 'exp-1', modo_ejecucion: 'apertura_siguiente', capital_inicial: 100000 } })
  })
  it('convierte los costes de % a fracción (también con coma) y recoge nombre y fecha', () => {
    const r = construirPeticion('exp-1', campos({ nombre: ' Mi sesión ', comisionPct: '0', slippagePct: '0,05', fechaAlta: '2026-11-23', modo: 'cierre' }))
    expect(r).toEqual({ peticion: { id_experimento: 'exp-1', modo_ejecucion: 'cierre', capital_inicial: 100000, comision: 0, slippage: 0.0005, nombre: 'Mi sesión', fecha_alta: '2026-11-23' } })
  })
  it('rechaza capital no positivo y costes fuera de rango', () => {
    expect(construirPeticion('e', campos({ capital: '-5' }))).toEqual({ error: expect.stringContaining('capital') })
    expect(construirPeticion('e', campos({ comisionPct: '20' }))).toEqual({ error: expect.stringContaining('comisión') })
    expect(construirPeticion('e', campos({ slippagePct: 'abc' }))).toEqual({ error: expect.stringContaining('slippage') })
  })
})

afterEach(() => vi.unstubAllGlobals())

describe('CrearSesionPaper', () => {
  it('crea la sesión y navega a su detalle', async () => {
    const fetchFalso = vi.fn(async () => ({ ok: true, status: 201, json: async () => ({ id: 'sesion-nueva' }) }))
    vi.stubGlobal('fetch', fetchFalso)
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={['/experimentos/exp-1']}>
          <Routes>
            <Route path="/experimentos/:id" element={<CrearSesionPaper idExperimento="exp-1" nombreSugerido="momentum" alCerrar={() => {}} />} />
            <Route path="/paper-interno/:id" element={<p>detalle de la sesión</p>} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    )
    await userEvent.selectOptions(screen.getByLabelText('Modo de ejecución'), 'cierre')
    await userEvent.click(screen.getByRole('button', { name: 'Crear sesión' }))
    await waitFor(() => expect(screen.getByText('detalle de la sesión')).toBeInTheDocument())
    const [ruta, opciones] = fetchFalso.mock.calls[0] as unknown as [string, RequestInit]
    expect(ruta).toBe('/api/paper/interno/sesiones')
    expect(JSON.parse(opciones.body as string)).toEqual({ id_experimento: 'exp-1', modo_ejecucion: 'cierre', capital_inicial: 100000, nombre: 'momentum' })
  })

  it('muestra el error de validación sin llamar a la API', async () => {
    const fetchFalso = vi.fn()
    vi.stubGlobal('fetch', fetchFalso)
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter><CrearSesionPaper idExperimento="exp-1" nombreSugerido="x" alCerrar={() => {}} /></MemoryRouter>
      </QueryClientProvider>,
    )
    await userEvent.clear(screen.getByLabelText('Capital inicial (USD)'))
    await userEvent.type(screen.getByLabelText('Capital inicial (USD)'), '-1')
    await userEvent.click(screen.getByRole('button', { name: 'Crear sesión' }))
    expect(screen.getByRole('alert')).toHaveTextContent('capital')
    expect(fetchFalso).not.toHaveBeenCalled()
  })
})
