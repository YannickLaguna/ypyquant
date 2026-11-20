/**
 * Tests del diálogo «Guardar como configuración»: valida el nombre, envía la fuente correcta
 * (experimento o barrido con su criterio) y avisa del sobreajuste cuando viene de un barrido.
 * Se relaciona con: componentes/GuardarConfiguracion.tsx, api/hooks.ts (useCrearConfiguracion).
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import GuardarConfiguracion, { type FuenteConfiguracion } from './GuardarConfiguracion'

function simularApi(estado = 201, detalle?: string) {
  const envios: Record<string, unknown>[] = []
  vi.stubGlobal('fetch', vi.fn(async (_ruta: string, opciones?: { method?: string; body?: string }) => {
    if (opciones?.method === 'POST') {
      envios.push(JSON.parse(opciones.body ?? '{}'))
      return estado === 201
        ? { ok: true, status: 201, json: async () => ({ id: 'mi-config' }) }
        : { ok: false, status: estado, json: async () => ({ detail: detalle }) }
    }
    return { ok: true, status: 200, json: async () => ({}) }
  }))
  return envios
}

function montar(fuente: FuenteConfiguracion) {
  const cliente = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <QueryClientProvider client={cliente}>
      <MemoryRouter initialEntries={['/origen']}>
        <Routes>
          <Route path="/origen" element={<GuardarConfiguracion fuente={fuente} nombreSugerido="momentum top_k=5" alCerrar={() => {}} />} />
          <Route path="/configuraciones/:id" element={<p>detalle de la configuración</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('GuardarConfiguracion', () => {
  it('no deja guardar con el nombre vacío', async () => {
    const envios = simularApi()
    montar({ tipo: 'experimento', id: 'momentum-abc' })
    const nombre = screen.getByLabelText('Nombre')
    expect(nombre).toHaveValue('momentum top_k=5')
    await userEvent.clear(nombre)
    await userEvent.click(screen.getByRole('button', { name: 'Guardar configuración' }))
    expect(screen.getByText('El nombre es obligatorio')).toBeInTheDocument()
    expect(envios).toHaveLength(0)
  })

  it('envía desde_experimento y navega a la configuración creada', async () => {
    const envios = simularApi()
    montar({ tipo: 'experimento', id: 'momentum-abc' })
    await userEvent.type(screen.getByLabelText('Notas'), 'prueba')
    await userEvent.click(screen.getByRole('button', { name: 'Guardar configuración' }))
    await waitFor(() => expect(screen.getByText('detalle de la configuración')).toBeInTheDocument())
    expect(envios[0]).toMatchObject({ nombre: 'momentum top_k=5', notas: 'prueba', desde_experimento: 'momentum-abc' })
    expect(envios[0]).not.toHaveProperty('desde_barrido')
  })

  it('desde un barrido envía el criterio y avisa del sobreajuste', async () => {
    const envios = simularApi()
    montar({ tipo: 'barrido', id: 'barrido-123', criterio: 'meseta', idExperimento: 'momentum-xyz' })
    expect(screen.getByRole('note')).toHaveTextContent(/sobreajust/i)
    await userEvent.click(screen.getByRole('button', { name: 'Guardar configuración' }))
    await waitFor(() => expect(envios).toHaveLength(1))
    expect(envios[0]).toMatchObject({ desde_barrido: 'barrido-123', criterio: 'meseta', id_experimento: 'momentum-xyz' })
  })

  it('muestra el error de la API (id duplicado)', async () => {
    simularApi(409, "Ya existe la configuración 'mi-config'")
    montar({ tipo: 'experimento', id: 'momentum-abc' })
    await userEvent.click(screen.getByRole('button', { name: 'Guardar configuración' }))
    expect(await screen.findByText("Ya existe la configuración 'mi-config'")).toBeInTheDocument()
  })
})
