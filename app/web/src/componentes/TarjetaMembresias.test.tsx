/**
 * Tests de la tarjeta de membresías del S&P 500: estado normal, pendientes de revisar con aviso y ausencia de datos.
 * Se relaciona con: componentes/TarjetaMembresias.tsx y api/universos.ts.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { EstadoMembresias } from '../api/universos'
import TarjetaMembresias from './TarjetaMembresias'

const ESTADO: EstadoMembresias = {
  activo: true, ultima_comprobacion: '2026-11-22T22:00:00+01:00', resultado: 'al_dia', error: null, n_vigentes: 503,
  ultima_comprobacion_correcta: '2026-11-22', antiguedad_dias: 0, cambios_aplicados_total: 24, cambios_aplicados_recientes: [],
  pendientes_de_revisar: [], avisos: [], avisos_calidad: [],
}

function montar(cuerpo: unknown) {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, json: async () => cuerpo })))
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><TarjetaMembresias /></QueryClientProvider>)
}

afterEach(() => vi.unstubAllGlobals())

describe('TarjetaMembresias', () => {
  it('muestra vigentes, comprobación y cambios aplicados', async () => {
    montar(ESTADO)
    const tarjeta = await screen.findByLabelText('Membresías del S&P 500')
    expect(tarjeta).toHaveTextContent('503')
    expect(tarjeta).toHaveTextContent('al día')
    expect(tarjeta).toHaveTextContent('24')
    expect(screen.queryByLabelText('Pendientes de revisar')).toBeNull()
  })

  it('lista los cambios pendientes de revisar con su motivo', async () => {
    montar({ ...ESTADO, resultado: 'pendiente', pendientes_de_revisar: [{ tipo: 'alta', simbolo: 'NEW', fecha: '2026-11-22', motivo_pendiente: 'spdr_spy discrepa: no incluye NEW' }] })
    expect(await screen.findByLabelText('Pendientes de revisar')).toHaveTextContent('alta de NEW')
    expect(screen.getByLabelText('Pendientes de revisar')).toHaveTextContent('spdr_spy discrepa')
  })

  it('no pinta nada si la API no devuelve el estado', async () => {
    montar({})
    await act(async () => { await new Promise((r) => setTimeout(r, 20)) })
    expect(screen.queryByLabelText('Membresías del S&P 500')).toBeNull()
  })
})
