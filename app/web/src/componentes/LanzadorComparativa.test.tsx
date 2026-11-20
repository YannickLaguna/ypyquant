/**
 * Tests del lanzador de comparativas: oculta de cada estrategia los parámetros que fija el marco, envía solo los
 * propios y muestra el informe de qué comunes se aplicaron (y los avisos).
 * Se relaciona con: componentes/LanzadorComparativa.tsx, lib/marcos.ts y api/hooks.ts.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Estrategia, Marco } from '../api/tipos'
import LanzadorComparativa from './LanzadorComparativa'

const base = { origen: 'incluida', archivo: null, frecuencia_rebalanceo: 'monthly', universo_sugerido: null, parametros_defecto: {} } as const
const ESTRATEGIAS: Estrategia[] = [
  {
    ...base, nombre: 'momentum', descripcion: 'Top-k',
    parametros_schema: { type: 'object', properties: { top_k: { type: 'integer', title: 'Top K', default: 5, minimum: 1 } } },
  },
  {
    ...base, nombre: 'bayes_stein', descripcion: 'BS',
    parametros_schema: {
      type: 'object',
      properties: {
        ventana_dias: { type: 'integer', title: 'Ventana', default: 756, minimum: 60 },
        penalizacion_turnover: { type: 'number', title: 'Penalización', default: 0, minimum: 0 },
      },
    },
  },
]
const MARCO = { id: 'marco-x', nombre: 'Marco X', parametros_comunes: { ventana_dias: 504 } } as unknown as Marco
const AVISO = 'No se aplicó en momentum: ventana_dias (la estrategia no lo declara)'

function simularApi() {
  const envios: Record<string, unknown>[] = []
  vi.stubGlobal('fetch', vi.fn(async (ruta: string, opciones?: { method?: string; body?: string }) => {
    if (opciones?.method === 'POST') {
      envios.push(JSON.parse(opciones.body ?? '{}'))
      return {
        ok: true, status: 202,
        json: async () => ({
          id: 'cmp-1', id_marco: 'marco-x', marco: MARCO, creada: '',
          experimentos: [{
            id_experimento: 'momentum-abc', estrategia: 'momentum', parametros: { top_k: 5 }, reutilizado: false,
            informe: { estrategia: 'momentum', aplicados: {}, no_aplicados: ['ventana_dias'], sobrescritos: [], avisos: [AVISO] },
          }],
        }),
      }
    }
    return { ok: true, status: 200, json: async () => (ruta.startsWith('/api/estrategias') ? { estrategias: ESTRATEGIAS, errores: {} } : {}) }
  }))
  return envios
}

function montar() {
  const cliente = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(<QueryClientProvider client={cliente}><MemoryRouter><LanzadorComparativa marco={MARCO} /></MemoryRouter></QueryClientProvider>)
}

afterEach(() => vi.unstubAllGlobals())

describe('LanzadorComparativa', () => {
  it('no permite lanzar sin estrategias', async () => {
    simularApi()
    montar()
    expect(await screen.findByRole('button', { name: 'Lanzar comparativa (0)' })).toBeDisabled()
  })

  it('oculta los parámetros que fija el marco y envía solo los propios', async () => {
    const envios = simularApi()
    montar()
    await userEvent.click(await screen.findByRole('checkbox', { name: /bayes_stein/ }))
    expect(screen.getByText(/Fijados por el marco: ventana_dias=504/)).toBeInTheDocument()
    expect(screen.queryByLabelText(/Ventana/)).not.toBeInTheDocument()
    expect(screen.getByLabelText(/Penalización/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('checkbox', { name: /momentum/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Lanzar comparativa (2)' }))
    await waitFor(() => expect(envios).toHaveLength(1))
    expect(envios[0]).toEqual({
      estrategias: [
        { estrategia: 'bayes_stein', parametros: { penalizacion_turnover: 0 } },
        { estrategia: 'momentum', parametros: { top_k: 5 } },
      ],
    })
  })

  it('muestra el informe con los avisos y el enlace a Comparar', async () => {
    simularApi()
    montar()
    await userEvent.click(await screen.findByRole('checkbox', { name: /momentum/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Lanzar comparativa (1)' }))
    expect(await screen.findByText(AVISO)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Abrir en Comparar' })).toHaveAttribute('href', '/comparar?ids=momentum-abc&comparativa=cmp-1')
  })
})
