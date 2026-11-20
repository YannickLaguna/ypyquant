/**
 * Tests del flujo «Lanzar contra otras estrategias» del detalle de configuración: la propia estrategia viene marcada
 * con sus parámetros propios, los comunes de la receta se fijan (no se piden), el envío lleva las fechas sugeridas y la
 * versión y el resultado enlaza con Comparar.
 * Se relaciona con: paginas/ConfiguracionDetalle.tsx, componentes/LanzadorComparativa.tsx y lib/marcos.ts.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Estrategia } from '../api/tipos'
import ConfiguracionDetalle from './ConfiguracionDetalle'

const base = { origen: 'incluida', archivo: null, frecuencia_rebalanceo: 'monthly', universo_sugerido: null, parametros_defecto: {} } as const
const ESTRATEGIAS: Estrategia[] = [
  {
    ...base, nombre: 'robusta_elipsoide', descripcion: 'Elipsoide',
    parametros_schema: {
      type: 'object',
      properties: {
        ventana_dias: { type: 'integer', title: 'Ventana', default: 756, minimum: 60 },
        delta: { type: 'number', title: 'Delta', default: 18, minimum: 0 },
      },
    },
  },
  {
    ...base, nombre: 'pesos_iguales', descripcion: 'Pesos iguales',
    parametros_schema: { type: 'object', properties: {} },
  },
]
const CONFIGURACION = {
  id: 'robusta-1', version: 3, nombre: 'Robusta 1', notas: '', estrategia: 'robusta_elipsoide',
  parametros: { ventana_dias: 756, delta: 18 }, universo: ['AAA', 'BBB'], universo_etiqueta: null, frecuencia_rebalanceo: 'monthly',
  comision: 0.001, slippage: 0.001, benchmark: 'SPY', capital_inicial: 100000, semilla: 42, dias_historia_previa: 1100,
  origen: { tipo: 'manual', metricas: {} }, creada: '2026-11-01T10:00:00', etiquetas: [], huella_estrategia: 'x', archivada: false, id_marco: null, deriva_de: null,
}

function simularApi() {
  const envios: { ruta: string; cuerpo: Record<string, unknown> }[] = []
  vi.stubGlobal('fetch', vi.fn(async (ruta: string, opciones?: { method?: string; body?: string }) => {
    if (opciones?.method === 'POST') {
      envios.push({ ruta, cuerpo: JSON.parse(opciones.body ?? '{}') })
      return {
        ok: true, status: 202,
        json: async () => ({
          id: 'cmp-9', id_marco: 'cfg-1', marco: { ...CONFIGURACION, id: 'cfg-1' }, creada: '',
          experimentos: [{
            id_experimento: 'robusta_elipsoide-abc', estrategia: 'robusta_elipsoide', parametros: { delta: 18 }, reutilizado: true,
            informe: { estrategia: 'robusta_elipsoide', aplicados: { ventana_dias: 756 }, no_aplicados: [], sobrescritos: [], avisos: [] },
          }],
        }),
      }
    }
    let cuerpo: unknown = {}
    if (ruta.startsWith('/api/estrategias/')) cuerpo = ESTRATEGIAS.find((e) => ruta.endsWith(e.nombre))
    else if (ruta.startsWith('/api/estrategias')) cuerpo = { estrategias: ESTRATEGIAS, errores: {} }
    if (ruta.startsWith('/api/configuraciones/robusta-1'))
      cuerpo = {
        configuracion: CONFIGURACION, versiones: [{ version: 3, creada: '2026-11-01T10:00:00', deriva_de: null, notas: '' }], marco: null,
        backtests: [], fechas_sugeridas: ['2020-01-01', '2024-12-31'], comparativas: [],
      }
    return { ok: true, status: 200, json: async () => cuerpo }
  }))
  return envios
}

function montar() {
  const cliente = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <QueryClientProvider client={cliente}>
      <MemoryRouter initialEntries={['/configuraciones/robusta-1']}>
        <Routes><Route path="/configuraciones/:id" element={<ConfiguracionDetalle />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('Lanzar contra otras estrategias', () => {
  it('abre el panel con las fechas sugeridas y la propia estrategia marcada sin pedir los comunes', async () => {
    simularApi()
    montar()
    await userEvent.click(await screen.findByRole('button', { name: 'Lanzar contra otras estrategias' }))
    expect(screen.getByLabelText('Inicio', { selector: '#cmp-inicio' })).toHaveValue('2020-01-01')
    expect(screen.getByLabelText('Fin', { selector: '#cmp-fin' })).toHaveValue('2024-12-31')
    expect(screen.getByRole('checkbox', { name: /robusta_elipsoide/ })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: /pesos_iguales/ })).not.toBeChecked()
    expect(screen.getByText(/Fijados por el marco: ventana_dias=756/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Lanzar comparativa (1)' })).toBeEnabled()
  })

  it('envía las estrategias con sus parámetros propios, las fechas y la versión, y enlaza con Comparar', async () => {
    const envios = simularApi()
    montar()
    await userEvent.click(await screen.findByRole('button', { name: 'Lanzar contra otras estrategias' }))
    await userEvent.click(screen.getByRole('checkbox', { name: /pesos_iguales/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Lanzar comparativa (2)' }))
    await waitFor(() => expect(envios).toHaveLength(1))
    expect(envios[0].ruta).toBe('/api/configuraciones/robusta-1/comparativa')
    expect(envios[0].cuerpo).toEqual({
      estrategias: [{ estrategia: 'robusta_elipsoide', parametros: { delta: 18 } }, { estrategia: 'pesos_iguales', parametros: {} }],
      fecha_inicio: '2020-01-01', fecha_fin: '2024-12-31', version: 3,
    })
    expect(await screen.findByRole('link', { name: 'Abrir en Comparar' })).toHaveAttribute('href', '/comparar?ids=robusta_elipsoide-abc&comparativa=cmp-9')
  })

  it('no deja lanzar con fechas incoherentes', async () => {
    simularApi()
    montar()
    await userEvent.click(await screen.findByRole('button', { name: 'Lanzar contra otras estrategias' }))
    const inicio = screen.getByLabelText('Inicio', { selector: '#cmp-inicio' })
    await userEvent.clear(inicio)
    await userEvent.type(inicio, '2025-01-01')
    expect(screen.getByRole('alert')).toHaveTextContent('El inicio debe ser anterior al fin.')
    expect(screen.getByRole('button', { name: 'Lanzar comparativa (1)' })).toBeDisabled()
  })
})
