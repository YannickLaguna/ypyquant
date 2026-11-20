/**
 * Tests de la página Comparar: el aviso «no es ceteris paribus» solo aparece cuando los experimentos difieren
 * del entorno o de los parámetros comunes, y los avisos del marco aparecen al abrir una comparativa.
 * Se relaciona con: paginas/Comparar.tsx, lib/marcos.ts (resumenDiferencias) y api/hooks.ts (useComparacion).
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { CeterisParibus, Comparacion } from '../api/tipos'
import Comparar from './Comparar'

// El gráfico usa canvas y matchMedia, que jsdom no implementa: aquí solo importan los avisos y la tabla
vi.mock('../componentes/Graficos', () => ({ GraficoSeries: () => <div role="img" aria-label="gráfico" /> }))

const resumen = (id: string) => ({
  id, estrategia: 'pesos_iguales', parametros: {}, universo: ['AAA'], fecha_inicio: '2020-01-01', fecha_fin: '2021-12-31', benchmark: 'SPY',
  frecuencia_rebalanceo: 'monthly', etiqueta: '', estado: 'hecho', error: null, creado: null, actualizado: null, metricas: {},
})

function simularApi(ceteris: CeterisParibus, avisosMarco: string[] = []) {
  const comparacion: Comparacion = {
    ids: ['a-1', 'b-2'], fechas: ['2020-01-02', '2020-01-03'], series: { 'a-1': [100, 101], 'b-2': [100, 102] }, benchmark: [100, 100],
    metricas: { 'a-1': {}, 'b-2': {} }, resumenes: { 'a-1': resumen('a-1'), 'b-2': resumen('b-2') } as never, ceteris_paribus: ceteris,
  }
  vi.stubGlobal('fetch', vi.fn(async (ruta: string) => {
    let cuerpo: unknown = { experimentos: [] }
    if (ruta.startsWith('/api/comparar')) cuerpo = comparacion
    if (ruta.startsWith('/api/comparativas/')) {
      cuerpo = {
        id: 'cmp-1', id_marco: 'marco-x', marco: {}, creada: '',
        experimentos: [{
          id_experimento: 'a-1', estrategia: 'momentum', parametros: {}, reutilizado: false,
          informe: { estrategia: 'momentum', aplicados: {}, no_aplicados: ['ventana_dias'], sobrescritos: [], avisos: avisosMarco },
        }],
      }
    }
    return { ok: true, status: 200, json: async () => cuerpo }
  }))
}

function montar(url: string) {
  const cliente = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={cliente}>
      <MemoryRouter initialEntries={[url]}><Comparar /></MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('Comparar', () => {
  it('avisa de que no es ceteris paribus y nombra las diferencias', async () => {
    simularApi({
      ok: false,
      diferencias: [
        { campo: 'fecha_fin', tipo: 'entorno', valores: { 'a-1': '2021-12-31', 'b-2': '2022-06-30' } },
        { campo: 'ventana_dias', tipo: 'comun', valores: { 'a-1': 504, 'b-2': 252 } },
      ],
    })
    montar('/comparar?ids=a-1,b-2')
    expect(await screen.findByText('No es ceteris paribus: difieren fecha de fin y el parámetro común ventana_dias.')).toBeInTheDocument()
  })

  it('no avisa cuando la comparación es ceteris paribus', async () => {
    simularApi({ ok: true, diferencias: [] })
    montar('/comparar?ids=a-1,b-2')
    expect(await screen.findByRole('table', { name: 'Métricas lado a lado' })).toBeInTheDocument()
    expect(screen.queryByText(/ceteris paribus/i)).not.toBeInTheDocument()
  })

  it('con una comparativa muestra los avisos del marco', async () => {
    simularApi({ ok: true, diferencias: [] }, ['No se aplicó en momentum: ventana_dias (la estrategia no lo declara)'])
    montar('/comparar?ids=a-1,b-2&comparativa=cmp-1')
    expect(await screen.findByText(/No se aplicó en momentum: ventana_dias/)).toBeInTheDocument()
  })
})
