/**
 * Tests de la página Comparar: el aviso «no es ceteris paribus» solo aparece cuando los experimentos difieren
 * del entorno o de los parámetros comunes, y los avisos del marco aparecen al abrir una comparativa.
 * Se relaciona con: paginas/Comparar.tsx, lib/marcos.ts (resumenDiferencias) y api/hooks.ts (useComparacion).
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { CeterisParibus, Comparacion } from '../api/tipos'
import Comparar from './Comparar'

// El gráfico usa canvas y matchMedia, que jsdom no implementa: aquí solo importan los avisos y la tabla
vi.mock('../componentes/Graficos', () => ({ GraficoSeries: ({ titulo, series }: { titulo: string; series: unknown[] }) => <div role="img" aria-label={`${titulo} (${series.length} series)`} /> }))

const resumen = (id: string) => ({
  id, estrategia: 'pesos_iguales', parametros: {}, universo: ['AAA'], fecha_inicio: '2020-01-01', fecha_fin: '2021-12-31', benchmark: 'SPY',
  frecuencia_rebalanceo: 'monthly', etiqueta: '', estado: 'hecho', error: null, creado: null, actualizado: null, metricas: {},
})

function simularApi(ceteris: CeterisParibus, avisosMarco: string[] = []) {
  const comparacion: Comparacion = {
    ids: ['a-1', 'b-2'], fechas: ['2020-01-02', '2020-01-03'], series: { 'a-1': [100, 101], 'b-2': [100, 102] }, benchmark: [100, 100],
    metricas: { 'a-1': { sharpe: 1.2, max_drawdown: -0.3, n_operaciones: 40 }, 'b-2': { sharpe: 0.4, max_drawdown: -0.1 } }, resumenes: { 'a-1': resumen('a-1'), 'b-2': resumen('b-2') } as never, ceteris_paribus: ceteris,
  }
  vi.stubGlobal('fetch', vi.fn(async (ruta: string) => {
    let cuerpo: unknown = { experimentos: [] }
    if (ruta.startsWith('/api/comparar/series'))
      cuerpo = {
        indicador: 'turnover_rebalanceo', etiqueta: 'Turnover por rebalanceo', ids: ['a-1', 'b-2'], periodos_por_anio: 252, sin_dato: ['b-2'],
        series: { 'a-1': { fechas: ['2020-01-02', '2020-01-03'], valores: [0.1, 0.2] }, 'b-2': { fechas: [], valores: [] } },
      }
    else if (ruta.startsWith('/api/comparar')) cuerpo = comparacion
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

function Ruta() {
  const l = useLocation()
  return <output aria-label="ruta">{l.search}</output>
}

function montar(url: string) {
  const cliente = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={cliente}>
      <MemoryRouter initialEntries={[url]}><Comparar /><Ruta /></MemoryRouter>
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

  it('con un indicador escalar de la URL dibuja barras ordenadas y «sin dato» para quien no lo tiene', async () => {
    simularApi({ ok: true, diferencias: [] })
    montar('/comparar?ids=a-1,b-2&indicador=n_operaciones')
    const figura = await screen.findByRole('figure', { name: 'Operaciones por experimento' })
    const filas = Array.from(figura.querySelectorAll('li')).map((li) => li.getAttribute('aria-label'))
    expect(filas).toEqual(['a-1: 40', 'b-2: sin dato'])
    expect(screen.getByRole('radio', { name: 'Operaciones' })).toBeChecked()
  })

  it('cambiar de indicador actualiza la URL y conserva los ids', async () => {
    simularApi({ ok: true, diferencias: [] })
    montar('/comparar?ids=a-1,b-2')
    await userEvent.click(await screen.findByRole('radio', { name: 'Sharpe' }))
    expect(screen.getByLabelText('ruta').textContent).toContain('indicador=sharpe')
    expect(screen.getByLabelText('ruta').textContent).toContain('ids=a-1%2Cb-2')
    await userEvent.click(screen.getByRole('radio', { name: 'Equity base 100' }))
    expect(screen.getByLabelText('ruta').textContent).not.toContain('indicador')
  })

  it('un indicador temporal pide la serie al servidor y avisa de los experimentos sin dato', async () => {
    simularApi({ ok: true, diferencias: [] })
    montar('/comparar?ids=a-1,b-2&indicador=turnover_rebalanceo')
    expect(await screen.findByRole('img', { name: 'Turnover por rebalanceo (1 series)' })).toBeInTheDocument()
    expect(screen.getByText(/Sin dato para Turnover por rebalanceo: b-2/)).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Métricas lado a lado' })).toBeInTheDocument()
  })
})
