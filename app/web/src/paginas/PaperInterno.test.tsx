/**
 * Tests de la página Paper interno: estado vacío, tabla de sesiones (retorno, Sharpe, «al día / n días pendientes»,
 * estado), botón «Avanzar ahora» con sondeo mientras está en curso y selección múltiple -> gráfico a base 100.
 * Se relaciona con: paginas/PaperInterno.tsx, api/paper_interno.ts y lib/paperInterno.ts.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { EstadoAvancePaper, ResumenSesionPaper } from '../api/paper_interno'
import PaperInterno from './PaperInterno'

// El gráfico usa canvas, que jsdom no implementa: basta saber qué series recibe
vi.mock('../componentes/Graficos', () => ({
  GraficoSeries: ({ titulo, series }: { titulo: string; series: { nombre: string }[] }) => <div role="img" aria-label={`${titulo}: ${series.map((s) => s.nombre).join(', ')}`} />,
}))

const sesion = (cambios: Partial<ResumenSesionPaper>): ResumenSesionPaper => ({
  id: 'a', nombre: 'Momentum A', estrategia: 'momentum', modo_ejecucion: 'apertura_siguiente', estado: 'activa', fecha_alta: '2026-11-02',
  capital_inicial: 100000, equity: 101500, dias: 12, ultima_fecha: '2026-11-20', retorno_total: 0.015, sharpe: 1.234, max_drawdown: -0.021,
  volatilidad: 0.12, costes: 85, dias_pendientes: 0, al_dia: true, id_experimento: 'exp-1', id_configuracion: null, ...cambios,
})

const SIN_AVANCE: EstadoAvancePaper = { en_curso: false, inicio: null, fin: null, resultados: [], error: null }

function simularApi(sesiones: ResumenSesionPaper[], avances: EstadoAvancePaper[] = [SIN_AVANCE]) {
  let n = 0
  const llamadas: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (ruta: string, opciones?: RequestInit) => {
    llamadas.push(`${opciones?.method ?? 'GET'} ${ruta}`)
    let cuerpo: unknown = {}
    if (ruta.endsWith('/sesiones')) cuerpo = { sesiones }
    else if (ruta.endsWith('/avanzar/estado')) cuerpo = avances[Math.min(n++, avances.length - 1)]
    else if (ruta.endsWith('/avanzar')) cuerpo = { ...SIN_AVANCE, en_curso: true, inicio: '2026-11-22T20:00:00' }
    else if (ruta.endsWith('/equity')) {
      const base = ruta.includes('/b/') ? 2000 : 1000
      cuerpo = { equity: [0, 1, 2].map((i) => ({ fecha: `2026-11-0${i + 2}`, equity: base * (1 + 0.01 * i), efectivo: 0, n_posiciones: 2, benchmark: 400 + i, retorno: null, drawdown: null })) }
    }
    return { ok: true, status: 200, json: async () => cuerpo }
  }))
  return llamadas
}

function montar() {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter><PaperInterno /></MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('Paper interno', () => {
  it('sin sesiones muestra el estado vacío con la indicación de dónde crearlas', async () => {
    simularApi([])
    montar()
    expect(await screen.findByText('Todavía no hay sesiones de paper interno')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Avanzar ahora' })).toBeEnabled()
  })

  it('la tabla muestra retorno, Sharpe, drawdown, costes, estado y días pendientes', async () => {
    simularApi([sesion({}), sesion({ id: 'b', nombre: 'Pesos B', estrategia: 'pesos_iguales', modo_ejecucion: 'cierre', estado: 'pausada', retorno_total: -0.03, dias_pendientes: 3, al_dia: false })])
    montar()
    const tabla = await screen.findByRole('table', { name: 'Sesiones de paper interno' })
    const fila = within(tabla).getByText('Momentum A').closest('tr')!
    expect(within(fila).getByText('1,50 %')).toBeInTheDocument()
    expect(within(fila).getByText('1,23')).toBeInTheDocument()
    expect(within(fila).getByText('Al día')).toBeInTheDocument()
    expect(within(fila).getByText('activa')).toBeInTheDocument()
    const otra = within(tabla).getByText('Pesos B').closest('tr')!
    expect(within(otra).getByText('3 días pendientes')).toBeInTheDocument()
    expect(within(otra).getByText('pausada')).toBeInTheDocument()
    expect(within(otra).getByText('Al cierre (optimista)')).toBeInTheDocument()
  })

  it('«Avanzar ahora» lanza el avance, se deshabilita mientras está en curso y muestra el resultado', async () => {
    const llamadas = simularApi([sesion({})], [
      SIN_AVANCE,
      { ...SIN_AVANCE, en_curso: true, inicio: '2026-11-22T20:00:00' },
      { en_curso: false, inicio: '2026-11-22T20:00:00', fin: '2026-11-22T20:00:03', error: null, resultados: [{ id_sesion: 'a', dias_procesados: 4, desde: '2026-11-17', hasta: '2026-11-20', avisos: [], omitida: null, error: null }] },
    ])
    montar()
    await screen.findByText('Momentum A')
    await userEvent.click(screen.getByRole('button', { name: 'Avanzar ahora' }))
    expect(llamadas).toContain('POST /api/paper/interno/avanzar')
    expect(await screen.findByRole('button', { name: 'Avanzando…' })).toBeDisabled()
    await waitFor(() => expect(screen.getByText(/4 días procesados en 1 sesión/)).toBeInTheDocument(), { timeout: 5000 })
    expect(screen.getByRole('button', { name: 'Avanzar ahora' })).toBeEnabled()
  })

  it('muestra los errores del avance', async () => {
    simularApi([sesion({})], [{ ...SIN_AVANCE, fin: '2026-11-22T20:00:03', resultados: [{ id_sesion: 'a', dias_procesados: 0, desde: null, hasta: null, avisos: [], omitida: null, error: 'RuntimeError: sin precios' }] }])
    montar()
    expect(await screen.findByRole('alert')).toHaveTextContent('RuntimeError: sin precios')
  })

  it('la selección múltiple dibuja las curvas elegidas y el benchmark', async () => {
    simularApi([sesion({}), sesion({ id: 'b', nombre: 'Pesos B' })])
    montar()
    await screen.findByText('Momentum A')
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    await userEvent.click(screen.getByLabelText('Seleccionar Momentum A'))
    await userEvent.click(screen.getByLabelText('Seleccionar Pesos B'))
    expect(await screen.findByRole('img', { name: 'Equity normalizada (base 100): Momentum A, Pesos B, Benchmark' })).toBeInTheDocument()
    await userEvent.click(screen.getByLabelText('Seleccionar Pesos B'))
    expect(await screen.findByRole('img', { name: 'Equity normalizada (base 100): Momentum A, Benchmark' })).toBeInTheDocument()
  })
})
