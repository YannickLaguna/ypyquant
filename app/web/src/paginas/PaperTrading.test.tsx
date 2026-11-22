/**
 * Tests de la página Paper trading multi-perfil con `fetch` simulado: pestañas por cuenta, cambio de perfil (consultas con
 * `?perfil=`), perfil «sin configurar» sin consultar su cuenta, y arranque del bot con el experimento elegido.
 * Se relaciona con: paginas/PaperTrading.tsx, componentes/{SelectorPerfil,PerfilSinConfigurar,BotPerfil}.tsx y api/alpaca.ts.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { PerfilAlpaca } from '../api/alpaca'
import { perfil } from '../test/perfiles'
import PaperTrading, { perfilInicial } from './PaperTrading'

vi.mock('../componentes/Graficos', () => ({ GraficoSeries: () => <div>gráfico</div> }))

const PERFILES: PerfilAlpaca[] = [
  perfil('A'),
  perfil('B', { configurado: false, ultimos4: null, faltan: ['ALPACA_B_API_KEY', 'ALPACA_B_SECRET_KEY'], conexion: { configurada: false, conectada: false, equity: null, error: null } }),
  perfil('C', { conexion: { configurada: true, conectada: true, equity: 50_000, error: null } }),
]
const EXPERIMENTO = { id: 'momentum-abc', estrategia: 'momentum', parametros: { top_k: 5 }, universo: ['A'], estado: 'hecho', metricas: {} }

function simular() {
  const llamadas: { ruta: string; metodo: string; cuerpo?: unknown }[] = []
  vi.stubGlobal('fetch', vi.fn(async (ruta: string, opciones?: { method?: string; body?: string }) => {
    const metodo = opciones?.method ?? 'GET'
    llamadas.push({ ruta, metodo, cuerpo: opciones?.body ? JSON.parse(opciones.body) : undefined })
    const ok = (cuerpo: unknown) => ({ ok: true, status: 200, json: async () => cuerpo })
    if (ruta.startsWith('/api/paper/alpaca/perfiles')) return ok({ perfiles: PERFILES, limite_activos: 3, activos: [], aviso: null })
    if (ruta.includes('/log')) return ok({ lineas: [], siguiente: 0 })
    if (ruta.includes('/arrancar')) return ok({ ...PERFILES[0].bot, en_marcha: true })
    if (ruta.startsWith('/api/paper/alpaca/bots/')) return ok({ ...PERFILES[0].bot, vinculo: null })
    if (ruta.startsWith('/api/experimentos')) return ok({ experimentos: [EXPERIMENTO] })
    if (ruta.startsWith('/api/paper/cuenta')) return ok({ equity: 1000, efectivo: 10, poder_compra: 20, pnl_dia: 1, pnl_dia_pct: 0.001 })
    if (ruta.startsWith('/api/paper/posiciones')) return ok({ posiciones: [] })
    if (ruta.startsWith('/api/paper/ordenes')) return ok({ ordenes: [] })
    if (ruta.startsWith('/api/paper/fills')) return ok({ fills: [] })
    if (ruta.startsWith('/api/paper/historial')) return ok({ fechas: [], equity: [], pnl: [], base: null })
    if (ruta.includes('/instantaneas')) return ok({ instantaneas: [] })
    return ok({})
  }))
  return llamadas
}

function montar(ruta = '/paper') {
  const cliente = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <QueryClientProvider client={cliente}>
      <MemoryRouter initialEntries={[ruta]}>
        <PaperTrading />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('perfilInicial', () => {
  it('prefiere el pedido en la URL, luego un configurado sin bot, luego el primero', () => {
    expect(perfilInicial(PERFILES, 'C')).toBe('C')
    expect(perfilInicial(PERFILES, 'ZZ')).toBe('A')
    const conBotEnA = [{ ...PERFILES[0], bot: { ...PERFILES[0].bot, en_marcha: true } }, PERFILES[1], PERFILES[2]]
    expect(perfilInicial(conBotEnA, null)).toBe('C')
    expect(perfilInicial([PERFILES[1]], null)).toBe('B')
    expect(perfilInicial([], null)).toBe('')
  })
})

describe('PaperTrading multi-perfil', () => {
  it('muestra una pestaña por cuenta y consulta la cuenta del perfil seleccionado', async () => {
    const llamadas = simular()
    montar()
    expect(await screen.findAllByRole('tab')).toHaveLength(3)
    expect(screen.getByRole('tab', { name: /^A/ })).toHaveAttribute('aria-selected', 'true')
    await waitFor(() => expect(llamadas.some((l) => l.ruta === '/api/paper/cuenta?perfil=A')).toBe(true))
    expect(await screen.findByText('Bot de la cuenta A')).toBeInTheDocument()
  })

  it('al cambiar de pestaña pide los datos del otro perfil', async () => {
    const llamadas = simular()
    montar()
    await userEvent.click(await screen.findByRole('tab', { name: /^C/ }))
    await waitFor(() => expect(llamadas.some((l) => l.ruta === '/api/paper/cuenta?perfil=C')).toBe(true))
    expect(await screen.findByText('Bot de la cuenta C')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /^C/ })).toHaveAttribute('aria-selected', 'true')
  })

  it('un perfil sin configurar se muestra «sin configurar» con las variables y no consulta su cuenta', async () => {
    const llamadas = simular()
    montar('/paper?perfil=B')
    const aviso = await screen.findByText('Perfil B: sin configurar')
    expect(aviso).toBeInTheDocument()
    expect(screen.getByLabelText('Variables del .env')).toHaveTextContent('ALPACA_B_API_KEY=')
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(screen.queryByText('Bot de la cuenta B')).toBeNull()
    expect(llamadas.some((l) => l.ruta.includes('perfil=B'))).toBe(false)
  })

  it('arranca el bot del perfil con el experimento elegido (preseleccionado desde la ficha) y el modo', async () => {
    const llamadas = simular()
    montar('/paper?perfil=A&experimento=momentum-abc')
    await screen.findByText('Bot de la cuenta A')
    await waitFor(() => expect(screen.getByLabelText('Experimento')).toHaveValue('momentum-abc'))
    await userEvent.click(screen.getByRole('radio', { name: /Diario/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Arrancar' }))
    await waitFor(() => expect(llamadas.some((l) => l.metodo === 'POST')).toBe(true))
    const envio = llamadas.find((l) => l.metodo === 'POST')!
    expect(envio.ruta).toBe('/api/paper/alpaca/bots/A/arrancar')
    expect(envio.cuerpo).toEqual({ experimento_id: 'momentum-abc', modo: 'diario', intervalo_s: 3600 })
  })

  it('el aviso del límite blando se muestra', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, json: async () => ({ perfiles: PERFILES, limite_activos: 3, activos: [], aviso: 'Hay 4 perfiles configurados' }) })))
    montar()
    expect(await screen.findByText('Hay 4 perfiles configurados')).toBeInTheDocument()
  })
})
