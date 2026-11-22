/**
 * Tests del selector de perfil de Alpaca: una pestaña por cuenta, la seleccionada marcada, el estado de cada una
 * («sin configurar», «sin conexión», «bot en marcha») y el cambio de perfil al pulsar.
 * Se relaciona con: componentes/SelectorPerfil.tsx y api/alpaca.ts (`PerfilAlpaca`).
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { PerfilAlpaca } from '../api/alpaca'
import { bot, perfil } from '../test/perfiles'
import SelectorPerfil, { estadoDePerfil } from './SelectorPerfil'

const TRES: PerfilAlpaca[] = [
  perfil('A', { bot: bot('A', true) }),
  perfil('B', { conexion: { configurada: true, conectada: false, equity: null, error: '401' } }),
  perfil('C', { configurado: false, ultimos4: null, faltan: ['ALPACA_C_API_KEY', 'ALPACA_C_SECRET_KEY'], conexion: { configurada: false, conectada: false, equity: null, error: null } }),
]

describe('SelectorPerfil', () => {
  it('muestra una pestaña por perfil con la seleccionada marcada', () => {
    render(<SelectorPerfil perfiles={TRES} seleccionado="B" alSeleccionar={() => {}} />)
    const pestanas = screen.getAllByRole('tab')
    expect(pestanas).toHaveLength(3)
    expect(screen.getByRole('tab', { name: /^B/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: /^A/ })).toHaveAttribute('aria-selected', 'false')
  })

  it('indica el estado de cada cuenta y solo enseña los 4 últimos de la API key', () => {
    render(<SelectorPerfil perfiles={TRES} seleccionado="A" alSeleccionar={() => {}} />)
    expect(screen.getByRole('tab', { name: /^A/ })).toHaveTextContent('bot en marcha')
    expect(screen.getByRole('tab', { name: /^B/ })).toHaveTextContent('sin conexión')
    expect(screen.getByRole('tab', { name: /^C/ })).toHaveTextContent('sin configurar')
    expect(screen.getByRole('tab', { name: /^A/ })).toHaveTextContent('…1234')
    expect(screen.getByRole('tab', { name: /^C/ })).not.toHaveTextContent('…')
  })

  it('avisa del perfil elegido al pulsar una pestaña', async () => {
    const alSeleccionar = vi.fn()
    render(<SelectorPerfil perfiles={TRES} seleccionado="A" alSeleccionar={alSeleccionar} />)
    await userEvent.click(screen.getByRole('tab', { name: /^C/ }))
    expect(alSeleccionar).toHaveBeenCalledWith('C')
  })

  it('el estado «sin configurar» tiene prioridad sobre los demás', () => {
    expect(estadoDePerfil(TRES[2]).texto).toBe('sin configurar')
    expect(estadoDePerfil(perfil('D')).texto).toBe('bot parado')
  })
})
