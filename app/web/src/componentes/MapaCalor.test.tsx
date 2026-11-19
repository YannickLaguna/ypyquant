/**
 * Tests del mapa de calor: marcas de mejor/meseta, clic que abre el experimento y celdas vacías.
 * Se relaciona con: componentes/MapaCalor.tsx y lib/barridos.ts (matrizMapaCalor).
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { FilaSuperficie } from '../api/tipos'
import { matrizMapaCalor } from '../lib/barridos'
import MapaCalor from './MapaCalor'

const fila = (id: string, a: number, b: number, objetivo: number): FilaSuperficie => ({
  id, etiqueta: id, valores: { a, b }, posiciones: { a: a - 1, b: b - 1 }, metricas: { sharpe: objetivo }, objetivo, meseta: objetivo, n_vecinos: 2,
})
const ejes = { a: [1, 2], b: [1, 2] }
const superficie = [fila('e1', 1, 1, 0.5), fila('e2', 2, 1, 1.5), fila('e3', 1, 2, 1.0)] // falta (2,2)

describe('MapaCalor', () => {
  it('marca la mejor, abre el experimento al pulsar y deshabilita las celdas vacías', async () => {
    const abrir = vi.fn()
    const matriz = matrizMapaCalor(superficie, ejes, 'a', 'b', {}, 'sharpe', 'sharpe')
    render(<MapaCalor titulo="sharpe" matriz={matriz} idMejor="e2" idMeseta="e3" onAbrir={abrir} />)
    const mejor = screen.getByRole('button', { name: /a=2, b=1.*mejor/ })
    await userEvent.click(mejor)
    expect(abrir).toHaveBeenCalledWith('e2')
    expect(screen.getByRole('button', { name: /a=1, b=2.*meseta/ })).toBeEnabled()
    expect(screen.getByRole('button', { name: /a=2, b=2/ })).toBeDisabled()
  })
})
