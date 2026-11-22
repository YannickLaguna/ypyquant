/**
 * Tests de la tabla «Salidas por motivo» de la ficha de un experimento de señales: una fila por motivo y lado de la posición con
 * nº, PnL, tasa de acierto y duración media; no se pinta sin filas.
 * Se relaciona con: componentes/TablaMotivosSalida.tsx y lib/senales.ts.
 */
import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { MotivoSalida } from '../api/tipos'
import TablaMotivosSalida from './TablaMotivosSalida'

const FILAS: MotivoSalida[] = [
  { motivo: 'take_profit', lado: 'largo', n: 4, pnl_total: 1200, pnl_medio: 300, tasa_acierto: 0.75, duracion_media_dias: 21.5 },
  { motivo: 'stop_loss', lado: 'corto', n: 2, pnl_total: -800, pnl_medio: -400, tasa_acierto: 0, duracion_media_dias: null },
]

describe('TablaMotivosSalida', () => {
  it('muestra el motivo, el lado, el número, el PnL, el acierto y la duración de cada fila', () => {
    render(<TablaMotivosSalida filas={FILAS} />)
    const tabla = screen.getByRole('table', { name: 'Salidas por motivo' })
    const take = within(tabla).getByText('Take profit').closest('tr') as HTMLElement
    expect(within(take).getByText('Largo')).toBeInTheDocument()
    expect(take.textContent).toContain('75,0 %')
    expect(take.textContent).toContain('21,5')
    const stop = within(tabla).getByText('Stop de pérdida').closest('tr') as HTMLElement
    expect(within(stop).getByText('Corto')).toBeInTheDocument()
    expect(stop.textContent).toContain('—')           // sin duración conocida
  })

  it('no pinta nada si no hay filas', () => {
    const { container } = render(<TablaMotivosSalida filas={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})
