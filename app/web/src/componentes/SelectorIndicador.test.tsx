/**
 * Tests del selector de indicador: grupos accesibles, selección marcada, cambio con clic y con teclado (flechas).
 * Se relaciona con: componentes/SelectorIndicador.tsx y lib/indicadores.ts.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import SelectorIndicador from './SelectorIndicador'

function Probar({ inicial = 'equity' }: { inicial?: string }) {
  const [valor, setValor] = useState(inicial)
  return (
    <>
      <SelectorIndicador valor={valor} onChange={setValor} />
      <output aria-label="elegido">{valor}</output>
    </>
  )
}

describe('SelectorIndicador', () => {
  it('ofrece los dos grupos y marca el indicador actual', () => {
    render(<Probar inicial="sharpe" />)
    expect(screen.getByRole('group', { name: /Resumen/ })).toBeInTheDocument()
    expect(screen.getByRole('group', { name: /En el tiempo/ })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Sharpe' })).toBeChecked()
    expect(screen.getByRole('radio', { name: 'Drawdown' })).not.toBeChecked()
  })

  it('cambia de indicador con el clic', async () => {
    render(<Probar />)
    await userEvent.click(screen.getByRole('radio', { name: 'Turnover anual' }))
    expect(screen.getByLabelText('elegido')).toHaveTextContent('turnover_anual')
  })

  it('se maneja con el teclado: Tab entra al grupo y las flechas cambian la selección', async () => {
    render(<Probar inicial="sharpe" />)
    screen.getByRole('radio', { name: 'Sharpe' }).focus()
    await userEvent.keyboard('{ArrowRight}')
    expect(screen.getByLabelText('elegido')).toHaveTextContent('sortino')
  })
})
