/**
 * Tests de la tabla de combinaciones del detalle de barrido: casillas de selección y botón
 * «Comparar seleccionados», que navega a `/comparar?ids=...` (la misma convención que Experimentos).
 * Se relaciona con: paginas/BarridoDetalle.tsx (SeccionTabla), paginas/Comparar.tsx y componentes/TablaOrdenable.tsx.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import type { FilaSuperficie } from '../api/tipos'
import { SeccionTabla } from './BarridoDetalle'

const fila = (id: string, objetivo: number): FilaSuperficie => ({
  id, etiqueta: `top_k=${id}`, valores: {}, posiciones: {}, metricas: { sharpe: objetivo }, objetivo, meseta: objetivo, n_vecinos: 2,
})
const SUPERFICIE = [fila('mom-1', 0.5), fila('mom-2', 1.2), fila('mom-3', 0.9)]

function DestinoComparar() {
  const ubicacion = useLocation()
  return <p>vista comparar {ubicacion.search}</p>
}

function montar() {
  render(
    <MemoryRouter initialEntries={['/barridos/b1']}>
      <Routes>
        <Route path="/barridos/b1" element={<SeccionTabla superficie={SUPERFICIE} metricaObjetivo="sharpe" />} />
        <Route path="/comparar" element={<DestinoComparar />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('Tabla de combinaciones del barrido', () => {
  it('«Comparar seleccionados» está desactivado con menos de 2 y navega a Comparar con los ids marcados', async () => {
    montar()
    const comparar = screen.getByRole('button', { name: /Comparar seleccionados/ })
    expect(comparar).toBeDisabled()
    const usuario = userEvent.setup()
    await usuario.click(screen.getByLabelText('Seleccionar mom-1 para comparar'))
    expect(comparar).toBeDisabled()
    await usuario.click(screen.getByLabelText('Seleccionar mom-3 para comparar'))
    expect(comparar).toBeEnabled()
    expect(comparar).toHaveTextContent('(2)')
    await usuario.click(comparar)
    expect(await screen.findByText('vista comparar ?ids=mom-1,mom-3')).toBeInTheDocument()
  })

  it('desmarcar una casilla la quita de la selección', async () => {
    montar()
    const usuario = userEvent.setup()
    const casilla = screen.getByLabelText('Seleccionar mom-2 para comparar')
    await usuario.click(casilla)
    expect(casilla).toBeChecked()
    await usuario.click(casilla)
    expect(casilla).not.toBeChecked()
    expect(screen.getByRole('button', { name: /Comparar seleccionados/ })).toHaveTextContent('(0)')
  })
})
