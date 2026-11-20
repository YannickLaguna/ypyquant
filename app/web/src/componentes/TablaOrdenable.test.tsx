/**
 * Tests de TablaOrdenable: orden por columna y límite de filas visibles («20 + Ver todas»).
 * Se relaciona con: componentes/TablaOrdenable.tsx (la usan Experimentos, ExperimentoDetalle, PaperTrading...).
 */
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import TablaOrdenable, { type Columna } from './TablaOrdenable'

interface Fila {
  id: string
  valor: number
}

const columnas: Columna<Fila>[] = [
  { clave: 'id', titulo: 'Id', valor: (f) => f.id, render: (f) => f.id },
  { clave: 'valor', titulo: 'Valor', alinear: 'der', valor: (f) => f.valor, render: (f) => String(f.valor) },
]

const filas = (n: number): Fila[] => Array.from({ length: n }, (_, i) => ({ id: `f${i + 1}`, valor: i + 1 }))
const celdasId = () => within(screen.getByRole('table', { name: 'Prueba' })).getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('cell')[0].textContent)

describe('TablaOrdenable con límite de filas', () => {
  it('sin límite muestra todas las filas y ningún botón', () => {
    render(<TablaOrdenable titulo="Prueba" columnas={columnas} filas={filas(25)} clave={(f) => f.id} />)
    expect(celdasId()).toHaveLength(25)
    expect(screen.queryByRole('button', { name: /Ver/ })).not.toBeInTheDocument()
  })

  it('con límite 20 muestra 20 filas y «Ver las N» las despliega todas; «Ver menos» las recoge', async () => {
    render(<TablaOrdenable titulo="Prueba" columnas={columnas} filas={filas(25)} clave={(f) => f.id} filasVisibles={20} />)
    expect(celdasId()).toHaveLength(20)
    const usuario = userEvent.setup()
    await usuario.click(screen.getByRole('button', { name: 'Ver las 25' }))
    expect(celdasId()).toHaveLength(25)
    await usuario.click(screen.getByRole('button', { name: 'Ver menos' }))
    expect(celdasId()).toHaveLength(20)
  })

  it('no muestra el botón si las filas caben en el límite', () => {
    render(<TablaOrdenable titulo="Prueba" columnas={columnas} filas={filas(20)} clave={(f) => f.id} filasVisibles={20} />)
    expect(celdasId()).toHaveLength(20)
    expect(screen.queryByRole('button', { name: /Ver/ })).not.toBeInTheDocument()
  })

  it('ordena TODAS las filas y después recorta (el mayor valor sale arriba aunque esté fuera de las primeras 20)', async () => {
    render(<TablaOrdenable titulo="Prueba" columnas={columnas} filas={filas(25)} clave={(f) => f.id} filasVisibles={20} />)
    const usuario = userEvent.setup()
    await usuario.click(screen.getByRole('button', { name: /Valor/ })) // primer clic: descendente
    expect(celdasId()[0]).toBe('f25')
    expect(celdasId()).toHaveLength(20)
  })
})
