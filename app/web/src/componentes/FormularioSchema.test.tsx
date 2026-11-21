/**
 * Tests del formulario autogenerado desde JSON Schema: controles por tipo, límites, validación y cambios.
 * Se relaciona con: componentes/FormularioSchema.tsx.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { EsquemaParametros } from '../api/tipos'
import FormularioSchema, { validarValores, valoresPorDefecto, type Valores } from './FormularioSchema'

const ESQUEMA: EsquemaParametros = {
  type: 'object',
  properties: {
    top_k: { type: 'integer', title: 'Top K', description: 'Número de ganadores', default: 5, minimum: 1, maximum: 200 },
    peso_maximo: { type: 'number', title: 'Peso máximo', default: 0.3, exclusiveMinimum: 0, maximum: 1 },
    modo: { type: 'string', title: 'Modo', enum: ['rapido', 'lento'], default: 'rapido' },
    activo: { type: 'boolean', title: 'Activo', default: true },
    opcional: { anyOf: [{ type: 'integer', minimum: 2 }, { type: 'null' as never }], title: 'Opcional', default: 3 },
  },
}

function Envoltorio({ alCambiar }: { alCambiar?: (v: Valores) => void }) {
  const [valores, setValores] = useState<Valores>(valoresPorDefecto(ESQUEMA))
  return (
    <FormularioSchema
      esquema={ESQUEMA}
      valores={valores}
      onChange={(v) => {
        setValores(v)
        alCambiar?.(v)
      }}
    />
  )
}

describe('FormularioSchema', () => {
  it('genera un control por propiedad con etiqueta, valor por defecto y límites', () => {
    render(<Envoltorio />)
    const topK = screen.getByLabelText('Top K') as HTMLInputElement
    expect(topK).toHaveValue(5)
    expect(topK).toHaveAttribute('min', '1')
    expect(topK).toHaveAttribute('max', '200')
    expect(topK).toHaveAttribute('step', '1')
    expect(screen.getByText(/Número de ganadores/)).toBeInTheDocument()
    expect(screen.getByLabelText('Peso máximo')).toHaveAttribute('step', 'any')
    expect(screen.getByLabelText('Modo')).toHaveValue('rapido')
    expect(screen.getByLabelText('Activo')).toBeChecked()
    expect(screen.getByLabelText('Opcional')).toHaveValue(3) // anyOf nullable resuelto a integer
  })

  it('propaga los cambios con el tipo correcto', async () => {
    const alCambiar = vi.fn()
    render(<Envoltorio alCambiar={alCambiar} />)
    const usuario = userEvent.setup()
    const topK = screen.getByLabelText('Top K')
    await usuario.clear(topK)
    await usuario.type(topK, '12')
    expect(alCambiar).toHaveBeenLastCalledWith(expect.objectContaining({ top_k: 12 }))
    await usuario.selectOptions(screen.getByLabelText('Modo'), 'lento')
    expect(alCambiar).toHaveBeenLastCalledWith(expect.objectContaining({ modo: 'lento' }))
    await usuario.click(screen.getByLabelText('Activo'))
    expect(alCambiar).toHaveBeenLastCalledWith(expect.objectContaining({ activo: false }))
  })

  it('muestra el error de validación cuando se sale del rango', async () => {
    render(<Envoltorio />)
    const usuario = userEvent.setup()
    const topK = screen.getByLabelText('Top K')
    await usuario.clear(topK)
    await usuario.type(topK, '500')
    expect(screen.getByText('Máximo 200')).toBeInTheDocument()
    expect(topK).toHaveAttribute('aria-invalid', 'true')
  })

  it('indica cuando no hay parámetros', () => {
    render(<FormularioSchema esquema={{ type: 'object', properties: {} }} valores={{}} onChange={() => {}} />)
    expect(screen.getByText(/no tiene parámetros/)).toBeInTheDocument()
  })
})

describe('validarValores / valoresPorDefecto', () => {
  it('los valores por defecto son válidos', () => {
    expect(validarValores(ESQUEMA, valoresPorDefecto(ESQUEMA))).toEqual({})
  })
  it('detecta mínimo, exclusivo, entero y vacío', () => {
    const errores = validarValores(ESQUEMA, { top_k: 0, peso_maximo: 0, opcional: 2.5, modo: 'rapido', activo: true })
    expect(errores.top_k).toBe('Mínimo 1')
    expect(errores.peso_maximo).toBe('Debe ser mayor que 0')
    expect(errores.opcional).toBe('Debe ser un entero')
    expect(validarValores(ESQUEMA, { ...valoresPorDefecto(ESQUEMA), top_k: NaN }).top_k).toBe('Introduce un número')
  })
})

// Parámetros opcionales con valor por defecto nulo (p. ej. `stop_perdida: float | None = None` de las estrategias de señales)
const ESQUEMA_OPCIONAL: EsquemaParametros = {
  type: 'object',
  properties: {
    stop_perdida: { anyOf: [{ type: 'number', exclusiveMinimum: 0, exclusiveMaximum: 1 }, { type: 'null' as never }], title: 'Stop de pérdida', default: null },
  },
}

describe('parámetros opcionales (anyOf con null)', () => {
  it('un opcional vacío con defecto nulo es válido y el control se pinta vacío', () => {
    expect(validarValores(ESQUEMA_OPCIONAL, valoresPorDefecto(ESQUEMA_OPCIONAL))).toEqual({})
    render(<FormularioSchema esquema={ESQUEMA_OPCIONAL} valores={{ stop_perdida: null }} onChange={() => {}} />)
    expect(screen.getByLabelText('Stop de pérdida')).toHaveValue(null)
  })

  it('al vaciar un opcional se envía null (no NaN) y se vuelve a validar el límite al escribir', async () => {
    const alCambiar = vi.fn()
    render(<FormularioSchema esquema={ESQUEMA_OPCIONAL} valores={{ stop_perdida: 0.1 }} onChange={alCambiar} />)
    const usuario = userEvent.setup()
    await usuario.clear(screen.getByLabelText('Stop de pérdida'))
    expect(alCambiar).toHaveBeenLastCalledWith({ stop_perdida: null })
    expect(validarValores(ESQUEMA_OPCIONAL, { stop_perdida: 1.5 }).stop_perdida).toBe('Debe ser menor que 1')
  })
})
