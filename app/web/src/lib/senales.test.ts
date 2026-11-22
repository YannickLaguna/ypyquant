/**
 * Tests de las utilidades de las estrategias de señales: etiquetas de motivo y de efectivo, y el resumen de motivos.
 * Se relaciona con: lib/senales.ts.
 */
import { describe, expect, it } from 'vitest'
import type { OperacionFila } from '../api/tipos'
import { etiquetaMotivo, exposiciones, nombreCapa, resumenMotivos } from './senales'

const op = (lado: 'compra' | 'venta', motivo?: string | null): OperacionFila => ({
  fecha: '2020-01-02', ticker: 'AAA', lado, cantidad: 1, precio: 1, comision: 0, pnl: null, motivo,
})

describe('etiquetas', () => {
  it('traduce los motivos conocidos y deja el resto tal cual', () => {
    expect(etiquetaMotivo('stop_perdida')).toBe('Stop de pérdida')
    expect(etiquetaMotivo('senal')).toBe('Señal')
    expect(etiquetaMotivo('ajuste')).toBe('Ajuste')
    expect(etiquetaMotivo('algo_nuevo')).toBe('algo_nuevo')
    expect(etiquetaMotivo(null)).toBe('')
  })
  it('CASH se muestra como Efectivo', () => {
    expect(nombreCapa('CASH')).toBe('Efectivo')
    expect(nombreCapa('AAPL')).toBe('AAPL')
  })
})

describe('resumenMotivos', () => {
  it('cuenta entradas, salidas y salidas por motivo (ignora los ajustes)', () => {
    const ops = [op('compra', 'senal'), op('compra', 'senal'), op('venta', 'stop_perdida'), op('venta', 'senal'), op('compra', 'ajuste'), op('venta', 'ajuste')]
    expect(resumenMotivos(ops)).toEqual({ entradas: 2, salidas: 2, porMotivo: { senal: 1, stop_perdida: 1 }, ajustes: 2 })
  })
  it('devuelve null si ninguna operación trae motivo (estrategias sin señales)', () => {
    expect(resumenMotivos([op('compra'), op('venta', null)])).toBeNull()
  })
})

describe('margen y cortos en los pesos', () => {
  it('rotula el CASH negativo como préstamo de margen y un peso negativo como corto', () => {
    expect(nombreCapa('CASH', -0.5)).toBe('Préstamo (margen)')
    expect(nombreCapa('CASH', 0.2)).toBe('Efectivo')
    expect(nombreCapa('AAPL', -0.15)).toBe('AAPL (corto)')
    expect(nombreCapa('AAPL', 0.3)).toBe('AAPL')
    expect(nombreCapa('AAPL')).toBe('AAPL')
  })

  it('calcula la exposición bruta y neta sin contar el CASH', () => {
    const e = exposiciones([{ ticker: 'A', peso: 1.3 }, { ticker: 'B', peso: -0.3 }, { ticker: 'CASH', peso: -0.2 }])
    expect(e?.bruta).toBeCloseTo(1.6)
    expect(e?.neta).toBeCloseTo(1.0)
    expect(exposiciones([])).toBeNull()
  })
})
