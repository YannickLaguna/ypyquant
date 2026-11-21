/**
 * Tests de las utilidades de las estrategias de señales: etiquetas de motivo y de efectivo, y el resumen de motivos.
 * Se relaciona con: lib/senales.ts.
 */
import { describe, expect, it } from 'vitest'
import type { OperacionFila } from '../api/tipos'
import { etiquetaMotivo, nombreCapa, resumenMotivos } from './senales'

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
