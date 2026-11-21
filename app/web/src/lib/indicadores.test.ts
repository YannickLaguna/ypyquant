/**
 * Tests de las utilidades puras de indicadores de Comparar (lib/indicadores.ts): indicador de la URL, orden de las
 * barras, «sin dato» (nunca 0) y geometría con valores negativos.
 * Se relaciona con: lib/indicadores.ts, componentes/GraficoBarras.tsx y paginas/Comparar.tsx.
 */
import { describe, expect, it } from 'vitest'
import { barrasOrdenadas, geometriaBarra, INDICADORES, indicadorDeUrl, rangoBarras } from './indicadores'

describe('indicadorDeUrl', () => {
  it('acepta un indicador conocido y cae en la equity con uno desconocido o ausente', () => {
    expect(indicadorDeUrl('sharpe').clave).toBe('sharpe')
    expect(indicadorDeUrl('drawdown').tipo).toBe('temporal')
    expect(indicadorDeUrl('inventado').clave).toBe('equity')
    expect(indicadorDeUrl(null).clave).toBe('equity')
  })

  it('no repite claves', () => {
    const claves = INDICADORES.map((i) => i.clave)
    expect(new Set(claves).size).toBe(claves.length)
  })
})

describe('barrasOrdenadas', () => {
  const ids = ['a', 'b', 'c', 'd']
  const metricas = { a: { sharpe: 0.5, n_operaciones: 10 }, b: { sharpe: 1.2, n_operaciones: null }, c: { sharpe: null }, d: { sharpe: 0.9, n_operaciones: 3 } }

  it('ordena de mayor a menor cuando mejor es max y deja los sin dato al final', () => {
    const barras = barrasOrdenadas(ids, metricas, 'sharpe', 'max')
    expect(barras.map((b) => b.id)).toEqual(['b', 'd', 'a', 'c'])
    expect(barras[3].valor).toBeNull()
  })

  it('ordena de menor a mayor cuando mejor es min', () => {
    expect(barrasOrdenadas(ids, metricas, 'n_operaciones', 'min').map((b) => b.id)).toEqual(['d', 'a', 'b', 'c'])
  })

  it('una métrica ausente, null o no finita es sin dato, no 0', () => {
    const barras = barrasOrdenadas(['x', 'y'], { x: { sharpe: 0 }, y: {} }, 'sharpe', 'max')
    expect(barras.map((b) => b.valor)).toEqual([0, null])
    expect(barrasOrdenadas(['z'], { z: { sharpe: Number.NaN } }, 'sharpe', 'max')[0].valor).toBeNull()
  })

  it('conserva el índice original para dar a cada experimento su color', () => {
    expect(barrasOrdenadas(ids, metricas, 'sharpe', 'max').map((b) => b.indice)).toEqual([1, 3, 0, 2])
  })
})

describe('geometriaBarra y rangoBarras', () => {
  it('con solo positivos el cero está a la izquierda', () => {
    expect(geometriaBarra(2, [1, 4])).toEqual({ izquierda: 0, ancho: 50, cero: 0 })
  })

  it('con negativos la barra sale a la izquierda del cero', () => {
    const g = geometriaBarra(-0.2, [-0.4, 0])
    expect(g.cero).toBe(100)
    expect(g.ancho).toBeCloseTo(50)
    expect(g.izquierda).toBeCloseTo(50)
  })

  it('con signos mixtos el cero queda dentro del eje', () => {
    const positiva = geometriaBarra(1, [-1, 3])
    expect(positiva.cero).toBeCloseTo(25)
    expect(positiva.izquierda).toBeCloseTo(25)
    expect(geometriaBarra(-1, [-1, 3]).izquierda).toBeCloseTo(0)
  })

  it('rangoBarras ignora los sin dato y es null si no hay ninguno', () => {
    expect(rangoBarras([{ id: 'a', indice: 0, valor: 2 }, { id: 'b', indice: 1, valor: null }, { id: 'c', indice: 2, valor: -1 }])).toEqual([-1, 2])
    expect(rangoBarras([{ id: 'a', indice: 0, valor: null }])).toBeNull()
  })
})
