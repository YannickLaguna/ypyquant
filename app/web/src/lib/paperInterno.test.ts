/**
 * Tests de lib/paperInterno.ts: curvas a base 100 en la fecha común, etiqueta de pendientes y de modo.
 * Se relaciona con: lib/paperInterno.ts.
 */
import { describe, expect, it } from 'vitest'
import type { PuntoEquitySesion } from '../api/paper_interno'
import { normalizarBase100, textoModo, textoPendientes } from './paperInterno'

const punto = (fecha: string, equity: number, benchmark: number | null = null): PuntoEquitySesion => ({
  fecha, equity, efectivo: 0, n_posiciones: 1, benchmark, retorno: null, drawdown: null,
})

describe('normalizarBase100', () => {
  it('lleva cada curva a 100 en la primera fecha común y descarta lo anterior', () => {
    const curvas = normalizarBase100([
      { id: 'a', equity: [punto('2021-01-04', 1000), punto('2021-01-05', 1100), punto('2021-01-06', 1210)] },
      { id: 'b', equity: [punto('2021-01-05', 50), punto('2021-01-06', 55)] },
    ])
    expect(curvas[0].puntos.map((p) => p.time)).toEqual(['2021-01-05', '2021-01-06'])
    expect(curvas[0].puntos[0].value).toBe(100)
    expect(curvas[0].puntos[1].value).toBeCloseTo(110, 9)
    expect(curvas[1].puntos.map((p) => p.time)).toEqual(['2021-01-05', '2021-01-06'])
    expect(curvas[1].puntos[1].value).toBeCloseTo(110, 9)
  })

  it('normaliza también el benchmark y no falla sin él', () => {
    const [con, sin] = normalizarBase100([
      { id: 'a', equity: [punto('2021-01-04', 100, 400), punto('2021-01-05', 100, 420)] },
      { id: 'b', equity: [punto('2021-01-04', 100), punto('2021-01-05', 100)] },
    ])
    expect(con.benchmark.map((p) => p.value)).toEqual([100, 105])
    expect(sin.benchmark).toEqual([])
  })

  it('sin fecha común cada curva usa su propio primer punto, y las vacías se omiten', () => {
    const curvas = normalizarBase100([
      { id: 'a', equity: [punto('2021-01-04', 10), punto('2021-01-05', 20)] },
      { id: 'b', equity: [punto('2021-02-01', 5), punto('2021-02-02', 10)] },
      { id: 'vacia', equity: [] },
    ])
    expect(curvas.map((c) => c.id)).toEqual(['a', 'b'])
    expect(curvas[0].puntos.map((p) => p.value)).toEqual([100, 200])
    expect(curvas[1].puntos.map((p) => p.value)).toEqual([100, 200])
  })
})

describe('textos', () => {
  it('pendientes: al día, singular y plural', () => {
    expect(textoPendientes(0, true)).toBe('Al día')
    expect(textoPendientes(1, false)).toBe('1 día pendiente')
    expect(textoPendientes(3, false)).toBe('3 días pendientes')
  })
  it('modo de ejecución', () => {
    expect(textoModo('cierre')).toContain('optimista')
    expect(textoModo('apertura_siguiente')).toBe('Apertura siguiente')
  })
})
