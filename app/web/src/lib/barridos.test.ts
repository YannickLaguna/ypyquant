/**
 * Tests de las utilidades de barridos: rejilla desde rangos, matriz del mapa de calor con ejes fijos,
 * meseta, semáforos, histograma y regresión.
 * Se relaciona con: lib/barridos.ts (código bajo prueba) y api/tipos.ts (FilaSuperficie).
 */
import { describe, expect, it } from 'vitest'
import type { FilaSuperficie } from '../api/tipos'
import {
  aniosPeriodoPermutacion, calcularMeseta, colorEscala, contarCombinaciones, costesAltos, elegirMejores, histograma,
  matrizMapaCalor, parsearValores, rangoAValores, regresionLineal, semaforoDsr, semaforoPbo, semaforoPValor, MAX_COMBINACIONES,
} from './barridos'

describe('rangoAValores', () => {
  it('genera el rango inclusivo', () => {
    expect(rangoAValores(1, 5, 2).valores).toEqual([1, 3, 5])
  })
  it('evita el ruido de coma flotante', () => {
    expect(rangoAValores(0.1, 0.5, 0.1).valores).toEqual([0.1, 0.2, 0.3, 0.4, 0.5])
  })
  it('admite pasos negativos', () => {
    expect(rangoAValores(10, 4, -3).valores).toEqual([10, 7, 4])
  })
  it('rechaza paso 0, signo contrario y rangos enormes', () => {
    expect(rangoAValores(1, 5, 0).error).toMatch(/paso/)
    expect(rangoAValores(1, 5, -1).error).toMatch(/signo/)
    expect(rangoAValores(0, 10_000, 1).error).toContain(String(MAX_COMBINACIONES))
    expect(rangoAValores(NaN, 5, 1).error).toBeDefined()
  })
})

describe('parsearValores y contarCombinaciones', () => {
  it('convierte según el tipo y elimina duplicados', () => {
    expect(parsearValores('3, 5; 5 8', 'integer').valores).toEqual([3, 5, 8])
    expect(parsearValores('0.5 1', 'number').valores).toEqual([0.5, 1])
    expect(parsearValores('true,false', 'boolean').valores).toEqual([true, false])
    expect(parsearValores('a,b', 'string').valores).toEqual(['a', 'b'])
  })
  it('informa de valores inválidos', () => {
    expect(parsearValores('3,x', 'integer').error).toMatch(/x/)
    expect(parsearValores('2.5', 'integer').error).toMatch(/entero/)
    expect(parsearValores('si', 'boolean').error).toBeDefined()
  })
  it('cuenta el producto cartesiano', () => {
    expect(contarCombinaciones({ a: [1, 2, 3], b: [1, 2] })).toBe(6)
    expect(contarCombinaciones({})).toBe(0)
    expect(contarCombinaciones({ a: [1], b: [] })).toBe(0)
  })
})

/** Superficie 3×2×2 (a × b × c) con objetivo = 10·a + b + 100·c para poder predecir cada celda. */
function superficiePrueba(): { superficie: FilaSuperficie[]; ejes: Record<string, unknown[]> } {
  const ejes = { a: [1, 2, 3], b: [10, 20], c: ['x', 'y'] }
  const superficie: FilaSuperficie[] = []
  ejes.a.forEach((a, ia) =>
    ejes.b.forEach((b, ib) =>
      ejes.c.forEach((c, ic) => {
        const objetivo = 10 * ia + ib + 100 * ic
        superficie.push({
          id: `e-${ia}${ib}${ic}`, etiqueta: '', valores: { a, b, c }, posiciones: { a: ia, b: ib, c: ic },
          metricas: { sharpe: objetivo, cagr: -objetivo }, objetivo, meseta: objetivo / 2, n_vecinos: 3,
        })
      }),
    ),
  )
  return { superficie, ejes }
}

describe('matrizMapaCalor', () => {
  it('con 2 ejes y el resto fijo selecciona la porción correcta', () => {
    const { superficie, ejes } = superficiePrueba()
    const m = matrizMapaCalor(superficie, ejes, 'a', 'b', { c: 1 }, 'sharpe', 'sharpe')
    expect(m.etiquetasX).toEqual(['1', '2', '3'])
    expect(m.etiquetasY).toEqual(['10', '20'])
    // celdas[iy][ix] con c=1 (ic=1): 100 + 10·ia + ib
    expect(m.celdas.map((fila) => fila.map((c) => c.valor))).toEqual([[100, 110, 120], [101, 111, 121]])
    expect(m.minimo).toBe(100)
    expect(m.maximo).toBe(121)
  })
  it('el eje fijo por defecto es la posición 0', () => {
    const { superficie, ejes } = superficiePrueba()
    const m = matrizMapaCalor(superficie, ejes, 'a', 'b', {}, 'sharpe', 'sharpe')
    expect(m.celdas[0][0].valor).toBe(0)
  })
  it('con 1 eje devuelve una única fila', () => {
    const { superficie, ejes } = superficiePrueba()
    const m = matrizMapaCalor(superficie, ejes, 'a', null, { b: 0, c: 0 }, 'sharpe', 'sharpe')
    expect(m.celdas).toHaveLength(1)
    expect(m.celdas[0].map((c) => c.valor)).toEqual([0, 10, 20])
  })
  it('modo meseta usa la meseta del servidor para la métrica objetivo', () => {
    const { superficie, ejes } = superficiePrueba()
    const m = matrizMapaCalor(superficie, ejes, 'a', 'b', { c: 0 }, 'sharpe', 'sharpe', 'meseta')
    expect(m.celdas[0][1].valor).toBe(5) // objetivo 10 / 2
  })
  it('otra métrica se lee de `metricas`; su meseta se calcula en cliente', () => {
    const { superficie, ejes } = superficiePrueba()
    const valor = matrizMapaCalor(superficie, ejes, 'a', 'b', { c: 0 }, 'cagr', 'sharpe')
    expect(valor.celdas[0][1].valor).toBe(-10)
    const meseta = matrizMapaCalor(superficie, ejes, 'a', 'b', { c: 0 }, 'cagr', 'sharpe', 'meseta')
    // vecinos de (a=1,b=0,c=0): a±1 -> (0,0,0)=0 y (2,0,0)=-20; b±1 -> (1,1,0)=-11; c no cuenta como eje de vecindad? sí cuenta: (1,0,1)=-110
    expect(meseta.celdas[0][1].valor).toBeCloseTo((0 + -20 + -11 + -110) / 4)
  })
  it('las celdas sin resultado (barrido en curso) quedan vacías', () => {
    const { superficie, ejes } = superficiePrueba()
    const m = matrizMapaCalor(superficie.filter((f) => f.id !== 'e-100'), ejes, 'a', 'b', { c: 0 }, 'sharpe', 'sharpe')
    expect(m.celdas[0][1]).toEqual({ fila: null, valor: null })
  })
})

describe('calcularMeseta y elegirMejores', () => {
  it('no cuenta la propia celda y devuelve null sin vecinos', () => {
    const solo: FilaSuperficie[] = [{ id: 'u', etiqueta: '', valores: { a: 1 }, posiciones: { a: 0 }, metricas: { sharpe: 1 }, objetivo: 1, meseta: null, n_vecinos: 0 }]
    expect(calcularMeseta(solo, 'sharpe', 'sharpe').get('u')).toBeNull()
  })
  it('elige el mejor por objetivo y por meseta', () => {
    const f = (id: string, objetivo: number, meseta: number | null): FilaSuperficie => ({
      id, etiqueta: id, valores: {}, posiciones: {}, metricas: {}, objetivo, meseta, n_vecinos: 2,
    })
    const { mejor, porMeseta } = elegirMejores([f('a', 3, 0.1), f('b', 1, 0.9), f('c', 2, null)])
    expect(mejor?.id).toBe('a')
    expect(porMeseta?.id).toBe('b')
    expect(elegirMejores([])).toEqual({ mejor: null, porMeseta: null })
  })
  it('colorEscala devuelve transparente sin datos y un color con datos', () => {
    expect(colorEscala(null, 0, 1)).toBe('transparent')
    expect(colorEscala(1, 0, 1)).toMatch(/^hsl\(/)
    expect(colorEscala(1, 1, 1)).toMatch(/^hsl\(/)
  })
})

describe('semáforos', () => {
  it('DSR', () => {
    expect(semaforoDsr(0.96)).toBe('verde')
    expect(semaforoDsr(0.95)).toBe('ambar')
    expect(semaforoDsr(0.3)).toBe('rojo')
    expect(semaforoDsr(null)).toBe('nulo')
  })
  it('PBO', () => {
    expect(semaforoPbo(0.05)).toBe('verde')
    expect(semaforoPbo(0.1)).toBe('ambar')
    expect(semaforoPbo(0.2)).toBe('ambar')
    expect(semaforoPbo(0.21)).toBe('rojo')
    expect(semaforoPbo(null)).toBe('nulo')
  })
  it('p-valor in-sample: < 1 % verde', () => {
    expect(semaforoPValor(0.009, 'in_sample', null)).toBe('verde')
    expect(semaforoPValor(0.03, 'in_sample', null)).toBe('ambar')
    expect(semaforoPValor(0.2, 'in_sample', null)).toBe('rojo')
  })
  it('p-valor walk-forward: umbral 5 % con 1 año y 1 % con ≥ 2 años', () => {
    expect(semaforoPValor(0.04, 'walk_forward', 1)).toBe('verde')
    expect(semaforoPValor(0.04, 'walk_forward', 2.5)).toBe('ambar')
    expect(semaforoPValor(0.005, 'walk_forward', 2.5)).toBe('verde')
    expect(semaforoPValor(0.04, 'walk_forward', null)).toBe('verde')
  })
  it('costes sobre ganancia bruta: ámbar por encima de 0,5', () => {
    expect(costesAltos(0.51)).toBe(true)
    expect(costesAltos(0.5)).toBe(false)
    expect(costesAltos(null)).toBe(false)
  })
})

describe('histograma, regresión y años', () => {
  it('reparte los datos y suma el total', () => {
    const h = histograma([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)
    expect(h).toHaveLength(5)
    expect(h.reduce((s, i) => s + i.cuenta, 0)).toBe(11)
    expect(h[4].cuenta).toBe(3) // 8, 9 y 10 (el máximo cae en el último intervalo)
  })
  it('el valor real extiende el rango del histograma', () => {
    const h = histograma([0, 1], 4, [5])
    expect(h[3].hasta).toBe(5)
  })
  it('vacío y datos constantes no fallan', () => {
    expect(histograma([])).toEqual([])
    expect(histograma([2, 2, 2], 3).reduce((s, i) => s + i.cuenta, 0)).toBe(3)
  })
  it('regresión exacta sobre una recta', () => {
    const r = regresionLineal([[0, 1], [1, 3], [2, 5]])
    expect(r?.pendiente).toBeCloseTo(2)
    expect(r?.intercepto).toBeCloseTo(1)
    expect(regresionLineal([[1, 1]])).toBeNull()
    expect(regresionLineal([[1, 1], [1, 2]])).toBeNull()
  })
  it('años del periodo de permutación', () => {
    expect(aniosPeriodoPermutacion({ modo: 'in_sample', inicio_permutacion: null, ventana_entreno_dias: null }, '2018-01-01', '2020-01-01')).toBeCloseTo(2, 1)
    expect(aniosPeriodoPermutacion({ modo: 'walk_forward', inicio_permutacion: null, ventana_entreno_dias: 365 }, '2018-01-01', '2020-01-01')).toBeCloseTo(1, 1)
    expect(aniosPeriodoPermutacion({ modo: 'walk_forward', inicio_permutacion: '2019-01-01', ventana_entreno_dias: 365 }, '2018-01-01', '2020-01-01')).toBeCloseTo(1, 1)
    expect(aniosPeriodoPermutacion({ modo: 'in_sample', inicio_permutacion: null, ventana_entreno_dias: null }, undefined, undefined)).toBeNull()
  })
})
