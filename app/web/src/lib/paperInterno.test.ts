/**
 * Tests de lib/paperInterno.ts: curvas a base 100 en la fecha común, etiquetas de pendientes y de modo, y los textos
 * de tiempo de la sesión (periodo, duración, frecuencia y próximo rebalanceo).
 * Se relaciona con: lib/paperInterno.ts.
 */
import { describe, expect, it } from 'vitest'
import type { PuntoEquitySesion } from '../api/paper_interno'
import {
  normalizarBase100,
  resumenTiempo,
  textoDuracion,
  textoFrecuencia,
  textoModo,
  textoPendientes,
  textoPeriodo,
  textoProximoRebalanceo,
  textoResultadoComprobacion,
  textoSesiones,
} from './paperInterno'

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

describe('tiempo de la sesión', () => {
  it('duración en días, semanas, meses y años (ambos extremos incluidos)', () => {
    expect(textoDuracion('2021-03-01', '2021-03-01')).toBe('1 día')
    expect(textoDuracion('2021-03-01', '2021-03-05')).toBe('5 días')
    expect(textoDuracion('2021-03-01', '2021-03-12')).toBe('1 semana y 5 días')
    expect(textoDuracion('2021-03-01', '2021-03-14')).toBe('2 semanas')
    expect(textoDuracion('2021-01-01', '2021-01-31')).toBe('1 mes')
    expect(textoDuracion('2021-01-01', '2021-04-05')).toBe('3 meses y 5 días')
    expect(textoDuracion('2020-01-01', '2021-03-10')).toBe('1 año y 2 meses')
  })

  it('sesiones de mercado: singular, plural y vacío', () => {
    expect(textoSesiones(11)).toBe('11 sesiones de mercado')
    expect(textoSesiones(1)).toBe('1 sesión de mercado')
    expect(textoSesiones(0)).toBe('sin sesiones procesadas')
  })

  it('periodo alta → última fecha, y sin días procesados', () => {
    expect(textoPeriodo('2021-03-01', '2021-03-12')).toMatch(/^1 mar\.? 2021 → 12 mar\.? 2021$/)
    expect(textoPeriodo('2021-03-01', null)).toMatch(/^1 mar\.? 2021 → sin días procesados$/)
  })

  it('frecuencia de rebalanceo en español', () => {
    expect(textoFrecuencia('monthly')).toBe('rebalanceo mensual')
    expect(textoFrecuencia('weekly')).toBe('rebalanceo semanal')
    expect(textoFrecuencia('quarterly')).toBe('rebalanceo trimestral')
    expect(textoFrecuencia('daily')).toBe('rebalanceo diario')
  })

  it('próximo rebalanceo con la distancia respecto a hoy', () => {
    expect(textoProximoRebalanceo('2021-04-01', '2021-03-12')).toMatch(/^1 abr\.? 2021 \(en 20 días\)$/)
    expect(textoProximoRebalanceo('2021-03-13', '2021-03-12')).toMatch(/\(mañana\)$/)
    expect(textoProximoRebalanceo('2021-03-12', '2021-03-12')).toMatch(/\(hoy\)$/)
    expect(textoProximoRebalanceo('2021-03-01', '2021-03-12')).toBe('en la próxima sesión que se procese')
    expect(textoProximoRebalanceo(null, '2021-03-12')).toBe('—')
  })

  it('resumen de tiempo reúne periodo, duración, pendientes, frecuencia y próximo rebalanceo', () => {
    const r = resumenTiempo({
      fecha_alta: '2021-03-01', ultima_fecha: '2021-03-12', dias: 10, dias_pendientes: 3, al_dia: false,
      frecuencia_rebalanceo: 'monthly', proximo_rebalanceo: '2021-04-01',
    }, '2021-03-15')
    expect(r.duracion).toBe('10 sesiones de mercado · 1 semana y 5 días')
    expect(r.pendientes).toBe('3 días pendientes')
    expect(r.frecuencia).toBe('rebalanceo mensual')
    expect(r.periodo).toMatch(/→/)
    expect(r.proximo).toMatch(/1 abr\.? 2021 \(en 17 días\)/)
  })

  it('sin días procesados la duración lo dice', () => {
    const r = resumenTiempo({ fecha_alta: '2021-03-01', ultima_fecha: null, dias: 0, dias_pendientes: 10, al_dia: false,
      frecuencia_rebalanceo: 'weekly', proximo_rebalanceo: '2021-03-01' }, '2021-03-12')
    expect(r.duracion).toBe('sin sesiones procesadas')
  })
})

describe('comprobación automática', () => {
  const base = { hora: '2026-11-22T14:00:00+01:00', dias_procesados: 0, error: null }
  it('resume el resultado de la última comprobación', () => {
    expect(textoResultadoComprobacion({ ...base, resultado: 'al_dia' })).toBe('no había días pendientes')
    expect(textoResultadoComprobacion({ ...base, resultado: 'avanzado', dias_procesados: 1 })).toBe('1 día procesado')
    expect(textoResultadoComprobacion({ ...base, resultado: 'avanzado', dias_procesados: 4 })).toBe('4 días procesados')
    expect(textoResultadoComprobacion({ ...base, resultado: 'avance_en_curso' })).toContain('avance en curso')
    expect(textoResultadoComprobacion({ ...base, resultado: 'sin_datos' })).toContain('aún no tiene sus precios')
    expect(textoResultadoComprobacion({ ...base, resultado: 'error', error: 'RuntimeError: sin red' })).toBe('error: RuntimeError: sin red')
  })
})
