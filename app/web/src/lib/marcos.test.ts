/**
 * Tests de las utilidades puras de marcos y comparación ceteris paribus (lib/marcos.ts).
 * Se relaciona con: lib/marcos.ts, componentes/FormularioMarco.tsx y paginas/Comparar.tsx.
 */
import { describe, expect, it } from 'vitest'
import type { ConfiguracionExperimento, EsquemaParametros } from '../api/tipos'
import { COMUNES, comunesActivos, comunesDeConfiguracion, esquemaSinComunes, marcoDesdeExperimento, parametrosPropios, resumenDiferencias } from './marcos'

describe('resumenDiferencias', () => {
  it('es null cuando no hay diferencias', () => {
    expect(resumenDiferencias([])).toBeNull()
    expect(resumenDiferencias(undefined)).toBeNull()
  })

  it('nombra las diferencias de entorno y de parámetros comunes en español', () => {
    const texto = resumenDiferencias([
      { campo: 'fecha_fin', tipo: 'entorno', valores: { a: '2021-12-31', b: '2022-06-30' } },
      { campo: 'comision', tipo: 'entorno', valores: { a: 0.001, b: 0.002 } },
      { campo: 'ventana_dias', tipo: 'comun', valores: { a: 504, b: 252 } },
    ])
    expect(texto).toBe('No es ceteris paribus: difieren fecha de fin, comisión y el parámetro común ventana_dias.')
  })

  it('usa el singular con una sola diferencia', () => {
    expect(resumenDiferencias([{ campo: 'universo', tipo: 'entorno', valores: {} }])).toBe('No es ceteris paribus: difiere universo.')
  })
})

describe('comunesActivos', () => {
  it('solo incluye los activados, con su valor', () => {
    const activos = comunesActivos({ ventana_dias: { activo: true, valor: 504 }, peso_maximo: { activo: false, valor: 0.3 } })
    expect(activos).toEqual({ ventana_dias: 504 })
  })

  it('los comunes sugeridos cubren los estimadores nuevos', () => {
    expect(COMUNES.map((c) => c.nombre)).toEqual(
      expect.arrayContaining(['ventana_dias', 'peso_maximo', 'tasa_libre_riesgo', 'estimador_sigma', 'frecuencia_estimacion']),
    )
  })
})

describe('esquemaSinComunes', () => {
  const esquema: EsquemaParametros = {
    type: 'object',
    properties: { ventana_dias: { type: 'integer', default: 756 }, penalizacion_turnover: { type: 'number', default: 0 } },
  }

  it('quita del formulario propio los parámetros que fija el marco', () => {
    const filtrado = esquemaSinComunes(esquema, { ventana_dias: 504 })
    expect(Object.keys(filtrado.properties ?? {})).toEqual(['penalizacion_turnover'])
    expect(Object.keys(esquema.properties ?? {})).toHaveLength(2) // no muta el original
  })
})

describe('parámetros comunes de una configuración', () => {
  const parametros = { ventana_dias: 756, peso_maximo: 0.3, estimador_sigma: 'ledoit_wolf', delta: 18, kappa: 0 }

  it('comunesDeConfiguracion toma solo los comunes conocidos que la receta trae', () => {
    expect(comunesDeConfiguracion(parametros)).toEqual({ ventana_dias: 756, peso_maximo: 0.3, estimador_sigma: 'ledoit_wolf' })
    expect(comunesDeConfiguracion({ top_k: 5 })).toEqual({})
  })

  it('parametrosPropios deja los que no son del entorno, para la propia estrategia', () => {
    expect(parametrosPropios(parametros)).toEqual({ delta: 18, kappa: 0 })
  })
})

describe('marcoDesdeExperimento', () => {
  const config: ConfiguracionExperimento = {
    estrategia: 'bayes_stein',
    parametros: { ventana_dias: 504, peso_maximo: 0.2, frecuencia_estimacion: 'mensual', n_replicas: 50 },
    universo: 'sp500_pit', fecha_inicio: '2015-01-01', fecha_fin: '2024-12-31', capital_inicial: 50000,
    comision: 0.002, slippage: 0.0005, frecuencia_rebalanceo: 'monthly', benchmark: 'SPY', semilla: 7,
    dias_historia_previa: 1500,
  }

  it('copia el entorno del experimento y solo los parámetros comunes', () => {
    const marco = marcoDesdeExperimento(config, 'bayes_stein-abc1234567')
    expect(marco).toMatchObject({
      universo: 'sp500_pit', fecha_inicio: '2015-01-01', fecha_fin: '2024-12-31', frecuencia_rebalanceo: 'monthly',
      comision: 0.002, slippage: 0.0005, benchmark: 'SPY', capital_inicial: 50000, semilla: 7, dias_historia_previa: 1500,
    })
    expect(marco.parametros_comunes).toEqual({ ventana_dias: 504, peso_maximo: 0.2, frecuencia_estimacion: 'mensual' })
    expect(marco.nombre).toContain('bayes_stein')
    expect(marco.notas).toContain('bayes_stein-abc1234567')
  })

  it('sin historia previa en la config usa el defecto del motor', () => {
    const { dias_historia_previa: _omitido, ...sinHistoria } = config
    expect(marcoDesdeExperimento(sinHistoria, 'x').dias_historia_previa).toBe(1100)
  })
})
