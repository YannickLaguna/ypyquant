/**
 * Tests de las utilidades puras de marcos y comparación ceteris paribus (lib/marcos.ts).
 * Se relaciona con: lib/marcos.ts, componentes/FormularioMarco.tsx y paginas/Comparar.tsx.
 */
import { describe, expect, it } from 'vitest'
import type { EsquemaParametros } from '../api/tipos'
import { COMUNES, comunesActivos, esquemaSinComunes, resumenDiferencias } from './marcos'

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
