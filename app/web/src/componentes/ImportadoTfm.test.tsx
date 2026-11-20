/**
 * Tests de la presentación de los experimentos importados del TFM: insignia «TFM · importado» y bloque
 * de procedencia con origen y huellas de los Parquet fuente.
 * Se relaciona con: componentes/Estado.tsx (InsigniaTfm, esImportadoTfm) y componentes/ProcedenciaExperimento.tsx.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { Procedencia } from '../api/tipos'
import { esImportadoTfm, InsigniaTfm } from './Estado'
import ProcedenciaExperimento from './ProcedenciaExperimento'

describe('InsigniaTfm', () => {
  it('se muestra solo para estrategias tfm_*', () => {
    expect(esImportadoTfm('tfm_mpt_muestral')).toBe(true)
    expect(esImportadoTfm('momentum')).toBe(false)
    const { rerender } = render(<InsigniaTfm estrategia="tfm_bl_prior" />)
    expect(screen.getByText('TFM · importado')).toBeInTheDocument()
    rerender(<InsigniaTfm estrategia="momentum" />)
    expect(screen.queryByText('TFM · importado')).not.toBeInTheDocument()
  })
})

describe('ProcedenciaExperimento de un importado', () => {
  it('muestra el origen y los SHA-256 en lugar del commit propio', () => {
    const procedencia: Procedencia = {
      commit: null, arbol_sucio: null, origen: 'importado del TFM', commit_tfm: 'abc123def456',
      archivos: { 'resultados_oos.parquet': { ruta: 'C:/tfm/resultados_oos.parquet', sha256: 'f'.repeat(64) } },
      importado: '2026-11-20T20:09:36',
    }
    const { container } = render(<ProcedenciaExperimento procedencia={procedencia} />)
    const texto = container.textContent ?? ''
    expect(texto).toContain('importado del TFM')
    expect(texto).toContain('abc123def456')
    expect(texto).toContain('f'.repeat(64))
    expect(texto).not.toContain('no disponible (la librería no está en un repo git)')
  })
})
