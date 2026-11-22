/**
 * Tests del estado «sin configurar» de un perfil de Alpaca: nombra las variables del `.env` que hay que definir y NO
 * ofrece ningún campo para pegar claves.
 * Se relaciona con: componentes/PerfilSinConfigurar.tsx y api/alpaca.ts (`PerfilAlpaca`).
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { PerfilAlpaca } from '../api/alpaca'
import PerfilSinConfigurar from './PerfilSinConfigurar'

const SIN_CLAVES: PerfilAlpaca = {
  nombre: 'B', configurado: false, ultimos4: null, faltan: ['ALPACA_B_API_KEY', 'ALPACA_B_SECRET_KEY'],
  variables: ['ALPACA_B_API_KEY', 'ALPACA_B_SECRET_KEY'], conexion: { configurada: false, conectada: false, equity: null, error: null },
  bot: { perfil: 'B', en_marcha: false, pid: null, codigo_salida: null, adoptado: false, modo: null, intervalo_s: null, experimento_id: null, configuracion_id: null, inicio: null },
  vinculo: null,
}

describe('PerfilSinConfigurar', () => {
  it('explica qué variables poner en el .env', () => {
    render(<PerfilSinConfigurar perfil={SIN_CLAVES} />)
    expect(screen.getByRole('alert')).toHaveTextContent('Perfil B: sin configurar')
    expect(screen.getByLabelText('Variables del .env')).toHaveTextContent('ALPACA_B_API_KEY=')
    expect(screen.getByLabelText('Variables del .env')).toHaveTextContent('ALPACA_B_SECRET_KEY=')
    expect(screen.getByRole('alert')).toHaveTextContent('.env')
    expect(screen.getByRole('alert')).toHaveTextContent('reinicia la API')
  })

  it('no pide pegar claves: no hay campos de entrada ni formularios', () => {
    render(<PerfilSinConfigurar perfil={SIN_CLAVES} />)
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(document.querySelector('input, textarea, form')).toBeNull()
  })

  it('si ya hay una de las dos variables, dice cuál falta', () => {
    render(<PerfilSinConfigurar perfil={{ ...SIN_CLAVES, faltan: ['ALPACA_B_SECRET_KEY'] }} />)
    expect(screen.getByText(/Ya hay una de las dos/)).toHaveTextContent('ALPACA_B_SECRET_KEY')
  })

  it('el perfil por defecto usa las variables sin nombre', () => {
    render(<PerfilSinConfigurar perfil={{ ...SIN_CLAVES, nombre: 'DEFECTO', variables: ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY'] }} />)
    expect(screen.getByLabelText('Variables del .env')).toHaveTextContent('ALPACA_API_KEY=')
  })
})
