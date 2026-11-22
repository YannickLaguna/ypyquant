/**
 * Datos de prueba compartidos de los tests de Alpaca multi-perfil: un `PerfilAlpaca` configurado por defecto.
 * Se relaciona con: componentes/SelectorPerfil.test.tsx, componentes/PerfilSinConfigurar.test.tsx y paginas/PaperTrading.test.tsx.
 */
import type { EstadoBotPerfil, PerfilAlpaca } from '../api/alpaca'

export const bot = (nombre: string, enMarcha = false): EstadoBotPerfil => ({
  perfil: nombre, en_marcha: enMarcha, pid: null, codigo_salida: null, adoptado: false, modo: null, intervalo_s: null,
  experimento_id: null, configuracion_id: null, inicio: null,
})

export const perfil = (nombre: string, cambios: Partial<PerfilAlpaca> = {}): PerfilAlpaca => ({
  nombre, configurado: true, ultimos4: '1234', faltan: [], variables: [`ALPACA_${nombre}_API_KEY`, `ALPACA_${nombre}_SECRET_KEY`],
  conexion: { configurada: true, conectada: true, equity: 100_000, error: null }, bot: bot(nombre), vinculo: null, ...cambios,
})
