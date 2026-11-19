/**
 * Configuración de Vite y Vitest.
 *
 * Responsabilidad: servidor de desarrollo con proxy `/api` -> API FastAPI (127.0.0.1:8010), build a
 * `dist/` (que la API sirve en `/` si existe) y entorno de tests (jsdom + testing-library).
 * Variables opcionales YPYQUANT_API_PUERTO / YPYQUANT_WEB_PUERTO cambian los puertos sin tocar el comportamiento por defecto.
 * Se relaciona con: app/api/main.py (puerto 8010 y servicio del `dist`), src/test/preparar.ts.
 */
import react from '@vitejs/plugin-react'
import { loadEnv } from 'vite'
import { defineConfig } from 'vitest/config'

// Por defecto: Vite en 5173 y API en 8010. Para convivir con otra instancia: YPYQUANT_API_PUERTO / YPYQUANT_WEB_PUERTO
export default defineConfig(({ mode }) => {
  // loadEnv también lee las variables del entorno del proceso con ese prefijo (sin necesitar @types/node)
  const entorno = loadEnv(mode, '.', 'YPYQUANT_')
  const puertoApi = entorno.YPYQUANT_API_PUERTO ?? '8010'
  const puertoWeb = Number(entorno.YPYQUANT_WEB_PUERTO ?? 5173)
  return {
    plugins: [react()],
    server: {
      host: '127.0.0.1',
      port: puertoWeb,
      proxy: { '/api': { target: `http://127.0.0.1:${puertoApi}`, changeOrigin: false } },
    },
    build: { chunkSizeWarningLimit: 900 },
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: ['./src/test/preparar.ts'],
      css: false,
    },
  }
})
