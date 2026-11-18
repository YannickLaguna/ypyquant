/**
 * Configuración de Vite y Vitest.
 *
 * Responsabilidad: servidor de desarrollo con proxy `/api` -> API FastAPI (127.0.0.1:8010), build a
 * `dist/` (que la API sirve en `/` si existe) y entorno de tests (jsdom + testing-library).
 * Se relaciona con: app/api/main.py (puerto 8010 y servicio del `dist`), src/test/preparar.ts.
 */
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8010', changeOrigin: false } },
  },
  build: { chunkSizeWarningLimit: 900 },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/preparar.ts'],
    css: false,
  },
})
