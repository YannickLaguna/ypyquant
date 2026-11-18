/**
 * Preparación de los tests de Vitest: matchers de jest-dom y limpieza automática del DOM.
 * Se relaciona con: vite.config.ts (setupFiles) y los *.test.tsx.
 */
import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

afterEach(() => cleanup())
