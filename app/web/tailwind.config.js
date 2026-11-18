/**
 * Tailwind: los colores salen de tokens CSS (variables en src/index.css) para tener tema oscuro por
 * defecto y claro opcional sin duplicar clases. Se relaciona con: src/index.css y src/lib/tema.ts.
 */
const token = (nombre) => `rgb(var(--${nombre}) / <alpha-value>)`

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        fondo: token('fondo'),
        panel: token('panel'),
        elevado: token('elevado'),
        borde: token('borde'),
        texto: token('texto'),
        suave: token('suave'),
        acento: token('acento'),
        pos: token('pos'),
        neg: token('neg'),
        aviso: token('aviso'),
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Cascadia Mono', 'Consolas', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
