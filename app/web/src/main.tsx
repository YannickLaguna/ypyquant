/**
 * Punto de entrada: aplica el tema guardado (oscuro por defecto) y monta la app con el router.
 * Se relaciona con: App.tsx, lib/tema.ts, index.css e index.html.
 */
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'
import { aplicarTema, temaGuardado } from './lib/tema'

aplicarTema(temaGuardado())

ReactDOM.createRoot(document.getElementById('raiz')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
