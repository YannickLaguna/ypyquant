/**
 * Raíz de la aplicación: proveedor de TanStack Query y rutas (react-router).
 * Responsabilidad: mapear URL -> página dentro del Layout; redirige `/` a Estrategias.
 * Se relaciona con: main.tsx (monta), componentes/Layout.tsx y paginas/* (incluye Barridos, Configuraciones y Marcos).
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './componentes/Layout'
import BarridoDetalle from './paginas/BarridoDetalle'
import Barridos from './paginas/Barridos'
import Comparar from './paginas/Comparar'
import ConfiguracionDetalle from './paginas/ConfiguracionDetalle'
import Configuraciones from './paginas/Configuraciones'
import EstrategiaDetalle from './paginas/EstrategiaDetalle'
import Estrategias from './paginas/Estrategias'
import ExperimentoDetalle from './paginas/ExperimentoDetalle'
import Experimentos from './paginas/Experimentos'
import MarcoDetalle from './paginas/MarcoDetalle'
import Marcos from './paginas/Marcos'
import PaperTrading from './paginas/PaperTrading'

export const clienteConsultas = new QueryClient({
  defaultOptions: { queries: { staleTime: 5_000, refetchOnWindowFocus: false, retry: 1 } },
})

export default function App() {
  return (
    <QueryClientProvider client={clienteConsultas}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/estrategias" replace />} />
          <Route path="estrategias" element={<Estrategias />} />
          <Route path="estrategias/:nombre" element={<EstrategiaDetalle />} />
          <Route path="experimentos" element={<Experimentos />} />
          <Route path="experimentos/:id" element={<ExperimentoDetalle />} />
          <Route path="barridos" element={<Barridos />} />
          <Route path="barridos/:id" element={<BarridoDetalle />} />
          <Route path="configuraciones" element={<Configuraciones />} />
          <Route path="configuraciones/:id" element={<ConfiguracionDetalle />} />
          <Route path="marcos" element={<Marcos />} />
          <Route path="marcos/:id" element={<MarcoDetalle />} />
          <Route path="comparar" element={<Comparar />} />
          <Route path="paper" element={<PaperTrading />} />
          <Route path="*" element={<p className="p-6 text-suave">Página no encontrada.</p>} />
        </Route>
      </Routes>
    </QueryClientProvider>
  )
}
