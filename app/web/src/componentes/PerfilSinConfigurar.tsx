/**
 * Aviso de «perfil sin configurar»: explica qué variables poner en el `.env` para activar una cuenta de Alpaca.
 * Responsabilidad: mostrar los nombres de las variables que faltan (`ALPACA_<PERFIL>_API_KEY` y `_SECRET_KEY`) y los pasos
 * (editar el `.env` de la raíz del repo y reiniciar la API). NO tiene ningún campo de texto: las claves no se pegan en la
 * interfaz ni viajan por la API, solo se leen del entorno del servidor.
 * Se relaciona con: api/alpaca.ts (`PerfilAlpaca`), paginas/PaperTrading.tsx y su test PerfilSinConfigurar.test.tsx.
 */
import type { PerfilAlpaca } from '../api/alpaca'

export default function PerfilSinConfigurar({ perfil }: { perfil: PerfilAlpaca }) {
  const [api, secreto] = perfil.variables
  return (
    <div role="alert" className="rounded-lg border border-aviso/50 bg-aviso/10 p-4">
      <p className="font-medium text-aviso">Perfil {perfil.nombre}: sin configurar</p>
      <p className="mt-1">
        Esta cuenta paper de Alpaca aún no tiene claves. Añade estas dos variables (de una cuenta <em>paper</em>) al archivo <code>.env</code> de la
        raíz del repo y reinicia la API:
      </p>
      <pre className="num mt-2 overflow-auto rounded bg-fondo-suave p-2 text-xs" aria-label="Variables del .env">
        {`${api}=tu_api_key\n${secreto}=tu_secret_key`}
      </pre>
      {perfil.faltan.length > 0 && perfil.faltan.length < 2 && (
        <p className="mt-2 text-xs">
          Ya hay una de las dos; falta <code>{perfil.faltan[0]}</code>.
        </p>
      )}
      <p className="mt-2 text-xs text-suave">Las claves nunca se escriben en esta pantalla: la API solo las lee de su entorno.</p>
    </div>
  )
}
