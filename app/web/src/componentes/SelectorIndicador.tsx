/**
 * Selector del indicador de Comparar: botones segmentados accesibles (grupo de radios nativos con estilo de botón).
 * Responsabilidad: elegir UN indicador entre los escalares («Resumen») y los temporales («En el tiempo»). Usa
 * `<input type="radio">` visualmente ocultos: el teclado funciona sin código (flechas dentro del grupo, Tab para
 * entrar y salir) y los lectores de pantalla anuncian «n de N». El estado vive fuera (la URL `?indicador=`).
 * Se relaciona con: lib/indicadores.ts (catálogo y grupos) y paginas/Comparar.tsx (lo monta y guarda la elección).
 */
import { GRUPOS, INDICADORES } from '../lib/indicadores'

interface Props {
  valor: string
  onChange: (clave: string) => void
}

export default function SelectorIndicador({ valor, onChange }: Props) {
  return (
    <fieldset className="space-y-2">
      <legend className="mb-1 font-medium">Indicador</legend>
      {GRUPOS.map(({ grupo, titulo }) => (
        <div key={grupo} role="group" aria-label={titulo} className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 w-full text-xs text-suave sm:w-auto sm:min-w-[15rem]">{titulo}</span>
          {INDICADORES.filter((i) => i.grupo === grupo).map((i) => (
            <label key={i.clave} className="cursor-pointer">
              <input type="radio" name="indicador" value={i.clave} checked={valor === i.clave} onChange={() => onChange(i.clave)} className="peer sr-only" />
              <span className="inline-block rounded border border-borde px-2.5 py-1 text-sm hover:bg-elevado peer-checked:border-acento peer-checked:bg-acento/15 peer-checked:font-semibold peer-checked:text-acento peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-acento">
                {i.etiqueta}
              </span>
            </label>
          ))}
        </div>
      ))}
    </fieldset>
  )
}
