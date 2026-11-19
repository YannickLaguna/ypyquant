/**
 * Mapa de calor de un barrido (1 o 2 ejes) como rejilla de botones accesibles.
 * Responsabilidad: pintar la matriz que calcula `matrizMapaCalor` (color por valor, rojo -> verde),
 * marcar la combinación elegida (★ mejor por objetivo) y la alternativa por meseta (◆), y avisar con
 * `onAbrir` al pulsar una celda (la página abre el experimento). Cada celda lleva `title` y `aria-label`
 * con su etiqueta y valor.
 * Se relaciona con: lib/barridos.ts (MatrizMapa, colorEscala), paginas/BarridoDetalle.tsx (lo usa).
 */
import { colorEscala, type MatrizMapa } from '../lib/barridos'
import { ratio } from '../lib/formato'

interface Props {
  matriz: MatrizMapa
  /** Id de la combinación con mejor métrica objetivo (★). */
  idMejor?: string | null
  /** Id de la combinación con mejor meseta (◆). */
  idMeseta?: string | null
  formato?: (v: number | null) => string
  onAbrir: (idExperimento: string) => void
  titulo: string
}

export default function MapaCalor({ matriz, idMejor, idMeseta, formato = (v) => ratio(v, 2), onAbrir, titulo }: Props) {
  const { ejeX, ejeY, etiquetasX, etiquetasY, celdas, minimo, maximo } = matriz
  return (
    <figure className="tarjeta p-3" aria-label={titulo}>
      <figcaption className="mb-2 text-xs font-semibold uppercase tracking-wide text-suave">{titulo}</figcaption>
      <div className="overflow-x-auto">
        <div
          className="grid items-stretch gap-1"
          style={{ gridTemplateColumns: `${ejeY ? 'auto ' : ''}repeat(${etiquetasX.length}, minmax(3.5rem, 8rem))` }}
        >
          {celdas.map((fila, iy) => (
            <Fila key={iy} cabecera={ejeY ? etiquetasY[iy] : null}>
              {fila.map((celda, ix) => {
                const id = celda.fila?.id
                const etiqueta = `${ejeX}=${etiquetasX[ix]}${ejeY ? `, ${ejeY}=${etiquetasY[iy]}` : ''}`
                const marca = id && id === idMejor ? '★' : id && id === idMeseta ? '◆' : ''
                const texto = celda.valor === null ? '—' : formato(celda.valor)
                return (
                  <button
                    key={ix}
                    type="button"
                    disabled={!id}
                    onClick={() => id && onAbrir(id)}
                    title={`${etiqueta}: ${texto}${marca === '★' ? ' (mejor)' : marca === '◆' ? ' (alternativa por meseta)' : ''}`}
                    aria-label={`${etiqueta}: ${texto}${marca === '★' ? ', mejor' : marca === '◆' ? ', alternativa por meseta' : ''}. Abrir experimento`}
                    className={`num relative flex h-12 items-center justify-center rounded text-xs text-white transition-transform enabled:hover:scale-105 disabled:cursor-default disabled:border disabled:border-dashed disabled:border-borde ${
                      marca ? 'ring-2 ring-white' : ''
                    }`}
                    style={{ background: colorEscala(celda.valor, minimo, maximo) }}
                  >
                    {texto}
                    {marca && <span className="absolute right-0.5 top-0 text-[10px] leading-none" aria-hidden>{marca}</span>}
                  </button>
                )
              })}
            </Fila>
          ))}
          {/* Pie con las etiquetas del eje X */}
          {ejeY && <span aria-hidden />}
          {etiquetasX.map((e, i) => (
            <span key={i} className="num text-center text-xs text-suave">{e}</span>
          ))}
        </div>
      </div>
      <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-suave">
        <span>Eje X: <span className="num text-texto">{ejeX}</span></span>
        {ejeY && <span>Eje Y: <span className="num text-texto">{ejeY}</span></span>}
        <span><span aria-hidden>★</span> mejor</span>
        <span><span aria-hidden>◆</span> alternativa por meseta</span>
        <span className="ml-auto">Rango: <span className="num">{formato(minimo)}</span> a <span className="num">{formato(maximo)}</span></span>
      </p>
    </figure>
  )
}

/** Fila del mapa: etiqueta del eje Y (si hay) y sus celdas, como hijos directos de la rejilla. */
function Fila({ cabecera, children }: { cabecera: string | null; children: React.ReactNode }) {
  return (
    <>
      {cabecera !== null && <span className="num flex items-center justify-end pr-1 text-xs text-suave">{cabecera}</span>}
      {children}
    </>
  )
}
