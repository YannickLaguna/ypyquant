/**
 * Gráfico de barras horizontales de un indicador escalar: una barra por experimento, ordenadas de mejor a peor.
 * Responsabilidad: dibujar con HTML/CSS (sin dependencias; tema claro/oscuro por los tokens) las barras que calcula
 * `lib/indicadores.ts`, con el valor en la etiqueta, el color del experimento (el mismo que su línea de equity) y
 * «sin dato» para quien no tiene la métrica (nunca una barra a 0). Admite valores negativos con el cero marcado.
 * Es una lista (`<ul>`) con el texto completo en cada fila: accesible para lectores de pantalla.
 * Se relaciona con: lib/indicadores.ts (Barra, geometriaBarra, rangoBarras), lib/tema.ts (PALETA_SERIES) y
 * paginas/Comparar.tsx (lo monta cuando el indicador es escalar).
 */
import { geometriaBarra, rangoBarras, type Barra, type Formato } from '../lib/indicadores'
import { PALETA_SERIES } from '../lib/tema'

interface Props {
  titulo: string
  barras: Barra[]
  formato: Formato
}

export default function GraficoBarras({ titulo, barras, formato }: Props) {
  const rango = rangoBarras(barras)
  return (
    <figure className="tarjeta p-3" aria-label={titulo}>
      <figcaption className="mb-2 text-sm font-medium">{titulo}</figcaption>
      <ul className="space-y-1.5">
        {barras.map((b) => {
          const color = PALETA_SERIES[b.indice % PALETA_SERIES.length]
          const g = b.valor !== null && rango ? geometriaBarra(b.valor, rango) : null
          return (
            <li key={b.id} className="grid grid-cols-[minmax(6rem,14rem)_1fr_6rem] items-center gap-2 text-sm" aria-label={`${b.id}: ${b.valor === null ? 'sin dato' : formato(b.valor)}`}>
              <span className="num truncate" title={b.id}>
                <span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: color }} aria-hidden />
                {b.id}
              </span>
              <span className="relative h-4 rounded bg-elevado" aria-hidden>
                {g && (
                  <>
                    <span className="absolute inset-y-0 w-px bg-borde" style={{ left: `${g.cero}%` }} />
                    <span className="absolute inset-y-0 rounded" style={{ left: `${g.izquierda}%`, width: `max(${g.ancho}%, 2px)`, background: color }} />
                  </>
                )}
              </span>
              <span className={`num text-right ${b.valor === null ? 'text-suave' : ''}`}>{b.valor === null ? 'sin dato' : formato(b.valor)}</span>
            </li>
          )
        })}
      </ul>
    </figure>
  )
}
