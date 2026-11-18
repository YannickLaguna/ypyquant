/**
 * Cliente HTTP mínimo de la API.
 * Responsabilidad: `fetch` con JSON, errores tipados (`ErrorApi` con estado y mensaje legible, también
 * para el 503 «sin claves de Alpaca») y construcción de querystrings.
 * Se relaciona con: api/hooks.ts (consume `pedir`) y api/tipos.ts.
 */

export class ErrorApi extends Error {
  estado: number

  constructor(estado: number, mensaje: string) {
    super(mensaje)
    this.name = 'ErrorApi'
    this.estado = estado
  }
}

/** Convierte el `detail` de FastAPI (texto o lista de errores de validación) en un mensaje legible. */
export function mensajeDeDetalle(detalle: unknown, porDefecto: string): string {
  if (typeof detalle === 'string') return detalle
  if (Array.isArray(detalle)) {
    return detalle
      .map((e) => {
        const campo = Array.isArray(e?.loc) ? e.loc.filter((p: unknown) => p !== 'body').join('.') : ''
        return campo ? `${campo}: ${e?.msg ?? ''}` : String(e?.msg ?? '')
      })
      .join(' · ')
  }
  return porDefecto
}

export async function pedir<T>(ruta: string, opciones: { metodo?: string; cuerpo?: unknown } = {}): Promise<T> {
  const respuesta = await fetch(ruta, {
    method: opciones.metodo ?? 'GET',
    headers: opciones.cuerpo !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: opciones.cuerpo !== undefined ? JSON.stringify(opciones.cuerpo) : undefined,
  })
  if (!respuesta.ok) {
    let detalle: unknown
    try {
      detalle = (await respuesta.json())?.detail
    } catch {
      detalle = undefined
    }
    throw new ErrorApi(respuesta.status, mensajeDeDetalle(detalle, `Error ${respuesta.status} en ${ruta}`))
  }
  if (respuesta.status === 204) return undefined as T
  return (await respuesta.json()) as T
}

export function querystring(parametros: Record<string, string | number | boolean | undefined | null>): string {
  const q = new URLSearchParams()
  for (const [clave, valor] of Object.entries(parametros)) {
    if (valor !== undefined && valor !== null && valor !== '') q.set(clave, String(valor))
  }
  const texto = q.toString()
  return texto ? `?${texto}` : ''
}
