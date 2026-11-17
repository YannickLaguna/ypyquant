"""
Utilidades comunes para trabajar con QuantLib.

Responsabilidad: proveer helpers de bajo nivel (conversión de fechas,
calendarios, convenciones de conteo de días) reutilizados por otros módulos
del subpaquete data/ y portfolio/.

Relacionado con:
  - data/yield_curves.py     (usa convertir_fecha_a_ql, crear_calendario_usa, etc.)
  - data/risk_free_rates.py  (usa tiene_quantlib)
  - portfolio/bond_pricing.py (usa convertir_fecha_a_ql, crear_calendario_usa, etc.)
"""

import warnings
from datetime import date, datetime


def tiene_quantlib() -> bool:
    """
    Verifica si QuantLib está disponible en el sistema.

    """
    try:
        import QuantLib as ql
        return True
    except ImportError:
        return False


def convertir_fecha_a_ql(fecha: str | datetime | date):
    """
    Convierte una fecha de Python a formato QuantLib.Date.

    Args:
        fecha: Fecha en formato 'YYYY-MM-DD', datetime, o date

    Returns:
        QuantLib.Date
    """
    try:
        import QuantLib as ql
    except ImportError:
        raise ImportError(
            "QuantLib no está instalado. Instalar con: pip install QuantLib"
        )

    # Convertir string a date si es necesario
    if isinstance(fecha, str):
        fecha_obj = datetime.strptime(fecha, '%Y-%m-%d').date()
    elif isinstance(fecha, datetime):
        fecha_obj = fecha.date()
    elif isinstance(fecha, date):
        fecha_obj = fecha
    else:
        raise ValueError(f"Formato de fecha no soportado: {type(fecha)}")

    # Crear QuantLib.Date
    return ql.Date(fecha_obj.day, fecha_obj.month, fecha_obj.year)


def convertir_ql_a_fecha(ql_date) -> date:
    """
    Convierte una fecha de QuantLib a datetime.date de Python.

    Args:
        ql_date: QuantLib.Date

    Returns:
        datetime.date
    """
    return date(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())


def crear_calendario_usa():
    """
    Crea un calendario de días hábiles de USA (Government Bond).

    Este calendario excluye:
    - Fines de semana
    - Festivos federales USA (New Year, Independence Day, Thanksgiving, etc.)

    Returns:
        QuantLib.Calendar

    Raises:
        ImportError: Si QuantLib no está instalado

    Example:
        >>> calendario = crear_calendario_usa()
        >>> fecha = ql.Date(4, 7, 2025)  # 4 de julio (Independence Day)
        >>> print(calendario.isHoliday(fecha))
        True
    """
    try:
        import QuantLib as ql
        return ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    except ImportError:
        raise ImportError("QuantLib no está instalado")


def obtener_day_count_convention(nombre: str = 'Actual/Actual'):
    """
    Obtiene una convención de conteo de días de QuantLib.

    Las convenciones de conteo de días determinan cómo se calculan los intereses
    devengados en bonos y otros instrumentos de renta fija.

    Args:
        nombre: Nombre de la convención. Opciones:
            - 'Actual/Actual' (ISDA) - Días reales / días reales (default)
            - 'Actual/360' - Días reales / 360 días (money market)
            - 'Actual/365' - Días reales / 365 días
            - '30/360' - Asume meses de 30 días

    Returns:
        QuantLib.DayCounter

    Raises:
        ImportError: Si QuantLib no está instalado

    Example:
        >>> day_count = obtener_day_count_convention('Actual/360')
        >>> # Usar en cálculos de bonos
    """
    try:
        import QuantLib as ql
    except ImportError:
        raise ImportError("QuantLib no está instalado")

    convenciones = {
        'Actual/Actual': ql.ActualActual(ql.ActualActual.ISDA),
        'Actual/360': ql.Actual360(),
        'Actual/365': ql.Actual365Fixed(),
        '30/360': ql.Thirty360(ql.Thirty360.BondBasis),
    }

    if nombre not in convenciones:
        warnings.warn(
            f"Convención '{nombre}' no reconocida. Usando Actual/Actual por defecto."
        )
        return convenciones['Actual/Actual']

    return convenciones[nombre]


def configurar_fecha_evaluacion(fecha: str | datetime | date | None = None):
    """
    Configura la fecha de evaluación global de QuantLib.

    QuantLib usa una fecha de evaluación global para todos los cálculos.
    Esta función permite configurarla.

    Args:
        fecha: Fecha a configurar. Si None, usa fecha actual del sistema.

    Example:
        >>> configurar_fecha_evaluacion('2025-01-15')
        >>> # Todos los cálculos de QuantLib usarán esta fecha
        >>> configurar_fecha_evaluacion()  # Volver a fecha actual
    """
    try:
        import QuantLib as ql
    except ImportError:
        raise ImportError("QuantLib no está instalado")

    if fecha is None:
        ql.Settings.instance().evaluationDate = ql.Date.todaysDate()
    else:
        ql_date = convertir_fecha_a_ql(fecha)
        ql.Settings.instance().evaluationDate = ql_date


def obtener_fecha_evaluacion() -> date:
    """
    Obtiene la fecha de evaluación actual configurada en QuantLib.

    Returns:
        datetime.date

    Example:
        >>> fecha = obtener_fecha_evaluacion()
        >>> print(f"Fecha de evaluación: {fecha}")
        Fecha de evaluación: 2025-11-18
    """
    try:
        import QuantLib as ql
    except ImportError:
        raise ImportError("QuantLib no está instalado")

    ql_date = ql.Settings.instance().evaluationDate
    return convertir_ql_a_fecha(ql_date)


def crear_quote_handle(valor: float):
    """
    Crea un QuoteHandle de QuantLib a partir de un valor flotante.

    Los QuoteHandles son usados en QuantLib para valores que pueden cambiar
    con el tiempo (como tasas de mercado).

    Args:
        valor: Valor numérico (ej: tasa de interés como decimal)

    Returns:
        QuantLib.QuoteHandle

    Example:
        >>> tasa = crear_quote_handle(0.045)  # 4.5%
        >>> # Usar en construcción de curvas
    """
    try:
        import QuantLib as ql
        return ql.QuoteHandle(ql.SimpleQuote(valor))
    except ImportError:
        raise ImportError("QuantLib no está instalado")


def obtener_periodo(plazo_str: str):
    """
    Convierte un string de plazo a QuantLib.Period.

    Args:
        plazo_str: Plazo en formato '3M', '1Y', '10Y', etc.

    Returns:
        QuantLib.Period

    Raises:
        ValueError: Si el formato es inválido

    Example:
        >>> periodo = obtener_periodo('6M')
        >>> periodo = obtener_periodo('10Y')
    """
    try:
        import QuantLib as ql
    except ImportError:
        raise ImportError("QuantLib no está instalado")

    # Parsear string
    if not plazo_str or len(plazo_str) < 2:
        raise ValueError(f"Formato de plazo inválido: {plazo_str}")

    numero = int(plazo_str[:-1])
    unidad = plazo_str[-1].upper()

    unidades_map = {
        'D': ql.Days,
        'W': ql.Weeks,
        'M': ql.Months,
        'Y': ql.Years
    }

    if unidad not in unidades_map:
        raise ValueError(
            f"Unidad de tiempo '{unidad}' no reconocida. "
            f"Usar: D (días), W (semanas), M (meses), Y (años)"
        )

    return ql.Period(numero, unidades_map[unidad])
