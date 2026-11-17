"""
Tests de data/fred.py (FREDFetcher): descarga de series macro de FRED.

Responsabilidad:
    - Offline: importación, configuración de series y comportamiento sin FRED_API_KEY.
    - TestFREDFetcherFetch (descarga y guardado en la BD Supabase, ya inexistente) se eliminó.

Se relaciona con:
    - data/fred.py (código bajo prueba); db/models.py (IndicatorData).
    - Es unittest: pytest lo recoge tal cual.
"""

import os
import unittest

from data.fred import SERIES_CONFIG, FREDFetcher


class TestFREDFetcherImport(unittest.TestCase):
    def test_importable(self):
        from data.fred import FREDFetcher
        self.assertTrue(callable(FREDFetcher))

    def test_has_api_key_attr(self):
        f = FREDFetcher()
        self.assertTrue(hasattr(f, "_api_key"))

    def test_series_config_present(self):
        self.assertIn("DFF", SERIES_CONFIG)
        self.assertIn("DGS10", SERIES_CONFIG)
        self.assertIn("CPIAUCSL", SERIES_CONFIG)
        self.assertIn("DEXUSEU", SERIES_CONFIG)


class TestFREDNoKey(unittest.TestCase):
    """Tests que funcionan sin FRED_API_KEY (unit tests puros)."""

    def test_fetcher_instantiates_without_key(self):
        # Debe instanciar sin error aunque no haya key
        original = os.environ.pop("FRED_API_KEY", None)
        try:
            f = FREDFetcher()
            self.assertIsNotNone(f)
        finally:
            if original is not None:
                os.environ["FRED_API_KEY"] = original

    def test_request_returns_none_without_key(self):
        original = os.environ.pop("FRED_API_KEY", None)
        try:
            f = FREDFetcher()
            result = f._request_fred("DFF", "2024-01-01", "2024-01-31")
            self.assertIsNone(result)
        finally:
            if original is not None:
                os.environ["FRED_API_KEY"] = original
