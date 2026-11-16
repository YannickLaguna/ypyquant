"""
Tests para FREDFetcher — data/fred.py

Ejecutar:
    python -X utf8 tests/test_fred.py
"""

import os
import sys
import unittest
from datetime import date

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fred import FREDFetcher, SERIES_CONFIG
from db.database import SessionLocal
from db.models import IndicatorData


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


class TestFREDFetcherFetch(unittest.TestCase):
    """Tests de integración — requieren FRED_API_KEY en .env y conexión a Supabase."""

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv
        load_dotenv()
        cls.has_key = bool(os.getenv("FRED_API_KEY"))
        cls.fetcher = FREDFetcher()

    def _skip_if_no_key(self):
        if not self.has_key:
            self.skipTest("FRED_API_KEY no configurada")

    def test_fetch_dff_returns_int(self):
        self._skip_if_no_key()
        with SessionLocal() as session:
            result = self.fetcher.fetch_and_save(["DFF"], "2024-01-01", "2024-01-31", session)
        self.assertIsInstance(result, int)

    def test_fetch_dff_idempotent(self):
        self._skip_if_no_key()
        with SessionLocal() as session:
            # Primera llamada
            self.fetcher.fetch_and_save(["DFF"], "2024-01-01", "2024-01-31", session)
            # Segunda llamada — debe devolver 0 (gap detection)
            result2 = self.fetcher.fetch_and_save(["DFF"], "2024-01-01", "2024-01-31", session)
        self.assertEqual(result2, 0)

    def test_data_in_indicator_data(self):
        self._skip_if_no_key()
        # Asegurar que hay datos primero
        with SessionLocal() as session:
            self.fetcher.fetch_and_save(["DFF"], "2024-01-02", "2024-01-05", session)

        with SessionLocal() as session:
            rows = session.query(IndicatorData).filter(
                IndicatorData.entity_id == "FRED",
                IndicatorData.metric == "DFF",
            ).limit(5).all()
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row.entity_id, "FRED")
            self.assertEqual(row.metric, "DFF")
            self.assertIsNotNone(row.date)

    def test_invalid_series_no_exception(self):
        self._skip_if_no_key()
        with SessionLocal() as session:
            # Serie inexistente — no debe lanzar excepción
            result = self.fetcher.fetch_and_save(
                ["SERIE_QUE_NO_EXISTE_XYZ123"],
                "2024-01-01", "2024-01-31",
                session,
            )
        self.assertIsInstance(result, int)

    def test_fetch_multiple_series(self):
        self._skip_if_no_key()
        with SessionLocal() as session:
            result = self.fetcher.fetch_and_save(
                ["DFF", "DGS10"],
                "2024-01-02", "2024-01-05",
                session,
            )
        self.assertIsInstance(result, int)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
