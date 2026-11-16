"""
Tests para load_indicators() — data/loader.py

Ejecutar:
    python -X utf8 tests/test_indicators.py
"""

import os
import sys
import unittest
from datetime import date

import pandas as pd

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_indicators
from db.database import SessionLocal
from db.models import IndicatorData


class TestLoadIndicatorsUnit(unittest.TestCase):
    """Tests sin necesidad de datos en DB."""

    def test_empty_result_no_exception(self):
        with SessionLocal() as session:
            df = load_indicators(
                entity_ids=["ENTITY_QUE_NO_EXISTE"],
                metrics=["METRIC_QUE_NO_EXISTE"],
                start_date="2099-01-01",
                end_date="2099-12-31",
                session=session,
            )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_returns_dataframe(self):
        with SessionLocal() as session:
            df = load_indicators(
                entity_ids=["FF_USA"],
                metrics=None,
                start_date="2099-01-01",
                end_date="2099-12-31",
                session=session,
            )
        self.assertIsInstance(df, pd.DataFrame)


class TestLoadIndicatorsIntegration(unittest.TestCase):
    """Tests de integración — requieren datos en indicator_data."""

    @classmethod
    def setUpClass(cls):
        """Inserta datos de prueba si no existen."""
        from dotenv import load_dotenv
        load_dotenv()

        with SessionLocal() as session:
            # Insertar datos mínimos de prueba
            test_rows = [
                IndicatorData(date=date(2023, 6, 1), entity_id="FF_USA", metric="MKT_RF",
                               value=0.003, frequency="daily", source="french_data_library"),
                IndicatorData(date=date(2023, 6, 1), entity_id="FF_USA", metric="RF",
                               value=0.0001, frequency="daily", source="french_data_library"),
                IndicatorData(date=date(2023, 6, 2), entity_id="FF_USA", metric="MKT_RF",
                               value=-0.002, frequency="daily", source="french_data_library"),
                IndicatorData(date=date(2023, 6, 2), entity_id="FF_USA", metric="RF",
                               value=0.0001, frequency="daily", source="french_data_library"),
                IndicatorData(date=date(2023, 6, 1), entity_id="FRED", metric="DFF",
                               value=5.08, frequency="daily", source="fred"),
                IndicatorData(date=date(2023, 6, 2), entity_id="FRED", metric="DFF",
                               value=5.08, frequency="daily", source="fred"),
            ]
            for row in test_rows:
                session.merge(row)
            session.commit()

    def test_load_ff_usa_returns_dataframe(self):
        with SessionLocal() as session:
            df = load_indicators(
                entity_ids=["FF_USA"],
                metrics=["MKT_RF", "RF"],
                start_date="2023-01-01",
                end_date="2023-12-31",
                session=session,
            )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_ff_usa_has_datetime_index(self):
        with SessionLocal() as session:
            df = load_indicators(
                entity_ids=["FF_USA"],
                metrics=["MKT_RF", "RF"],
                start_date="2023-01-01",
                end_date="2023-12-31",
                session=session,
            )
        if not df.empty:
            self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_ff_usa_columns_correct(self):
        with SessionLocal() as session:
            df = load_indicators(
                entity_ids=["FF_USA"],
                metrics=["MKT_RF", "RF"],
                start_date="2023-01-01",
                end_date="2023-12-31",
                session=session,
            )
        if not df.empty:
            # entity_id único → columnas planas
            self.assertIn("MKT_RF", df.columns)
            self.assertIn("RF", df.columns)

    def test_multiple_entities_multiindex_columns(self):
        with SessionLocal() as session:
            df = load_indicators(
                entity_ids=["FF_USA", "FRED"],
                metrics=["MKT_RF", "RF", "DFF"],
                start_date="2023-01-01",
                end_date="2023-12-31",
                session=session,
            )
        if not df.empty:
            # Múltiples entity_ids → MultiIndex en columnas
            self.assertIsInstance(df.columns, pd.MultiIndex)

    def test_metrics_none_loads_all(self):
        with SessionLocal() as session:
            df = load_indicators(
                entity_ids=["FF_USA"],
                metrics=None,
                start_date="2023-01-01",
                end_date="2023-12-31",
                session=session,
            )
        self.assertIsInstance(df, pd.DataFrame)

    def test_date_range_filtering(self):
        with SessionLocal() as session:
            df = load_indicators(
                entity_ids=["FF_USA"],
                metrics=["MKT_RF"],
                start_date="2023-06-01",
                end_date="2023-06-02",
                session=session,
            )
        if not df.empty:
            self.assertGreaterEqual(df.index.min(), pd.Timestamp("2023-06-01"))
            self.assertLessEqual(df.index.max(), pd.Timestamp("2023-06-02"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
