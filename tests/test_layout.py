"""Smoke checks for the documented repository layout, without real UCI data."""
import unittest
from pathlib import Path
from unittest.mock import patch
from src import data, models

ROOT = Path(__file__).resolve().parents[1]

class LayoutTests(unittest.TestCase):
    def test_data_paths_are_inside_repository(self):
        self.assertEqual(data.ROOT, ROOT)
        self.assertEqual(data.RAW_FILE.parent, ROOT / 'data')

    def test_documented_sample_exists(self):
        self.assertTrue((ROOT / 'samples' / 'sample_hourly.csv').is_file())

    def test_synthetic_fallback_is_labelled(self):
        with patch.object(data, 'CACHE_FILE', ROOT / 'tests' / '_missing_cache.csv'), \
             patch.object(data, 'RAW_FILE', ROOT / 'tests' / '_missing_raw.txt'):
            hourly, source = data.load_hourly()
        self.assertEqual(source, 'synthetic')
        self.assertEqual(len(hourly), 1100 * 24)
        daily = data.to_daily(hourly)
        self.assertEqual(len(daily), 1100)
        self.assertEqual(models.RS, 42)

if __name__ == '__main__':
    unittest.main()
