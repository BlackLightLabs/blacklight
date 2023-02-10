import unittest
from src.dataLoaders.utils import *


class TestDataloaderUtils_read(unittest.TestCase):
    def test_read_error_type(self):
        with self.assertRaises(ValueError):
            read_data_from_file(
                "tests/dataloaders/data/Iris.csv",
                "otherdatatype")
        return

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_data_from_file("tests/dataloaders/data/doesnotexist", "csv")
        return

    def test_read_csv_data(self):
        data = read_data_from_file("tests/dataloaders/data/Iris.csv", "csv")
        self.assertEqual(data.shape, (150, 6))
        return

    def test_read_json_data(self):
        data = read_data_from_file("tests/dataloaders/data/Iris.json", "json")
        self.assertEqual(data.shape, (150, 6))
        return

    def test_read_parquet(self):
        data = read_data_from_file(
            "tests/dataloaders/data/Iris.parquet", "parquet")
        self.assertEqual(data.shape, (150, 6))
        return

    def test_read_excel(self):
        data = read_data_from_file("tests/dataloaders/data/Iris.xlsx", "excel")
        self.assertEqual(data.shape, (150, 6))
        return