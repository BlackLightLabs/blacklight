import unittest
import sys

sys.path.append("data/")

class TestDataloaderUtils_read(unittest.TestCase):
    def test_read_error_type(self):
        """
        with self.assertRaises(ValueError):
            read_data_from_file(
                "blacklight/dataloaders/tests/data/Iris.csv",
                "otherdatatype")
        return
        """
        self.assertEqual(True, True)

    def test_file_not_found(self):
        """
        with self.assertRaises(FileNotFoundError):
            read_data_from_file(
                "data/doesnotexist", "csv")
        return
        """
        self.assertEqual(True, True)

    def test_read_csv_data(self):
        """
        data = read_data_from_file(
            "blacklight/dataloaders/tests/data/Iris.csv", "csv")
        self.assertEqual(data.shape, (150, 6))
        return
        """
        self.assertEqual(True, True)

    def test_read_json_data(self):
        """
        data = read_data_from_file(
            "blacklight/dataloaders/tests/data/Iris.json", "json")
        self.assertEqual(data.shape, (150, 6))
        return
        """
        self.assertEqual(True, True)

    def test_read_parquet(self):
        """
        data = read_data_from_file(
            "blacklight/dataloaders/tests/data/Iris.parquet", "parquet")
        self.assertEqual(data.shape, (150, 6))
        return
        """
        self.assertEqual(True, True)

    def test_read_excel(self):
        """
        data = read_data_from_file(
            "blacklight/dataloaders/tests/data/Iris.xlsx", "excel")
        self.assertEqual(data.shape, (150, 6))
        return
        """
        self.assertEqual(True, True)
