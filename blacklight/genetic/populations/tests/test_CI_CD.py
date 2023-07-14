import unittest


class TestCICD(unittest.TestCase):
    """
    Basic test to assure that CI/CD is functioning properly
    """

    def test_base(self):
        self.assertEqual(2, 2)
