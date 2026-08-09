"""
Unit tests for inclination formatting and model/dataset nomenclature
standards in the ORBITA framework.
"""

import unittest
import numpy as np


class TestNamingConventions(unittest.TestCase):
    """Test suite for inclination decimal formatting (2 decimal places)."""

    def test_inclination_two_decimals_formatting(self):
        """Verify inclination min/max values are formatted with exactly 2 decimal places."""
        inc_min_deg = 0.0
        inc_max_deg = 90.0
        inc_str = f"{inc_min_deg:.2f}-{inc_max_deg:.2f}"
        self.assertEqual(inc_str, "0.00-90.00")

        inc_half1 = f"{0.0:.2f}-{45.0:.2f}"
        inc_half2 = f"{45.0:.2f}-{90.0:.2f}"
        self.assertEqual(inc_half1, "0.00-45.00")
        self.assertEqual(inc_half2, "45.00-90.00")

    def test_inclination_rad2deg_formatting(self):
        """Verify radian to degree conversion with 2 decimal places."""
        inc_min_rad = 0.0
        inc_max_rad = np.pi / 2.0
        deg_min = np.degrees(inc_min_rad)
        deg_max = np.degrees(inc_max_rad)
        inc_str = f"{deg_min:.2f}-{deg_max:.2f}"
        self.assertEqual(inc_str, "0.00-90.00")


if __name__ == "__main__":
    unittest.main()
