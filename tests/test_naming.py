import sys
import unittest
from pathlib import Path

import numpy as np

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (  # noqa: E402
    TOTAL_ECC_BOUNDS,
    TOTAL_INC_BOUNDS,
    TOTAL_SMA_BOUNDS,
    format_domain_suffix,
    get_global_dataset_filename,
    get_global_model_filename,
)
from physics.oracle import R_EQ  # noqa: E402
from simulate_mission import find_expert_system  # noqa: E402


class TestNamingConventions(unittest.TestCase):
    """Test suite for inclination decimal formatting (2 decimal places)."""

    def test_inclination_two_decimals_formatting(self):
        """Verify inclination min/max formatting (2 decimal places)."""
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

    def test_format_domain_suffix_helpers(self):
        """Verify central domain suffix helper formats properly."""
        suffix = format_domain_suffix(
            TOTAL_SMA_BOUNDS, TOTAL_ECC_BOUNDS, TOTAL_INC_BOUNDS
        )
        self.assertEqual(suffix, "300-2000_0.0000-0.1000_0.00-90.00")

        ds_fn = get_global_dataset_filename()
        exp_ds = "orbita_dataset_300-2000_0.0000-0.1000_0.00-90.00.csv"
        self.assertTrue(ds_fn.endswith(exp_ds))

        model_fn = get_global_model_filename("resnet")
        exp_model = (
            "orbita_predictor_resnet_300-2000_0.0000-0.1000_0.00-90.00.pth"
        )
        self.assertTrue(exp_model in model_fn)

    def test_global_only_routing_selection(self):
        """Verify global_only=True enforces selection of the global model."""
        sma = R_EQ + 500e3  # 500 km (falls inside local expert 300-640 km)
        ecc = 0.01
        inc = np.radians(20.0)

        # When global_only=True, it MUST select global model, not local expert
        model_path, dataset_path = find_expert_system(
            sma, ecc, inc, target_model_type="resnet", global_only=True
        )
        self.assertIn("300-2000_0.0000-0.1000", model_path)
        self.assertIn("300-2000_0.0000-0.1000", dataset_path)


if __name__ == "__main__":
    unittest.main()
