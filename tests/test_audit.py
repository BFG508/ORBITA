import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from audit_results import main as audit_main, _expected_valid_grid_cells


class TestAuditLogic(unittest.TestCase):
    """Test suite for audit checks and metrics structure validation."""

    def test_audit_main_callable(self):
        """Verify audit main entry point is callable."""
        self.assertTrue(callable(audit_main))

    def test_expected_grid_cells(self):
        """Verify calculation of valid orbital grid cells."""
        valid_cells, invalid_cells = _expected_valid_grid_cells()
        self.assertIsInstance(valid_cells, list)
        self.assertIsInstance(invalid_cells, list)
        self.assertGreater(len(valid_cells), 0)

    def test_metrics_cv_structure(self):
        """Verify data/metrics_cv.csv or data/metrics/metrics_cv.csv column schema."""
        cv_path = "data/metrics/metrics_cv.csv"
        if not os.path.exists(cv_path):
            cv_path = "data/metrics_cv.csv"

        if os.path.exists(cv_path):
            df = pd.read_csv(cv_path)
            self.assertIn("architecture", df.columns)
            self.assertTrue(
                "mean_val_mse" in df.columns or "val_mse_mean" in df.columns
            )
            self.assertTrue(
                "std_val_mse" in df.columns or "val_mse_std" in df.columns
            )


if __name__ == "__main__":
    unittest.main()
