import csv
import tempfile
import unittest
from pathlib import Path

from extract_case4_taus import extract_five_fold_taus


class ExtractCase4TausTests(unittest.TestCase):
    def test_two_swaps_are_averaged_within_each_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            fieldnames = ["Fold", "Swap", "Case", "Cost", "Target", "Threshold"]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for fold in range(1, 6):
                    writer.writerow({
                        "Fold": fold, "Swap": "A->B", "Case": "Case 4 (FNR+Cost)",
                        "Cost": 5, "Target": 0.05, "Threshold": fold / 10,
                    })
                    writer.writerow({
                        "Fold": fold, "Swap": "B->A", "Case": "Case 4 (FNR+Cost)",
                        "Cost": 5, "Target": 0.05, "Threshold": fold / 10 + 0.1,
                    })
            details, fold_taus, mean_tau = extract_five_fold_taus(path)
            self.assertEqual(len(details), 5)
            self.assertAlmostEqual(fold_taus[0], 0.15)
            self.assertAlmostEqual(mean_tau, 0.35)


if __name__ == "__main__":
    unittest.main()
