import argparse
import tempfile
import unittest
from pathlib import Path

from clinical_app.fold_tau_policy import build_average_tau_policy


class FoldTauPolicyTests(unittest.TestCase):
    def test_exactly_five_taus_are_averaged_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = []
            for index in range(5):
                path = Path(tmp) / f"fold-{index + 1}.pth"
                path.write_bytes(f"fold-{index + 1}".encode())
                checkpoints.append(path)
            args = argparse.Namespace(
                tau=[0.2, 0.3, 0.4, 0.5, 0.6],
                checkpoint=checkpoints,
                model_version="ensemble-v1",
                policy_version="mean-tau-test",
                prediction_threshold=0.5,
                target_fnr=0.05,
                delta=0.05,
            )
            policy = build_average_tau_policy(args)
            self.assertAlmostEqual(policy["acceptance_threshold"], 0.4)
            self.assertEqual(policy["certification"]["fold_taus"], args.tau)
            self.assertFalse(policy["certified"])

    def test_five_values_are_required(self):
        args = argparse.Namespace(
            tau=[0.2], checkpoint=[], model_version="v", policy_version=None,
            prediction_threshold=0.5, target_fnr=0.05, delta=0.05,
        )
        with self.assertRaisesRegex(ValueError, "five"):
            build_average_tau_policy(args)


if __name__ == "__main__":
    unittest.main()
