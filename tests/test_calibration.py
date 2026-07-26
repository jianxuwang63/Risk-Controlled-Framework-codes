import json
import tempfile
import unittest
from pathlib import Path

from clinical_app.calibration import (
    PredictionRow,
    select_np_threshold,
    system_failures,
    tune_acceptance_threshold,
)
from clinical_app.policy import DeploymentPolicy, PolicyError, sha256_file
from clinical_app.statistics import (
    clopper_pearson_upper,
    required_positives_for_zero_failures,
)


class CalibrationTests(unittest.TestCase):
    def test_np_uses_alpha_and_delta_not_naive_percentile(self):
        result = select_np_threshold(
            [index / 100 for index in range(1, 101)], alpha=0.05, delta=0.05
        )
        self.assertEqual(result.order_index, 2)
        self.assertAlmostEqual(result.threshold, 0.02)
        self.assertLessEqual(result.binomial_tail, 0.05)

    def test_one_percent_requires_more_than_one_hundred_positives(self):
        self.assertEqual(required_positives_for_zero_failures(0.01, 0.05), 299)
        with self.assertRaisesRegex(ValueError, "at least 299"):
            select_np_threshold([0.9] * 100, alpha=0.01, delta=0.05)

    def test_zero_failure_upper_bound_matches_sample_requirement(self):
        self.assertLessEqual(clopper_pearson_upper(0, 59, 0.05), 0.05)
        self.assertGreater(clopper_pearson_upper(0, 58, 0.05), 0.05)

    def test_tuning_maximizes_coverage_subject_to_empirical_target(self):
        rows = [
            PredictionRow(1, 0.2, 0.2),
            PredictionRow(1, 0.8, 0.4),
            PredictionRow(1, 0.9, 0.8),
            PredictionRow(0, 0.1, 0.3),
        ]
        threshold = tune_acceptance_threshold(rows, 0.5, tuning_target=0.0)
        self.assertEqual(threshold, 0.3)
        failures, positives, accepted = system_failures(rows, 0.5, threshold)
        self.assertEqual((failures, positives, accepted), (0, 3, 3))

    def test_policy_rejects_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint-a")
            policy_file = root / "policy.json"
            policy_file.write_text(
                """{
                  "schema_version": 1,
                  "policy_version": "test",
                  "model_version": "v1",
                  "checkpoint_sha256": ["deadbeef"],
                  "prediction_threshold": 0.5,
                  "acceptance_threshold": 0.7,
                  "target_system_fnr": 0.05,
                  "confidence_delta": 0.05,
                  "certified": true,
                  "certification": {}
                }""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PolicyError, "hashes"):
                DeploymentPolicy.load(
                    policy_file,
                    expected_model_version="v1",
                    checkpoints=(checkpoint,),
                )
            self.assertEqual(len(sha256_file(checkpoint)), 64)

    def test_uncertified_policy_is_validation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint-a")
            policy_file = root / "validation-policy.json"
            policy_file.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "policy_version": "validation-test",
                        "model_version": "v1",
                        "checkpoint_sha256": [sha256_file(checkpoint)],
                        "prediction_threshold": 0.5,
                        "acceptance_threshold": 0.5,
                        "target_system_fnr": 0.05,
                        "confidence_delta": 0.05,
                        "certified": False,
                        "certification": {"method": "none"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PolicyError, "requires a certified policy"):
                DeploymentPolicy.load(
                    policy_file,
                    expected_model_version="v1",
                    checkpoints=(checkpoint,),
                )
            policy = DeploymentPolicy.load(
                policy_file,
                expected_model_version="v1",
                checkpoints=(checkpoint,),
                require_certified=False,
            )
            self.assertFalse(policy.certified)


if __name__ == "__main__":
    unittest.main()
