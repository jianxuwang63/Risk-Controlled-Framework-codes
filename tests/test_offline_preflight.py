import argparse
import json
import tempfile
import unittest
from pathlib import Path

from clinical_app.fold_tau_policy import build_average_tau_policy
from clinical_app.offline_preflight import validate_offline_pilot


class OfflinePreflightTests(unittest.TestCase):
    def _mean_policy(self, root: Path) -> tuple[Path, list[Path]]:
        checkpoints = []
        for index in range(5):
            path = root / f"fold-{index + 1}.pth"
            path.write_bytes(f"checkpoint-{index + 1}".encode())
            checkpoints.append(path)
        args = argparse.Namespace(
            tau=[0.2, 0.3, 0.4, 0.5, 0.6],
            checkpoint=checkpoints,
            target_fnr=0.05,
            delta=0.05,
            model_version="ensemble-v1",
            prediction_threshold=0.5,
            policy_version="test",
        )
        policy = build_average_tau_policy(args)
        path = root / "policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        return path, checkpoints

    def test_valid_mean_tau_policy_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy, checkpoints = self._mean_policy(root)
            result = validate_offline_pilot(policy, checkpoints, root / "data")
            self.assertAlmostEqual(result["mean_tau"], 0.4)
            self.assertEqual(result["database_integrity"], "new")

    def test_provisional_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy, checkpoints = self._mean_policy(root)
            content = json.loads(policy.read_text(encoding="utf-8"))
            content["certification"]["method"] = "none - engineering validation only"
            policy.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not the required five-fold"):
                validate_offline_pilot(policy, checkpoints, root / "data")


if __name__ == "__main__":
    unittest.main()
