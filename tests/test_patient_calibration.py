import unittest

from clinical_app.patient_calibration import (
    PatientImageScore,
    patient_operating_counts,
    tune_patient_threshold,
)


class PatientCalibrationTests(unittest.TestCase):
    def test_multi_image_rule_counts_patients_not_images(self):
        rows = [
            PatientImageScore("p1", 1, 0.1, 0.9),
            PatientImageScore("p1", 1, 0.4, 0.8),
            PatientImageScore("p2", 0, 0.1, 0.9),
            PatientImageScore("p2", 0, 0.2, 0.9),
        ]
        failures, positives, accepted, patients = patient_operating_counts(
            rows, prediction_threshold=0.5, acceptance_threshold=0.5
        )
        self.assertEqual((failures, positives, accepted, patients), (1, 1, 2, 2))

        threshold = tune_patient_threshold(
            rows, prediction_threshold=0.5, target=0.0
        )
        self.assertEqual(threshold, 0.9)
        failures, positives, accepted, patients = patient_operating_counts(
            rows, prediction_threshold=0.5, acceptance_threshold=threshold
        )
        self.assertEqual((failures, positives, accepted, patients), (0, 1, 1, 2))


if __name__ == "__main__":
    unittest.main()
