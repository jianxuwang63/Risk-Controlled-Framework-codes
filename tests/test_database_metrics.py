import json
import tempfile
import unittest
from pathlib import Path

from clinical_app.database import PilotDatabase, utc_now
from clinical_app.metrics import (
    paper_summary_rows,
    publication_validation_rows,
    summarize_postlaunch,
)


def case_record(case_id: str, *, predicted: int, accepted: bool) -> dict:
    return {
        "case_id": case_id,
        "case_ref": case_id,
        "submitted_at": utc_now(),
        "submitted_by": "doctor-1",
        "use_context": "pathologist_self_review",
        "evaluation_cohort": "VALIDATION-TEST",
        "evidence_role": "held_out_validation",
        "image_sha256": "a" * 64,
        "image_width": 512,
        "image_height": 512,
        "model_version": "v1",
        "model_hashes_json": json.dumps(["b" * 64]),
        "policy_version": "p1",
        "prediction_threshold": 0.5,
        "acceptance_threshold": 0.5,
        "target_system_fnr": 0.05,
        "confidence_delta": 0.05,
        "backend_mode": "clinical",
        "p_mip": 0.8 if predicted else 0.2,
        "selection_score": 0.9 if accepted else 0.2,
        "predicted_label": predicted,
        "accepted": int(accepted),
        "decision": "ai_mip_present" if predicted else "ai_mip_absent",
        "inference_ms": 100.0,
        "tile_count": 4,
    }


class DatabaseMetricTests(unittest.TestCase):
    def test_physician_reviews_drive_postlaunch_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = PilotDatabase(Path(tmp) / "pilot.db")
            database.insert_case(case_record("c1", predicted=0, accepted=True))
            database.insert_case(case_record("c2", predicted=1, accepted=True))
            database.insert_case(case_record("c3", predicted=0, accepted=False))
            database.review_case(
                "c1", reviewer_id="d1", ground_truth_label=1,
                ai_usefulness="unhelpful", notes=None
            )
            database.review_case(
                "c2", reviewer_id="d1", ground_truth_label=1,
                ai_usefulness="helpful", notes=None
            )
            database.review_case(
                "c3", reviewer_id="d1", ground_truth_label=1,
                ai_usefulness="neutral", notes=None
            )
            summary = summarize_postlaunch(database.all_cases_for_metrics(), 0.05)
            self.assertEqual(summary["reviewed_positive_images"], 3)
            self.assertEqual(summary["image_system_fnr_failures"], 1)
            self.assertAlmostEqual(summary["system_fnr"], 1 / 3)
            self.assertEqual(summary["image_fnr_denominator_positive_images"], 3)
            self.assertAlmostEqual(summary["operational_coverage"], 2 / 3)
            self.assertEqual(summary["image_true_positive_count"], 1)
            self.assertEqual(summary["image_false_negative_count"], 1)
            self.assertEqual(summary["image_deferred_doctor_positive"], 1)
            self.assertEqual(summary["primary_metric_unit"], "image")
            exported = database.export_reviewed_csv()
            self.assertIn("case_ref", exported)
            self.assertIn("is_false_negative", exported)
            self.assertIn("system_fnr_denominator", exported)

    def test_multi_image_session_keeps_independent_image_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = PilotDatabase(Path(tmp) / "pilot.db")
            records = []
            for index, predicted in enumerate((0, 1), start=1):
                record = case_record(
                    f"multi-{index}", predicted=predicted, accepted=True
                )
                record.update(
                    {
                        "submission_id": "patient-1",
                        "case_ref": "patient-1",
                        "pilot_phase": "silent",
                        "image_index": index,
                        "image_count": 2,
                        "use_context": "pathologist_self_review",
                    }
                )
                records.append(record)
            database.insert_submission(records)
            reviewed = database.diagnose_submission(
                "patient-1",
                reviewer_id="d1",
                image_reviews=[
                    {
                        "case_id": "multi-1",
                        "ground_truth_label": 0,
                        "notes": None,
                    },
                    {
                        "case_id": "multi-2",
                        "ground_truth_label": 1,
                        "notes": None,
                    },
                ],
                patient_notes=None,
                surgical_procedure="segmentectomy",
            )
            self.assertIsNotNone(reviewed)
            self.assertEqual(reviewed["image_count"], 2)
            self.assertIsNone(reviewed["patient_ground_truth_label"])
            self.assertEqual(reviewed["surgical_procedure"], "segmentectomy")
            self.assertIsNone(reviewed["patient_predicted_label"])
            self.assertIsNone(reviewed["patient_accepted"])
            self.assertEqual(reviewed["aggregation_rule"], "none_image_primary")
            self.assertEqual(
                [image["ground_truth_label"] for image in reviewed["images"]],
                [0, 1],
            )
            summary = summarize_postlaunch(
                database.all_cases_for_metrics(), 0.05
            )
            self.assertEqual(summary["all_images"], 2)
            self.assertEqual(summary["all_review_sessions"], 1)
            self.assertEqual(summary["aggregation_rule"], "none_image_primary")
            self.assertEqual(summary["system_fnr"], 0.0)
            self.assertIsNotNone(summary["system_fnr_upper_bound"])
            self.assertEqual(summary["image_true_positive_count"], 1)
            self.assertEqual(summary["image_true_negative_count"], 1)
            self.assertEqual(summary["image_auroc"], 1.0)

            fields, rows = paper_summary_rows(
                summary,
                generated_at_utc=utc_now(),
                app_version="test",
                model_display_name="HistoNexa-MIP",
                model_version="v1",
                policy_version="p1",
                pilot_phase="silent",
                target_system_fnr=0.05,
                confidence_delta=0.05,
                certified_prelaunch=True,
            )
            self.assertIn("numerator", fields)
            metrics = {row["metric_name"]: row for row in rows}
            self.assertEqual(metrics["system_fnr"]["numerator"], 0)
            self.assertEqual(metrics["system_fnr"]["denominator"], 1)
            self.assertEqual(metrics["auroc_all_evaluated_images"]["value"], 1.0)

            publication_fields, publication_rows = publication_validation_rows(
                database.all_cases_for_metrics(),
                generated_at_utc=utc_now(),
                app_version="test",
                model_display_name="HistoNexa-MIP",
                pilot_phase="silent",
                target_system_fnr=0.05,
                confidence_delta=0.05,
                certified_prelaunch=True,
                current_prediction_threshold=0.5,
                current_acceptance_threshold=0.5,
                certification={
                    "deployment_rule": (
                        "mean ensemble selection score >= mean_tau"
                    ),
                    "fold_taus": [0.1, 0.2, 0.3, 0.4, 0.5],
                },
            )
            self.assertIn("evaluation_cohort", publication_fields)
            self.assertEqual(len(publication_rows), 1)
            publication = publication_rows[0]
            self.assertEqual(publication["evaluation_cohort"], "VALIDATION-TEST")
            self.assertEqual(publication["positive_images"], 1)
            self.assertEqual(publication["negative_images"], 1)
            self.assertEqual(
                publication[
                    "positive_images_required_for_zero_fn_certification"
                ],
                59,
            )
            self.assertEqual(
                publication["positive_image_shortfall_if_zero_fn"], 58
            )

    def test_publication_export_separates_primary_and_sensitivity_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = PilotDatabase(Path(tmp) / "pilot.db")
            for case_id, predicted, reference in (
                ("overlap", 1, 1),
                ("ambiguous", 0, 1),
                ("positive", 1, 1),
                ("negative", 0, 0),
            ):
                database.insert_case(
                    case_record(
                        case_id, predicted=predicted, accepted=True
                    )
                )
                database.review_case(
                    case_id,
                    reviewer_id="d1",
                    ground_truth_label=reference,
                    ai_usefulness=None,
                    notes=None,
                )
            with database.connect() as connection:
                connection.execute(
                    """
                    UPDATE cases
                    SET training_overlap = 1,
                        original_dataset_label = 1
                    WHERE case_id = 'overlap'
                    """
                )
                connection.execute(
                    """
                    UPDATE cases
                    SET pre_ai_ambiguity_flag = 1,
                        reference_adjudication_status = 'pending'
                    WHERE case_id = 'ambiguous'
                    """
                )

            fields, rows = publication_validation_rows(
                database.all_cases_for_metrics(),
                generated_at_utc=utc_now(),
                app_version="test",
                model_display_name="HistoNexa-MIP",
                pilot_phase="silent",
                target_system_fnr=0.05,
                confidence_delta=0.05,
                certified_prelaunch=False,
                current_prediction_threshold=0.5,
                current_acceptance_threshold=0.5,
                certification={},
            )
            self.assertIn("recommended_for_main_paper_table", fields)
            self.assertEqual(len(rows), 2)
            by_scope = {row["analysis_scope"]: row for row in rows}
            primary = by_scope["independence_audited_primary"]
            self.assertEqual(primary["candidate_images_total"], 4)
            self.assertEqual(primary["training_overlap_images_excluded"], 1)
            self.assertEqual(primary["evaluated_images"], 3)
            self.assertEqual(primary["pending_reference_adjudication_images"], 1)
            self.assertEqual(primary["formal_analysis_ready"], 0)
            sensitivity = by_scope[
                "predefined_high_agreement_sensitivity"
            ]
            self.assertEqual(
                sensitivity["pre_ai_ambiguous_images_excluded"], 1
            )
            self.assertEqual(sensitivity["evaluated_images"], 2)
            self.assertEqual(sensitivity["formal_analysis_ready"], 1)
            self.assertEqual(
                sensitivity["recommended_for_main_paper_table"], 0
            )


if __name__ == "__main__":
    unittest.main()
