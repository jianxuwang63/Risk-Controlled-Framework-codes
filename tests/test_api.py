import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path


HAS_WEB_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("fastapi", "multipart", "PIL", "httpx")
)


@unittest.skipUnless(HAS_WEB_DEPS, "optional web test dependencies are not installed")
class ApiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        os.environ["APP_MODE"] = "demo"
        os.environ["PILOT_PHASE"] = "silent"
        os.environ["MAX_IMAGES_PER_CASE"] = "300"
        os.environ["DATA_DIR"] = cls.tempdir.name
        os.environ["DATABASE_PATH"] = str(Path(cls.tempdir.name) / "api-test.db")
        from fastapi.testclient import TestClient
        from PIL import Image
        from clinical_app.api import app

        cls.client = TestClient(app)
        cls.image_bytes = []
        for color in ((170, 80, 110), (80, 150, 120)):
            image = Image.new("RGB", (64, 64), color)
            output = io.BytesIO()
            image.save(output, format="PNG")
            cls.image_bytes.append(output.getvalue())

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.tempdir.cleanup()

    def test_submit_review_and_measure(self):
        self.assertEqual(self.client.get("/health/live").status_code, 200)
        response = self.client.post(
            "/api/v1/cases",
            data={"case_ref": "PILOT-001", "submitted_by": "DOC-1"},
            files=[
                ("images", ("roi-1.png", self.image_bytes[0], "image/png")),
                ("images", ("roi-2.png", self.image_bytes[1], "image/png")),
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        case = response.json()
        self.assertEqual(case["model_display_name"], "HistoNexa-MIP")
        self.assertEqual(case["use_context"], "pathologist_self_review")
        self.assertEqual(case["backend_mode"], "demo")
        self.assertEqual(case["pilot_phase"], "silent")
        self.assertEqual(case["evaluation_cohort"], "UNSPECIFIED")
        self.assertEqual(case["evidence_role"], "held_out_validation")
        self.assertEqual(case["image_count"], 2)
        self.assertEqual(len(case["images"]), 2)
        self.assertFalse(case["ai_result_visible"])
        self.assertIsNone(case["p_mip"])
        self.assertEqual(case["decision"], "silent_ai_hidden")
        self.assertTrue(all(image["image_available"] for image in case["images"]))
        stored_image = self.client.get(case["images"][0]["image_url"])
        self.assertEqual(stored_image.status_code, 200)
        self.assertEqual(stored_image.content, self.image_bytes[0])

        system = self.client.get("/api/v1/system").json()
        self.assertTrue(system["offline_only"])
        self.assertEqual(system["deployment_scope"], "offline_local")
        self.assertEqual(system["max_images_per_case"], 300)

        duplicate = self.client.post(
            "/api/v1/cases",
            data={"case_ref": "PILOT-001", "submitted_by": "DOC-1"},
            files=[("images", ("roi-3.png", self.image_bytes[0], "image/png"))],
        )
        self.assertEqual(duplicate.status_code, 409, duplicate.text)
        self.assertIn("already been submitted", duplicate.json()["detail"])

        pending = self.client.get("/api/v1/cases?status=pending")
        self.assertEqual(pending.status_code, 200, pending.text)
        pending_case = pending.json()["items"][0]
        self.assertFalse(pending_case["ai_result_visible"])
        self.assertTrue(
            all(image["p_mip"] is None for image in pending_case["images"])
        )

        started = self.client.post(
            f"/api/v1/cases/{case['submission_id']}/workflow/start-diagnosis",
            json={"reviewer_id": "DOC-1"},
        )
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(started.json()["diagnosis_started_by"], "DOC-1")

        diagnosis = self.client.post(
            f"/api/v1/cases/{case['submission_id']}/diagnosis",
            json={
                "reviewer_id": "DOC-1",
                "images": [
                    {
                        "case_id": case["images"][0]["case_id"],
                        "ground_truth_label": 0,
                        "notes": "image negative",
                    },
                    {
                        "case_id": case["images"][1]["case_id"],
                        "ground_truth_label": 1,
                        "notes": "focal MIP",
                    },
                ],
                "patient_notes": "image review session note",
                "surgical_procedure": "lobectomy",
            },
        )
        self.assertEqual(diagnosis.status_code, 200, diagnosis.text)
        diagnosed = diagnosis.json()
        self.assertEqual(diagnosed["review_status"], "reviewed")
        self.assertEqual(diagnosed["feedback_status"], "pending")
        self.assertIsNone(diagnosed["patient_ground_truth_label"])
        self.assertEqual(diagnosed["aggregation_rule"], "none_image_primary")
        self.assertEqual(
            diagnosed["patient_decision"], "not_applicable_image_level_service"
        )
        self.assertEqual(diagnosed["surgical_procedure"], "lobectomy")
        self.assertTrue(diagnosed["ai_result_visible"])
        self.assertTrue(
            all(image["p_mip"] is not None for image in diagnosed["images"])
        )
        self.assertEqual(
            [image["ground_truth_label"] for image in diagnosed["images"]],
            [0, 1],
        )

        feedback = self.client.post(
            f"/api/v1/cases/{case['submission_id']}/feedback",
            json={
                "reviewer_id": "DOC-1",
                "images": [
                    {
                        "case_id": case["images"][0]["case_id"],
                        "notes": None,
                    },
                    {
                        "case_id": case["images"][1]["case_id"],
                        "notes": "highlighted positive image",
                    },
                ],
                "patient_notes": "image comparison session complete",
                "diagnosis_changed_after_ai": False,
            },
        )
        self.assertEqual(feedback.status_code, 200, feedback.text)
        reviewed = feedback.json()
        self.assertEqual(reviewed["feedback_status"], "completed")
        self.assertIsNone(reviewed["patient_ai_usefulness"])
        self.assertFalse(reviewed["diagnosis_changed_after_ai"])
        self.assertIsNone(reviewed["post_ai_patient_label"])
        self.assertTrue(
            all(
                image["ai_usefulness"] in {"effective", "neutral", "unhelpful"}
                for image in reviewed["images"]
            )
        )

        metrics = self.client.get("/api/v1/metrics/summary")
        self.assertEqual(metrics.status_code, 200, metrics.text)
        summary = metrics.json()
        self.assertEqual(summary["reviewed_images"], 2)
        self.assertEqual(summary["feedback_completed_sessions"], 1)
        self.assertEqual(summary["image_fnr_denominator_positive_images"], 1)
        self.assertEqual(summary["primary_metric_unit"], "image")
        self.assertEqual(summary["aggregation_rule"], "none_image_primary")
        self.assertEqual(summary["physician_summaries"][0]["reviewer_id"], "DOC-1")

        image_export = self.client.get("/api/v1/metrics/export.csv")
        self.assertEqual(image_export.status_code, 200)
        self.assertIn("use_context", image_export.text)
        self.assertIn("diagnosis_changed_after_ai", image_export.text)
        self.assertIn("is_false_negative", image_export.text)
        self.assertIn("rejection_score", image_export.text)
        self.assertIn("lobectomy", image_export.text)

        paper_export = self.client.get("/api/v1/metrics/export-paper.csv")
        self.assertEqual(paper_export.status_code, 200)
        self.assertIn(
            "histonexa_validation_table.csv",
            paper_export.headers["content-disposition"],
        )
        self.assertIn("system_fnr", paper_export.text)
        self.assertIn("auroc_all_evaluated_images", paper_export.text)
        self.assertIn("evaluation_cohort", paper_export.text)
        self.assertIn(
            "positive_images_required_for_zero_fn_certification",
            paper_export.text,
        )

        metric_dictionary = self.client.get(
            "/api/v1/metrics/export-paper-long.csv"
        )
        self.assertEqual(metric_dictionary.status_code, 200)
        self.assertIn("numerator", metric_dictionary.text)
        self.assertIn("denominator", metric_dictionary.text)

    def test_patient_self_check_is_immediately_visible_and_excluded(self):
        response = self.client.post(
            "/api/v1/cases",
            data={
                "case_ref": "SELF-CHECK-001",
                "submitted_by": "USER-001",
                "use_context": "patient_self_check",
                "interaction_mode": "self_service",
            },
            files=[("images", ("self-check.png", self.image_bytes[0], "image/png"))],
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["use_context"], "patient_self_check")
        self.assertEqual(result["interaction_mode"], "self_service")
        self.assertIsNone(result["diagnosing_pathologist_id"])
        self.assertTrue(result["ai_result_visible"])
        self.assertIsNotNone(result["images"][0]["p_mip"])
        self.assertIn("not a pathology diagnosis", result["clinical_notice"])

        rejected = self.client.post(
            f"/api/v1/cases/{result['submission_id']}/workflow/start-diagnosis",
            json={"reviewer_id": "DOC-1"},
        )
        self.assertEqual(rejected.status_code, 409)

        summary = self.client.get("/api/v1/metrics/summary").json()
        self.assertGreaterEqual(summary["patient_information_images"], 1)

    def test_offline_host_policy(self):
        from clinical_app.api import _client_host_allowed

        self.assertTrue(
            _client_host_allowed(
                "127.0.0.1", offline_only=True, allow_private_lan=False
            )
        )
        self.assertFalse(
            _client_host_allowed(
                "192.168.50.20", offline_only=True, allow_private_lan=False
            )
        )
        self.assertTrue(
            _client_host_allowed(
                "192.168.50.20", offline_only=True, allow_private_lan=True
            )
        )
        self.assertFalse(
            _client_host_allowed(
                "8.8.8.8", offline_only=True, allow_private_lan=True
            )
        )

    def test_zz_researcher_mediated_workflow_is_attributed_and_attested(self):
        response = self.client.post(
            "/api/v1/cases",
            data={
                "case_ref": "PILOT-MEDIATED-001",
                "submitted_by": "RESEARCHER-2",
                "diagnosing_pathologist_id": "PATH-9",
                "interaction_mode": "researcher_mediated",
            },
            files=[("images", ("roi.png", self.image_bytes[0], "image/png"))],
        )
        self.assertEqual(response.status_code, 200, response.text)
        case = response.json()
        self.assertEqual(case["interaction_mode"], "researcher_mediated")
        self.assertEqual(case["diagnosing_pathologist_id"], "PATH-9")
        self.assertEqual(case["submitted_by"], "RESEARCHER-2")

        started = self.client.post(
            f"/api/v1/cases/{case['submission_id']}/workflow/start-diagnosis",
            json={"reviewer_id": "PATH-9", "operator_id": "RESEARCHER-2"},
        )
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(
            started.json()["diagnosis_started_operator"], "RESEARCHER-2"
        )

        payload = {
            "reviewer_id": "PATH-9",
            "operator_id": "RESEARCHER-2",
            "source_attested": False,
            "images": [
                {
                    "case_id": case["images"][0]["case_id"],
                    "ground_truth_label": 0,
                    "notes": "transcribed from pre-AI source record",
                }
            ],
            "patient_notes": None,
            "surgical_procedure": "biopsy_or_other",
        }
        missing_attestation = self.client.post(
            f"/api/v1/cases/{case['submission_id']}/diagnosis", json=payload
        )
        self.assertEqual(missing_attestation.status_code, 409)
        self.assertIn("attestation", missing_attestation.json()["detail"])

        payload["source_attested"] = True
        diagnosis = self.client.post(
            f"/api/v1/cases/{case['submission_id']}/diagnosis", json=payload
        )
        self.assertEqual(diagnosis.status_code, 200, diagnosis.text)
        diagnosed = diagnosis.json()
        self.assertTrue(diagnosed["source_attested"])
        self.assertEqual(diagnosed["diagnosis_entered_by"], "RESEARCHER-2")
        self.assertEqual(diagnosed["reviewer_id"], "PATH-9")

        feedback = self.client.post(
            f"/api/v1/cases/{case['submission_id']}/feedback",
            json={
                "reviewer_id": "PATH-9",
                "operator_id": "RESEARCHER-2",
                "images": [
                    {"case_id": case["images"][0]["case_id"], "notes": None}
                ],
                "patient_notes": None,
                "diagnosis_changed_after_ai": False,
            },
        )
        self.assertEqual(feedback.status_code, 200, feedback.text)
        self.assertEqual(feedback.json()["feedback_entered_by"], "RESEARCHER-2")

        image_export = self.client.get("/api/v1/metrics/export.csv")
        self.assertIn("interaction_mode", image_export.text)
        self.assertIn("researcher_mediated", image_export.text)
        self.assertIn("diagnosing_pathologist_id", image_export.text)


if __name__ == "__main__":
    unittest.main()
