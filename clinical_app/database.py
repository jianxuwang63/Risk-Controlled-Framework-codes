from __future__ import annotations

import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def elapsed_ms(started_at: str | None, ended_at: str) -> int | None:
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(ended_at)
    except (TypeError, ValueError):
        return None
    return max(0, int((end - start).total_seconds() * 1000))


def objective_ai_rating(*, accepted: bool, predicted: int, reference: int) -> str:
    if not accepted:
        return "neutral"
    return "effective" if predicted == reference else "unhelpful"


class PilotDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    case_ref TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    submitted_by TEXT,
                    use_context TEXT NOT NULL DEFAULT 'legacy_research',
                    interaction_mode TEXT NOT NULL DEFAULT 'legacy_unspecified',
                    diagnosing_pathologist_id TEXT,
                    submission_id TEXT,
                    pilot_phase TEXT NOT NULL DEFAULT 'assisted',
                    evaluation_cohort TEXT NOT NULL DEFAULT 'UNSPECIFIED',
                    evidence_role TEXT NOT NULL DEFAULT 'legacy_unspecified',
                    image_index INTEGER NOT NULL DEFAULT 1,
                    image_count INTEGER NOT NULL DEFAULT 1,
                    image_name TEXT NOT NULL DEFAULT '',
                    image_storage_key TEXT,
                    image_content_type TEXT,
                    image_size_bytes INTEGER,
                    image_sha256 TEXT NOT NULL,
                    image_width INTEGER NOT NULL,
                    image_height INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    model_hashes_json TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    prediction_threshold REAL,
                    acceptance_threshold REAL,
                    target_system_fnr REAL,
                    confidence_delta REAL,
                    training_overlap INTEGER NOT NULL DEFAULT 0,
                    original_dataset_label INTEGER,
                    pre_ai_ambiguity_flag INTEGER NOT NULL DEFAULT 0,
                    reference_adjudication_status TEXT NOT NULL DEFAULT 'not_required',
                    backend_mode TEXT NOT NULL,
                    p_mip REAL NOT NULL,
                    selection_score REAL NOT NULL,
                    predicted_label INTEGER NOT NULL,
                    accepted INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    inference_ms REAL NOT NULL,
                    tile_count INTEGER NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    reviewed_at TEXT,
                    reviewer_id TEXT,
                    ground_truth_label INTEGER,
                    ai_usefulness TEXT,
                    review_notes TEXT,
                    image_feedback_notes TEXT,
                    patient_ground_truth_label INTEGER,
                    patient_ai_usefulness TEXT,
                    patient_diagnosis_notes TEXT,
                    surgical_procedure TEXT,
                    patient_feedback_notes TEXT,
                    feedback_status TEXT NOT NULL DEFAULT 'pending',
                    feedback_at TEXT,
                    diagnosis_started_at TEXT,
                    diagnosis_started_by TEXT,
                    diagnosis_started_operator TEXT,
                    diagnosis_entered_by TEXT,
                    feedback_entered_by TEXT,
                    source_attested INTEGER NOT NULL DEFAULT 0,
                    diagnosis_duration_ms INTEGER,
                    feedback_duration_ms INTEGER,
                    diagnosis_changed_after_ai INTEGER,
                    post_ai_patient_label INTEGER,
                    aggregation_rule TEXT NOT NULL DEFAULT 'none_image_primary'
                );

                CREATE INDEX IF NOT EXISTS idx_cases_status
                    ON cases(review_status, submitted_at DESC);
                CREATE INDEX IF NOT EXISTS idx_cases_submitted
                    ON cases(submitted_at DESC);

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT,
                    event_type TEXT NOT NULL,
                    event_at TEXT NOT NULL,
                    actor TEXT,
                    details_json TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id)
                );
                """
            )

            # Migrate databases created by app versions before patient-level multi-image
            # submissions were introduced. Existing records remain valid single-image cases.
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(cases)").fetchall()
            }
            needs_legacy_review_migration = "feedback_status" not in columns
            migrations = {
                "submission_id": "TEXT",
                "use_context": "TEXT NOT NULL DEFAULT 'legacy_research'",
                "interaction_mode": "TEXT NOT NULL DEFAULT 'legacy_unspecified'",
                "diagnosing_pathologist_id": "TEXT",
                "pilot_phase": "TEXT NOT NULL DEFAULT 'assisted'",
                "evaluation_cohort": "TEXT NOT NULL DEFAULT 'UNSPECIFIED'",
                "evidence_role": "TEXT NOT NULL DEFAULT 'legacy_unspecified'",
                "image_index": "INTEGER NOT NULL DEFAULT 1",
                "image_count": "INTEGER NOT NULL DEFAULT 1",
                "image_name": "TEXT NOT NULL DEFAULT ''",
                "image_storage_key": "TEXT",
                "image_content_type": "TEXT",
                "image_size_bytes": "INTEGER",
                "prediction_threshold": "REAL",
                "acceptance_threshold": "REAL",
                "target_system_fnr": "REAL",
                "confidence_delta": "REAL",
                "training_overlap": "INTEGER NOT NULL DEFAULT 0",
                "original_dataset_label": "INTEGER",
                "pre_ai_ambiguity_flag": "INTEGER NOT NULL DEFAULT 0",
                "reference_adjudication_status": (
                    "TEXT NOT NULL DEFAULT 'not_required'"
                ),
                "image_feedback_notes": "TEXT",
                "patient_ground_truth_label": "INTEGER",
                "patient_ai_usefulness": "TEXT",
                "patient_diagnosis_notes": "TEXT",
                "surgical_procedure": "TEXT",
                "patient_feedback_notes": "TEXT",
                "feedback_status": "TEXT NOT NULL DEFAULT 'pending'",
                "feedback_at": "TEXT",
                "diagnosis_started_at": "TEXT",
                "diagnosis_started_by": "TEXT",
                "diagnosis_started_operator": "TEXT",
                "diagnosis_entered_by": "TEXT",
                "feedback_entered_by": "TEXT",
                "source_attested": "INTEGER NOT NULL DEFAULT 0",
                "diagnosis_duration_ms": "INTEGER",
                "feedback_duration_ms": "INTEGER",
                "diagnosis_changed_after_ai": "INTEGER",
                "post_ai_patient_label": "INTEGER",
                "aggregation_rule": "TEXT NOT NULL DEFAULT 'none_image_primary'",
            }
            for column, definition in migrations.items():
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE cases ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                """
                UPDATE cases SET submission_id = case_id
                WHERE submission_id IS NULL OR submission_id = ''
                """
            )
            connection.execute(
                """
                UPDATE cases
                SET diagnosing_pathologist_id = COALESCE(
                    diagnosing_pathologist_id, reviewer_id, submitted_by
                )
                WHERE diagnosing_pathologist_id IS NULL
                   OR diagnosing_pathologist_id = ''
                """
            )
            # Run this compatibility rewrite only once for databases that genuinely
            # predate the separate diagnosis/feedback workflow. Re-running it on
            # every startup would incorrectly convert newly reviewed sessions whose
            # post-AI feedback is still pending into legacy records.
            if needs_legacy_review_migration:
                connection.execute(
                    """
                    UPDATE cases
                    SET patient_ground_truth_label = COALESCE(
                            patient_ground_truth_label, ground_truth_label
                        ),
                        patient_ai_usefulness = COALESCE(
                            patient_ai_usefulness, ai_usefulness
                        ),
                        feedback_status = CASE
                            WHEN review_status = 'reviewed' THEN 'completed'
                            ELSE feedback_status
                        END
                    """
                )
                connection.execute(
                    """
                    UPDATE cases SET feedback_status = 'legacy_not_collected'
                    WHERE feedback_status = 'completed' AND feedback_at IS NULL
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cases_submission
                ON cases(submission_id, image_index)
                """
            )
    def case_ref_exists(self, case_ref: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM cases WHERE case_ref = ? LIMIT 1", (case_ref,)
            ).fetchone()
        return row is not None

    def insert_case(self, record: dict[str, Any]) -> None:
        self.insert_submission([record])

    def insert_submission(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("records cannot be empty")
        normalized: list[dict[str, Any]] = []
        image_count = len(records)
        for image_index, original in enumerate(records, start=1):
            record = dict(original)
            record.setdefault("submission_id", record["case_id"])
            record.setdefault("pilot_phase", "assisted")
            record.setdefault("evaluation_cohort", "UNSPECIFIED")
            record.setdefault("evidence_role", "legacy_unspecified")
            record.setdefault("use_context", "legacy_research")
            record.setdefault("interaction_mode", "legacy_unspecified")
            record.setdefault(
                "diagnosing_pathologist_id", record.get("submitted_by")
            )
            record.setdefault("image_index", image_index)
            record.setdefault("image_count", image_count)
            record.setdefault("image_name", f"image_{image_index}")
            record.setdefault("aggregation_rule", "none_image_primary")
            normalized.append(record)
        submission_ids = {record["submission_id"] for record in normalized}
        if len(submission_ids) != 1:
            raise ValueError("all records must share one submission_id")
        case_refs = {record["case_ref"] for record in normalized}
        if len(case_refs) != 1:
            raise ValueError("all records must share one case_ref")

        with self.connect() as connection:
            duplicate = connection.execute(
                "SELECT 1 FROM cases WHERE case_ref = ? LIMIT 1",
                (normalized[0]["case_ref"],),
            ).fetchone()
            if duplicate:
                raise ValueError("case_ref has already been submitted")
            for record in normalized:
                columns = tuple(record.keys())
                placeholders = ", ".join("?" for _ in columns)
                sql = (
                    f"INSERT INTO cases ({', '.join(columns)}) "
                    f"VALUES ({placeholders})"
                )
                connection.execute(
                    sql, tuple(record[column] for column in columns)
                )
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        case_id, event_type, event_at, actor, details_json
                    )
                    VALUES (?, 'inference_completed', ?, ?, ?)
                    """,
                    (
                        record["case_id"],
                        utc_now(),
                        record.get("submitted_by"),
                        json.dumps(
                            {
                                "submission_id": record["submission_id"],
                                "image_index": record["image_index"],
                                "image_count": record["image_count"],
                                "pilot_phase": record["pilot_phase"],
                                "evaluation_cohort": record["evaluation_cohort"],
                                "evidence_role": record["evidence_role"],
                                "use_context": record["use_context"],
                                "model_version": record["model_version"],
                                "policy_version": record["policy_version"],
                                "decision": record["decision"],
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_submission(self, identifier: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            resolved = connection.execute(
                """
                SELECT submission_id FROM cases
                WHERE submission_id = ? OR case_id = ?
                LIMIT 1
                """,
                (identifier, identifier),
            ).fetchone()
            if not resolved:
                return None
            rows = connection.execute(
                """
                SELECT * FROM cases WHERE submission_id = ?
                ORDER BY image_index
                """,
                (resolved["submission_id"],),
            ).fetchall()
        return self._rows_to_submission(rows)

    def list_cases(self, status: str | None, limit: int) -> list[dict[str, Any]]:
        return self.list_submissions(status, limit)

    def list_submissions(
        self, status: str | None, limit: int
    ) -> list[dict[str, Any]]:
        sql = "SELECT submission_id, MAX(submitted_at) AS latest FROM cases"
        params: list[Any] = []
        if status:
            sql += " WHERE review_status = ?"
            params.append(status)
        sql += " GROUP BY submission_id ORDER BY latest DESC LIMIT ?"
        params.append(limit)
        with self.connect() as connection:
            submission_rows = connection.execute(sql, params).fetchall()
            submission_ids = [row["submission_id"] for row in submission_rows]
            if not submission_ids:
                return []
            placeholders = ",".join("?" for _ in submission_ids)
            rows = connection.execute(
                f"""
                SELECT * FROM cases WHERE submission_id IN ({placeholders})
                ORDER BY submitted_at DESC, image_index
                """,
                submission_ids,
            ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {
            submission_id: [] for submission_id in submission_ids
        }
        for row in rows:
            grouped[row["submission_id"]].append(row)
        return [
            self._rows_to_submission(grouped[submission_id])
            for submission_id in submission_ids
        ]

    def review_case(
        self,
        case_id: str,
        *,
        reviewer_id: str,
        ground_truth_label: int,
        ai_usefulness: str,
        notes: str | None,
    ) -> dict[str, Any] | None:
        return self.review_submission(
            case_id,
            reviewer_id=reviewer_id,
            ground_truth_label=ground_truth_label,
            ai_usefulness=ai_usefulness,
            notes=notes,
        )

    def review_submission(
        self,
        identifier: str,
        *,
        reviewer_id: str,
        ground_truth_label: int,
        ai_usefulness: str,
        notes: str | None,
    ) -> dict[str, Any] | None:
        """Compatibility helper for legacy single-label callers.

        New clinical workflow code should use ``diagnose_submission`` followed by
        ``record_feedback`` so image labels and patient feedback remain distinct.
        """
        existing = self.get_submission(identifier)
        if existing is None:
            return None
        image_reviews = [
            {
                "case_id": image["case_id"],
                "ground_truth_label": ground_truth_label,
                "notes": notes,
            }
            for image in existing["images"]
        ]
        diagnosed = self.diagnose_submission(
            identifier,
            reviewer_id=reviewer_id,
            image_reviews=image_reviews,
            patient_notes=notes,
        )
        if diagnosed is None:
            return None
        return self.record_feedback(
            identifier,
            reviewer_id=reviewer_id,
            image_feedback=[
                {
                    "case_id": image["case_id"],
                    "notes": None,
                }
                for image in diagnosed["images"]
            ],
            patient_notes=None,
            diagnosis_changed_after_ai=False,
        )

    def start_diagnosis(
        self, identifier: str, *, reviewer_id: str, operator_id: str | None = None
    ) -> dict[str, Any] | None:
        operator_id = operator_id or reviewer_id
        with self.connect() as connection:
            resolved = connection.execute(
                """
                SELECT submission_id FROM cases
                WHERE submission_id = ? OR case_id = ? LIMIT 1
                """,
                (identifier, identifier),
            ).fetchone()
            if not resolved:
                return None
            submission_id = resolved["submission_id"]
            rows = connection.execute(
                """
                SELECT review_status, diagnosis_started_at, diagnosis_started_by,
                       diagnosis_started_operator, interaction_mode,
                       diagnosing_pathologist_id
                FROM cases WHERE submission_id = ? ORDER BY image_index
                """,
                (submission_id,),
            ).fetchall()
            if any(row["review_status"] == "reviewed" for row in rows):
                raise ValueError("independent diagnosis has already been submitted")
            pathologist_id = rows[0]["diagnosing_pathologist_id"]
            interaction_mode = rows[0]["interaction_mode"]
            if (
                interaction_mode != "legacy_unspecified"
                and pathologist_id
                and pathologist_id != reviewer_id
            ):
                raise ValueError("reviewer_id does not match the assigned pathologist")
            if interaction_mode == "direct_on_device" and operator_id != reviewer_id:
                raise ValueError("direct workflow requires operator and pathologist IDs to match")
            started_by = rows[0]["diagnosis_started_by"]
            if started_by and started_by != reviewer_id:
                raise ValueError("this case is already being reviewed by another physician")
            started_operator = rows[0]["diagnosis_started_operator"]
            if started_operator and started_operator != operator_id:
                raise ValueError("this case is already being entered by another operator")
            if not rows[0]["diagnosis_started_at"]:
                started_at = utc_now()
                connection.execute(
                    """
                    UPDATE cases SET diagnosis_started_at = ?, diagnosis_started_by = ?,
                                     diagnosis_started_operator = ?
                    WHERE submission_id = ?
                    """,
                    (started_at, reviewer_id, operator_id, submission_id),
                )
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        case_id, event_type, event_at, actor, details_json
                    ) VALUES (
                        (SELECT case_id FROM cases WHERE submission_id = ?
                         ORDER BY image_index LIMIT 1),
                        'independent_diagnosis_started', ?, ?, ?
                    )
                    """,
                    (
                        submission_id,
                        started_at,
                        operator_id,
                        json.dumps(
                            {
                                "submission_id": submission_id,
                                "diagnosing_pathologist_id": reviewer_id,
                                "data_entry_operator_id": operator_id,
                                "interaction_mode": interaction_mode,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
        return self.get_submission(submission_id)

    def diagnose_submission(
        self,
        identifier: str,
        *,
        reviewer_id: str,
        operator_id: str | None = None,
        source_attested: bool = False,
        image_reviews: list[dict[str, Any]],
        patient_notes: str | None,
        surgical_procedure: str | None = None,
    ) -> dict[str, Any] | None:
        operator_id = operator_id or reviewer_id
        with self.connect() as connection:
            resolved = connection.execute(
                """
                SELECT submission_id FROM cases
                WHERE submission_id = ? OR case_id = ?
                LIMIT 1
                """,
                (identifier, identifier),
            ).fetchone()
            if not resolved:
                return None
            submission_id = resolved["submission_id"]
            rows = connection.execute(
                """
                SELECT case_id, review_status, diagnosis_started_at,
                       diagnosis_started_by, diagnosis_started_operator,
                       interaction_mode, diagnosing_pathologist_id, use_context FROM cases
                WHERE submission_id = ? ORDER BY image_index
                """,
                (submission_id,),
            ).fetchall()
            if any(row["review_status"] == "reviewed" for row in rows):
                raise ValueError("independent diagnosis has already been submitted")
            pathologist_id = rows[0]["diagnosing_pathologist_id"]
            interaction_mode = rows[0]["interaction_mode"]
            if (
                interaction_mode != "legacy_unspecified"
                and pathologist_id
                and pathologist_id != reviewer_id
            ):
                raise ValueError("reviewer_id does not match the assigned pathologist")
            if interaction_mode == "direct_on_device" and operator_id != reviewer_id:
                raise ValueError("direct workflow requires operator and pathologist IDs to match")
            if interaction_mode == "researcher_mediated" and not source_attested:
                raise ValueError(
                    "mediated workflow requires attestation that the independent "
                    "diagnosis was recorded before AI reveal"
                )
            if rows[0]["diagnosis_started_by"] and rows[0]["diagnosis_started_by"] != reviewer_id:
                raise ValueError("diagnosis must be submitted by the physician who started it")
            if rows[0]["diagnosis_started_operator"] and rows[0]["diagnosis_started_operator"] != operator_id:
                raise ValueError("diagnosis must be entered by the operator who started it")
            expected_case_ids = {row["case_id"] for row in rows}
            supplied = {str(review["case_id"]): review for review in image_reviews}
            if set(supplied) != expected_case_ids or len(supplied) != len(image_reviews):
                raise ValueError("one image diagnosis is required for every image")
            labels = [int(review["ground_truth_label"]) for review in supplied.values()]
            if any(label not in (0, 1) for label in labels):
                raise ValueError("image ground_truth_label must be 0 or 1")
            reviewed_at = utc_now()
            diagnosis_duration_ms = elapsed_ms(
                rows[0]["diagnosis_started_at"], reviewed_at
            )
            for row in rows:
                review = supplied[row["case_id"]]
                label = int(review["ground_truth_label"])
                notes = review.get("notes")
                connection.execute(
                    """
                    UPDATE cases
                    SET review_status = 'reviewed', reviewed_at = ?, reviewer_id = ?,
                        ground_truth_label = ?, review_notes = ?,
                        patient_ground_truth_label = NULL, patient_diagnosis_notes = ?,
                        surgical_procedure = ?,
                        ai_usefulness = NULL, image_feedback_notes = NULL,
                        patient_ai_usefulness = NULL, patient_feedback_notes = NULL,
                        feedback_status = 'pending', feedback_at = NULL,
                        diagnosis_duration_ms = ?, feedback_duration_ms = NULL,
                        diagnosis_entered_by = ?, source_attested = ?,
                        diagnosis_changed_after_ai = NULL,
                        post_ai_patient_label = NULL,
                        aggregation_rule = 'none_image_primary'
                    WHERE case_id = ?
                    """,
                    (
                        reviewed_at,
                        reviewer_id,
                        label,
                        notes,
                        patient_notes,
                        surgical_procedure,
                        diagnosis_duration_ms,
                        operator_id,
                        int(source_attested),
                        row["case_id"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        case_id, event_type, event_at, actor, details_json
                    )
                    VALUES (?, 'image_diagnosis_recorded', ?, ?, ?)
                    """,
                    (
                        row["case_id"],
                        reviewed_at,
                        operator_id,
                        json.dumps(
                            {
                                "submission_id": submission_id,
                                "image_ground_truth_label": label,
                                "analysis_unit": "image",
                                "aggregation_rule": "none_image_primary",
                                "surgical_procedure": surgical_procedure,
                                "diagnosis_duration_ms": diagnosis_duration_ms,
                                "diagnosing_pathologist_id": reviewer_id,
                                "data_entry_operator_id": operator_id,
                                "interaction_mode": interaction_mode,
                                "use_context": rows[0]["use_context"],
                                "source_attested": bool(source_attested),
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            connection.execute(
                """
                INSERT INTO audit_events(
                    case_id, event_type, event_at, actor, details_json
                ) VALUES (?, 'image_review_session_recorded', ?, ?, ?)
                """,
                (
                    rows[0]["case_id"],
                    reviewed_at,
                    operator_id,
                    json.dumps(
                        {
                            "submission_id": submission_id,
                            "image_review_count": len(rows),
                            "positive_image_count": sum(labels),
                            "negative_image_count": len(labels) - sum(labels),
                            "aggregation_rule": "none_image_primary",
                            "diagnosis_duration_ms": diagnosis_duration_ms,
                            "diagnosing_pathologist_id": reviewer_id,
                            "data_entry_operator_id": operator_id,
                            "interaction_mode": interaction_mode,
                            "use_context": rows[0]["use_context"],
                            "source_attested": bool(source_attested),
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        return self.get_submission(submission_id)

    def record_feedback(
        self,
        identifier: str,
        *,
        reviewer_id: str,
        operator_id: str | None = None,
        image_feedback: list[dict[str, Any]],
        patient_notes: str | None,
        diagnosis_changed_after_ai: bool,
    ) -> dict[str, Any] | None:
        operator_id = operator_id or reviewer_id
        with self.connect() as connection:
            resolved = connection.execute(
                """
                SELECT submission_id FROM cases
                WHERE submission_id = ? OR case_id = ?
                LIMIT 1
                """,
                (identifier, identifier),
            ).fetchone()
            if not resolved:
                return None
            submission_id = resolved["submission_id"]
            rows = connection.execute(
                """
                SELECT case_id, review_status, reviewer_id, feedback_status,
                       accepted, predicted_label, ground_truth_label,
                       patient_ground_truth_label, reviewed_at,
                       interaction_mode, diagnosing_pathologist_id,
                       diagnosis_entered_by, use_context
                FROM cases WHERE submission_id = ? ORDER BY image_index
                """,
                (submission_id,),
            ).fetchall()
            if any(row["review_status"] != "reviewed" for row in rows):
                raise ValueError("independent diagnosis must be submitted first")
            if any(row["reviewer_id"] != reviewer_id for row in rows):
                raise ValueError("feedback must be submitted by the diagnosing physician")
            interaction_mode = rows[0]["interaction_mode"]
            pathologist_id = rows[0]["diagnosing_pathologist_id"]
            if (
                interaction_mode != "legacy_unspecified"
                and pathologist_id
                and pathologist_id != reviewer_id
            ):
                raise ValueError("reviewer_id does not match the assigned pathologist")
            if interaction_mode == "direct_on_device" and operator_id != reviewer_id:
                raise ValueError("direct workflow requires operator and pathologist IDs to match")
            if rows[0]["diagnosis_entered_by"] and rows[0]["diagnosis_entered_by"] != operator_id:
                raise ValueError("feedback must be entered by the diagnosis data-entry operator")
            if any(row["feedback_status"] == "completed" for row in rows):
                raise ValueError("AI feedback has already been submitted")
            expected_case_ids = {row["case_id"] for row in rows}
            supplied = {str(item["case_id"]): item for item in image_feedback}
            if set(supplied) != expected_case_ids or len(supplied) != len(image_feedback):
                raise ValueError("one optional note record is required for every image")

            feedback_at = utc_now()
            feedback_duration_ms = elapsed_ms(rows[0]["reviewed_at"], feedback_at)
            for row in rows:
                feedback = supplied[row["case_id"]]
                image_rating = objective_ai_rating(
                    accepted=bool(row["accepted"]),
                    predicted=int(row["predicted_label"]),
                    reference=int(row["ground_truth_label"]),
                )
                connection.execute(
                    """
                    UPDATE cases
                    SET ai_usefulness = ?, image_feedback_notes = ?,
                        patient_ai_usefulness = NULL, patient_feedback_notes = ?,
                        feedback_status = 'completed', feedback_at = ?,
                        feedback_entered_by = ?,
                        feedback_duration_ms = ?, diagnosis_changed_after_ai = ?,
                        post_ai_patient_label = NULL,
                        aggregation_rule = 'none_image_primary'
                    WHERE case_id = ?
                    """,
                    (
                        image_rating,
                        feedback.get("notes"),
                        patient_notes,
                        feedback_at,
                        operator_id,
                        feedback_duration_ms,
                        int(diagnosis_changed_after_ai),
                        row["case_id"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        case_id, event_type, event_at, actor, details_json
                    ) VALUES (?, 'image_ai_feedback_recorded', ?, ?, ?)
                    """,
                    (
                        row["case_id"],
                        feedback_at,
                        operator_id,
                        json.dumps(
                            {
                                "submission_id": submission_id,
                                "ai_usefulness": image_rating,
                                "rating_rule": "effective_if_correct_neutral_if_deferred",
                                "diagnosing_pathologist_id": reviewer_id,
                                "data_entry_operator_id": operator_id,
                                "interaction_mode": interaction_mode,
                                "use_context": rows[0]["use_context"],
                                "any_image_diagnosis_changed_after_ai": diagnosis_changed_after_ai,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            connection.execute(
                """
                INSERT INTO audit_events(
                    case_id, event_type, event_at, actor, details_json
                ) VALUES (?, 'image_review_session_feedback_recorded', ?, ?, ?)
                """,
                (
                    rows[0]["case_id"],
                    feedback_at,
                    operator_id,
                    json.dumps(
                        {
                            "submission_id": submission_id,
                            "image_review_count": len(rows),
                            "any_image_diagnosis_changed_after_ai": diagnosis_changed_after_ai,
                            "feedback_duration_ms": feedback_duration_ms,
                            "diagnosing_pathologist_id": reviewer_id,
                            "data_entry_operator_id": operator_id,
                            "interaction_mode": interaction_mode,
                            "use_context": rows[0]["use_context"],
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        return self.get_submission(submission_id)

    def all_cases_for_metrics(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cases ORDER BY submitted_at"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _write_csv(fields: list[str], rows: list[dict[str, Any]]) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
        # UTF-8 BOM keeps Chinese text readable when opened directly in Windows Excel.
        return "\ufeff" + output.getvalue()

    def export_images_csv(self, *, legacy_only: bool = False) -> str:
        fields = [
            "analysis_included",
            "analysis_unit",
            "analysis_population",
            "evaluation_cohort",
            "evidence_role",
            "independent_validation_included",
            "high_agreement_sensitivity_included",
            "training_overlap",
            "original_dataset_label",
            "training_overlap_label_disagreement",
            "pre_ai_ambiguity_flag",
            "reference_adjudication_status",
            "ground_truth_source",
            "reference_label_text",
            "model_output",
            "rejection_score",
            "positive_probability_margin",
            "selection_threshold_margin",
            "is_deferred",
            "prediction_correct_when_accepted",
            "is_true_positive",
            "is_true_negative",
            "is_false_positive",
            "is_false_negative",
            "is_deferred_positive",
            "is_deferred_negative",
            "system_fnr_numerator",
            "system_fnr_denominator",
            "accepted_risk_numerator",
            "accepted_risk_denominator",
            "system_accuracy_numerator",
            "system_accuracy_denominator",
            "submission_id",
            "case_id",
            "case_ref",
            "submitted_by",
            "use_context",
            "interaction_mode",
            "diagnosing_pathologist_id",
            "reviewer_id",
            "pilot_phase",
            "image_index",
            "image_count",
            "image_name",
            "submitted_at",
            "reviewed_at",
            "feedback_at",
            "diagnosis_started_at",
            "diagnosis_started_by",
            "diagnosis_started_operator",
            "diagnosis_entered_by",
            "feedback_entered_by",
            "source_attested",
            "diagnosis_duration_ms",
            "feedback_duration_ms",
            "diagnosis_changed_after_ai",
            "post_ai_patient_label",
            "model_version",
            "model_hashes_json",
            "policy_version",
            "prediction_threshold",
            "acceptance_threshold",
            "target_system_fnr",
            "confidence_delta",
            "backend_mode",
            "p_mip",
            "selection_score",
            "predicted_label",
            "accepted",
            "decision",
            "ground_truth_label",
            "review_notes",
            "ai_usefulness",
            "image_feedback_notes",
            "patient_ground_truth_label",
            "patient_ai_usefulness",
            "patient_diagnosis_notes",
            "surgical_procedure",
            "patient_feedback_notes",
            "feedback_status",
            "aggregation_rule",
            "inference_ms",
            "tile_count",
            "image_sha256",
            "image_width",
            "image_height",
            "image_content_type",
            "image_size_bytes",
            "review_status",
        ]
        predicate = (
            "(feedback_status = 'legacy_not_collected' OR use_context = 'legacy_research')"
            if legacy_only
            else (
                "feedback_status <> 'legacy_not_collected' "
                "AND use_context = 'pathologist_self_review' "
                "AND evidence_role IN ("
                "'controlled_post_launch_pilot', 'held_out_validation'"
                ") "
                "AND review_status = 'reviewed' "
                "AND ground_truth_label IN (0, 1)"
            )
        )
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM cases WHERE {predicate}
                ORDER BY submitted_at, submission_id, image_index
                """
            ).fetchall()
        exported: list[dict[str, Any]] = []
        for database_row in rows:
            row = dict(database_row)
            if not legacy_only:
                accepted = bool(row.get("accepted"))
                training_overlap = bool(row.get("training_overlap"))
                ambiguity = bool(row.get("pre_ai_ambiguity_flag"))
                predicted = int(row["predicted_label"])
                reference = int(row["ground_truth_label"])
                correct = accepted and predicted == reference
                false_positive = accepted and predicted == 1 and reference == 0
                false_negative = accepted and predicted == 0 and reference == 1
                row.update(
                    {
                        "analysis_included": int(not training_overlap),
                        "independent_validation_included": int(
                            not training_overlap
                        ),
                        "high_agreement_sensitivity_included": int(
                            not training_overlap and not ambiguity
                        ),
                        "analysis_unit": "image",
                        "analysis_population": (
                            "formal_image_evaluation_with_pre_ai_reference_label"
                        ),
                        "ground_truth_source": (
                            "attested_pre_ai_source_record"
                            if row.get("interaction_mode") == "researcher_mediated"
                            and bool(row.get("source_attested"))
                            else "recorded_pre_ai_image_assessment"
                        ),
                        "training_overlap_label_disagreement": (
                            int(
                                int(row["original_dataset_label"])
                                != reference
                            )
                            if training_overlap
                            and row.get("original_dataset_label") is not None
                            else None
                        ),
                        "reference_label_text": (
                            "MIP_present" if reference == 1 else "MIP_absent"
                        ),
                        "model_output": (
                            "deferred"
                            if not accepted
                            else "MIP_present"
                            if predicted == 1
                            else "MIP_absent"
                        ),
                        "rejection_score": 1.0 - float(row["selection_score"]),
                        "positive_probability_margin": (
                            float(row["p_mip"])
                            - float(row["prediction_threshold"])
                            if row.get("prediction_threshold") is not None
                            else None
                        ),
                        "selection_threshold_margin": (
                            float(row["selection_score"])
                            - float(row["acceptance_threshold"])
                            if row.get("acceptance_threshold") is not None
                            else None
                        ),
                        "is_deferred": int(not accepted),
                        "prediction_correct_when_accepted": (
                            int(correct) if accepted else None
                        ),
                        "is_true_positive": int(
                            accepted and predicted == 1 and reference == 1
                        ),
                        "is_true_negative": int(
                            accepted and predicted == 0 and reference == 0
                        ),
                        "is_false_positive": int(false_positive),
                        "is_false_negative": int(false_negative),
                        "is_deferred_positive": int(
                            not accepted and reference == 1
                        ),
                        "is_deferred_negative": int(
                            not accepted and reference == 0
                        ),
                        "system_fnr_numerator": int(false_negative),
                        "system_fnr_denominator": int(reference == 1),
                        "accepted_risk_numerator": int(
                            accepted and predicted != reference
                        ),
                        "accepted_risk_denominator": int(accepted),
                        "system_accuracy_numerator": int(correct),
                        "system_accuracy_denominator": 1,
                    }
                )
            exported.append(row)
        return self._write_csv(fields, exported)

    def export_reviewed_csv(self) -> str:
        """Backward-compatible name; formal export excludes legacy engineering data."""
        return self.export_images_csv(legacy_only=False)

    def export_patients_csv(self) -> str:
        fields = [
            "submission_id",
            "case_ref",
            "submitted_by",
            "use_context",
            "interaction_mode",
            "evaluation_cohort",
            "evidence_role",
            "diagnosing_pathologist_id",
            "reviewer_id",
            "pilot_phase",
            "submitted_at",
            "reviewed_at",
            "feedback_at",
            "image_count",
            "model_version",
            "model_hashes",
            "policy_version",
            "prediction_threshold",
            "acceptance_threshold",
            "target_system_fnr",
            "confidence_delta",
            "backend_mode",
            "patient_p_mip",
            "patient_predicted_label",
            "patient_accepted",
            "patient_decision",
            "patient_ground_truth_label",
            "patient_ai_usefulness",
            "patient_diagnosis_notes",
            "surgical_procedure",
            "patient_feedback_notes",
            "post_ai_patient_label",
            "diagnosis_changed_after_ai",
            "diagnosis_started_at",
            "diagnosis_started_operator",
            "diagnosis_entered_by",
            "feedback_entered_by",
            "source_attested",
            "diagnosis_duration_ms",
            "feedback_duration_ms",
            "review_status",
            "feedback_status",
            "aggregation_rule",
            "inference_ms",
            "tile_count",
        ]
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM cases
                WHERE feedback_status <> 'legacy_not_collected'
                  AND use_context <> 'legacy_research'
                ORDER BY submitted_at, submission_id, image_index
                """
            ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(str(row["submission_id"]), []).append(row)
        patients = [self._rows_to_submission(group) for group in grouped.values()]
        for patient in patients:
            patient["model_hashes"] = json.dumps(
                patient.get("model_hashes", []), ensure_ascii=False
            )
        return self._write_csv(fields, patients)

    def export_audit_csv(self) -> str:
        fields = [
            "event_id", "case_id", "case_ref", "submission_id", "event_type",
            "event_at", "actor", "details_json",
        ]
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT a.event_id, a.case_id, c.case_ref, c.submission_id,
                       a.event_type, a.event_at, a.actor, a.details_json
                FROM audit_events a JOIN cases c ON c.case_id = a.case_id
                WHERE c.feedback_status <> 'legacy_not_collected'
                  AND c.use_context <> 'legacy_research'
                ORDER BY a.event_id
                """
            ).fetchall()
        return self._write_csv(fields, [dict(row) for row in rows])

    @classmethod
    def _rows_to_submission(
        cls, rows: list[sqlite3.Row]
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError("rows cannot be empty")
        images = [cls._row_to_dict(row) for row in rows]
        first = images[0]
        reviewed = all(image["review_status"] == "reviewed" for image in images)
        accepted_count = sum(bool(image["accepted"]) for image in images)
        positive_count = sum(
            bool(image["accepted"]) and int(image["predicted_label"]) == 1
            for image in images
        )
        negative_count = sum(
            bool(image["accepted"]) and int(image["predicted_label"]) == 0
            for image in images
        )
        submission: dict[str, Any] = {
            "submission_id": first["submission_id"],
            "case_id": first["submission_id"],
            "case_ref": first["case_ref"],
            "submitted_at": first["submitted_at"],
            "submitted_by": first["submitted_by"],
            "use_context": first.get("use_context", "legacy_research"),
            "interaction_mode": first["interaction_mode"],
            "diagnosing_pathologist_id": first["diagnosing_pathologist_id"],
            "pilot_phase": first["pilot_phase"],
            "evaluation_cohort": first["evaluation_cohort"],
            "evidence_role": first["evidence_role"],
            "image_count": len(images),
            "review_status": "reviewed" if reviewed else "pending",
            "reviewed_at": first["reviewed_at"] if reviewed else None,
            "reviewer_id": first["reviewer_id"] if reviewed else None,
            "feedback_status": first["feedback_status"],
            "feedback_at": first["feedback_at"],
            "diagnosis_started_at": first["diagnosis_started_at"],
            "diagnosis_started_by": first["diagnosis_started_by"],
            "diagnosis_started_operator": first["diagnosis_started_operator"],
            "diagnosis_entered_by": first["diagnosis_entered_by"],
            "feedback_entered_by": first["feedback_entered_by"],
            "source_attested": bool(first["source_attested"]),
            "diagnosis_duration_ms": first["diagnosis_duration_ms"],
            "feedback_duration_ms": first["feedback_duration_ms"],
            "diagnosis_changed_after_ai": first["diagnosis_changed_after_ai"],
            "post_ai_patient_label": first["post_ai_patient_label"],
            "ground_truth_label": None,
            "patient_ground_truth_label": None,
            "ai_usefulness": None,
            "patient_ai_usefulness": None,
            "review_notes": first["patient_diagnosis_notes"],
            "patient_diagnosis_notes": first["patient_diagnosis_notes"],
            "surgical_procedure": first["surgical_procedure"],
            "patient_feedback_notes": first["patient_feedback_notes"],
            "aggregation_rule": "none_image_primary",
            "model_version": first["model_version"],
            "model_hashes": first["model_hashes"],
            "policy_version": first["policy_version"],
            "prediction_threshold": first["prediction_threshold"],
            "acceptance_threshold": first["acceptance_threshold"],
            "target_system_fnr": first["target_system_fnr"],
            "confidence_delta": first["confidence_delta"],
            "backend_mode": first["backend_mode"],
            "inference_ms": sum(float(image["inference_ms"]) for image in images),
            "tile_count": sum(int(image["tile_count"]) for image in images),
            "accepted_image_count": accepted_count,
            "positive_image_count": positive_count,
            "negative_image_count": negative_count,
            "deferred_image_count": len(images) - accepted_count,
            "patient_p_mip": None,
            "patient_predicted_label": None,
            "patient_accepted": None,
            "patient_decision": "not_applicable_image_level_service",
            "images": images,
        }
        submission.update(
            {
                "p_mip": first["p_mip"] if len(images) == 1 else None,
                "selection_score": (
                    first["selection_score"] if len(images) == 1 else None
                ),
                "predicted_label": (
                    first["predicted_label"] if len(images) == 1 else None
                ),
                "accepted": first["accepted"] if len(images) == 1 else None,
                "decision": (
                    first["decision"] if len(images) == 1 else "image_level_results"
                ),
            }
        )
        return submission

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["accepted"] = bool(item["accepted"])
        if item.get("diagnosis_changed_after_ai") is not None:
            item["diagnosis_changed_after_ai"] = bool(
                item["diagnosis_changed_after_ai"]
            )
        try:
            item["model_hashes"] = json.loads(item.pop("model_hashes_json"))
        except (json.JSONDecodeError, TypeError):
            item["model_hashes"] = []
            item.pop("model_hashes_json", None)
        return item
