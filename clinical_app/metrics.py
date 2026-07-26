from __future__ import annotations

import json
from collections import defaultdict
from statistics import median
from typing import Any

from .statistics import (
    clopper_pearson_interval,
    clopper_pearson_upper,
    required_positives_for_zero_failures,
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _wilson_95_interval(
    successes: int, total: int
) -> tuple[float | None, float | None]:
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * (
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        ** 0.5
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _rank_auc(rows: list[dict[str, Any]]) -> float | None:
    """Tie-aware Mann-Whitney AUROC from reference labels and p_mip scores."""
    observations = sorted(
        (
            float(row["p_mip"]),
            int(row["ground_truth_label"]),
        )
        for row in rows
        if row.get("p_mip") is not None
        and row.get("ground_truth_label") in {0, 1}
    )
    positives = sum(label == 1 for _, label in observations)
    negatives = sum(label == 0 for _, label in observations)
    if not positives or not negatives:
        return None
    positive_rank_sum = 0.0
    index = 0
    while index < len(observations):
        end = index + 1
        while (
            end < len(observations)
            and observations[end][0] == observations[index][0]
        ):
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for _, label in observations[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _rating(value: str | None) -> str | None:
    if value == "helpful":
        return "effective"
    return value if value in {"effective", "neutral", "unhelpful"} else None


def summarize_postlaunch(
    cases: list[dict[str, Any]], confidence_delta: float
) -> dict[str, Any]:
    legacy_submission_ids = {
        str(case.get("submission_id") or case["case_id"])
        for case in cases
        if case.get("feedback_status") == "legacy_not_collected"
        or case.get("use_context") == "legacy_research"
    }
    legacy_images_excluded = sum(
        str(case.get("submission_id") or case["case_id"])
        in legacy_submission_ids
        for case in cases
    )
    current = [
        case
        for case in cases
        if str(case.get("submission_id") or case["case_id"])
        not in legacy_submission_ids
    ]
    patient_information_images = [
        case for case in current if case.get("use_context") == "patient_self_check"
    ]
    workflow_rehearsal_images = [
        case
        for case in current
        if case.get("evidence_role") == "workflow_rehearsal"
    ]
    evaluation_images = [
        case
        for case in current
        if case.get("use_context") != "patient_self_check"
        and case.get("evidence_role") != "workflow_rehearsal"
    ]

    reviewed = [
        case for case in evaluation_images if case.get("review_status") == "reviewed"
    ]
    accepted = [case for case in evaluation_images if bool(case.get("accepted"))]
    accepted_reviewed = [case for case in reviewed if bool(case.get("accepted"))]
    positive = [case for case in reviewed if case.get("ground_truth_label") == 1]
    negative = [case for case in reviewed if case.get("ground_truth_label") == 0]
    false_negatives = [
        case
        for case in positive
        if bool(case.get("accepted")) and int(case.get("predicted_label", 0)) == 0
    ]
    errors = [
        case
        for case in accepted_reviewed
        if int(case.get("predicted_label", 0))
        != int(case.get("ground_truth_label", 0))
    ]
    false_positives = [
        case
        for case in negative
        if bool(case.get("accepted")) and int(case.get("predicted_label", 0)) == 1
    ]
    true_positives = [
        case
        for case in positive
        if bool(case.get("accepted")) and int(case.get("predicted_label", 0)) == 1
    ]
    true_negatives = [
        case
        for case in negative
        if bool(case.get("accepted")) and int(case.get("predicted_label", 0)) == 0
    ]
    deferred_reviewed = [case for case in reviewed if not bool(case.get("accepted"))]
    deferred_positive = [
        case for case in deferred_reviewed if case.get("ground_truth_label") == 1
    ]
    deferred_negative = [
        case for case in deferred_reviewed if case.get("ground_truth_label") == 0
    ]
    rated = [case for case in reviewed if _rating(case.get("ai_usefulness"))]
    ratings = [_rating(case.get("ai_usefulness")) for case in rated]

    submissions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in evaluation_images:
        submissions[str(case.get("submission_id") or case["case_id"])].append(case)
    completed_sessions = [
        images
        for images in submissions.values()
        if images and images[0].get("feedback_status") == "completed"
    ]
    changed_sessions = [
        images
        for images in completed_sessions
        if images[0].get("diagnosis_changed_after_ai") is not None
    ]

    physician_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in reviewed:
        physician_groups[str(case.get("reviewer_id") or "unknown")].append(case)
    physician_summaries = []
    for reviewer_id in sorted(physician_groups):
        images = physician_groups[reviewer_id]
        doctor_ratings = [
            _rating(image.get("ai_usefulness"))
            for image in images
            if _rating(image.get("ai_usefulness")) is not None
        ]
        doctor_submissions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for image in images:
            doctor_submissions[
                str(image.get("submission_id") or image["case_id"])
            ].append(image)
        completed = [
            group
            for group in doctor_submissions.values()
            if group[0].get("feedback_status") == "completed"
        ]
        changed = [
            group
            for group in completed
            if group[0].get("diagnosis_changed_after_ai") is not None
        ]
        diagnosis_times = [
            int(group[0]["diagnosis_duration_ms"])
            for group in doctor_submissions.values()
            if group[0].get("diagnosis_duration_ms") is not None
        ]
        physician_summaries.append(
            {
                "reviewer_id": reviewer_id,
                "image_reviews": len(images),
                "positive_image_reviews": sum(
                    image.get("ground_truth_label") == 1 for image in images
                ),
                "review_sessions": len(doctor_submissions),
                "feedback_completed_sessions": len(completed),
                "image_rating_count": len(doctor_ratings),
                "image_effective_count": doctor_ratings.count("effective"),
                "image_neutral_count": doctor_ratings.count("neutral"),
                "image_unhelpful_count": doctor_ratings.count("unhelpful"),
                "image_effective_rate": _ratio(
                    doctor_ratings.count("effective"), len(doctor_ratings)
                ),
                "diagnosis_changed_session_count": sum(
                    bool(group[0].get("diagnosis_changed_after_ai"))
                    for group in changed
                ),
                "diagnosis_changed_rate": _ratio(
                    sum(
                        bool(group[0].get("diagnosis_changed_after_ai"))
                        for group in changed
                    ),
                    len(changed),
                ),
                "median_independent_review_ms": (
                    median(diagnosis_times) if diagnosis_times else None
                ),
            }
        )

    inference_times = [float(case["inference_ms"]) for case in evaluation_images]
    reviewed_inference_times = [float(case["inference_ms"]) for case in reviewed]
    diagnosis_times = [
        int(images[0]["diagnosis_duration_ms"])
        for images in submissions.values()
        if images[0].get("diagnosis_duration_ms") is not None
    ]
    feedback_times = [
        int(images[0]["feedback_duration_ms"])
        for images in completed_sessions
        if images[0].get("feedback_duration_ms") is not None
    ]
    coverage = _ratio(len(accepted), len(evaluation_images))
    reviewed_coverage = _ratio(len(accepted_reviewed), len(reviewed))
    image_fnr = _ratio(len(false_negatives), len(positive))
    image_risk = _ratio(len(errors), len(accepted_reviewed))
    accepted_positive_predictions = len(true_positives) + len(false_positives)
    accepted_negative_predictions = len(true_negatives) + len(false_negatives)
    accepted_reference_positives = len(true_positives) + len(false_negatives)
    correct_accepted = len(true_positives) + len(true_negatives)
    sensitivity = _ratio(len(true_positives), len(positive))
    specificity = _ratio(len(true_negatives), len(negative))
    system_accuracy = _ratio(correct_accepted, len(reviewed))
    accepted_accuracy = _ratio(correct_accepted, len(accepted_reviewed))
    accepted_subset_fnr = _ratio(
        len(false_negatives), accepted_reference_positives
    )
    positive_predictive_value = _ratio(
        len(true_positives), accepted_positive_predictions
    )
    negative_predictive_value = _ratio(
        len(true_negatives), accepted_negative_predictions
    )
    deferred_rate = _ratio(len(deferred_reviewed), len(reviewed))
    positive_deferred_rate = _ratio(len(deferred_positive), len(positive))
    negative_deferred_rate = _ratio(len(deferred_negative), len(negative))
    reviewed_coverage_ci = _wilson_95_interval(
        len(accepted_reviewed), len(reviewed)
    )
    sensitivity_ci = _wilson_95_interval(len(true_positives), len(positive))
    specificity_ci = _wilson_95_interval(len(true_negatives), len(negative))
    system_accuracy_ci = _wilson_95_interval(correct_accepted, len(reviewed))
    accepted_accuracy_ci = _wilson_95_interval(
        correct_accepted, len(accepted_reviewed)
    )
    accepted_risk_ci = _wilson_95_interval(
        len(errors), len(accepted_reviewed)
    )
    system_fnr_exact_ci = clopper_pearson_interval(
        len(false_negatives), len(positive), confidence_delta
    )

    return {
        "legacy_images_excluded": legacy_images_excluded,
        "all_uploaded_images": len(current),
        "patient_information_images": len(patient_information_images),
        "workflow_rehearsal_images_excluded": len(
            workflow_rehearsal_images
        ),
        "all_images": len(evaluation_images),
        "reviewed_images": len(reviewed),
        "pending_review_images": len(evaluation_images) - len(reviewed),
        "all_review_sessions": len(submissions),
        "feedback_completed_sessions": len(completed_sessions),
        "reviewed_positive_images": len(positive),
        "reviewed_negative_images": len(negative),
        "image_accepted_reviewed_count": len(accepted_reviewed),
        "image_true_positive_count": len(true_positives),
        "image_true_negative_count": len(true_negatives),
        "image_false_positive_count": len(false_positives),
        "image_false_negative_count": len(false_negatives),
        "image_correct_accepted_count": correct_accepted,
        "image_operational_coverage": coverage,
        "image_reviewed_coverage": reviewed_coverage,
        "image_reviewed_coverage_ci95_lower": reviewed_coverage_ci[0],
        "image_reviewed_coverage_ci95_upper": reviewed_coverage_ci[1],
        "image_system_fnr": image_fnr,
        "image_system_fnr_failures": len(false_negatives),
        "image_fnr_denominator_positive_images": len(positive),
        "image_system_fnr_upper_bound": clopper_pearson_upper(
            len(false_negatives), len(positive), confidence_delta
        ),
        "image_system_fnr_exact_ci_lower": system_fnr_exact_ci[0],
        "image_system_fnr_exact_ci_upper": system_fnr_exact_ci[1],
        "image_accepted_risk": image_risk,
        "image_accepted_risk_ci95_lower": accepted_risk_ci[0],
        "image_accepted_risk_ci95_upper": accepted_risk_ci[1],
        "image_system_sensitivity": sensitivity,
        "image_system_sensitivity_ci95_lower": sensitivity_ci[0],
        "image_system_sensitivity_ci95_upper": sensitivity_ci[1],
        "image_system_specificity": specificity,
        "image_system_specificity_ci95_lower": specificity_ci[0],
        "image_system_specificity_ci95_upper": specificity_ci[1],
        "image_system_accuracy": system_accuracy,
        "image_system_accuracy_ci95_lower": system_accuracy_ci[0],
        "image_system_accuracy_ci95_upper": system_accuracy_ci[1],
        "image_accepted_accuracy": accepted_accuracy,
        "image_accepted_accuracy_ci95_lower": accepted_accuracy_ci[0],
        "image_accepted_accuracy_ci95_upper": accepted_accuracy_ci[1],
        "image_accepted_subset_fnr": accepted_subset_fnr,
        "image_positive_predictive_value": positive_predictive_value,
        "image_negative_predictive_value": negative_predictive_value,
        "image_auroc": _rank_auc(reviewed),
        "image_false_positive_rate": _ratio(len(false_positives), len(negative)),
        "image_feedback_count": len(ratings),
        "image_effective_count": ratings.count("effective"),
        "image_effective_rate": _ratio(ratings.count("effective"), len(ratings)),
        "image_deferred_reviewed_count": len(deferred_reviewed),
        "image_deferred_rate": deferred_rate,
        "image_deferred_doctor_positive": len(deferred_positive),
        "image_deferred_doctor_negative": len(deferred_negative),
        "image_positive_deferred_rate": positive_deferred_rate,
        "image_negative_deferred_rate": negative_deferred_rate,
        "diagnosis_changed_session_count": sum(
            bool(images[0].get("diagnosis_changed_after_ai"))
            for images in changed_sessions
        ),
        "diagnosis_changed_recorded_sessions": len(changed_sessions),
        "diagnosis_changed_rate": _ratio(
            sum(
                bool(images[0].get("diagnosis_changed_after_ai"))
                for images in changed_sessions
            ),
            len(changed_sessions),
        ),
        "physician_summaries": physician_summaries,
        "median_inference_ms": median(inference_times) if inference_times else None,
        "median_reviewed_inference_ms": (
            median(reviewed_inference_times) if reviewed_inference_times else None
        ),
        "median_independent_review_ms": (
            median(diagnosis_times) if diagnosis_times else None
        ),
        "median_feedback_ms": median(feedback_times) if feedback_times else None,
        "metric_unit": "image",
        "primary_metric_unit": "image",
        "aggregation_rule": "none_image_primary",
        "system_fnr": image_fnr,
        "system_fnr_upper_bound": clopper_pearson_upper(
            len(false_negatives), len(positive), confidence_delta
        ),
        "operational_coverage": coverage,
        "reviewed_coverage": reviewed_coverage,
        "accepted_risk": image_risk,
        "all_cases": len(evaluation_images),
        "reviewed_cases": len(reviewed),
        "pending_review_cases": len(evaluation_images) - len(reviewed),
        "interpretation": (
            "All operational and error metrics use pathology images as the analysis "
            "unit. Each uploaded image receives its own result; a batch is "
            "only an upload and review session and does not produce a patient-level "
            "diagnosis. Patient-information uploads are excluded from performance "
            "evaluation because they do not provide a pathologist label recorded before result disclosure."
        ),
    }


def paper_summary_rows(
    summary: dict[str, Any],
    *,
    generated_at_utc: str,
    app_version: str,
    model_display_name: str,
    model_version: str,
    policy_version: str,
    pilot_phase: str,
    target_system_fnr: float,
    confidence_delta: float,
    certified_prelaunch: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return a tidy, paper-oriented metric table with explicit denominators."""
    fields = [
        "generated_at_utc",
        "app_version",
        "model_display_name",
        "model_version",
        "policy_version",
        "pilot_phase",
        "analysis_unit",
        "analysis_population",
        "target_system_fnr",
        "confidence_level",
        "certified_prelaunch",
        "metric_group",
        "metric_name",
        "value",
        "numerator",
        "denominator",
        "ci_method",
        "ci_lower",
        "ci_upper",
        "display_value",
        "definition",
    ]
    reviewed = int(summary["reviewed_images"])
    positives = int(summary["reviewed_positive_images"])
    negatives = int(summary["reviewed_negative_images"])
    accepted = int(summary["image_accepted_reviewed_count"])
    deferred = int(summary["image_deferred_reviewed_count"])
    tp = int(summary["image_true_positive_count"])
    tn = int(summary["image_true_negative_count"])
    fp = int(summary["image_false_positive_count"])
    fn = int(summary["image_false_negative_count"])
    correct = int(summary["image_correct_accepted_count"])

    def add(
        group: str,
        name: str,
        value: Any,
        numerator: int | None,
        denominator: int | None,
        definition: str,
        *,
        ci_method: str = "",
        ci_lower: Any = None,
        ci_upper: Any = None,
        display: str | None = None,
    ) -> dict[str, Any]:
        if display is None:
            if value is None:
                display = "NA"
            elif denominator is None:
                display = str(value)
            else:
                display = f"{100.0 * float(value):.2f}%"
        return {
            "generated_at_utc": generated_at_utc,
            "app_version": app_version,
            "model_display_name": model_display_name,
            "model_version": model_version,
            "policy_version": policy_version,
            "pilot_phase": pilot_phase,
            "analysis_unit": "image",
            "analysis_population": (
                "pathologist self-review images with a recorded pre-AI image label"
            ),
            "target_system_fnr": target_system_fnr,
            "confidence_level": 1.0 - confidence_delta,
            "certified_prelaunch": int(bool(certified_prelaunch)),
            "metric_group": group,
            "metric_name": name,
            "value": value,
            "numerator": numerator,
            "denominator": denominator,
            "ci_method": ci_method,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "display_value": display,
            "definition": definition,
        }

    rows = [
        add(
            "cohort",
            "uploaded_evaluation_images",
            int(summary["all_images"]),
            None,
            None,
            "Uploaded pathologist self-review images after excluding legacy engineering records and patient-information sessions.",
        ),
        add(
            "cohort", "evaluated_images", reviewed, None, None,
            "Images with a pathologist image-level label recorded before AI disclosure.",
        ),
        add(
            "cohort",
            "pending_reference_label_images",
            int(summary["pending_review_images"]),
            None,
            None,
            "Uploaded evaluation images excluded from performance metrics because no pre-AI image-level reference label was recorded.",
        ),
        add(
            "cohort",
            "patient_information_images_excluded",
            int(summary["patient_information_images"]),
            None,
            None,
            "Patient-information uploads excluded from model-performance evaluation.",
        ),
        add(
            "cohort",
            "legacy_engineering_images_excluded",
            int(summary["legacy_images_excluded"]),
            None,
            None,
            "Legacy engineering or training-era records excluded from formal evaluation.",
        ),
        add(
            "cohort", "reference_positive_images", positives, None, None,
            "Evaluated images labelled MIP-associated morphology present.",
        ),
        add(
            "cohort", "reference_negative_images", negatives, None, None,
            "Evaluated images labelled MIP-associated morphology absent.",
        ),
        add(
            "selective_system", "accepted_images", accepted, None, None,
            "Evaluated images receiving a binary model result.",
        ),
        add(
            "selective_system", "deferred_images", deferred, None, None,
            "Evaluated images for which the model withheld a binary result.",
        ),
        add(
            "selective_system",
            "reviewed_image_coverage",
            summary["image_reviewed_coverage"],
            accepted,
            reviewed,
            "Accepted evaluated images divided by all evaluated images.",
            ci_method="Wilson two-sided 95%",
            ci_lower=summary["image_reviewed_coverage_ci95_lower"],
            ci_upper=summary["image_reviewed_coverage_ci95_upper"],
        ),
        add(
            "selective_system",
            "deferred_image_rate",
            summary["image_deferred_rate"],
            deferred,
            reviewed,
            "Deferred evaluated images divided by all evaluated images.",
        ),
        add(
            "confusion_matrix", "true_positives", tp, None, None,
            "Accepted model-positive images with a positive reference label.",
        ),
        add(
            "confusion_matrix", "true_negatives", tn, None, None,
            "Accepted model-negative images with a negative reference label.",
        ),
        add(
            "confusion_matrix", "false_positives", fp, None, None,
            "Accepted model-positive images with a negative reference label.",
        ),
        add(
            "confusion_matrix", "false_negatives", fn, None, None,
            "Accepted model-negative images with a positive reference label.",
        ),
        add(
            "performance",
            "system_fnr",
            summary["image_system_fnr"],
            fn,
            positives,
            "Accepted false negatives divided by all reference-positive evaluated images; deferrals are not counted as misses but remain in the denominator.",
            ci_method=f"Clopper-Pearson one-sided {100.0 * (1.0 - confidence_delta):.0f}% upper",
            ci_upper=summary["image_system_fnr_upper_bound"],
        ),
        add(
            "performance",
            "system_sensitivity",
            summary["image_system_sensitivity"],
            tp,
            positives,
            "Accepted true positives divided by all reference-positive evaluated images.",
            ci_method="Wilson two-sided 95%",
            ci_lower=summary["image_system_sensitivity_ci95_lower"],
            ci_upper=summary["image_system_sensitivity_ci95_upper"],
        ),
        add(
            "performance",
            "system_specificity",
            summary["image_system_specificity"],
            tn,
            negatives,
            "Accepted true negatives divided by all reference-negative evaluated images.",
            ci_method="Wilson two-sided 95%",
            ci_lower=summary["image_system_specificity_ci95_lower"],
            ci_upper=summary["image_system_specificity_ci95_upper"],
        ),
        add(
            "performance",
            "system_accuracy",
            summary["image_system_accuracy"],
            correct,
            reviewed,
            "Correct accepted binary results divided by all evaluated images.",
            ci_method="Wilson two-sided 95%",
            ci_lower=summary["image_system_accuracy_ci95_lower"],
            ci_upper=summary["image_system_accuracy_ci95_upper"],
        ),
        add(
            "performance",
            "accepted_image_accuracy",
            summary["image_accepted_accuracy"],
            correct,
            accepted,
            "Correct binary results divided by accepted evaluated images.",
            ci_method="Wilson two-sided 95%",
            ci_lower=summary["image_accepted_accuracy_ci95_lower"],
            ci_upper=summary["image_accepted_accuracy_ci95_upper"],
        ),
        add(
            "performance",
            "accepted_image_risk",
            summary["image_accepted_risk"],
            fp + fn,
            accepted,
            "Incorrect binary results divided by accepted evaluated images.",
        ),
        add(
            "performance",
            "accepted_subset_fnr",
            summary["image_accepted_subset_fnr"],
            fn,
            tp + fn,
            "False negatives divided by accepted reference-positive images.",
        ),
        add(
            "performance",
            "false_positive_rate",
            summary["image_false_positive_rate"],
            fp,
            negatives,
            "Accepted false positives divided by all reference-negative evaluated images.",
        ),
        add(
            "performance",
            "positive_predictive_value",
            summary["image_positive_predictive_value"],
            tp,
            tp + fp,
            "True positives divided by all accepted model-positive images.",
        ),
        add(
            "performance",
            "negative_predictive_value",
            summary["image_negative_predictive_value"],
            tn,
            tn + fn,
            "True negatives divided by all accepted model-negative images.",
        ),
        add(
            "discrimination",
            "auroc_all_evaluated_images",
            summary["image_auroc"],
            None,
            None,
            "Tie-aware AUROC calculated from p_mip across all evaluated images, irrespective of deferral.",
            display=(
                "NA"
                if summary["image_auroc"] is None
                else f"{float(summary['image_auroc']):.4f}"
            ),
        ),
        add(
            "deferral",
            "positive_image_deferred_rate",
            summary["image_positive_deferred_rate"],
            summary["image_deferred_doctor_positive"],
            positives,
            "Deferred reference-positive images divided by all reference-positive evaluated images.",
        ),
        add(
            "deferral",
            "negative_image_deferred_rate",
            summary["image_negative_deferred_rate"],
            summary["image_deferred_doctor_negative"],
            negatives,
            "Deferred reference-negative images divided by all reference-negative evaluated images.",
        ),
        add(
            "workflow",
            "effective_result_rate",
            summary["image_effective_rate"],
            int(summary["image_effective_count"]),
            int(summary["image_feedback_count"]),
            "Objectively effective AI results divided by images with recorded post-AI feedback.",
        ),
        add(
            "workflow",
            "diagnosis_changed_session_rate",
            summary["diagnosis_changed_rate"],
            int(summary["diagnosis_changed_session_count"]),
            int(summary["diagnosis_changed_recorded_sessions"]),
            "Completed review sessions in which any initial image assessment changed after AI disclosure.",
        ),
        add(
            "workflow",
            "median_inference_ms",
            summary["median_reviewed_inference_ms"],
            None,
            None,
            "Median model inference time per evaluated image in milliseconds.",
            display=(
                "NA"
                if summary["median_reviewed_inference_ms"] is None
                else f"{float(summary['median_reviewed_inference_ms']):.1f} ms"
            ),
        ),
    ]
    return fields, rows


FORMAL_EVIDENCE_ROLES = {
    "controlled_post_launch_pilot",
    "held_out_validation",
}


def publication_validation_rows(
    cases: list[dict[str, Any]],
    *,
    generated_at_utc: str,
    app_version: str,
    model_display_name: str,
    pilot_phase: str,
    target_system_fnr: float,
    confidence_delta: float,
    certified_prelaunch: bool,
    current_prediction_threshold: float,
    current_acceptance_threshold: float,
    certification: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Create one paper-ready wide result row per evaluation cohort."""
    fields = [
        "generated_at_utc",
        "app_version",
        "model_display_name",
        "analysis_scope",
        "evidence_role",
        "evaluation_cohort",
        "analysis_unit",
        "pilot_phase",
        "model_version",
        "policy_version",
        "prediction_threshold",
        "acceptance_threshold",
        "target_system_fnr",
        "confidence_level",
        "certified_prelaunch",
        "threshold_derivation",
        "fold_taus",
        "candidate_images_total",
        "candidate_pre_ai_ambiguous_images_total",
        "training_overlap_images_excluded",
        "pre_ai_ambiguous_images_excluded",
        "uploaded_images",
        "evaluated_images",
        "pending_reference_labels",
        "pending_reference_adjudication_images",
        "reference_label_completion_rate",
        "positive_images",
        "negative_images",
        "accepted_images",
        "deferred_images",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "deferred_positive_images",
        "deferred_negative_images",
        "coverage",
        "coverage_ci95_lower",
        "coverage_ci95_upper",
        "test_risk",
        "test_risk_ci95_lower",
        "test_risk_ci95_upper",
        "system_fnr",
        "system_fnr_exact_ci95_lower",
        "system_fnr_exact_ci95_upper",
        "system_fnr_exact_one_sided_upper_95",
        "observed_fnr_at_or_below_target",
        "fnr_upper_bound_at_or_below_target",
        "positive_images_required_for_zero_fn_certification",
        "positive_image_shortfall_if_zero_fn",
        "system_sensitivity",
        "system_sensitivity_ci95_lower",
        "system_sensitivity_ci95_upper",
        "system_specificity",
        "system_specificity_ci95_lower",
        "system_specificity_ci95_upper",
        "system_accuracy",
        "system_accuracy_ci95_lower",
        "system_accuracy_ci95_upper",
        "accepted_image_accuracy",
        "accepted_subset_fnr",
        "false_positive_rate",
        "positive_predictive_value",
        "negative_predictive_value",
        "auroc_all_evaluated_images",
        "positive_image_deferred_rate",
        "negative_image_deferred_rate",
        "review_sessions",
        "feedback_completed_sessions",
        "feedback_completion_rate",
        "effective_result_rate",
        "diagnosis_changed_recorded_sessions",
        "diagnosis_changed_session_count",
        "diagnosis_changed_session_rate",
        "median_inference_ms",
        "median_initial_review_seconds",
        "median_feedback_seconds",
        "mixed_model_version_flag",
        "mixed_policy_version_flag",
        "formal_analysis_ready",
        "recommended_for_main_paper_table",
        "ads_post_launch_evidence_ready",
        "analysis_note",
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        role = str(case.get("evidence_role") or "legacy_unspecified")
        if (
            case.get("use_context") != "pathologist_self_review"
            or case.get("feedback_status") == "legacy_not_collected"
            or role not in FORMAL_EVIDENCE_ROLES
        ):
            continue
        cohort = str(case.get("evaluation_cohort") or "UNSPECIFIED")
        grouped[(role, cohort)].append(case)

    required_positive_images = required_positives_for_zero_failures(
        target_system_fnr, confidence_delta
    )
    threshold_derivation = str(
        certification.get("deployment_rule")
        or certification.get("method")
        or "not_recorded"
    )
    fold_taus = certification.get("fold_taus")
    rows: list[dict[str, Any]] = []

    def distinct(group: list[dict[str, Any]], key: str) -> list[Any]:
        values = {
            item.get(key)
            for item in group
            if item.get(key) not in (None, "")
        }
        return sorted(values, key=str)

    scoped_groups: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = {}
    scope_context: dict[tuple[str, str, str], dict[str, int]] = {}
    for (role, cohort), candidate_group in grouped.items():
        independent_group = [
            case for case in candidate_group
            if not bool(case.get("training_overlap"))
        ]
        independent_ambiguous = [
            case for case in independent_group
            if bool(case.get("pre_ai_ambiguity_flag"))
        ]
        primary_key = (role, cohort, "independence_audited_primary")
        scoped_groups[primary_key] = independent_group
        scope_context[primary_key] = {
            "candidate_images_total": len(candidate_group),
            "candidate_pre_ai_ambiguous_images_total": sum(
                bool(case.get("pre_ai_ambiguity_flag"))
                for case in candidate_group
            ),
            "training_overlap_images_excluded": (
                len(candidate_group) - len(independent_group)
            ),
            "pre_ai_ambiguous_images_excluded": 0,
        }
        if independent_ambiguous:
            sensitivity_key = (
                role,
                cohort,
                "predefined_high_agreement_sensitivity",
            )
            scoped_groups[sensitivity_key] = [
                case for case in independent_group
                if not bool(case.get("pre_ai_ambiguity_flag"))
            ]
            scope_context[sensitivity_key] = {
                "candidate_images_total": len(candidate_group),
                "candidate_pre_ai_ambiguous_images_total": sum(
                    bool(case.get("pre_ai_ambiguity_flag"))
                    for case in candidate_group
                ),
                "training_overlap_images_excluded": (
                    len(candidate_group) - len(independent_group)
                ),
                "pre_ai_ambiguous_images_excluded": len(
                    independent_ambiguous
                ),
            }

    for (role, cohort, analysis_scope), group in sorted(
        scoped_groups.items()
    ):
        context = scope_context[(role, cohort, analysis_scope)]
        summary = summarize_postlaunch(group, confidence_delta)
        reviewed = int(summary["reviewed_images"])
        uploaded = int(summary["all_images"])
        positives = int(summary["reviewed_positive_images"])
        model_versions = distinct(group, "model_version")
        policy_versions = distinct(group, "policy_version")
        prediction_thresholds = distinct(group, "prediction_threshold")
        acceptance_thresholds = distinct(group, "acceptance_threshold")
        prediction_threshold = (
            prediction_thresholds[0]
            if len(prediction_thresholds) == 1
            else current_prediction_threshold
        )
        acceptance_threshold = (
            acceptance_thresholds[0]
            if len(acceptance_thresholds) == 1
            else current_acceptance_threshold
        )
        feedback_sessions = int(summary["feedback_completed_sessions"])
        review_sessions = int(summary["all_review_sessions"])
        pending_adjudication = sum(
            str(case.get("reference_adjudication_status") or "")
            == "pending"
            for case in group
        )
        is_primary = analysis_scope == "independence_audited_primary"
        formal_ready = (
            reviewed > 0
            and positives > 0
            and int(summary["reviewed_negative_images"]) > 0
            and int(summary["pending_review_images"]) == 0
            and pending_adjudication == 0
            and len(model_versions) == 1
            and len(policy_versions) == 1
        )
        ads_ready = (
            role == "controlled_post_launch_pilot"
            and is_primary
            and formal_ready
            and feedback_sessions > 0
        )
        fnr = summary["image_system_fnr"]
        fnr_upper = summary["image_system_fnr_upper_bound"]
        rows.append(
            {
                "generated_at_utc": generated_at_utc,
                "app_version": app_version,
                "model_display_name": model_display_name,
                "analysis_scope": analysis_scope,
                "evidence_role": role,
                "evaluation_cohort": cohort,
                "analysis_unit": "pathology_image",
                "pilot_phase": pilot_phase,
                "model_version": (
                    model_versions[0] if len(model_versions) == 1 else "MIXED"
                ),
                "policy_version": (
                    policy_versions[0] if len(policy_versions) == 1 else "MIXED"
                ),
                "prediction_threshold": prediction_threshold,
                "acceptance_threshold": acceptance_threshold,
                "target_system_fnr": target_system_fnr,
                "confidence_level": 1.0 - confidence_delta,
                "certified_prelaunch": int(bool(certified_prelaunch)),
                "threshold_derivation": threshold_derivation,
                "fold_taus": (
                    json.dumps(fold_taus, ensure_ascii=False)
                    if fold_taus is not None
                    else ""
                ),
                "candidate_images_total": context[
                    "candidate_images_total"
                ],
                "candidate_pre_ai_ambiguous_images_total": context[
                    "candidate_pre_ai_ambiguous_images_total"
                ],
                "training_overlap_images_excluded": context[
                    "training_overlap_images_excluded"
                ],
                "pre_ai_ambiguous_images_excluded": context[
                    "pre_ai_ambiguous_images_excluded"
                ],
                "uploaded_images": uploaded,
                "evaluated_images": reviewed,
                "pending_reference_labels": int(
                    summary["pending_review_images"]
                ),
                "pending_reference_adjudication_images": (
                    pending_adjudication
                ),
                "reference_label_completion_rate": _ratio(reviewed, uploaded),
                "positive_images": positives,
                "negative_images": int(summary["reviewed_negative_images"]),
                "accepted_images": int(
                    summary["image_accepted_reviewed_count"]
                ),
                "deferred_images": int(
                    summary["image_deferred_reviewed_count"]
                ),
                "true_positives": int(summary["image_true_positive_count"]),
                "true_negatives": int(summary["image_true_negative_count"]),
                "false_positives": int(summary["image_false_positive_count"]),
                "false_negatives": int(summary["image_false_negative_count"]),
                "deferred_positive_images": int(
                    summary["image_deferred_doctor_positive"]
                ),
                "deferred_negative_images": int(
                    summary["image_deferred_doctor_negative"]
                ),
                "coverage": summary["image_reviewed_coverage"],
                "coverage_ci95_lower": summary[
                    "image_reviewed_coverage_ci95_lower"
                ],
                "coverage_ci95_upper": summary[
                    "image_reviewed_coverage_ci95_upper"
                ],
                "test_risk": summary["image_accepted_risk"],
                "test_risk_ci95_lower": summary[
                    "image_accepted_risk_ci95_lower"
                ],
                "test_risk_ci95_upper": summary[
                    "image_accepted_risk_ci95_upper"
                ],
                "system_fnr": fnr,
                "system_fnr_exact_ci95_lower": summary[
                    "image_system_fnr_exact_ci_lower"
                ],
                "system_fnr_exact_ci95_upper": summary[
                    "image_system_fnr_exact_ci_upper"
                ],
                "system_fnr_exact_one_sided_upper_95": fnr_upper,
                "observed_fnr_at_or_below_target": (
                    int(float(fnr) <= target_system_fnr)
                    if fnr is not None
                    else ""
                ),
                "fnr_upper_bound_at_or_below_target": (
                    int(float(fnr_upper) <= target_system_fnr)
                    if fnr_upper is not None
                    else ""
                ),
                "positive_images_required_for_zero_fn_certification": (
                    required_positive_images
                ),
                "positive_image_shortfall_if_zero_fn": max(
                    0, required_positive_images - positives
                ),
                "system_sensitivity": summary["image_system_sensitivity"],
                "system_sensitivity_ci95_lower": summary[
                    "image_system_sensitivity_ci95_lower"
                ],
                "system_sensitivity_ci95_upper": summary[
                    "image_system_sensitivity_ci95_upper"
                ],
                "system_specificity": summary["image_system_specificity"],
                "system_specificity_ci95_lower": summary[
                    "image_system_specificity_ci95_lower"
                ],
                "system_specificity_ci95_upper": summary[
                    "image_system_specificity_ci95_upper"
                ],
                "system_accuracy": summary["image_system_accuracy"],
                "system_accuracy_ci95_lower": summary[
                    "image_system_accuracy_ci95_lower"
                ],
                "system_accuracy_ci95_upper": summary[
                    "image_system_accuracy_ci95_upper"
                ],
                "accepted_image_accuracy": summary["image_accepted_accuracy"],
                "accepted_subset_fnr": summary["image_accepted_subset_fnr"],
                "false_positive_rate": summary["image_false_positive_rate"],
                "positive_predictive_value": summary[
                    "image_positive_predictive_value"
                ],
                "negative_predictive_value": summary[
                    "image_negative_predictive_value"
                ],
                "auroc_all_evaluated_images": summary["image_auroc"],
                "positive_image_deferred_rate": summary[
                    "image_positive_deferred_rate"
                ],
                "negative_image_deferred_rate": summary[
                    "image_negative_deferred_rate"
                ],
                "review_sessions": review_sessions,
                "feedback_completed_sessions": feedback_sessions,
                "feedback_completion_rate": _ratio(
                    feedback_sessions, review_sessions
                ),
                "effective_result_rate": summary["image_effective_rate"],
                "diagnosis_changed_recorded_sessions": int(
                    summary["diagnosis_changed_recorded_sessions"]
                ),
                "diagnosis_changed_session_count": int(
                    summary["diagnosis_changed_session_count"]
                ),
                "diagnosis_changed_session_rate": summary[
                    "diagnosis_changed_rate"
                ],
                "median_inference_ms": summary["median_reviewed_inference_ms"],
                "median_initial_review_seconds": (
                    float(summary["median_independent_review_ms"]) / 1000.0
                    if summary["median_independent_review_ms"] is not None
                    else None
                ),
                "median_feedback_seconds": (
                    float(summary["median_feedback_ms"]) / 1000.0
                    if summary["median_feedback_ms"] is not None
                    else None
                ),
                "mixed_model_version_flag": int(len(model_versions) != 1),
                "mixed_policy_version_flag": int(len(policy_versions) != 1),
                "formal_analysis_ready": int(formal_ready),
                "recommended_for_main_paper_table": int(
                    is_primary and formal_ready
                ),
                "ads_post_launch_evidence_ready": int(ads_ready),
                "analysis_note": (
                    (
                        "Primary independence-audited analysis: exact "
                        f"development-set duplicates excluded ({context['training_overlap_images_excluded']}). "
                        f"{pending_adjudication} pre-AI ambiguous independent "
                        "images still require blinded adjudication before this "
                        "row is final."
                        if is_primary and pending_adjudication
                        else "Primary independence-audited analysis: exact "
                        f"development-set duplicates excluded ({context['training_overlap_images_excluded']})."
                        if is_primary
                        else "Predefined high-agreement sensitivity analysis: "
                        f"{context['pre_ai_ambiguous_images_excluded']} pre-AI "
                        "ambiguous independent images excluded in addition to "
                        "development-set overlaps; do not substitute this row "
                        "for the primary analysis."
                    )
                    + (
                        " Controlled post-launch evidence may be reported only "
                        "with completed expert-workflow outcomes."
                        if role == "controlled_post_launch_pilot"
                        else " Held-out validation only: do not describe this "
                        "row as post-launch ADS evidence."
                    )
                ),
            }
        )
    return fields, rows
