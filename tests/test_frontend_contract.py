from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "clinical_app" / "static" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "clinical_app" / "static" / "index.html").read_text(encoding="utf-8")


def test_six_clinician_model_report_scenarios_are_present():
    scenarios = {
        "pathologist-positive-model-positive",
        "pathologist-negative-model-negative",
        "pathologist-negative-model-positive",
        "pathologist-positive-model-negative",
        "pathologist-positive-model-review",
        "pathologist-negative-model-review",
    }

    for scenario in scenarios:
        assert APP_JS.count(f'scenario: "{scenario}"') == 1


def test_report_uses_review_sections_and_excludes_unsafe_old_content():
    report_contract = APP_JS[
        APP_JS.index("function decisionContent")
        : APP_JS.index("function imagePreviewUrl")
    ]

    assert "finding:" in report_contract
    assert "reviewPlan:" in report_contract
    assert "limitations" in report_contract
    assert "predict prognosis" in report_contract
    assert "determine treatment" in report_contract

    banned_old_sections = (
        "SURVIVAL TIME",
        "LIFE EXPECTANCY",
        "SURGICAL CONSIDERATIONS",
        "RECOMMEND SURGERY",
    )
    upper_contract = report_contract.upper()
    for phrase in banned_old_sections:
        assert phrase not in upper_contract


def test_frontend_labels_are_image_level_and_versioned():
    combined = f"{INDEX_HTML}\n{APP_JS}"

    assert "v0.10.15" in INDEX_HTML
    assert "Image-specific" in INDEX_HTML
    assert "patient-level disease probabilities" in APP_JS
    assert "predict prognosis" in INDEX_HTML
    assert "prescribe surgery or treatment" in INDEX_HTML
    assert "Open ${Number(item.image_count)}-image session" in APP_JS
