import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_full_reviewer_launchers_use_real_local_validation_mode():
    launchers = {
        "macOS": _read("start_full_reviewer_macos.command"),
        "Windows": _read("start_full_reviewer_windows.cmd"),
        "Linux": _read("start_full_reviewer_linux.sh"),
    }
    for platform, script in launchers.items():
        assert "APP_MODE=validation" in script, platform
        assert "PILOT_PHASE=assisted" in script, platform
        assert "OFFLINE_ONLY=true" in script, platform
        assert "ALLOW_PRIVATE_LAN=false" in script, platform
        assert "PUBLIC_INTERNET_MODE=false" in script, platform
        assert "127.0.0.1" in script, platform
        assert "PERSIST_IMAGES=false" in script, platform
        assert "deployment_policy.cost5.validation.json" in script, platform
        assert "cost_5.0_fold_" in script, platform
        if platform == "Windows":
            assert "for /L %%G in (1,1,5)" in script
        else:
            assert "1 2 3 4 5" in script or "{1..5}" in script


def test_full_reviewer_launchers_do_not_embed_a_password():
    for name in [
        "start_full_reviewer_macos.command",
        "start_full_reviewer_windows.cmd",
        "start_full_reviewer_linux.sh",
    ]:
        script = _read(name)
        assert not re.search(r"APP_API_KEY=.*[A-Za-z0-9]{20,}", script)
        assert "PUBLIC_INTERNET_MODE=true" not in script


def test_full_reviewer_documentation_is_explicit_about_scope():
    guide = _read("START_HERE_FULL_REVIEWER.md")
    required_phrases = (
        "real five-model inference",
        "not an independently certified guarantee",
        "Uploaded image bytes",
        "127.0.0.1",
        "contains no pathology image",
    )
    for phrase in required_phrases:
        assert phrase in guide


def test_public_artifact_contains_no_pathology_examples():
    examples = ROOT / "clinical_app" / "static" / "examples"
    assert not examples.exists() or not any(path.is_file() for path in examples.rglob("*"))
    for name in [
        "README.md",
        "START_HERE_FULL_REVIEWER.md",
        "REVIEWER_QUICKSTART.md",
        "clinical_app/static/index.html",
    ]:
        text = _read(name)
        assert "reference-mip-associated.jpg" not in text
        assert "reference-comparator.jpg" not in text


def test_setup_launchers_verify_package_integrity_before_installing():
    for name in [
        "setup_full_reviewer_macos.command",
        "setup_full_reviewer_windows.cmd",
        "setup_full_reviewer_linux.sh",
    ]:
        script = _read(name)
        assert "verify_full_reviewer_integrity.py" in script
        assert "(3, 10)" in script
        assert "(3, 14)" in script

    windows = _read("setup_full_reviewer_windows.cmd")
    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert f"py -{version}" in windows
