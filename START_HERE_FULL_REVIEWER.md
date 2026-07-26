# HistoNexa-MIP full reviewer package

This package runs the complete local HistoNexa-MIP website with the five frozen
Cost-5 checkpoints and the frozen Case 4 FNR-5% deployment policy. It is a
password-free, localhost-only research artifact for reviewer inspection.

## What this package contains

- the complete English image-review website and FastAPI application;
- five frozen Cost-5 model checkpoints;
- the hash-bound deployment policy;
- password-free launchers for Windows, macOS, and Linux;
- checksum and release-manifest files.

It contains no pathology image, hospital or pilot database, physician record,
credential, access key, or historical training/development image. Reviewers
must use only local images that they are authorized to process.

## Important interpretation

The package performs real five-model inference. It does not use placeholder
scores. Any locally supplied image remains outside the paper's independent
pilot cohort and must not be reported as paper validation data.

The packaged acceptance threshold is the advisor-prespecified arithmetic mean
of five fold-specific Case 4 FNR-5% thresholds. It is an empirical operating
point and is not an independently certified guarantee that deployed FNR is at
most 5%.

## Requirements

- 64-bit Python 3.10-3.13; Python 3.11 is recommended.
- At least 8 GB RAM; 16 GB is recommended.
- Approximately 6 GB free disk space for the package, Python environment, and
  local records.
- Internet access during the one-time dependency installation.

After setup, inference runs locally and can be used with networking disabled.
CPU inference is supported and may take tens of seconds per image depending on
image dimensions and computer performance.

## Windows

1. Extract the complete ZIP file before running anything.
2. Double-click `setup_full_reviewer_windows.cmd`.
3. Double-click `start_full_reviewer_windows.cmd`.
4. The browser opens at <http://127.0.0.1:8000/>.

The setup launcher selects a supported Python installation and verifies all
five checkpoints and the policy before installing dependencies.

Keep the terminal window open while using the website. Press `Ctrl+C` in that
window to stop the service.

## macOS

1. Extract the complete ZIP file.
2. Double-click `setup_full_reviewer_macos.command`.
3. Double-click `start_full_reviewer_macos.command`.
4. The browser opens at <http://127.0.0.1:8000/>.

The launcher ignores an unsupported system Python when a supported
`python3.10`, `python3.11`, `python3.12`, or `python3.13` is installed.

If macOS blocks a launcher downloaded from the internet, right-click it,
choose **Open**, and confirm once. Keep the Terminal window open while using
the website.

## Linux

```bash
chmod +x setup_full_reviewer_linux.sh start_full_reviewer_linux.sh
./setup_full_reviewer_linux.sh
./start_full_reviewer_linux.sh
```

Then open <http://127.0.0.1:8000/>.

All platform setup launchers run `verify_full_reviewer_integrity.py` before
creating the local environment. A failed integrity check means the ZIP was
incompletely downloaded, modified, or incompletely extracted; download and
extract it again rather than continuing.

## Trying the interface

No pathology image is bundled. Use only a permission-cleared local image. For
immediate per-image educational output, select **Patient image information**.
To inspect the blinded pathologist workflow, select **Pathologist self-review**,
enter a pseudonymous reviewer ID, record the initial image assessments, and
then reveal and compare the model results.

## Local data behavior

- Uploaded image bytes are processed locally and are not retained by the
  supplied full-reviewer launchers.
- Pseudonymous workflow records and exports are stored only in
  `runtime/full_reviewer/pilot.db`.
- Delete `runtime/full_reviewer/` to reset the reviewer artifact after the
  application has been stopped.
- The service listens only on `127.0.0.1`; other computers cannot access it.
- No password is required because the service is not exposed to a LAN or the
  public internet.

## Integrity

Every startup verifies:

1. all five model files are present;
2. their SHA-256 hashes match the deployment policy;
3. the recorded acceptance threshold equals the arithmetic mean of the five
   fold-specific thresholds;
4. the policy remains explicitly marked `certified=false`; and
5. the local data directory and existing database are usable.

The package-level file hashes are recorded in `SHA256SUMS.txt`. The ZIP archive
is accompanied by its own `.sha256` file on the GitHub Release.

## Troubleshooting

- **Port already in use:** stop the other local service using port 8000 and
  restart this package.
- **Missing checkpoint:** re-extract the complete Release ZIP; do not download
  only the GitHub source-code archive.
- **Hash mismatch:** delete the affected package and download the Release ZIP
  again.
- **Slow inference:** use fewer images per session or run on a faster CPU.
- **Blank page or Failed to fetch:** keep the launcher terminal open and use
  `http://127.0.0.1:8000/`, not a local `file://` copy of `index.html`.

## Research-use boundary

This software is a research artifact, not a medical device and not a
standalone diagnostic system. Final interpretation remains the responsibility
of qualified clinicians. Read `RESEARCH_USE_NOTICE.md` and
`THIRD_PARTY_MODEL_NOTICE.md` before use. The included fine-tuned checkpoints
incorporate Phikon and are limited to non-commercial research use by non-profit
entities under the Owkin license.
