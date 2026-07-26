# HistoNexa-MIP reviewer quick start

## Choose the correct artifact

- For complete real-model verification, download the full
  [v0.10.15 GitHub Release](https://github.com/jianxuwang63/Risk-Controlled-Framework-codes/releases/tag/v0.10.15)
  and follow [START_HERE_FULL_REVIEWER.md](START_HERE_FULL_REVIEWER.md). It
  contains the exact five Cost=5 checkpoints and frozen deployment policy.
- For interface-only inspection, use the lightweight source-repository demo
  described below. It uses deterministic placeholder scores and must not be
  used to reproduce model outputs.

This repository contains a local, password-free reviewer demonstration of the
deployed HistoNexa-MIP interface. It is intended to make the application
workflow, image-level reports, abstention presentation, audit fields, and CSV
exports inspectable without transmitting pathology images to an online service.

## Important scope

- `APP_MODE=demo` uses deterministic placeholder scores. It does **not** load
  the five trained checkpoints and must not be used to verify paper metrics.
- No pathology image is bundled. Use only a local image that you are authorized
  to process.
- The normal source checkout does not contain pathology images, pilot
  databases, access credentials, model checkpoints, or retained physician
  records. The five checkpoints are distributed only in the full v0.10.15
  Release package.
- The ADS deployment evidence is the controlled physician pilot and its
  post-launch measurements described in the paper. Repository availability is
  complementary reproducibility evidence, not the deployment itself.

## macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
./start_reviewer_demo_macos.command
```

Open <http://127.0.0.1:8000/>. No password is required.

## Windows

1. Double-click `setup_windows.cmd`.
2. Double-click `start_reviewer_demo_windows.cmd`.
3. Open <http://127.0.0.1:8000/>.

No password is required.

## Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
./start_reviewer_demo.sh
```

## Trying the interface

Upload a permission-cleared local image through the web interface. The
interface will produce a structured image-level report. Because the reviewer
package runs in demo mode, the displayed score and label are placeholders
derived from the file hash and do not represent the real model.

## Automated verification

```bash
python -m pytest -q
```

The tests cover API workflows, authentication boundaries, calibration helpers,
database metrics, backup integrity, and frontend contracts.
