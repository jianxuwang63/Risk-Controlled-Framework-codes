# Model files

The source repository does not track trained checkpoint binaries. The
v0.10.15 GitHub Release includes the exact five checkpoints inside the full
reviewer ZIP. The package includes no pathology image, hospital data, access
credential, physician record, or active pilot database.

The password-free reviewer demonstration runs with `APP_MODE=demo` and does not
require this directory. Its deterministic placeholder scores are provided only
to exercise the interface.

Real-model validation requires the exact five Cost=5 checkpoints and a
hash-bound FNR-5% deployment policy. A full reviewer Release must verify every
checkpoint hash before startup. The lightweight demo must not be used to
reproduce or audit paper performance claims.
