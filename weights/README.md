# Model files

The source repository does not track trained checkpoint binaries. GitHub
Release builds may include the exact five checkpoints as downloadable binary
assets or inside the full reviewer ZIP. Neither distribution includes hospital
data, access credentials, or the active pilot database.

The password-free reviewer demonstration runs with `APP_MODE=demo` and does not
require this directory. Its deterministic placeholder scores are provided only
to exercise the interface.

Real-model validation requires the exact five Cost=5 checkpoints and a
hash-bound FNR-5% deployment policy. A full reviewer Release must verify every
checkpoint hash before startup. The lightweight demo must not be used to
reproduce or audit paper performance claims.
