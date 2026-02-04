# Log benchmark map

Goal: implement the analytically tractable benchmark (log utility, controlled setting)
and use it as a validation target for ML solvers.

## Steps
1) Write down closed-form objects (r, phi, p_over_y, pi) from notes/Ehling.
2) Implement in src/solvers/analytic_log.py
3) Add sanity checks (signs, limiting cases).
4) Use as training target in scripts/train_log_policy.py

## TODO
- Paste exact formulas + equation references (Ehling eq. ... / notes eq. ...).
