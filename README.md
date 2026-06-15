# ML Equilibrium OLG

**Physics-informed neural network methods for general-equilibrium asset pricing in heterogeneous-agent Blanchard–Yaari OLG economies.**

Master's thesis — *Cecilia Trojani, UZH / ETH Zurich, 2025–2026.*

---

## Overview

This repository contains the numerical implementation accompanying the
thesis. The methodology applies physics-informed neural networks (PINNs)
to three asset-pricing equilibrium problems of increasing complexity:

1. **Homogeneous log-utility benchmark** ($\gamma = 1$) — closed-form
   reference.
2. **Homogeneous CRRA benchmark** ($\gamma = 2$) — closed-form
   reference.
3. **Heterogeneous two-agent economy** ($\gamma^1 = 1$, $\gamma^2 = 2$) —
   production result; no closed-form solution exists.

The homogeneous cases are solved through their HJB dynamic-programming
formulation; the heterogeneous case is solved directly from the
stationary martingale ODE system derived in the thesis, using a
structural-baseline architecture with multiplicative neural correction.

---

## Methodology highlights

- **Hard boundary conditions through architectural ansatz.** Trial
  solutions are constructed so that all relevant boundary conditions
  hold exactly for any network output, eliminating boundary-penalty
  terms.
  - *Homogeneous (HJB).* Log-of-affine ansatz (log utility) and
    power-of-affine ansatz (CRRA), each enforcing the
    Merton-with-mortality boundary at $y = 0$ by construction.
  - *Heterogeneous (ODE).* Multiplicative correction
    $\widehat\phi^i(f) = \phi^i_{\mathrm{base}}(f) \cdot \exp\bigl(f(1-f)\,N^i(f)\bigr)$,
    with a common $f(1-f)$ envelope that anchors each valuation ratio at
    both endpoints — the same-type homogeneous equilibrium at one end and
    the alien-type representative-agent W2C ratio at the other.

- **Minimal shape penalty.** Loss enforces only monotonicity
  $V_w > 0$ and $V_y > 0$ — no concavity, cross-derivative or higher-order
  constraints. Numerical guards on $V_w$ and $V_{ww}$ inside the FOC
  formulas are formula-level safeguards (analogous to
  `c.clamp(min=1e-8)` inside `log c`), not penalty terms in the loss.

- **Two-stage training.** Adam optimisation in FP32 for warm-up, then
  strong-Wolfe L-BFGS in FP64 for polishing on a residual-weighted
  frozen collocation grid.

- **Endogenous β closure (heterogeneous solver).** The newborn-consumption
  share is reconstructed from the learned valuation ratios at every
  collocation point and re-enters the ODE coefficients, turning the
  equilibrium into an implicit fixed-point problem resolved through
  residual minimisation.

- **Validation through homogeneous-limit tests.** The heterogeneous
  solver is run at $\gamma^1 = \gamma^2 = 1$ and $\gamma^1 = \gamma^2 = 2$
  and must recover the corresponding closed-form constants throughout
  the interior — the architecture is forced to do so exactly when
  $\gamma^1 = \gamma^2$.

---

## Notebooks

### Production solvers
| Notebook | Problem |
|---|---|
| `hjb_2d_hardBC_LOG_minimal.ipynb` | Homogeneous log-utility HJB-PINN ($\gamma = 1$). |
| `hjb_2d_hardBC_CRRA_minimal.ipynb` | Homogeneous CRRA HJB-PINN ($\gamma = 2$). |
| `hjb_1d_hardBC_het_structuralBaseline.ipynb` | Heterogeneous two-agent stationary ODE solver. Includes the Garleanu–Panageas (2015) calibration reproduction ($\gamma^1 = 1.5$, $\gamma^2 = 10$) and the interest-rate decomposition figures. |

### Methodology ablations
| Notebook | Purpose |
|---|---|
| `hjb_2d_hardBC_LOG_minimal_noclamps.ipynb` | Log-utility solver with the FOC-stability clamps removed. Empirical test of whether the $V_w$ and $V_{ww}$ clamps are correctness requirements or optimisation accelerators. |
| `hjb_2d_hardBC_CRRA_minimal_noclamps.ipynb` | Same ablation for CRRA($\gamma = 2$). |

The ablation runs converge to a stable but biased equilibrium with a
$\sim 3\%$ portfolio overshoot, providing empirical evidence that the
clamps act as economically-motivated FOC safeguards rather than mere
numerical heuristics.

---

## Running

```bash
git clone https://github.com/ctrojani/ml-equilibrium-olg.git
cd ml-equilibrium-olg
pip install -r requirements.txt
jupyter notebook
```

Tested on Python 3.10+ with PyTorch 2.x. Random seeds are fixed (42)
throughout; training is deterministic on CPU. On CUDA, small
reduction-order non-determinism may appear at the $\sim 10^{-9}$ level
without affecting the qualitative results.

Each notebook is self-contained: its title cell summarises the
architecture, training pipeline, loss components, and validation
strategy used in that specific run. All hyperparameters
($N_{\mathrm{epochs}}$, learning rates, collocation grid sizes,
L-BFGS settings, etc.) are documented in the configuration cell at the
top of the training phase.

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{trojani2026olg,
  author  = {Trojani, Cecilia},
  title   = {Physics-Informed Neural Networks for General-Equilibrium
             Asset Pricing in Heterogeneous-Agent OLG Economies},
  school  = {University of Zurich and ETH Zurich},
  year    = {2026}
}
```

---

## License

Released for academic use accompanying the thesis. See `LICENSE` for
details.
