# ML Equilibrium OLG

**Physics-Informed Neural Networks for General-Equilibrium Asset Pricing in Heterogeneous-Agent Overlapping-Generations Economies**

Master's Thesis — *Cecilia Trojani*  
Master in Quantitative Finance, University of Zurich & ETH Zurich (2025–2026)  
Supervisor: Prof. Yucheng Yang

---

## Overview

This repository contains the code accompanying my Master's thesis on
physics-informed neural network (PINN) methods for solving continuous-time
general-equilibrium asset-pricing models with heterogeneous agents in
Blanchard–Yaari overlapping-generations (OLG) economies.

The thesis combines asset-pricing theory, heterogeneous-agent
equilibrium analysis, and scientific machine learning. The objective is
to develop neural-network-based solvers capable of computing equilibrium
asset prices, portfolio allocations, and valuation ratios in economic
environments where analytical solutions are unavailable.

The framework builds on the heterogeneous-agent OLG model of
Ehling, Graniero, and Heyerdahl-Larsen (2018) and focuses on
heterogeneity in risk aversion as the primary application.

---

## Economic Motivation

Continuous-time asset-pricing models are often analytically tractable
when all investors share identical preferences. In such homogeneous
economies, equilibrium quantities can frequently be expressed in
closed form using Merton-type arguments.

Introducing preference heterogeneity fundamentally changes the problem.
When agents differ in risk aversion, equilibrium prices depend not only
on aggregate risk but also on how risk is distributed across investors.
The resulting equilibrium is characterized by a system of nonlinear
differential equations whose solution determines risk premia, interest
rates, wealth distributions, and asset valuations.

These mechanisms play a central role in modern heterogeneous-agent
asset-pricing models. In particular, the influential framework of
Gârleanu and Panageas (2015) shows how differences in risk tolerance
can generate state-dependent risk premia and excess return volatility.

Solving such equilibria accurately therefore becomes a prerequisite for
quantitative analysis.

---

## Main Contributions

The repository implements three methodological contributions developed
in the thesis.

### 1. Hard-Boundary PINN Architecture

The trial solutions incorporate analytical equilibrium limits directly
into the network architecture. Known boundary conditions are satisfied
exactly rather than approximately through penalty terms.

As a result, the neural network is responsible only for learning the
unknown heterogeneous interior of the equilibrium.

### 2. Economically Structured PINN Formulation

Economic restrictions are embedded directly into the learning problem.
Preference monotonicity is enforced explicitly, while numerical
stabilization terms are introduced only when they admit a clear
economic interpretation.

This separates economically meaningful structure from purely numerical
regularization.

### 3. Heterogeneous-Equilibrium Solver

A stationary ODE formulation is developed for the heterogeneous OLG
economy. The solver incorporates an endogenous newborn-consumption-share
closure and computes equilibrium valuation ratios, risk premia, and
interest rates directly from the equilibrium conditions.

The methodology is validated through homogeneous-limit consistency
checks built into the architecture itself.

---

## Repository Structure

The methodology is developed and validated through three notebooks of
increasing complexity.

| Notebook | Description |
|-----------|-------------|
| `hjb_2d_hardBC_LOG_minimal.ipynb` | Homogeneous log-utility benchmark ($begin:math:text$\\gamma \= 1$end:math:text$). Two-dimensional HJB formulation with analytical closed-form solution. |
| `hjb_2d_hardBC_CRRA_minimal.ipynb` | Homogeneous CRRA benchmark ($begin:math:text$\\gamma \= 2$end:math:text$). Two-dimensional HJB formulation with analytical closed-form solution. |
| `hjb_1d_hardBC_heterogeneous.ipynb` | Heterogeneous two-agent economy ($begin:math:text$\\gamma\^1\,\\gamma\^2$end:math:text$). Stationary equilibrium ODE formulation with structural baselines and multiplicative neural corrections. Includes the Gârleanu–Panageas (2015) calibration and interest-rate decomposition analysis. |

The homogeneous notebooks serve as controlled benchmark environments in
which the neural solutions can be compared directly against analytical
equilibria.

The heterogeneous notebook contains the full production solver and
implements its own internal validation strategy. Running the
heterogeneous architecture under homogeneous-limit parameterizations,

$begin:math:display$
\\gamma\^1\=\\gamma\^2\=1
\\qquad\\text\{and\}\\qquad
\\gamma\^1\=\\gamma\^2\=2\,
$end:math:display$

must recover the corresponding homogeneous equilibria throughout the
entire state space. These tests provide a stringent consistency check
before applying the solver to genuinely heterogeneous economies.

---

## Numerical Methodology

Key numerical features include:

- **Hard-boundary trial solutions** that satisfy analytical boundary
  conditions exactly.

- **Structural analytical baselines** that anchor the heterogeneous
  solution to economically meaningful reference equilibria.

- **Endogenous equilibrium closure**, where the newborn-consumption
  share is reconstructed from the learned valuation ratios at each
  collocation point.

- **Physics-informed training**, in which the neural networks minimize
  equilibrium residuals rather than fitting data.

- **Two-stage optimization**, combining Adam warm-up iterations with
  strong-Wolfe L-BFGS refinement in double precision.

Each notebook is self-contained and documents the architecture,
equilibrium equations, loss functions, training procedure, and
validation methodology used in that experiment.

---

## Reference

If you use this repository, please cite:

```bibtex
@mastersthesis{trojani2026olg,
  author  = {Trojani, Cecilia},
  title   = {Deep Learning for Solving Heterogeneous-Agent Overlapping-Generations Asset Pricing Models},
  school  = {University of Zurich and ETH Zurich},
  year    = {2026}
}
```
