# ML Equilibrium OLG

**Physics-Informed Neural Networks for General-Equilibrium Asset Pricing in Heterogeneous-Agent Overlapping-Generations Economies**

Master's Thesis — *Cecilia Trojani*  
Master in Quantitative Finance, University of Zurich & ETH Zurich (2025–2026)  
Supervisor: Prof. Yucheng Yang


## Overview

This repository contains the code developed for my Master's thesis on
physics-informed neural network (PINN) methods for continuous-time
general-equilibrium asset-pricing models with heterogeneous agents in
Blanchard--Yaari overlapping-generations (OLG) economies.

The project combines asset-pricing theory, heterogeneous-agent
equilibrium analysis, and scientific machine learning. Its objective is
to develop neural-network-based solvers for computing equilibrium asset
prices, portfolio allocations, and valuation ratios in settings where
analytical solutions are no longer available.

The framework builds on the heterogeneous-agent OLG model of
Ehling, Graniero, and Heyerdahl-Larsen (2018), with heterogeneity in
risk aversion as the main application.



## Economic Motivation

Continuous-time asset-pricing models are typically analytically
tractable when all investors share identical preferences. In such
homogeneous economies, equilibrium asset prices and portfolio
allocations can often be characterized in closed form using
Merton-type arguments.

Introducing preference heterogeneity substantially enriches the
equilibrium structure. When agents differ in risk aversion,
equilibrium prices depend not only on aggregate risk but also on how
risk is distributed across investors. The resulting equilibrium is
typically characterized by a system of coupled nonlinear differential
equations whose solution determines risk premia, interest rates,
wealth distributions, and asset valuations.

These mechanisms play a central role in modern heterogeneous-agent
asset-pricing theory. In particular, the framework of
Gârleanu and Panageas (2015) illustrates how differences in risk
tolerance can generate state-dependent risk premia, excess return
volatility, and time-varying aggregate risk-bearing capacity.

Accurately solving such equilibria is therefore essential for the
quantitative analysis of heterogeneous-agent asset-pricing models.

## Main Contributions

The repository implements the methodology developed in the thesis for
solving continuous-time equilibrium asset-pricing models using
physics-informed neural networks.

A central feature of the approach is a hard-boundary PINN architecture
in which analytical equilibrium limits are embedded directly into the
trial solutions. Boundary conditions are therefore satisfied exactly by
construction rather than approximately through penalty terms, allowing
the networks to focus on the unknown interior equilibrium dynamics.

The methodology is developed and evaluated in two stages.

### 1. Validation in Homogeneous Economies

The framework is first applied to homogeneous economies with
logarithmic and CRRA preferences, where closed-form equilibrium
solutions are available. These benchmark environments provide a
controlled setting in which the PINN solutions can be compared
directly against analytical results, establishing the ability of the
methodology to recover equilibrium value functions, optimal policies,
and asset-pricing quantities.

### 2. Application to Heterogeneous-Agent Equilibria

The validated framework is then applied to a heterogeneous two-agent
OLG economy with different levels of risk aversion. In this setting,
equilibrium is characterized by a system of coupled differential
equations and no closed-form solution is available. The repository
demonstrates that the proposed methodology can recover economically
meaningful heterogeneous equilibria and reproduce the expected
state-dependent behavior of equilibrium asset prices, risk premia,
interest rates, and valuation ratios.

## Repository Structure

The methodology is developed and validated through three notebooks of
increasing economic and numerical complexity.

| Notebook | Description |
|-----------|-------------|
| `hjb_2d_hardBC_LOG_minimal.ipynb` | Homogeneous log-utility benchmark ($\gamma = 1$). Two-dimensional HJB formulation with a closed-form equilibrium solution. |
| `hjb_2d_hardBC_CRRA_minimal.ipynb` | Homogeneous CRRA benchmark ($\gamma = 2$). Two-dimensional HJB formulation with a closed-form equilibrium solution. |
| `hjb_1d_hardBC_heterogeneous.ipynb` | Heterogeneous two-agent economy ($\gamma^1,\gamma^2$). Stationary equilibrium ODE formulation with structural baselines and multiplicative neural corrections. Includes the Gârleanu–Panageas (2015) calibration and an interest-rate decomposition analysis. |

The homogeneous notebooks provide controlled benchmark environments in
which the neural approximations can be compared directly against
analytical equilibrium solutions. They serve both as validation cases
for the PINN methodology and as illustrations of the hard-boundary
architecture in settings where the exact solution is known.

The heterogeneous notebook implements the full equilibrium solver.
Because no closed-form solution is available in the interior of the
state space, validation relies on homogeneous-limit consistency checks.
Specifically, the heterogeneous architecture is evaluated under the
parameterizations

$$
\gamma^1=\gamma^2=1,
\qquad
\gamma^1=\gamma^2=2,
$$

for which the equilibrium must collapse to the corresponding
homogeneous benchmark throughout the entire state space. These tests
provide a stringent verification of both the architecture and the
equilibrium implementation before the solver is applied to genuinely
heterogeneous economies.

## Numerical Methodology

The numerical framework combines analytical structure with
physics-informed neural-network approximation. Its main features are:

- **Hard-boundary trial solutions**, which satisfy analytical boundary
  conditions exactly by construction rather than through penalty terms.

- **Structural analytical baselines**, which anchor the heterogeneous
  equilibrium to economically meaningful reference solutions and reduce
  the complexity of the learning problem.

- **Endogenous equilibrium closure**, whereby the newborn-consumption
  share is reconstructed from the learned valuation ratios and fed back
  into the equilibrium coefficients at each collocation point.

- **Physics-informed training**, in which the networks are trained by
  minimizing equilibrium residuals rather than fitting observed data.

- **Two-stage optimization**, combining Adam warm-up iterations with
  strong-Wolfe L-BFGS refinement in double precision.

Each notebook is fully self-contained and documents the equilibrium
formulation, network architecture, loss specification, training
procedure, and validation strategy associated with the corresponding
economic model.

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
