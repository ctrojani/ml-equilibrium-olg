# ml-equilibrium-olg
Master’s thesis project on general equilibrium asset pricing with heterogeneous agents.  Focus on Blanchard–Yaari OLG models, wealth-to-consumption ratios, and neural-network-based solution methods.

# Master Thesis – General Equilibrium with Heterogeneous Agents

This repository contains the code and notes for my Master’s thesis in Quantitative Finance 
(University of Zurich / ETH Zurich).

## Project Overview

The thesis studies general equilibrium asset pricing in overlapping-generations economies 
with heterogeneous agents, building on the framework of Ehling et al. (2018).

The main objective is to:
- re-derive analytically tractable benchmark cases (log utility),
- validate neural-network-based numerical solvers against closed-form solutions,
- extend the framework to richer preference structures (e.g. CRRA utility) where analytical solutions are no longer available.

Neural networks are used **as numerical solution tools**, not as black-box predictors.

## Structure (work in progress)

- `src/` – core model and equilibrium definitions  
- `solvers/` – neural-network solvers for equilibrium objects  
- `experiments/` – numerical experiments and diagnostics  

## Status

Work in progress (Master’s thesis).
