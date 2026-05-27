cd /tmp/ml-equilibrium-olg
cat > README.md << 'EOF'
# ML Equilibrium OLG

Master's thesis (UZH/ETH, 2025–2026): physics-informed neural network
(PINN) methods for general-equilibrium asset pricing with heterogeneous
agents, based on the Blanchard–Yaari overlapping-generations framework.

## Notebooks

### Production
- **`hjb_1d_hardBC_het_structuralBaseline.ipynb`** — Heterogeneous
  two-agent solver. Uses a structural-baseline ansatz with multiplicative
  lifting envelope to enforce six hard boundary conditions exactly, and
  an endogenous β closure derived from the per-type per-capita newborn
  consumption identity. Validates by recovering the homogeneous log and
  CRRA(γ=2) closed-form constants in their respective limits.

### Methodology ablation
- **`hjb_2d_hardBC_LOG_minimal.ipynb`** — Homogeneous log-utility
  HJB-PINN trained with a minimal loss: HJB residual + monotonicity
  penalties (V_w > 0, V_y > 0) only. Tests whether CRRA-specific
  analytical penalties are strict correctness requirements or training
  accelerators.

- **`hjb_2d_hardBC_CRRA_minimal.ipynb`** — Same ablation for the
  homogeneous CRRA(γ=2) HJB-PINN.

## Methodology summary

All notebooks implement hard boundary conditions through a closed-form
trial-solution ansatz that satisfies the Merton boundary at y = Y_MIN
exactly, by construction. Networks are trained with a two-phase pipeline:
Phase 1 Adam (FP32) for warm-up, Phase 2 L-BFGS (FP64) for polish.
EOF

git add README.md
git commit -m "Update README to reflect kept notebooks"
git push origin main
