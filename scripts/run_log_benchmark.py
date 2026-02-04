from __future__ import annotations

from src.model.parameters import Params
from src.model.log_analytic import solve_log
from src.model.economy import simulate_output
from src.solvers.analytic_log import solve_log_benchmark


def main() -> None:
    params = Params()
    t, Y, extras = simulate_output(params)
    sol = solve_log_benchmark(params)

    print("=== LOG BENCHMARK ===")
    print("Params:", params)
    print("Simulated Y:", Y[:5], "...", Y[-5:])
    print("Extras:", extras)
    print("Analytic objects (log closed-form):", sol)

    # Basic sanity checks (once formulas are real)
    # assert sol["phi"] > 0, "Wealth-to-consumption ratio should be positive"


if __name__ == "__main__":
    main()
