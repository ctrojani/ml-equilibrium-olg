from __future__ import annotations

from src.model.parameters import Params
from src.solvers.analytic_log import solve_log_benchmark


def main() -> None:
    params = Params()
    target = solve_log_benchmark(params)
    print("Training stub. Target objects:", target)
    print("Next: define NN inputs/outputs + loss to match benchmark.")


if __name__ == "__main__":
    main()
