import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model.parameters import Params
from src.model.log_analytic import solve_log


def main():
    params = Params()
    sol = solve_log(params)

    print("Log benchmark solution:")
    print(f"beta     = {sol.beta:.6f}")
    print(f"r        = {sol.r:.6f}")
    print(f"theta    = {sol.theta:.6f}")
    print(f"phi      = {sol.phi:.6f}")
    print(f"p_over_y = {sol.p_over_y:.6f}")
    print(f"c_coeff  = {sol.c_coeff:.6f}")
    print(f"pi       = {sol.pi:.6f}")
    print(f"pi_coeff = {sol.pi_coeff:.6f}")
    print(f"mu_s     = {sol.mu_s:.6f}")
    print(f"sigma_s  = {sol.sigma_s:.6f}")


if __name__ == "__main__":
    main()
