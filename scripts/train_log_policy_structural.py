import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model.parameters import Params
from scripts.train_log_policy import main as train_log_policy_main


def main():
    params = Params(lambda_levels=0.5, lambda_rates=2.0)
    return train_log_policy_main(params)


if __name__ == "__main__":
    main()
