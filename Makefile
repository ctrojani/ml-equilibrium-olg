.PHONY: help log-bench train-log test lint fmt

help:
@echo "Targets:"
@echo "  make log-bench   - run log benchmark"
@echo "  make train-log   - run training stub"
@echo "  make test        - run pytest"

log-bench:
python -m scripts.run_log_benchmark

train-log:
python -m scripts.train_log_policy

test:
pytest -q
