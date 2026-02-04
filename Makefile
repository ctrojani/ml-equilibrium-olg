.PHONY: help log-bench smoke

help:
@echo "Targets:"
@echo "  make smoke      - run smoke test"
@echo "  make log-bench  - run log benchmark"

smoke:
python -m scripts.smoke_test

log-bench:
python -m scripts.run_log_benchmark
