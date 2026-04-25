.PHONY: test lint format-check smoke clean

PYTHON ?= python

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

smoke:
	$(PYTHON) -m mujoco_continuous_control.train --config configs/smoke_test.yaml

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info outputs runs videos checkpoints
