.PHONY: install test lint format-check smoke train-walker train-ant eval-walker eval-ant video-walker video-ant clean

PYTHON ?= python

install:
	pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

smoke:
	$(PYTHON) -m mujoco_continuous_control.train --config configs/smoke_test.yaml

train-walker:
	$(PYTHON) -m mujoco_continuous_control.train --config configs/walker2d.yaml

train-ant:
	$(PYTHON) -m mujoco_continuous_control.train --config configs/ant.yaml

eval-walker:
	$(PYTHON) -m mujoco_continuous_control.evaluate --checkpoint runs/Walker2d-v5/walker2d_seed1/checkpoints/best.pt --episodes 20

eval-ant:
	$(PYTHON) -m mujoco_continuous_control.evaluate --checkpoint runs/Ant-v5/ant_seed1/checkpoints/best.pt --episodes 20

video-walker:
	$(PYTHON) -m mujoco_continuous_control.record_video --checkpoint runs/Walker2d-v5/walker2d_seed1/checkpoints/best.pt --episodes 3 --output-dir assets/videos/walker2d

video-ant:
	$(PYTHON) -m mujoco_continuous_control.record_video --checkpoint runs/Ant-v5/ant_seed1/checkpoints/best.pt --episodes 3 --output-dir assets/videos/ant

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info outputs runs videos checkpoints
