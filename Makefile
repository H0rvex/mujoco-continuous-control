.PHONY: install test lint format-check smoke train-walker train-ant eval-walker eval-ant video-walker video-ant docker-build docker-test docker-smoke docker-shell clean

PYTHON ?= python
DOCKER ?= docker
DOCKER_IMAGE ?= mujoco-continuous-control:dev
DOCKER_BUILD_ARGS ?= --build-arg APP_UID=$(shell id -u) --build-arg APP_GID=$(shell id -g)
WALKER_RUN ?= walker2d_seed1
ANT_RUN ?= ant_seed1
WALKER_CHECKPOINT ?= runs/Walker2d-v5/$(WALKER_RUN)/checkpoints/best.pt
ANT_CHECKPOINT ?= runs/Ant-v5/$(ANT_RUN)/checkpoints/best.pt

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
	$(PYTHON) -m mujoco_continuous_control.train --config configs/walker2d.yaml --run-name $(WALKER_RUN)

train-ant:
	$(PYTHON) -m mujoco_continuous_control.train --config configs/ant.yaml --run-name $(ANT_RUN)

eval-walker:
	@test -f "$(WALKER_CHECKPOINT)" || { echo "Missing $(WALKER_CHECKPOINT). Run make train-walker first, or pass WALKER_CHECKPOINT=/path/to/best.pt."; exit 1; }
	$(PYTHON) -m mujoco_continuous_control.evaluate --checkpoint $(WALKER_CHECKPOINT) --episodes 20

eval-ant:
	@test -f "$(ANT_CHECKPOINT)" || { echo "Missing $(ANT_CHECKPOINT). Run make train-ant first, or pass ANT_CHECKPOINT=/path/to/best.pt."; exit 1; }
	$(PYTHON) -m mujoco_continuous_control.evaluate --checkpoint $(ANT_CHECKPOINT) --episodes 20

video-walker:
	@test -f "$(WALKER_CHECKPOINT)" || { echo "Missing $(WALKER_CHECKPOINT). Run make train-walker first, or pass WALKER_CHECKPOINT=/path/to/best.pt."; exit 1; }
	$(PYTHON) -m mujoco_continuous_control.record_video --checkpoint $(WALKER_CHECKPOINT) --episodes 3 --output-dir assets/videos/walker2d

video-ant:
	@test -f "$(ANT_CHECKPOINT)" || { echo "Missing $(ANT_CHECKPOINT). Run make train-ant first, or pass ANT_CHECKPOINT=/path/to/best.pt."; exit 1; }
	$(PYTHON) -m mujoco_continuous_control.record_video --checkpoint $(ANT_CHECKPOINT) --episodes 3 --output-dir assets/videos/ant

docker-build:
	$(DOCKER) build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE) .

docker-test:
	$(DOCKER) run --rm $(DOCKER_IMAGE) make test

docker-smoke:
	$(DOCKER) run --rm -v "$(CURDIR)/runs:/app/runs" -v "$(CURDIR)/assets:/app/assets" $(DOCKER_IMAGE) make smoke

docker-shell:
	$(DOCKER) run --rm -it -v "$(CURDIR)/runs:/app/runs" -v "$(CURDIR)/assets:/app/assets" $(DOCKER_IMAGE) bash

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info outputs runs videos checkpoints
