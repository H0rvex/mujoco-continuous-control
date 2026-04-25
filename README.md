# MuJoCo Continuous Control

Portfolio-grade continuous-control reinforcement learning scaffold for Gymnasium
MuJoCo locomotion environments. PPO is intentionally not implemented yet; this
initial version provides packaging, configuration, smoke execution, and the
environment factory that later training code will use.

## Quick Start

```bash
python -m mujoco_continuous_control.train --config configs/smoke_test.yaml
pytest
```

The smoke config uses `Pendulum-v1` so the repository can be checked quickly
without requiring MuJoCo assets. Main MuJoCo targets are `Walker2d-v5` and
`Ant-v5`.
