# MuJoCo Continuous Control with PPO

This project is a reproducible MuJoCo continuous-control benchmark using
from-scratch PPO. It is designed as a bridge from classic Gymnasium RL projects
toward robotics simulation stacks such as Isaac Lab.

The implementation focuses on bounded continuous actions, tanh-squashed
Gaussian policies, vectorized locomotion rollouts, deterministic evaluation,
rollout media, and diagnostic plots. Ant-v5 is the primary benchmark because it
is the harder multi-limb locomotion task; Walker2d-v5 is the lower-dimensional
locomotion baseline.

> **Status:** Walker2d-v5 has three completed PPO seeds with deterministic
> evaluation, curves, and a representative rollout GIF. Ant-v5 remains the next
> harder benchmark.

## Rollout Videos

Walker2d-v5 is used as the lower-dimensional locomotion baseline, while Ant-v5
is the harder multi-limb benchmark and primary portfolio target. Recordings are
generated from deterministic policy rollouts and saved as GIFs for quick
inspection.

| Environment | Rollout GIF | Command |
| --- | --- | --- |
| Walker2d-v5 | `assets/videos/Walker2d-v5/walker2d_seed1/Walker2d-v5_episode_1.gif` | Representative rollout from the strongest seed |
| Ant-v5 | `assets/videos/ant/Ant-v5_episode_1.gif` | `make video-ant` after Ant training |

Walker2d seed 1 is used as the representative video because it has the highest
20-episode deterministic evaluation mean and survives the full 1000-step
horizon in all evaluation episodes. Ant rollout media will be added after the
Ant training run.

```bash
make video-walker
make video-ant
```

## Highlights

- From-scratch PPO implementation in PyTorch for continuous-control locomotion.
- Tanh-squashed Gaussian actor with log-probability correction for bounded
  action spaces.
- Separate actor and critic networks with configurable hidden sizes and
  activation.
- Vectorized Gymnasium environments for rollout collection.
- GAE-Lambda returns, clipped PPO policy objective, clipped value loss,
  entropy monitoring with configurable entropy regularization, gradient clipping, and target-KL early stopping.
- Observation normalization with frozen statistics during evaluation and video
  recording.
- Deterministic evaluation protocol for reproducible policy comparison.
- CSV metrics, JSON evaluation summaries, rollout GIFs, and Matplotlib curves
  for reviewable experiment artifacts.

## Results

Walker2d-v5 results are from `best.pt` checkpoints evaluated deterministically
for 20 episodes per seed with frozen observation normalization and evaluation
seed `1000`. The seed-level table reports the post-training
`eval_results.json` summary for each trained seed.

| Environment | Role | Train seeds | Steps / seed | Eval episodes / seed | Mean return | Std across seeds | Best return | Curves | Video |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Walker2d-v5 | Locomotion baseline | 3 | 2,000,000 | 20 | 2951.31 | 872.80 | 3720.55 | `assets/curves/Walker2d-v5/` | `assets/videos/Walker2d-v5/walker2d_seed1/` |
| Ant-v5 | Harder benchmark | TBD | 5,000,000 | 20 | TBD | TBD | TBD | `assets/curves/Ant-v5/` | `assets/videos/ant/` |

Walker2d seed-level deterministic evaluation:

| Seed | Checkpoint | Eval mean | Eval std | Eval min | Eval max | Episode length note |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `runs/Walker2d-v5/walker2d_seed1/checkpoints/best.pt` | 3720.55 | 57.10 | 3609.85 | 3848.61 | Full horizon in all 20 episodes, 1000/1000 steps |
| 2 | `runs/Walker2d-v5/walker2d_seed2/checkpoints/best.pt` | 3130.59 | 12.98 | 3104.85 | 3155.95 | Full horizon in all 20 episodes, 1000/1000 steps |
| 3 | `runs/Walker2d-v5/walker2d_seed3/checkpoints/best.pt` | 2002.79 | 422.43 | 1457.71 | 2830.42 | Early termination in all 20 episodes, 453-808 steps |

The aggregate Walker2d mean is `2951.31` with sample standard deviation
`872.80` across the three seed-level means. Seed 3 is a useful negative case:
it learns partial locomotion but remains dynamically unstable, falls before the
time limit, and shows the seed sensitivity expected from PPO on contact-rich
locomotion tasks at this training budget.

Recommended result-generation flow:

```bash
python -m mujoco_continuous_control.train --config configs/walker2d.yaml --run-name walker2d_seed1
make eval-walker
python -m mujoco_continuous_control.plotting --run-dir runs/Walker2d-v5/walker2d_seed1
make video-walker

python -m mujoco_continuous_control.train --config configs/ant.yaml --run-name ant_seed1
make eval-ant
python -m mujoco_continuous_control.plotting --run-dir runs/Ant-v5/ant_seed1
make video-ant
```

## Evaluation Protocol

Development runs use one seed first to debug the full training, evaluation,
plotting, and video pipeline.

Final reported results use multiple training seeds where compute allows:

- Walker2d-v5: 3 training seeds
- Ant-v5: 3 training seeds

Example multi-seed training commands:

```bash
for seed in 1 2 3; do
  python -m mujoco_continuous_control.train \
    --config configs/walker2d.yaml \
    --seed $seed \
    --run-name walker2d_seed${seed}
done

for seed in 1 2 3; do
  python -m mujoco_continuous_control.train \
    --config configs/ant.yaml \
    --seed $seed \
    --run-name ant_seed${seed}
done
```

Each trained policy is evaluated deterministically with:

- deterministic raw action from the actor mean
- bounded policy action: tanh(mean)
- environment action: scale(tanh(mean))
- frozen observation normalization statistics
- fixed evaluation seeds
- 10–20 episodes per training seed

The final report aggregates mean return, standard deviation, minimum return,
maximum return, and best deterministic evaluation performance across seeds.

## Training and Evaluation Curves

Plotting reads each run's `metrics.csv` and writes PNGs under
`assets/curves/{env_id}/{run_name}/`.

Ant curve set, after Ant training:

- `assets/curves/Ant-v5/ant_seed1/training_return.png`
- `assets/curves/Ant-v5/ant_seed1/evaluation_return.png`
- `assets/curves/Ant-v5/ant_seed1/losses.png`
- `assets/curves/Ant-v5/ant_seed1/entropy.png`
- `assets/curves/Ant-v5/ant_seed1/approx_kl.png`
- `assets/curves/Ant-v5/ant_seed1/clip_fraction.png`
- `assets/curves/Ant-v5/ant_seed1/action_std.png`

Walker2d curve sets:

- `assets/curves/Walker2d-v5/walker2d_seed1/training_return.png`
- `assets/curves/Walker2d-v5/walker2d_seed1/evaluation_return.png`
- `assets/curves/Walker2d-v5/walker2d_seed1/losses.png`
- `assets/curves/Walker2d-v5/walker2d_seed1/entropy.png`
- `assets/curves/Walker2d-v5/walker2d_seed1/approx_kl.png`
- `assets/curves/Walker2d-v5/walker2d_seed1/clip_fraction.png`
- `assets/curves/Walker2d-v5/walker2d_seed1/action_std.png`

The same seven-plot set is included for `walker2d_seed2` and
`walker2d_seed3`.

## Why MuJoCo Continuous Control

MuJoCo locomotion is a compact testbed for continuous-control policy
engineering. Unlike discrete-action control, these environments require stable
stochastic policies over bounded real-valued action spaces, careful treatment of
action scaling, and diagnostics that show when the policy update is becoming too
aggressive.

This demonstrates core continuous-control foundations used in robotics
simulation and embodied-AI research workflows: locomotion control, normalized
observations, rollout-based policy optimization, deterministic evaluation, and
artifact-driven experiment review.

## Why Walker2d and Ant

Walker2d-v5 is a useful locomotion baseline because it is lower-dimensional and
easier to inspect visually. It exposes balance, gait formation, and collapse
failures without the full complexity of a multi-limb agent.

Ant-v5 is the harder target and the more important benchmark in this project.
It has a higher-dimensional action space, more contacts, more opportunities for
unstable gaits, and more meaningful diagnostic value when PPO begins to drift.
Strong Ant behavior is a better signal that the implementation handles
continuous-control locomotion rather than only a small baseline.

## Algorithm Overview

Training follows the standard PPO rollout-update loop:

1. Create vectorized Gymnasium environments.
2. Normalize observations using running mean and variance statistics.
3. Sample tanh-squashed Gaussian actions from the actor.
4. Store observations, raw actions, scaled actions, rewards, dones, values, and
   log probabilities.
5. Compute GAE-Lambda advantages and returns.
6. Run PPO minibatch updates with clipped policy and value objectives.
7. Log training returns, policy losses, value losses, entropy, KL, clip
   fraction, explained variance, action statistics, and throughput.
8. Evaluate the deterministic policy on schedule.
9. Generate curves and rollout GIFs from saved run artifacts.

## Squashed Gaussian Policy

The actor outputs a mean vector and a learnable log standard deviation. During
training, the policy samples a raw Gaussian action, applies `tanh`, and then
scales the squashed action into the environment's action bounds.

```text
raw_action ~ Normal(mean, std)
squashed_action = tanh(raw_action)
env_action = action_low + 0.5 * (squashed_action + 1) * (action_high - action_low)
```

The PPO log-probability includes the tanh correction:

```text
log_prob = Normal(mean, std).log_prob(raw_action)
log_prob -= log(1 - tanh(raw_action)^2 + eps)
```

The affine scaling from `[-1, 1]` to the environment action range is constant
for a fixed environment, so it cancels in PPO probability ratios. Deterministic
evaluation uses the actor mean as the raw action, then applies the same
`tanh` and action-bound scaling.

## PPO Diagnostics

The training loop records diagnostics that make policy behavior inspectable:

- `train/episodic_return` and `train/episodic_length` track rollout progress.
- `eval/mean_return`, `eval/std_return`, `eval/min_return`, and
  `eval/max_return` summarize deterministic policy quality.
- `loss/policy_loss` and `loss/value_loss` show actor and critic optimization.
- `loss/entropy` tracks exploration pressure.
- `loss/approx_kl` and `loss/clip_fraction` show whether updates are too large.
- `loss/explained_variance` gives a quick view of value-function fit.
- `policy/action_std` and `policy/log_std_mean` help catch action collapse or
  overly noisy policies.
- `system/fps` tracks rollout and update throughput.

## Installation

Use Python 3.10 or newer.

From the repository root:

```bash
pip install -e ".[dev]"

python -m mujoco_continuous_control.train --config configs/smoke_test.yaml

pytest
```

The main MuJoCo targets use Gymnasium MuJoCo v5 environments. The smoke config
uses `Pendulum-v1` so the package can be checked quickly without launching a
long MuJoCo run.

## Training

```bash
make smoke
```

Use explicit run names for the canonical Ant and Walker2d artifact paths used
throughout this README:

```bash
python -m mujoco_continuous_control.train --config configs/smoke_test.yaml
python -m mujoco_continuous_control.train --config configs/walker2d.yaml --run-name walker2d_seed1
python -m mujoco_continuous_control.train --config configs/ant.yaml --run-name ant_seed1
```

The Makefile also includes `make train-ant` and `make train-walker` convenience
targets for launching the default environment configs.

## Evaluation

```bash
make eval-walker
make eval-ant
```

Evaluation uses deterministic actions and writes JSON summaries containing
mean, standard deviation, minimum, maximum, per-episode returns, and per-episode
lengths.

## Video Recording

```bash
make video-walker
make video-ant
```

Video recording uses `render_mode="rgb_array"` and deterministic actions. The
published Walker2d representative rollout is
`assets/videos/Walker2d-v5/walker2d_seed1/Walker2d-v5_episode_1.gif`; `make
video-walker` can regenerate default Walker2d GIFs under
`assets/videos/walker2d/`.

## Plotting

```bash
python -m mujoco_continuous_control.plotting --run-dir runs/Walker2d-v5/walker2d_seed1
python -m mujoco_continuous_control.plotting --run-dir runs/Ant-v5/ant_seed1
```

The plotting command generates training return, evaluation return, loss,
entropy, approximate KL, clip fraction, and action standard-deviation curves.

## Failure Analysis

Expected PPO and locomotion failure modes include:

- Ant policies that learn movement but fail to form a stable gait.
- Walker2d policies that improve briefly or learn partial gaits, then terminate
  early when balance becomes unstable.
- Value estimates that lag behind rapidly changing returns.
- Entropy collapse, visible through falling action standard deviation.
- Excessive approximate KL or high clip fraction, indicating overly aggressive
  policy updates.
- Action saturation near the environment bounds.

Detailed environment-specific reports can be added after the Ant run:

- `reports/ant_report.md`
- `reports/walker2d_report.md`
- `reports/failure_analysis.md`

## Limitations

This is not a full robotics stack. It mainly focuses on MuJoCo
continuous-control locomotion with PPO, deterministic evaluation, rollout media,
and experiment diagnostics.

Future extensions could include `Humanoid-v5`, custom MuJoCo morphologies,
domain randomization, or Isaac Lab experiments. Those would add more demanding
embodiment, robustness, and simulation-workflow coverage beyond the current
Walker2d and Ant locomotion benchmark.
