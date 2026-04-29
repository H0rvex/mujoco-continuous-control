# MuJoCo Continuous Control with PPO

This project is a reproducible MuJoCo continuous-control benchmark using a
from-scratch PyTorch PPO implementation. 
It is designed as a bridge from classic Gymnasium RL projects
toward robotics simulation stacks such as Isaac Lab.

The goal is not to maximize one lucky benchmark run, but to build a
reproducible continuous-control PPO training system with multi-seed evaluation,
diagnostics, and visual rollout validation.

Humanoid-v5 is the primary stretch benchmark because it is the highest
dimensional locomotion task in this run set. Ant-v5 is the harder multi-limb
benchmark after Walker2d-v5, and Walker2d-v5 is the lower-dimensional
locomotion baseline.

> **Status:** Walker2d-v5, Ant-v5, and Humanoid-v5 each have three completed
> PPO seeds with deterministic evaluation, curves, and representative rollout
> GIFs.

## Results Summary

Walker2d-v5, Ant-v5, and Humanoid-v5 results are from `best.pt` checkpoints
evaluated deterministically for 20 episodes per seed with frozen observation
normalization and evaluation seed `1000`. The seed-level tables report the
post-training `eval_results.json` summary for each trained seed.

| Environment | Role | Train seeds | Steps / seed | Eval episodes / seed | Mean return | Std across seeds | Best 20-episode eval mean | Curves | Video |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Humanoid-v5 | Highest-dimensional stretch benchmark | 3 | 5,000,000 | 20 | 7456.04 | 573.61 | 8049.33 | `assets/curves/Humanoid-v5/` | `assets/videos/Humanoid-v5/humanoid_seed3/` |
| Ant-v5 | Harder benchmark | 3 | 5,000,000 | 20 | 4750.40 | 70.15 | 4822.60 | `assets/curves/Ant-v5/` | `assets/videos/Ant-v5/ant_seed2/` |
| Walker2d-v5 | Locomotion baseline | 3 | 2,000,000 | 20 | 2951.31 | 872.80 | 3720.55 | `assets/curves/Walker2d-v5/` | `assets/videos/Walker2d-v5/walker2d_seed1/` |

Humanoid seed-level deterministic evaluation:

| Seed | Checkpoint | Eval mean | Eval std | Eval min | Eval max | Episode length note |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `runs/Humanoid-v5/humanoid_seed1/checkpoints/best.pt` | 6904.37 | 12.09 | 6883.05 | 6921.63 | Full horizon in all 20 episodes, 1000/1000 steps |
| 2 | `runs/Humanoid-v5/humanoid_seed2/checkpoints/best.pt` | 7414.43 | 15.23 | 7370.18 | 7446.10 | Full horizon in all 20 episodes, 1000/1000 steps |
| 3 | `runs/Humanoid-v5/humanoid_seed3/checkpoints/best.pt` | 8049.33 | 20.50 | 8006.55 | 8088.44 | Full horizon in all 20 episodes, 1000/1000 steps |

The aggregate Humanoid mean is `7456.04` with sample standard deviation
`573.61` across the three seed-level means. Humanoid seed 3 is the strongest
external 20-episode checkpoint evaluation and is used as the representative
Humanoid rollout. Its deterministic evaluation is stable across all 20
full-horizon episodes, even though the last stochastic on-policy training rows
remain noisy.

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

Ant seed-level deterministic evaluation:

| Seed | Checkpoint | Eval mean | Eval std | Eval min | Eval max | Episode length note |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `runs/Ant-v5/ant_seed1/checkpoints/best.pt` | 4746.07 | 91.40 | 4555.38 | 4877.88 | Full horizon in all 20 episodes, 1000/1000 steps |
| 2 | `runs/Ant-v5/ant_seed2/checkpoints/best.pt` | 4822.60 | 177.25 | 4340.23 | 5204.72 | Full horizon in 19/20 episodes; one 928-step early termination |
| 3 | `runs/Ant-v5/ant_seed3/checkpoints/best.pt` | 4682.51 | 955.00 | 570.18 | 5306.86 | Full horizon in 19/20 episodes; one 141-step catastrophic termination |

The aggregate Ant mean is `4750.40` with sample standard deviation `70.15`
across the three seed-level means. Ant seed 2 is the strongest external
20-episode checkpoint evaluation. Ant seed 3 has the highest train-time
deterministic evaluation checkpoint (`4993.54`) and the highest single external
episode return (`5306.86`), but one catastrophic deterministic rollout makes it
less robust as the representative result.

Interesting observations and edge cases:

- Humanoid seed 3 is the strongest result in the repository by deterministic
  evaluation mean. It also highlights why checkpointed deterministic evaluation
  is reported separately from stochastic rollout collection: its final training
  row is short, but the saved `best.pt` policy reaches the 1000-step horizon in
  all 20 evaluation episodes.
- Ant is substantially stronger and more consistent across seeds than Walker2d
  in this run set: all three Ant seed means are clustered within about 140
  return points, while Walker2d seed 3 is a clear instability case.
- Ant seed 3 shows a useful edge case for reporting: most rollouts are strong,
  but a single 141-step fall dominates the 20-episode standard deviation. This
  is why the README reports min return and episode lengths, not only mean.
- Walker2d seed 3 is an honest failure/partial-success example. It learns some
  locomotion but never reaches the full horizon during deterministic
  evaluation, which is visible in both the episode lengths and rollout quality.
- Training returns are noisy for both environments because stochastic rollout
  collection can terminate early even when scheduled deterministic evaluation is
  improving.

## Rollout Videos

Recordings are generated from deterministic policy rollouts and saved as GIFs
for quick inspection.

| Environment | Rollout GIF | Selection |
| --- | --- | --- |
| Humanoid-v5 | `assets/videos/Humanoid-v5/humanoid_seed3/Humanoid-v5_episode_1.gif` | Representative rollout from the strongest Humanoid seed |
| Ant-v5 | `assets/videos/Ant-v5/ant_seed2/Ant-v5_episode_1.gif` | Representative rollout from the strongest stable Ant seed |
| Walker2d-v5 | `assets/videos/Walker2d-v5/walker2d_seed1/Walker2d-v5_episode_1.gif` | Representative rollout from the strongest Walker2d seed |

<p align="center">
  <img src="assets/videos/Humanoid-v5/humanoid_seed3/Humanoid-v5_episode_1.gif" width="620" alt="Humanoid-v5 deterministic rollout from seed 3">
</p>

<p align="center">
  <img src="assets/videos/Ant-v5/ant_seed2/Ant-v5_episode_1.gif" width="620" alt="Ant-v5 deterministic rollout from seed 2">
</p>

<p align="center">
  <img src="assets/videos/Walker2d-v5/walker2d_seed1/Walker2d-v5_episode_1.gif" width="620" alt="Walker2d-v5 deterministic rollout from seed 1">
</p>

Humanoid seed 3 is used as the representative Humanoid video because it has the
highest external 20-episode deterministic evaluation mean and survives the full
1000-step horizon in all evaluation episodes. Ant seed 2 is used as the
representative Ant video because it has the highest external 20-episode
deterministic evaluation mean while remaining visually stable in rollout media.
Walker2d seed 1 is used as the representative Walker2d video because it has the
highest Walker2d deterministic evaluation mean and survives the full 1000-step
horizon in all evaluation episodes.

```bash
make video-walker
make video-ant
python -m mujoco_continuous_control.record_video \
  --checkpoint runs/Humanoid-v5/humanoid_seed3/checkpoints/best.pt \
  --episodes 3 \
  --seed 1000 \
  --output-dir assets/videos/Humanoid-v5/humanoid_seed3
```

## Key Engineering Details

- tanh-squashed Gaussian policy with log-probability correction
- GAE-Lambda
- clipped PPO policy and value losses
- observation normalization with frozen eval/video statistics
- reward normalization and clipping
- target-KL early stopping
- deterministic multi-seed evaluation
- rollout GIFs, CSV logs, JSON eval summaries, and diagnostic curves

## Core Hyperparameters

Values are pulled from `configs/walker2d.yaml`, `configs/ant.yaml`, and
`configs/humanoid.yaml`.

| Setting | Walker2d-v5 | Ant-v5 | Humanoid-v5 |
| --- | ---: | ---: | ---: |
| `total_timesteps` | 2000000 | 5000000 | 5000000 |
| `num_envs` | 8 | 8 | 8 |
| `rollout_steps` | 2048 | 2048 | 2048 |
| Rollout batch size | 16384 | 16384 | 16384 |
| `num_minibatches` | 16 | 16 | 16 |
| `update_epochs` | 5 | 5 | 5 |
| `learning_rate` | 2.0e-4 | 2.0e-4 | 2.0e-4 |
| `gamma` | 0.99 | 0.99 | 0.99 |
| `gae_lambda` | 0.95 | 0.95 | 0.95 |
| `clip_coef` | 0.2 | 0.2 | 0.2 |
| `target_kl` | 0.02 | 0.02 | 0.02 |
| `ent_coef` | 0.0 | 0.0 | 0.0 |
| `vf_coef` | 0.5 | 0.5 | 0.5 |
| `max_grad_norm` | 0.5 | 0.5 | 0.5 |
| `normalize_obs` | true | true | true |
| `normalize_rewards` | true | true | true |
| `normalize_advantages` | true | true | true |
| `reward_clip` | 10.0 | 10.0 | 10.0 |
| `hidden_sizes` | [256, 256] | [256, 256] | [256, 256] |
| `activation` | tanh | tanh | tanh |

## Reproducing Results

Recommended result-generation flow:

```bash
python -m mujoco_continuous_control.train --config configs/walker2d.yaml --run-name walker2d_seed1
python -m mujoco_continuous_control.evaluate \
  --checkpoint runs/Walker2d-v5/walker2d_seed1/checkpoints/best.pt \
  --episodes 20 \
  --seed 1000 \
  --output runs/Walker2d-v5/walker2d_seed1/eval_results.json
python -m mujoco_continuous_control.plotting \
  --run-dir runs/Walker2d-v5/walker2d_seed1 \
  --output-dir assets/curves
python -m mujoco_continuous_control.record_video \
  --checkpoint runs/Walker2d-v5/walker2d_seed1/checkpoints/best.pt \
  --episodes 3 \
  --seed 1000 \
  --output-dir assets/videos/Walker2d-v5/walker2d_seed1

python -m mujoco_continuous_control.train --config configs/ant.yaml --run-name ant_seed2
python -m mujoco_continuous_control.evaluate \
  --checkpoint runs/Ant-v5/ant_seed2/checkpoints/best.pt \
  --episodes 20 \
  --seed 1000 \
  --output runs/Ant-v5/ant_seed2/eval_results.json
python -m mujoco_continuous_control.plotting \
  --run-dir runs/Ant-v5/ant_seed2 \
  --output-dir assets/curves
python -m mujoco_continuous_control.record_video \
  --checkpoint runs/Ant-v5/ant_seed2/checkpoints/best.pt \
  --episodes 3 \
  --seed 1000 \
  --output-dir assets/videos/Ant-v5/ant_seed2

python -m mujoco_continuous_control.train --config configs/humanoid.yaml --run-name humanoid_seed3
python -m mujoco_continuous_control.evaluate \
  --checkpoint runs/Humanoid-v5/humanoid_seed3/checkpoints/best.pt \
  --episodes 20 \
  --seed 1000 \
  --output runs/Humanoid-v5/humanoid_seed3/eval_results.json
python -m mujoco_continuous_control.plotting \
  --run-dir runs/Humanoid-v5/humanoid_seed3 \
  --output-dir assets/curves
python -m mujoco_continuous_control.record_video \
  --checkpoint runs/Humanoid-v5/humanoid_seed3/checkpoints/best.pt \
  --episodes 3 \
  --seed 1000 \
  --output-dir assets/videos/Humanoid-v5/humanoid_seed3
```

## Evaluation Protocol

Development runs use one seed first to debug the full training, evaluation,
plotting, and video pipeline.

Final reported results use multiple training seeds where compute allows:

- Walker2d-v5: 3 training seeds
- Ant-v5: 3 training seeds
- Humanoid-v5: 3 training seeds

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

for seed in 1 2 3; do
  python -m mujoco_continuous_control.train \
    --config configs/humanoid.yaml \
    --seed $seed \
    --run-name humanoid_seed${seed}
done
```

Each trained policy is evaluated deterministically with:

- deterministic raw action from the actor mean
- bounded policy action: tanh(mean)
- environment action: scale(tanh(mean))
- frozen observation normalization statistics
- fixed evaluation seeds
- final reported results use 20 deterministic evaluation episodes per training seed

The final report aggregates mean return, standard deviation, minimum return,
maximum return, and best deterministic evaluation performance across seeds.

## Training and Evaluation Curves

Plotting reads each run's `metrics.csv` and writes PNGs under
`assets/curves/{env_id}/{run_name}/`.

The primary Humanoid comparison is the deterministic evaluation curve across
the three training seeds:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Humanoid seed 1 evaluation return](assets/curves/Humanoid-v5/humanoid_seed1/evaluation_return.png) | ![Humanoid seed 2 evaluation return](assets/curves/Humanoid-v5/humanoid_seed2/evaluation_return.png) | ![Humanoid seed 3 evaluation return](assets/curves/Humanoid-v5/humanoid_seed3/evaluation_return.png) |

Humanoid training-return curves are noisier than the deterministic evaluation
curves, especially late in seed 3, which is why the final report separates
training rollout noise from frozen-checkpoint evaluation:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Humanoid seed 1 training return](assets/curves/Humanoid-v5/humanoid_seed1/training_return.png) | ![Humanoid seed 2 training return](assets/curves/Humanoid-v5/humanoid_seed2/training_return.png) | ![Humanoid seed 3 training return](assets/curves/Humanoid-v5/humanoid_seed3/training_return.png) |

<details>
<summary>Humanoid diagnostic curves</summary>

Approximate KL:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Humanoid seed 1 approximate KL](assets/curves/Humanoid-v5/humanoid_seed1/approx_kl.png) | ![Humanoid seed 2 approximate KL](assets/curves/Humanoid-v5/humanoid_seed2/approx_kl.png) | ![Humanoid seed 3 approximate KL](assets/curves/Humanoid-v5/humanoid_seed3/approx_kl.png) |

Action standard deviation:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Humanoid seed 1 action standard deviation](assets/curves/Humanoid-v5/humanoid_seed1/action_std.png) | ![Humanoid seed 2 action standard deviation](assets/curves/Humanoid-v5/humanoid_seed2/action_std.png) | ![Humanoid seed 3 action standard deviation](assets/curves/Humanoid-v5/humanoid_seed3/action_std.png) |

Additional plots are available for `losses`, `entropy`, and `clip_fraction` in
each `assets/curves/Humanoid-v5/humanoid_seed*/` directory.

</details>

The primary Walker2d comparison is the deterministic evaluation curve across
the three training seeds:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Walker2d seed 1 evaluation return](assets/curves/Walker2d-v5/walker2d_seed1/evaluation_return.png) | ![Walker2d seed 2 evaluation return](assets/curves/Walker2d-v5/walker2d_seed2/evaluation_return.png) | ![Walker2d seed 3 evaluation return](assets/curves/Walker2d-v5/walker2d_seed3/evaluation_return.png) |

Training-return curves show how noisy rollout collection remains even when
deterministic evaluation improves:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Walker2d seed 1 training return](assets/curves/Walker2d-v5/walker2d_seed1/training_return.png) | ![Walker2d seed 2 training return](assets/curves/Walker2d-v5/walker2d_seed2/training_return.png) | ![Walker2d seed 3 training return](assets/curves/Walker2d-v5/walker2d_seed3/training_return.png) |

<details>
<summary>Walker2d diagnostic curves</summary>

Approximate KL:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Walker2d seed 1 approximate KL](assets/curves/Walker2d-v5/walker2d_seed1/approx_kl.png) | ![Walker2d seed 2 approximate KL](assets/curves/Walker2d-v5/walker2d_seed2/approx_kl.png) | ![Walker2d seed 3 approximate KL](assets/curves/Walker2d-v5/walker2d_seed3/approx_kl.png) |

Action standard deviation:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Walker2d seed 1 action standard deviation](assets/curves/Walker2d-v5/walker2d_seed1/action_std.png) | ![Walker2d seed 2 action standard deviation](assets/curves/Walker2d-v5/walker2d_seed2/action_std.png) | ![Walker2d seed 3 action standard deviation](assets/curves/Walker2d-v5/walker2d_seed3/action_std.png) |

Additional plots are available for `losses`, `entropy`, and `clip_fraction` in
each `assets/curves/Walker2d-v5/walker2d_seed*/` directory.

</details>

The primary Ant comparison is the deterministic evaluation curve across the
three training seeds:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Ant seed 1 evaluation return](assets/curves/Ant-v5/ant_seed1/evaluation_return.png) | ![Ant seed 2 evaluation return](assets/curves/Ant-v5/ant_seed2/evaluation_return.png) | ![Ant seed 3 evaluation return](assets/curves/Ant-v5/ant_seed3/evaluation_return.png) |

Ant training-return curves show the same rollout noise as Walker2d, but the
scheduled deterministic evaluations remain clustered near strong locomotion
returns for all three seeds:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Ant seed 1 training return](assets/curves/Ant-v5/ant_seed1/training_return.png) | ![Ant seed 2 training return](assets/curves/Ant-v5/ant_seed2/training_return.png) | ![Ant seed 3 training return](assets/curves/Ant-v5/ant_seed3/training_return.png) |

<details>
<summary>Ant diagnostic curves</summary>

Approximate KL:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Ant seed 1 approximate KL](assets/curves/Ant-v5/ant_seed1/approx_kl.png) | ![Ant seed 2 approximate KL](assets/curves/Ant-v5/ant_seed2/approx_kl.png) | ![Ant seed 3 approximate KL](assets/curves/Ant-v5/ant_seed3/approx_kl.png) |

Action standard deviation:

| Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- |
| ![Ant seed 1 action standard deviation](assets/curves/Ant-v5/ant_seed1/action_std.png) | ![Ant seed 2 action standard deviation](assets/curves/Ant-v5/ant_seed2/action_std.png) | ![Ant seed 3 action standard deviation](assets/curves/Ant-v5/ant_seed3/action_std.png) |

Additional plots are available for `losses`, `entropy`, and `clip_fraction` in
each `assets/curves/Ant-v5/ant_seed*/` directory.

</details>

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

## Why Walker2d, Ant, and Humanoid

Walker2d-v5 is a useful locomotion baseline because it is lower-dimensional and
easier to inspect visually. It exposes balance, gait formation, and collapse
failures without the full complexity of a multi-limb agent.

Ant-v5 is the harder multi-limb target after Walker2d. It has a
higher-dimensional action space, more contacts, more opportunities for unstable
gaits, and more meaningful diagnostic value when PPO begins to drift. Strong
Ant behavior is a better signal that the implementation handles
continuous-control locomotion rather than only a small baseline.

Humanoid-v5 is the completed stretch benchmark for the repository. It adds a
much larger observation and action space, upright balance, and a harder
gait-learning problem. Three strong Humanoid seeds make the project read less
like a single environment demo and more like a reusable PPO continuous-control
benchmark.

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

## Docker

The Docker image installs the package with development dependencies and sets
headless MuJoCo rendering defaults for smoke tests and GIF recording.

```bash
make docker-build
make docker-test
make docker-smoke
```

For an interactive container with `runs/` and `assets/` mounted from the host:

```bash
make docker-shell
```

## Training

```bash
make smoke
```

Use explicit run names for the canonical Walker2d, Ant, and Humanoid artifact
paths used throughout this README:

```bash
python -m mujoco_continuous_control.train --config configs/smoke_test.yaml
python -m mujoco_continuous_control.train --config configs/walker2d.yaml --run-name walker2d_seed1
python -m mujoco_continuous_control.train --config configs/ant.yaml --run-name ant_seed1
python -m mujoco_continuous_control.train --config configs/humanoid.yaml --run-name humanoid_seed1
```

The Makefile also includes `make train-ant` and `make train-walker`
convenience targets for launching the default Ant and Walker2d configs. Use
the explicit command above for Humanoid.

## Evaluation

```bash
make eval-walker
make eval-ant
python -m mujoco_continuous_control.evaluate \
  --checkpoint runs/Humanoid-v5/humanoid_seed3/checkpoints/best.pt \
  --episodes 20 \
  --seed 1000 \
  --output runs/Humanoid-v5/humanoid_seed3/eval_results.json
```

Evaluation uses deterministic actions and writes JSON summaries containing
mean, standard deviation, minimum, maximum, per-episode returns, and per-episode
lengths.

## Video Recording

```bash
make video-walker
make video-ant
python -m mujoco_continuous_control.record_video \
  --checkpoint runs/Humanoid-v5/humanoid_seed3/checkpoints/best.pt \
  --episodes 3 \
  --seed 1000 \
  --output-dir assets/videos/Humanoid-v5/humanoid_seed3
```

Video recording uses `render_mode="rgb_array"` and deterministic actions. The
published Humanoid representative rollout is
`assets/videos/Humanoid-v5/humanoid_seed3/Humanoid-v5_episode_1.gif`, the
published Ant representative rollout is
`assets/videos/Ant-v5/ant_seed2/Ant-v5_episode_1.gif`, and the published
Walker2d representative rollout is
`assets/videos/Walker2d-v5/walker2d_seed1/Walker2d-v5_episode_1.gif`.
`make video-ant` and `make video-walker` regenerate default GIFs under
`assets/videos/ant/` and `assets/videos/walker2d/`; use explicit
`record_video` commands when regenerating the Humanoid or seed-specific
portfolio paths.

## Plotting

```bash
python -m mujoco_continuous_control.plotting --run-dir runs/Walker2d-v5/walker2d_seed1
python -m mujoco_continuous_control.plotting --run-dir runs/Ant-v5/ant_seed1
python -m mujoco_continuous_control.plotting --run-dir runs/Humanoid-v5/humanoid_seed1
```

The plotting command generates training return, evaluation return, loss,
entropy, approximate KL, clip fraction, and action standard-deviation curves.

## Failure Analysis

Expected PPO and locomotion failure modes include:

- Ant policies that mostly learn stable movement but still have rare
  catastrophic deterministic rollouts.
- Humanoid training rollouts that remain noisy late in training even when the
  checkpointed deterministic policy is strong.
- Walker2d policies that improve briefly or learn partial gaits, then terminate
  early when balance becomes unstable.
- Value estimates that lag behind rapidly changing returns.
- Entropy collapse, visible through falling action standard deviation.
- Excessive approximate KL or high clip fraction, indicating overly aggressive
  policy updates.
- Action saturation near the environment bounds.

The top-level README is the current aggregate report: it includes seed-level
metrics, cross-seed aggregates, rollout media, curves, and the main failure
cases. Separate environment reports are not required unless the project grows
into a longer experiment write-up.

## Limitations

This is not a full robotics stack. It mainly focuses on MuJoCo
continuous-control locomotion with PPO, deterministic evaluation, rollout media,
and experiment diagnostics.

Future extensions could include custom MuJoCo morphologies, domain
randomization, Isaac Lab experiments, or experiment tracking integrations such
as Weights & Biases. Those would add more demanding embodiment, robustness, and
simulation-workflow coverage beyond the current Walker2d, Ant, and Humanoid
locomotion benchmarks.
