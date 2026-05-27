# rotom-rl

`rotom-rl` is a local MuJoCo workspace for two related pieces:

- A generated MuJoCo model of the Rotom mechanism exported from Onshape.
- A PPO training script for end-effector reaching on that mechanism.

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="imgs/rotom_insim.png" alt="Rotom in simulation" height="400"/>
  <img src="imgs/rotomv2_cad.png" alt="Rotom CAD model" height="400"/>
</div>

## Repo Layout

- `robot.xml`: generated MuJoCo model of the Rotom mechanism.
- `scene.xml`: lightweight scene wrapper that includes `robot.xml`, lighting, and a floor plane.
- `assets/`: CAD meshes for the robot plus older MuJoCo experiments.
- `config.json`: `onshape-to-robot` export configuration for the Onshape document.
- `basline.py`: Gymnasium + Stable-Baselines3 training script for Rotom end-effector reaching.
- `artifacts/`: saved PPO checkpoints and generated plots. This directory is gitignored.
- `justfile`: convenience recipes for regenerating the robot and launching the MuJoCo helper.
- `mujoco/`: vendored upstream MuJoCo source tree. It is not required to run `basline.py`, but it is useful for local reference and engine work.
- `mujoco_tutorial.ipynb`: local notebook for MuJoCo experimentation.

## Setup

This repo is currently set up around Python 3.12.

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

The main Python dependencies are:

- `mujoco`
- `gymnasium`
- `stable-baselines3`
- `torch`
- `onshape-to-robot`

## Working With The Rotom Model

The generated robot assets come from the Onshape document referenced in `config.json`. The usual flow is:

1. Export or refresh the MuJoCo model from Onshape.
2. Inspect the generated `robot.xml` and mesh assets in `assets/`.
3. Open the scene wrapper in MuJoCo.

Direct commands:

```bash
./.venv/bin/onshape-to-robot .
./.venv/bin/mjpython ./.venv/bin/onshape-to-robot-mujoco .
```

Optional `just` wrappers:

```bash
just onshape_api
just test_mujoco
```

Relevant files:

- `config.json` points at the Onshape assembly and overrides joint and geometry properties.
- `robot.xml` contains four actuated hinge joints: `O`, `A`, `B`, and `C`.
- `scene.xml` adds a floor plane and basic lighting around the generated robot.

## RL Starter

`basline.py` is the current executable entry point for reinforcement learning experiments. The filename is spelled `basline.py` in the repo, so the commands below use that exact path.

What the script currently does:

- Defines a single `RotomReachEnv`.
- Samples a reachable Cartesian target for the `ee` site.
- Moves the red target marker in the MuJoCo model to that sampled position.
- Trains PPO with Stable-Baselines3.
- Supports environment validation, random rollouts, policy evaluation, live viewer playback, and rollout plots.

Default task files:

```text
training/eval -> robot.xml
viewer -> scene.xml
checkpoint -> artifacts/ppo_rotom_reacher.zip
```

## RL Commands

Validate the environment:

```bash
./.venv/bin/mjpython basline.py check-env
```

Run a short random rollout:

```bash
./.venv/bin/mjpython basline.py random --steps 10
```

Train PPO:

```bash
./.venv/bin/mjpython basline.py train --timesteps 50000
```

Evaluate a saved policy:

```bash
./.venv/bin/mjpython basline.py eval --episodes 5
```

Open the MuJoCo viewer and watch the trained policy:

```bash
./.venv/bin/mjpython basline.py play --xml-path scene.xml --episodes 3
```

Save a headless-safe rollout plot:

```bash
./.venv/bin/mjpython basline.py plot-rollout --episodes 3 --output-path artifacts/rotom_rollout.png
```

Available subcommands:

- `train`
- `eval`
- `play`
- `plot-rollout`
- `random`
- `check-env`

## Current State

What is already working:

- `robot.xml` and `scene.xml` load successfully through MuJoCo.
- `basline.py check-env` passes.
- `basline.py train` learns a working end-effector reaching policy with PPO.
- `basline.py play --xml-path scene.xml` shows the cyan end-effector marker and red target marker in the viewer.
- `basline.py plot-rollout` saves a trajectory figure without requiring an OpenGL display.

What is not wired up yet:

- The task is still pure reaching; it does not manipulate objects or use contacts.
- The target is a visual marker only; there is no obstacle field or curriculum yet.
- The exploratory notebook `rotom.ipynb` is not the recommended training entry point.

## Suggested Next Steps

- Add camera presets in `scene.xml` if you want deterministic cinematic recordings instead of the default free-camera render.
- Consider switching from joint-setpoint actions to velocity or torque control if you want a harder but more physically direct policy.
- Add curriculum over target distance or height if you want better sample efficiency on harder reaches.
- Extend the observation with end-effector orientation if the task should care about pose, not just position.
- Rename `basline.py` to `baseline.py` later if you want cleaner ergonomics, after updating references.
