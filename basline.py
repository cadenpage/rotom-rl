from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env


DEFAULT_XML_PATH = ROOT / "assets" / "point_reacher.xml"
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "ppo_point_reacher"


class PointReacherEnv(gym.Env[np.ndarray, np.ndarray]):
    """A small MuJoCo reach task that is easy to modify and reason about."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML_PATH,
        frame_skip: int = 5,
        max_episode_steps: int = 200,
        target_radius: float = 0.08,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.target_radius = target_radius
        self.render_mode = render_mode

        self._elapsed_steps = 0
        self._renderer: mujoco.Renderer | None = None

        self.target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target",
        )
        if self.target_body_id < 0:
            raise ValueError("MuJoCo model is missing a body named 'target'.")

        action_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        action_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

    def _target_xy(self) -> np.ndarray:
        return self.model.body_pos[self.target_body_id, :2].copy()

    def _set_target_xy(self, target_xy: np.ndarray) -> None:
        self.model.body_pos[self.target_body_id, :2] = target_xy

    def _sample_target_xy(self, agent_xy: np.ndarray) -> np.ndarray:
        for _ in range(64):
            candidate = self.np_random.uniform(low=-0.75, high=0.75, size=2)
            if np.linalg.norm(candidate - agent_xy) > 0.25:
                return candidate.astype(np.float64)
        return np.array([0.6, 0.6], dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        target_xy = self._target_xy()
        delta = target_xy - qpos
        obs = np.concatenate([qpos, qvel, target_xy, delta])
        return obs.astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._elapsed_steps = 0

        mujoco.mj_resetData(self.model, self.data)

        agent_xy = self.np_random.uniform(low=-0.5, high=0.5, size=2)
        target_xy = self._sample_target_xy(agent_xy)

        self.data.qpos[:] = agent_xy
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._set_target_xy(target_xy)
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        distance = float(np.linalg.norm(target_xy - agent_xy))
        info = {"distance": distance, "success": False}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._elapsed_steps += 1

        agent_xy = self.data.qpos.copy()
        target_xy = self._target_xy()
        distance = float(np.linalg.norm(target_xy - agent_xy))
        control_cost = float(0.05 * np.square(action).sum())
        success = distance < self.target_radius

        reward = -distance - control_cost
        if success:
            reward += 5.0

        terminated = success
        truncated = self._elapsed_steps >= self.max_episode_steps
        info = {
            "distance": distance,
            "reward_distance": -distance,
            "reward_control": -control_cost,
            "success": success,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise NotImplementedError("This environment only implements rgb_array rendering.")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=480)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(max_episode_steps: int) -> PointReacherEnv:
    return PointReacherEnv(max_episode_steps=max_episode_steps)


def train(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        PointReacherEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"max_episode_steps": args.max_episode_steps},
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.rollout_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        verbose=1,
        policy_kwargs={"net_arch": [128, 128]},
        device="auto",
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(str(model_path))
    env.close()

    print(f"Saved PPO policy to {model_path}.zip")


def evaluate(args: argparse.Namespace) -> None:
    env = PointReacherEnv(max_episode_steps=args.max_episode_steps)
    model = PPO.load(str(args.model_path), device="auto")

    returns: list[float] = []
    successes = 0

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=None if args.seed is None else args.seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        final_info: dict[str, Any] = {"distance": float("nan"), "success": False}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            final_info = info

        returns.append(episode_return)
        successes += int(bool(final_info["success"]))
        print(
            f"episode={episode + 1} "
            f"return={episode_return:.3f} "
            f"distance={final_info['distance']:.3f} "
            f"success={final_info['success']}"
        )

    env.close()
    print(
        f"mean_return={np.mean(returns):.3f} "
        f"success_rate={successes / max(args.episodes, 1):.2%}"
    )


def play(args: argparse.Namespace) -> None:
    try:
        import mujoco.viewer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MuJoCo viewer is not available in this environment.") from exc

    env = PointReacherEnv(max_episode_steps=args.max_episode_steps)
    model = PPO.load(str(args.model_path), device="auto")
    obs, _ = env.reset(seed=args.seed)

    episodes_left = args.episodes
    timestep = env.model.opt.timestep * env.frame_skip

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and episodes_left > 0:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(timestep)

            if terminated or truncated:
                print(f"episode complete: success={info['success']} distance={info['distance']:.3f}")
                episodes_left -= 1
                if episodes_left > 0:
                    obs, _ = env.reset()

    env.close()


def run_random_policy(args: argparse.Namespace) -> None:
    env = PointReacherEnv(max_episode_steps=args.max_episode_steps)
    obs, info = env.reset(seed=args.seed)
    print(f"reset obs_shape={obs.shape} distance={info['distance']:.3f}")

    episode_return = 0.0
    for step_idx in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        print(
            f"step={step_idx + 1} reward={reward:.3f} "
            f"distance={info['distance']:.3f} success={info['success']}"
        )
        if terminated or truncated:
            break

    print(f"random rollout return={episode_return:.3f}")
    env.close()


def run_env_check(_: argparse.Namespace) -> None:
    env = PointReacherEnv()
    check_env(env, warn=True, skip_render_check=True)
    env.close()
    print("Gymnasium / Stable-Baselines3 environment check passed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo + PPO starter for this repo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train PPO on the custom point reacher task.")
    train_parser.add_argument("--timesteps", type=int, default=50_000)
    train_parser.add_argument("--n-envs", type=int, default=4)
    train_parser.add_argument("--rollout-steps", type=int, default=512)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--ent-coef", type=float, default=0.0)
    train_parser.add_argument("--max-episode-steps", type=int, default=200)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved PPO policy.")
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--max-episode-steps", type=int, default=200)
    eval_parser.add_argument("--seed", type=int, default=7)
    eval_parser.add_argument("--stochastic", action="store_true")
    eval_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH.with_suffix(".zip"))
    eval_parser.set_defaults(func=evaluate)

    play_parser = subparsers.add_parser("play", help="Run the trained policy in the MuJoCo viewer.")
    play_parser.add_argument("--episodes", type=int, default=3)
    play_parser.add_argument("--max-episode-steps", type=int, default=200)
    play_parser.add_argument("--seed", type=int, default=7)
    play_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH.with_suffix(".zip"))
    play_parser.set_defaults(func=play)

    random_parser = subparsers.add_parser("random", help="Smoke test the environment with random actions.")
    random_parser.add_argument("--steps", type=int, default=25)
    random_parser.add_argument("--max-episode-steps", type=int, default=200)
    random_parser.add_argument("--seed", type=int, default=7)
    random_parser.set_defaults(func=run_random_policy)

    check_parser = subparsers.add_parser("check-env", help="Run Gymnasium environment validation.")
    check_parser.set_defaults(func=run_env_check)

    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
