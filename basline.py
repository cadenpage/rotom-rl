from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env


DEFAULT_XML_PATH = ROOT / "robot.xml"
DEFAULT_SCENE_PATH = ROOT / "scene.xml"
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "ppo_rotom_reacher"


class RotomReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Reach a visible 3D target with the Rotom end-effector."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    JOINT_NAMES = ("O", "A", "B", "C")

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML_PATH,
        frame_skip: int = 20,
        max_episode_steps: int = 100,
        target_radius: float = 0.04,
        action_step_scale: float = 0.15,
        min_target_distance: float = 0.04,
        max_target_distance: float = 0.14,
        render_mode: str | None = None,
        camera_name: str | None = None,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.fk_data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.target_radius = target_radius
        self.action_step_scale = action_step_scale
        self.min_target_distance = min_target_distance
        self.max_target_distance = max_target_distance
        self.render_mode = render_mode
        self.camera_name = camera_name

        self._elapsed_steps = 0
        self._previous_distance = 0.0
        self._renderer: mujoco.Renderer | None = None

        self.joint_ids = np.array(
            [self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name, "joint") for name in self.JOINT_NAMES],
            dtype=np.int32,
        )
        self.qpos_ids = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_ids = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)

        self.joint_limits = self.model.jnt_range[self.joint_ids].astype(np.float64)
        self.joint_low = self.joint_limits[:, 0]
        self.joint_high = self.joint_limits[:, 1]
        self.joint_mid = 0.5 * (self.joint_low + self.joint_high)
        self.joint_half = 0.5 * (self.joint_high - self.joint_low)
        self.safe_joint_half = np.maximum(self.joint_half, 1e-6)

        self.ee_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "ee", "site")
        self.target_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "target", "body")

        workspace_min, workspace_max = self._estimate_workspace()
        self.workspace_center = 0.5 * (workspace_min + workspace_max)
        self.workspace_half = np.maximum(0.5 * (workspace_max - workspace_min), 1e-3)
        self.target_pos = self.workspace_center.copy()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.JOINT_NAMES),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

    def _require_id(self, obj_type: mujoco.mjtObj, name: str, label: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"MuJoCo model is missing {label!r} named {name!r}.")
        return obj_id

    def current_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].copy()

    def current_qd(self) -> np.ndarray:
        return self.data.qvel[self.qvel_ids].copy()

    def current_ee(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _normalize_joints(self, q: np.ndarray) -> np.ndarray:
        return (q - self.joint_mid) / self.safe_joint_half

    def _normalize_position(self, position: np.ndarray) -> np.ndarray:
        return (position - self.workspace_center) / self.workspace_half

    def _set_configuration(self, data: mujoco.MjData, q: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, data)
        data.qpos[self.qpos_ids] = q
        data.qvel[self.qvel_ids] = 0.0
        data.ctrl[:] = q
        mujoco.mj_forward(self.model, data)

    def _forward_ee(self, q: np.ndarray) -> np.ndarray:
        self._set_configuration(self.fk_data, q)
        return self.fk_data.site_xpos[self.ee_site_id].copy()

    def _sample_configuration(self) -> np.ndarray:
        margin = 0.10 * self.joint_half
        low = self.joint_low + margin
        high = self.joint_high - margin
        invalid = high <= low
        low[invalid] = self.joint_low[invalid]
        high[invalid] = self.joint_high[invalid]
        return self.np_random.uniform(low=low, high=high).astype(np.float64)

    def _estimate_workspace(self, samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        positions = []
        for _ in range(samples):
            q = np.array([rng.uniform(lo, hi) for lo, hi in self.joint_limits], dtype=np.float64)
            positions.append(self._forward_ee(q))
        stacked = np.asarray(positions, dtype=np.float64)
        return stacked.min(axis=0), stacked.max(axis=0)

    def _sample_target(self, start_q: np.ndarray, start_ee: np.ndarray) -> np.ndarray:
        for _ in range(128):
            joint_delta = self.np_random.uniform(low=-0.18, high=0.18, size=len(self.JOINT_NAMES))
            q_target = np.clip(start_q + joint_delta * self.joint_half, self.joint_low, self.joint_high)
            ee_target = self._forward_ee(q_target)
            distance = float(np.linalg.norm(ee_target - start_ee))
            if self.min_target_distance <= distance <= self.max_target_distance:
                return ee_target
        return self._forward_ee(self._sample_configuration())

    def _set_target(self, target_pos: np.ndarray) -> None:
        self.target_pos = target_pos.astype(np.float64)
        self.model.body_pos[self.target_body_id] = self.target_pos

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self.target_pos - self.current_ee()))

    def _get_obs(self) -> np.ndarray:
        q = self.current_q()
        qd = self.current_qd()
        ee = self.current_ee()

        joint_obs = self._normalize_joints(q)
        velocity_obs = 0.05 * qd
        ee_obs = self._normalize_position(ee)
        target_obs = self._normalize_position(self.target_pos)
        delta_obs = (self.target_pos - ee) / self.workspace_half

        return np.concatenate([joint_obs, velocity_obs, ee_obs, target_obs, delta_obs]).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._elapsed_steps = 0

        q0 = self._sample_configuration()
        self._set_configuration(self.data, q0)

        ee0 = self.current_ee()
        self._set_target(self._sample_target(q0, ee0))
        mujoco.mj_forward(self.model, self.data)

        distance = self._distance_to_target()
        self._previous_distance = distance
        return self._get_obs(), {"distance": distance, "success": False}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        q = self.current_q()
        ctrl = q + self.action_step_scale * self.joint_half * action
        ctrl = np.clip(ctrl, self.joint_low, self.joint_high)

        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._elapsed_steps += 1

        distance = self._distance_to_target()
        progress = self._previous_distance - distance
        control_cost = float(0.005 * np.square(action).sum())
        velocity_cost = float(0.0005 * np.square(self.current_qd()).sum())
        success = distance < self.target_radius

        reward = 20.0 * progress - 2.0 * distance - control_cost - velocity_cost
        if success:
            reward += 10.0

        self._previous_distance = distance

        info = {
            "distance": distance,
            "success": success,
            "ee_position": self.current_ee().astype(np.float32),
            "target_position": self.target_pos.astype(np.float32),
        }
        terminated = success
        truncated = self._elapsed_steps >= self.max_episode_steps
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise NotImplementedError("This environment only implements rgb_array rendering.")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=480)
        camera = self.camera_name if self.camera_name is not None else -1
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def resolve_model_path(model_path: Path | None, *, for_training: bool) -> Path:
    path = model_path or DEFAULT_MODEL_PATH
    return path if for_training or path.suffix == ".zip" else path.with_suffix(".zip")


def make_env(
    xml_path: Path,
    max_episode_steps: int,
    *,
    render_mode: str | None = None,
    camera_name: str | None = None,
) -> RotomReachEnv:
    return RotomReachEnv(
        xml_path=xml_path,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        camera_name=camera_name,
    )


def run_policy_episode(
    env: RotomReachEnv,
    model: PPO,
    *,
    seed: int | None = None,
    deterministic: bool = True,
    collect_trace: bool = False,
) -> tuple[float, dict[str, Any], np.ndarray | None, np.ndarray | None]:
    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {"distance": float("nan"), "success": False}
    trace: list[np.ndarray] | None = [env.current_ee()] if collect_trace else None

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        final_info = info
        if trace is not None:
            trace.append(np.asarray(info["ee_position"], dtype=np.float64))

    target = env.target_pos.copy() if collect_trace else None
    trace_array = np.asarray(trace, dtype=np.float64) if trace is not None else None
    return episode_return, final_info, trace_array, target


def train(args: argparse.Namespace) -> None:
    model_path = resolve_model_path(args.model_path, for_training=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        RotomReachEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"xml_path": args.xml_path, "max_episode_steps": args.max_episode_steps},
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

    saved_path = model_path if model_path.suffix == ".zip" else model_path.with_suffix(".zip")
    print(f"Saved PPO policy to {saved_path}")


def evaluate(args: argparse.Namespace) -> None:
    env = make_env(args.xml_path, args.max_episode_steps)
    model = PPO.load(str(resolve_model_path(args.model_path, for_training=False)), device="auto")

    returns: list[float] = []
    successes = 0
    for episode in range(args.episodes):
        episode_return, info, _, _ = run_policy_episode(
            env,
            model,
            seed=None if args.seed is None else args.seed + episode,
            deterministic=not args.stochastic,
        )
        returns.append(episode_return)
        successes += int(bool(info["success"]))
        print(
            f"episode={episode + 1} "
            f"return={episode_return:.3f} "
            f"distance={info['distance']:.3f} "
            f"success={info['success']}"
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

    env = make_env(args.xml_path, args.max_episode_steps)
    model = PPO.load(str(resolve_model_path(args.model_path, for_training=False)), device="auto")
    obs, _ = env.reset(seed=args.seed)

    timestep = env.model.opt.timestep * env.frame_skip
    episodes_left = args.episodes

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


def plot_rollout(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    env = make_env(args.xml_path, args.max_episode_steps)
    model = PPO.load(str(resolve_model_path(args.model_path, for_training=False)), device="auto")
    output_path = args.output_path or (ROOT / "artifacts" / "rotom_rollout.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traces: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for episode in range(args.episodes):
        _, info, trace, target = run_policy_episode(
            env,
            model,
            seed=None if args.seed is None else args.seed + episode,
            collect_trace=True,
        )
        assert trace is not None
        assert target is not None
        traces.append((trace, target, bool(info["success"])))

    env.close()

    fig = plt.figure(figsize=(13, 4.5))
    ax_3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 3, 2)
    ax_xz = fig.add_subplot(1, 3, 3)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(len(traces), 1)))

    for idx, (trace, target, success) in enumerate(traces):
        color = colors[idx]
        label = f"ep {idx + 1} {'ok' if success else 'miss'}"

        ax_3d.plot(trace[:, 0], trace[:, 1], trace[:, 2], color=color, linewidth=2, label=label)
        ax_3d.scatter(*trace[0], color=color, marker="o", s=24)
        ax_3d.scatter(*trace[-1], color=color, marker="x", s=48)
        ax_3d.scatter(*target, color="#d62728", marker="*", s=120)

        ax_xy.plot(trace[:, 0], trace[:, 1], color=color, linewidth=2)
        ax_xy.scatter(trace[0, 0], trace[0, 1], color=color, marker="o", s=20)
        ax_xy.scatter(trace[-1, 0], trace[-1, 1], color=color, marker="x", s=40)
        ax_xy.scatter(target[0], target[1], color="#d62728", marker="*", s=80)

        ax_xz.plot(trace[:, 0], trace[:, 2], color=color, linewidth=2)
        ax_xz.scatter(trace[0, 0], trace[0, 2], color=color, marker="o", s=20)
        ax_xz.scatter(trace[-1, 0], trace[-1, 2], color=color, marker="x", s=40)
        ax_xz.scatter(target[0], target[2], color="#d62728", marker="*", s=80)

    ax_3d.set_title("3D End-Effector Path")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.legend(loc="best", fontsize=8)

    ax_xy.set_title("Top View (x/y)")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.axis("equal")

    ax_xz.set_title("Side View (x/z)")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.axis("equal")

    fig.suptitle("Rotom End-Effector Reaching Rollout", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rollout plot to {output_path}")


def run_random_policy(args: argparse.Namespace) -> None:
    env = make_env(args.xml_path, args.max_episode_steps)
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


def run_env_check(args: argparse.Namespace) -> None:
    env = make_env(args.xml_path, args.max_episode_steps)
    check_env(env, warn=True, skip_render_check=True)
    env.close()
    print("Gymnasium / Stable-Baselines3 environment check passed.")


def add_common_arguments(parser: argparse.ArgumentParser, *, include_model_path: bool, default_xml_path: Path) -> None:
    parser.add_argument("--xml-path", type=Path, default=default_xml_path)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    if include_model_path:
        parser.add_argument("--model-path", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and inspect PPO on the Rotom reach task.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train PPO on the Rotom reach task.")
    add_common_arguments(train_parser, include_model_path=True, default_xml_path=DEFAULT_XML_PATH)
    train_parser.add_argument("--timesteps", type=int, default=50_000)
    train_parser.add_argument("--n-envs", type=int, default=4)
    train_parser.add_argument("--rollout-steps", type=int, default=512)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--ent-coef", type=float, default=0.0)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved PPO policy.")
    add_common_arguments(eval_parser, include_model_path=True, default_xml_path=DEFAULT_XML_PATH)
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--stochastic", action="store_true")
    eval_parser.set_defaults(func=evaluate)

    play_parser = subparsers.add_parser("play", help="Watch the policy in the MuJoCo viewer.")
    add_common_arguments(play_parser, include_model_path=True, default_xml_path=DEFAULT_SCENE_PATH)
    play_parser.add_argument("--episodes", type=int, default=3)
    play_parser.set_defaults(func=play)

    plot_parser = subparsers.add_parser("plot-rollout", help="Save rollout traces to a PNG.")
    add_common_arguments(plot_parser, include_model_path=True, default_xml_path=DEFAULT_XML_PATH)
    plot_parser.add_argument("--episodes", type=int, default=3)
    plot_parser.add_argument("--output-path", type=Path, default=None)
    plot_parser.set_defaults(func=plot_rollout)

    random_parser = subparsers.add_parser("random", help="Smoke test the environment with random actions.")
    add_common_arguments(random_parser, include_model_path=False, default_xml_path=DEFAULT_XML_PATH)
    random_parser.add_argument("--steps", type=int, default=25)
    random_parser.set_defaults(func=run_random_policy)

    check_parser = subparsers.add_parser("check-env", help="Run Gymnasium environment validation.")
    add_common_arguments(check_parser, include_model_path=False, default_xml_path=DEFAULT_XML_PATH)
    check_parser.set_defaults(func=run_env_check)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
