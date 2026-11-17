"""Native Isaac Lab/RSL-RL training. --check only reads local readiness."""
import argparse
import json
import os
from pathlib import Path
import sys
from _bootstrap import ROOT
from _stage import mark
mark("03b-train-module")
from f1_beyondmimic.contract import assert_can_run, readiness, load_contract
mark("03c-contract-imported")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Explicit F1 continuation checkpoint; never latest")
    if "--check" in sys.argv:
        parser.parse_args()
        print(json.dumps(readiness(), ensure_ascii=False, indent=2))
        return
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
        print("After authorization, AppLauncher additionally accepts --headless and --device.")
        return
    launch = assert_can_run("training")  # Must precede Isaac imports and simulator launch.
    mark("04-contract")
    for key in ("num_envs", "max_iterations", "seed"):
        value = launch.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError("Explicit authorized integer launch setting required: " + key)
    os.environ["F1_BM_EXECUTION_SCOPE"] = "training"
    from isaaclab.app import AppLauncher
    mark("05-isaac-import")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args)
    mark("06-app-ready")
    env = None
    try:
        import gymnasium as gym
        import f1_beyondmimic.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        from f1_beyondmimic.utils.checkpoint import load_checkpoint
        from f1_beyondmimic.utils.runner import F1OnPolicyRunner
        from f1_beyondmimic.utils.runtime_policy import verify_runtime_checkpoint
        from datetime import datetime
        mark("07-task-modules")
        task = "Tracking-VelocityReference-F1-Proprio-v0"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        cfg.scene.num_envs, cfg.seed, agent.seed = launch["num_envs"], launch["seed"], launch["seed"]
        cfg.sim.device = args.device
        agent.device = args.device
        agent.max_iterations = launch["max_iterations"]
        agent.logger = "tensorboard"
        if args.checkpoint:
            load_checkpoint(args.checkpoint)
        log_dir = ROOT / "logs" / datetime.now().strftime("%Y%m%dT%H%M%S")
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
        mark("08-env")
        runner = F1OnPolicyRunner(env, agent.to_dict(), log_dir=str(log_dir), device=agent.device)
        if args.checkpoint:
            runner.load(str(args.checkpoint))
        dump_yaml(str(log_dir / "params/env.yaml"), cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent)
        mark("09-learn")
        runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
        final = log_dir / f"model_{runner.current_learning_iteration}.pt"
        if not final.is_file():
            runner.save(str(final))
        print(json.dumps({"checkpoint": str(final), "parity": verify_runtime_checkpoint(runner.alg.policy, final)}))
    finally:
        if env is not None:
            env.close()
        app.app.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
