"""Generate or execute one condition's Harbor job for either experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from settings import MODEL_SPECS

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harbor/scripts"))
from generate_tasks import TASK_IDS as PILOT_TASKS
from generate_extension_tasks import TASK_IDS as NATIVE_TASKS


def build_job(
    experiment: int,
    condition: str,
    harness: str,
    candidates: list[int],
    tasks: list[str],
    api_base: str,
    jobs_dir: str,
    environment: str = "modal",
) -> dict:
    spec = MODEL_SPECS[condition]
    maximum = 4 if experiment == 1 else 2
    expected_tasks = PILOT_TASKS if experiment == 1 else NATIVE_TASKS
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(c not in range(maximum) for c in candidates)
    ):
        raise ValueError(f"Candidates must be unique integers in [0, {maximum - 1}]")
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or not set(tasks) <= set(expected_tasks)
    ):
        raise ValueError("Tasks must be a unique subset of the selected experiment")
    if experiment == 1 and harness != "terminus":
        raise ValueError("Experiment 1 uses only Terminus")
    endpoint = urlparse(api_base)
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or endpoint.path.rstrip("/") != "/v1"
    ):
        raise ValueError("Supply the Modal endpoint as https://HOST/v1")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError(
            "Do not embed credentials or query parameters in the endpoint URL"
        )
    family, stage = condition.split("_")
    wall_seconds = 900 if experiment == 1 else 1800
    agents = []
    for candidate in candidates:
        kwargs = {"condition_id": condition, "candidate_index": candidate}
        if experiment == 1:
            kwargs.update(
                {
                    "model_family": family,
                    "model_stage": stage,
                    "checkpoint_model_id": spec["model_id"],
                    "checkpoint_revision": spec["revision"],
                    "prompt_sha256": hashlib.sha256(
                        (ROOT / "harbor/prompts/terminus-json-plain.txt").read_bytes()
                    ).hexdigest(),
                    "pool_seed": 20260819,
                    "parser_name": "json",
                    "temperature": 0.7,
                    "trajectory_config": {
                        "raw_content": False,
                        "linear_history": False,
                    },
                }
            )
            import_path = "olmo_harbor.seeded_terminus:SeededTerminus2"
        elif harness == "mini-swe":
            kwargs.update({"version": "2.4.6", "cost_limit": "0"})
            import_path = "olmo_harbor.seeded_mini_swe:SeededMiniSweAgent"
        else:
            import_path = "olmo_harbor.native_terminus:SeededNativeTerminus2"
        if harness == "terminus":
            kwargs.update(
                {
                    "api_base": api_base,
                    "max_turns": 30 if experiment == 1 else 20,
                    "enable_summarize": False,
                    "collect_rollout_details": True,
                    "suppress_max_turns_warning": True,
                    "model_info": {
                        "max_tokens": 65536,
                        "max_input_tokens": 61440,
                        "max_output_tokens": 4096,
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "llm_call_kwargs": {"top_p": 0.95, "max_tokens": 4096},
                }
            )
            if experiment == 2:
                kwargs["record_terminal_session"] = True
                kwargs["llm_call_kwargs"].update(
                    {
                        "tool_choice": "auto",
                        "top_logprobs": 0,
                        "num_retries": 0,
                        "extra_body": {
                            "return_token_ids": True,
                            "return_tokens_as_token_ids": True,
                            "include_reasoning": True,
                        },
                    }
                )
        agent = {
            "import_path": import_path,
            "model_name": f"openai/{spec['model_id']}@{spec['revision']}",
            "n_concurrent": 1,
            "concurrency_group": condition,
            "override_timeout_sec": wall_seconds,
            "extra_allowed_hosts": [endpoint.hostname],
            "env": {
                "OPENAI_API_KEY": "${OPENAI_API_KEY}",
                "OPENAI_BASE_URL": api_base,
                "OPENAI_API_BASE": api_base,
            },
            "kwargs": kwargs,
        }
        if harness == "mini-swe":
            agent["env"]["MSWEA_API_KEY"] = "${OPENAI_API_KEY}"
            agent["include_logs"] = [
                "mini-swe-agent.trajectory.json",
                "trajectory.json",
                "mini-swe-agent.txt",
            ]
        agents.append(agent)
    env = {"type": "docker", "force_build": False, "delete": True}
    if environment == "modal":
        env = {
            "import_path": "olmo_harbor.modal_dynamic:DynamicModalEnvironment",
            "force_build": False,
            "delete": True,
            "kwargs": {
                "app_name": f"olmohack-mats-experiment{experiment}-sandboxes",
                "sandbox_timeout_secs": 2700,
                "sandbox_idle_timeout_secs": 2100,
            },
        }
    dataset = "evilgenie-pilot" if experiment == 1 else "evilgenie-24"
    selector = hashlib.sha256(json.dumps([tasks, candidates]).encode()).hexdigest()[:10]
    return {
        "job_name": f"experiment{experiment}-{condition}-{harness}-{selector}",
        "jobs_dir": jobs_dir,
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "retry": {"max_retries": 1 if experiment == 1 else 0},
        "environment": env,
        "agents": agents,
        "datasets": [
            {"path": str(ROOT / "harbor/datasets" / dataset), "task_names": tasks}
        ],
        "metrics": [
            {
                "type": "uv-script",
                "kwargs": {"script_path": str(ROOT / "harbor/metric.py")},
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=(1, 2), required=True)
    parser.add_argument("--condition", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument(
        "--harness", choices=("terminus", "mini-swe"), default="terminus"
    )
    parser.add_argument(
        "--candidates",
        help="Comma-separated candidate indices; defaults to the complete pool",
    )
    parser.add_argument(
        "--tasks", help="Comma-separated task IDs; defaults to the full slate"
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment", choices=("modal", "docker"), default="modal")
    parser.add_argument(
        "--run", action="store_true", help="Execute the generated job using Harbor"
    )
    args = parser.parse_args()
    candidates = list(range(4 if args.experiment == 1 else 2))
    if args.candidates:
        candidates = [int(value) for value in args.candidates.split(",")]
    tasks = list(PILOT_TASKS if args.experiment == 1 else NATIVE_TASKS)
    if args.tasks:
        tasks = args.tasks.split(",")
    output = args.output_dir.resolve()
    job = build_job(
        args.experiment,
        args.condition,
        args.harness,
        candidates,
        tasks,
        args.api_base,
        str(output / "raw"),
        args.environment,
    )
    if args.run:
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error(
                "--run requires OPENAI_API_KEY matching the deployment's VLLM_API_KEY"
            )
        dataset = Path(job["datasets"][0]["path"])
        missing = [
            task for task in tasks if not (dataset / task / "task.toml").is_file()
        ]
        if missing:
            parser.error(
                f"Generate the dataset first; missing tasks: {', '.join(missing)}"
            )
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "job.json"
    config_path.write_text(json.dumps(job, indent=2) + "\n")
    print(f"Prepared {len(tasks) * len(candidates)} trials: {config_path}", flush=True)
    if args.run:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = (
            str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        )
        subprocess.run(
            [
                str(Path(sys.executable).parent / "harbor"),
                "run",
                "-c",
                str(config_path),
                "--yes",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
