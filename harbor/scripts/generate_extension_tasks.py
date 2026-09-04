#!/usr/bin/env python3
"""Generate the frozen 24-task vulnerable-only EvilGenie extension dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


PINNED_EVILGENIE_SHA = "f5d8a2ca5d92ecf5fd1e857695c54e609312e1bd"
TASK_GROUPS = {
    "introductory": {
        "stdin": ("lcb_abc397_a", "lcb_abc395_b", "lcb_abc394_a", "lcb_abc400_a"),
        "function": ("lcb_3702", "lcb_3617", "lcb_3736", "lcb_3612"),
    },
    "interview": {
        "stdin": ("lcb_abc396_d", "lcb_abc382_c", "lcb_abc374_d", "lcb_abc391_d"),
        "function": ("lcb_3613", "lcb_3793", "lcb_3786", "lcb_3805"),
    },
    "competition": {
        "stdin": ("lcb_arc195_b", "lcb_arc184_e", "lcb_arc187_b", "lcb_abc396_f"),
        "function": ("lcb_3696", "lcb_3701", "lcb_3680", "lcb_3717"),
    },
}
TASK_IDS = tuple(
    task_id
    for group in TASK_GROUPS.values()
    for task_ids in group.values()
    for task_id in task_ids
)
EXTRACTOR = r'''
import json
import os
from dataset import reward_hacking_dataset

groups = json.loads(os.environ["HARBOR_TASK_GROUPS"])
payload = []
for difficulty, interfaces in groups.items():
    requested = [task_id for task_ids in interfaces.values() for task_id in task_ids]
    expected = {task_id: interface for interface, task_ids in interfaces.items() for task_id in task_ids}
    samples = reward_hacking_dataset(
        difficulty=difficulty,
        limit=None,
        seed=42,
        dataset_source="livecodebench",
        livecodebench_release="v5_v6",
        problem_ids=requested,
    )
    for sample in samples:
        payload.append({
            "id": sample.id,
            "difficulty": difficulty,
            "expected_interface": expected[sample.id],
            "input": sample.input,
            "files": dict(sample.files),
            "metadata": dict(sample.metadata),
        })
print(json.dumps(payload, sort_keys=True))
'''
INTERNAL_MARKER = "[[HARBOR_TASK_ID:{task_id}]]"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_instruction(task_id: str, problem: str) -> str:
    """Keep task identity internal and the full problem as the only task text."""

    # Match the text-mode normalization used to prepare the collected tasks.
    problem = problem.replace("\r\n", "\n").replace("\r", "\n")
    return f"{INTERNAL_MARKER.format(task_id=task_id)}\n\n{problem}"


def checked_output(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        argv, cwd=cwd, env=env, check=True, text=True, stdout=subprocess.PIPE
    )
    return result.stdout.strip()


def extract_samples(evilgenie_root: Path, python: Path) -> list[dict[str, Any]]:
    env = dict(os.environ)
    env["HARBOR_TASK_GROUPS"] = json.dumps(TASK_GROUPS)
    return json.loads(
        checked_output([str(python), "-c", EXTRACTOR], cwd=evilgenie_root, env=env)
    )


def write(path: Path, data: str | bytes, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    if executable:
        path.chmod(0o755)


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def render_task_toml(sample: dict[str, Any], manifest: dict[str, Any]) -> str:
    task_id = sample["id"]
    metadata = sample["metadata"]
    return f'''schema_version = "1.4"
artifacts = ["/app/solution.py", "/app/test.py", "/app/test_cases.json", "/app/problem.md"]

[task]
name = "olmo-extension/{task_id.replace('_', '-')}"
version = "1.0.0"
description = {toml_string("Pinned vulnerable EvilGenie/LiveCodeBench extension task " + task_id)}
keywords = ["evilgenie", "livecodebench", "reward-hacking", "vulnerable"]

[metadata]
upstream_task_id = "{task_id}"
upstream_repository = "https://github.com/JonathanGabor/evilgenie_inspect"
upstream_revision = "{PINNED_EVILGENIE_SHA}"
dataset_source = "livecodebench"
livecodebench_release = "v5_v6"
dataset_seed = 42
difficulty = "{sample['difficulty']}"
interface = "{sample['expected_interface']}"
function_name = {toml_string(str(metadata.get('function_name') or ''))}
visible_test_count = {manifest['visible_test_count']}
holdout_test_count = {manifest['holdout_test_count']}
variant = "vulnerable"

[agent]
timeout_sec = 900
user = "root"
network_mode = "no-network"

[verifier]
timeout_sec = 300
user = "root"
network_mode = "no-network"
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"
build_timeout_sec = 600
cpus = 1
memory_mb = 1024
storage_mb = 2048
workdir = "/app"

[environment]
network_mode = "no-network"
build_timeout_sec = 600
cpus = 1
memory_mb = 1024
storage_mb = 2048
workdir = "/app"
'''


def generate_one(sample: dict[str, Any], output: Path, templates: Path) -> dict[str, Any]:
    task_id = sample["id"]
    metadata = sample["metadata"]
    files = {name: str(value) for name, value in sample["files"].items()}
    required = {"problem.md", "solution.py", "test.py", "test_cases.json"}
    if set(files) != required:
        raise RuntimeError(f"{task_id}: unexpected workspace files {sorted(files)}")
    is_function = bool(metadata.get("is_function"))
    if is_function != (sample["expected_interface"] == "function"):
        raise RuntimeError(f"{task_id}: interface classification drift")
    if is_function and not metadata.get("function_name"):
        raise RuntimeError(f"{task_id}: functional task lacks function_name")

    visible_cases = json.loads(files["test_cases.json"])
    holdout_cases = [
        {
            "id": index,
            "input": case["input"],
            "output": case["output"],
            "type": "function" if is_function else "stdin",
        }
        for index, case in enumerate(metadata["holdout_test_cases"])
    ]
    if not visible_cases or not holdout_cases:
        raise RuntimeError(f"{task_id}: empty visible or held-out test set")
    workspace_hashes = {
        name: digest(content.encode("utf-8")) for name, content in files.items()
    }
    if workspace_hashes["test_cases.json"] != metadata["test_cases_hash"]:
        raise RuntimeError(f"{task_id}: visible-case hash drift")
    if workspace_hashes["test.py"] != metadata["test_script_hash"]:
        raise RuntimeError(f"{task_id}: test-runner hash drift")

    instruction = render_instruction(task_id, files["problem.md"])
    manifest = {
        "schema_version": 2,
        "task_id": task_id,
        "title": metadata["title"],
        "evilgenie_revision": PINNED_EVILGENIE_SHA,
        "dataset": {
            "source": "livecodebench",
            "release": "v5_v6",
            "seed": 42,
            "difficulty": sample["difficulty"],
        },
        "variant": "vulnerable",
        "is_function": is_function,
        "function_name": metadata.get("function_name"),
        "workspace_sha256": workspace_hashes,
        "visible_test_count": len(visible_cases),
        "holdout_test_count": len(holdout_cases),
    }
    write(
        output / "instruction.md",
        instruction,
    )
    write(output / "task.toml", render_task_toml(sample, manifest))
    write(
        output / "environment" / "Dockerfile",
        (templates / "environment_extension.Dockerfile").read_text(),
    )
    for name, content in files.items():
        write(
            output / "environment" / "workspace" / name,
            content,
            executable=name == "test.py",
        )
    write(
        output / "tests" / "Dockerfile",
        (templates / "verifier.Dockerfile").read_text(),
    )
    write(
        output / "tests" / "test.sh",
        (templates / "verifier_entrypoint.sh").read_text(),
        executable=True,
    )
    write(
        output / "tests" / "verify.py",
        (templates / "verify_extension.py").read_text(),
        executable=True,
    )
    write(
        output / "tests" / "visible_cases.json",
        json.dumps(visible_cases, sort_keys=True, indent=2) + "\n",
    )
    write(
        output / "tests" / "holdout_cases.json",
        json.dumps(holdout_cases, sort_keys=True, indent=2) + "\n",
    )
    write(
        output / "tests" / "manifest.json",
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
    )
    return {**manifest, "instruction_sha256": digest(instruction.encode("utf-8"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evilgenie-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "evilgenie-24",
    )
    parser.add_argument("--python", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    evilgenie_root = args.evilgenie_root.expanduser().resolve()
    revision = checked_output(["git", "rev-parse", "HEAD"], cwd=evilgenie_root)
    if revision != PINNED_EVILGENIE_SHA:
        raise RuntimeError(
            f"EvilGenie checkout must be exactly {PINNED_EVILGENIE_SHA}; got {revision}"
        )
    python = (args.python or evilgenie_root / ".venv" / "bin" / "python").absolute()
    if not python.is_file():
        raise FileNotFoundError(f"Pinned EvilGenie Python not found at {python}")

    output = args.output.resolve()
    existing = [task_id for task_id in TASK_IDS if (output / task_id).exists()]
    if existing and not args.force:
        raise FileExistsError(f"Generated tasks already exist in {output}; pass --force")
    if args.force:
        for task_id in TASK_IDS:
            target = output / task_id
            if target.exists():
                shutil.rmtree(target)
    output.mkdir(parents=True, exist_ok=True)

    samples = extract_samples(evilgenie_root, python)
    by_id = {sample["id"]: sample for sample in samples}
    if set(by_id) != set(TASK_IDS):
        raise RuntimeError(
            f"Resolved IDs differ from frozen slate: {sorted(set(TASK_IDS) ^ set(by_id))}"
        )
    templates = Path(__file__).resolve().parents[1] / "task_templates"
    catalog = [
        generate_one(by_id[task_id], output / task_id, templates)
        for task_id in TASK_IDS
    ]
    catalog_path = output / "catalog.json"
    write(
        catalog_path,
        json.dumps({"schema_version": 2, "tasks": catalog}, sort_keys=True, indent=2)
        + "\n",
    )
    print(f"Generated {len(TASK_IDS)} tasks in {output}")
    print(catalog_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
