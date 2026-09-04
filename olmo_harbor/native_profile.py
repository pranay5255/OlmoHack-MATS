"""The two prompt and inference configurations used for Experiment 2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from settings import MODEL_SPECS
from olmo_harbor.native_contract import COMPLETION_MATCH, CONTRACT_VERSION
from olmo_harbor.native_examples import EXAMPLE_MODE
from olmo_harbor.released_chat_templates import (
    compose_recommended_system_prompt,
    verified_template_bytes,
)
from olmo_harbor.selected_telemetry import TELEMETRY_CONTRACT

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def native_profile(
    condition_id: str, candidate_index: int, pool_seed: int
) -> dict[str, Any]:
    if condition_id not in MODEL_SPECS or candidate_index not in (0, 1):
        raise ValueError(
            "Experiment 2 requires a catalog checkpoint and candidate 0 or 1"
        )
    family, stage = condition_id.split("_")
    spec = MODEL_SPECS[condition_id]
    prompts = ROOT / "harbor/prompts"
    if family == "think":
        system_path = prompts / "think-native-coding-system-v1.txt"
        instance_path = prompts / "think-native-coding-instance-v1.txt"
        template = (prompts / "olmo3-think-native-multiturn-v3.jinja").read_bytes()
        if (
            digest(template)
            != "8cf820848852adb5bbd4964f4aa16ebac44c4ba0bc9331cef2ba8aef646f542b"
        ):
            raise ValueError("Think chat template hash mismatch")
        system = system_path.read_text().strip()
        temperature = 0.0
        parser = "olmo3"
        example_mode = "template_inline_sequential_two_call"
    else:
        system_path = prompts / "native-bash-system.txt"
        instance_path = prompts / "native-bash-instance.txt"
        template = verified_template_bytes(condition_id, prompts)
        system = compose_recommended_system_prompt(
            condition_id, system_path.read_text()
        )
        temperature = 0.6
        parser = "olmo3_strict"
        example_mode = EXAMPLE_MODE
    return {
        "system_prompt": system,
        "instance_prompt": instance_path.read_text(),
        "temperature": temperature,
        "identity": {
            "condition_id": condition_id,
            "model_family": family,
            "model_stage": stage,
            "model_id": spec["model_id"],
            "model_revision": spec["revision"],
            "candidate_index": candidate_index,
            "pool_seed": pool_seed,
            "trial_kind": "experiment2_rollout",
            "native_contract_version": CONTRACT_VERSION,
            "completion_match": COMPLETION_MATCH,
            "tool_call_parser": parser,
            "tool_choice": "auto",
            "reasoning_parser": "olmo3" if family == "think" else None,
            "sampling_temperature": temperature,
            "system_prompt_sha256": digest(system_path.read_bytes()),
            "instance_prompt_sha256": digest(instance_path.read_bytes()),
            "composed_system_prompt_sha256": digest(system.encode()),
            "chat_template_sha256": digest(template),
            "prompt_example_mode": example_mode,
            "prompt_example_count": 2,
            "prompt_examples_injected": True,
            "telemetry_contract": TELEMETRY_CONTRACT,
            "request_retry_count": 0,
        },
    }
