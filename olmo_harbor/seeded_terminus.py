"""A Terminus-2 adapter that preserves the experiment's task-level seeds.

Harbor expands a job as task x agent x attempt, but its stock Terminus-2 agent
does not derive a different sampling seed for each task.  This adapter computes
the per-task sampling seed from the stable task marker embedded in
instruction.md and records the derivation in AgentContext metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from olmo_harbor.plain_json import RepositoryTerminusJSONParser


TASK_MARKER = re.compile(r"^Harbor task ID:\s*(lcb_[a-z0-9_]+)\s*$", re.MULTILINE)


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for sampling seeds."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_sampling_seed(
    pool_seed: int,
    task_id: str,
    condition_id: str,
    candidate_index: int,
) -> int:
    payload = canonical_json([pool_seed, task_id, condition_id, candidate_index])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


class SeededTerminus2(Terminus2):
    """Terminus-2 with deterministic per-task candidate sampling."""

    def __init__(
        self,
        *args: Any,
        condition_id: str,
        model_family: str,
        model_stage: str,
        checkpoint_model_id: str,
        checkpoint_revision: str,
        prompt_sha256: str,
        candidate_index: int,
        pool_seed: int,
        **kwargs: Any,
    ) -> None:
        if condition_id not in {
            "think_sft", "think_dpo", "think_rlvr",
            "instruct_sft", "instruct_dpo", "instruct_rlvr",
        }:
            raise ValueError(f"Unsupported experiment condition: {condition_id!r}")
        if condition_id != f"{model_family}_{model_stage}":
            raise ValueError("condition axes do not match condition_id")
        if model_stage not in {"sft", "dpo", "rlvr"}:
            raise ValueError(f"Unsupported experiment stage: {model_stage!r}")
        if not 0 <= candidate_index < 4:
            raise ValueError("candidate_index must be in [0, 3] for the pilot")
        if not re.fullmatch(r"[0-9a-f]{40}", checkpoint_revision):
            raise ValueError("checkpoint_revision must be a 40-character commit SHA")
        prompt_path = self._get_prompt_template_path()
        actual_prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if actual_prompt_sha256 != prompt_sha256:
            raise ValueError("frozen Terminus prompt hash mismatch")
        self._condition_id = condition_id
        self._model_family = model_family
        self._experiment_stage = model_stage
        self._checkpoint_model_id = checkpoint_model_id
        self._checkpoint_revision = checkpoint_revision
        self._prompt_sha256 = prompt_sha256
        self._candidate_index = candidate_index
        self._pool_seed = pool_seed
        super().__init__(*args, **kwargs)

    def _get_parser(self) -> RepositoryTerminusJSONParser:
        if self._parser_name != "json":
            raise ValueError("the experiment requires the repository plain-JSON parser")
        return RepositoryTerminusJSONParser()

    def _get_prompt_template_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "harbor/prompts/terminus-json-plain.txt"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        match = TASK_MARKER.search(instruction)
        if match is None:
            raise ValueError(
                "Task instruction is missing the required 'Harbor task ID:' marker"
            )
        task_id = match.group(1)
        sampling_seed = stable_sampling_seed(
            self._pool_seed,
            task_id,
            self._condition_id,
            self._candidate_index,
        )

        # Passed to LiteLLM on every generation and therefore to the OpenAI-
        # compatible vLLM endpoint. Harbor resets conversation state itself.
        self._llm_call_kwargs["seed"] = sampling_seed

        try:
            await super().run(instruction, environment, context)
        finally:
            # Terminus2 writes its own metadata in its finally block. Extend it
            # afterwards so these fields survive successes, timeouts, and errors.
            metadata = dict(context.metadata or {})
            metadata.update(
                {
                    "task_id": task_id,
                    "condition_id": self._condition_id,
                    "model_family": self._model_family,
                    "model_stage": self._experiment_stage,
                    "model_id": self._checkpoint_model_id,
                    "model_revision": self._checkpoint_revision,
                    "candidate_index": self._candidate_index,
                    "pool_seed": self._pool_seed,
                    "sampling_seed": sampling_seed,
                    "prompt_sha256": self._prompt_sha256,
                    "native_function_calling": False,
                }
            )
            context.metadata = metadata
