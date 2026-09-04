"""Mini-SWE-Agent v2 adapter for paired native-tool EvilGenie rollouts."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Mapping, override

import yaml
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from settings import NATIVE_BASH_TOOLS, stable_sampling_seed
from olmo_harbor.native_profile import native_profile
from olmo_harbor.native_contract import (
    completion_evidence,
    split_internal_task_instruction,
    stopping_condition,
)
from olmo_harbor.native_examples import TOOL_FORMAT_REMINDER
from olmo_harbor.completed_commands import mini_swe_completed_commands
from olmo_harbor.selected_telemetry import (
    TELEMETRY_CONTRACT,
    compact_json_bytes,
    selected_turn_statistics,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    return str(value)


def extract_rollout_details(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract telemetry from every persisted model response, regardless of role.

    Mini-SWE attaches a response that fails action parsing to the generated
    user-role FormatError message. Restricting extraction to assistant messages
    therefore drops valid model tokens, most notably finish_reason=length turns.
    """
    prompt_ids: list[list[int]] = []
    completion_ids: list[list[int]] = []
    selected_logprobs: list[list[float]] = []
    turn_statistics: list[dict[str, Any]] = []
    response_metadata: list[dict[str, Any]] = []

    for message_index, message in enumerate(trajectory.get("messages") or []):
        if not isinstance(message, Mapping):
            continue
        extra = message.get("extra")
        response = extra.get("response") if isinstance(extra, Mapping) else None
        response = _plain(response)
        if not isinstance(response, Mapping):
            continue
        choices = response.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], Mapping)
        ):
            continue
        choice = choices[0]
        logprobs = choice.get("logprobs")
        content = logprobs.get("content") if isinstance(logprobs, Mapping) else None
        if not isinstance(content, list) or not content:
            raise RuntimeError(
                f"Mini-SWE response at message {message_index} contains no token logprobs"
            )
        if not all(isinstance(item, Mapping) for item in content):
            raise RuntimeError(
                f"Mini-SWE response at message {message_index} has malformed token logprobs"
            )
        normalized = [dict(item) for item in content]
        if any(
            not isinstance(item.get("logprob"), (int, float)) for item in normalized
        ):
            raise RuntimeError(
                f"Mini-SWE response at message {message_index} lacks selected logprobs"
            )
        if any(item.get("top_logprobs") not in (None, []) for item in normalized):
            raise RuntimeError(
                f"Mini-SWE response at message {message_index} contains forbidden top-k candidates"
            )

        response_prompt_ids = response.get("prompt_token_ids")
        if not (
            isinstance(response_prompt_ids, list)
            and response_prompt_ids
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in response_prompt_ids
            )
        ):
            raise RuntimeError(
                f"Mini-SWE response at message {message_index} lacks prompt token IDs"
            )
        fields = choice.get("provider_specific_fields")
        token_ids = fields.get("token_ids") if isinstance(fields, Mapping) else None
        if not (
            isinstance(token_ids, list)
            and token_ids
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in token_ids
            )
        ):
            raise RuntimeError(
                f"Mini-SWE response at message {message_index} lacks completion token IDs"
            )
        if len(token_ids) != len(normalized):
            raise RuntimeError(
                f"Mini-SWE response at message {message_index} has misaligned completion telemetry"
            )

        selected = [float(item["logprob"]) for item in normalized]
        selected_logprobs.append(selected)
        prompt_ids.append(list(response_prompt_ids))
        completion_ids.append(list(token_ids))
        usage = response.get("usage")
        message_payload = choice.get("message")
        tool_calls = (
            message_payload.get("tool_calls")
            if isinstance(message_payload, Mapping)
            else None
        )
        turn_statistics.append(
            selected_turn_statistics(
                selected,
                finish_reason=choice.get("finish_reason"),
                stop_reason=choice.get("stop_reason"),
                output_truncated=choice.get("finish_reason") == "length",
                tool_calls=tool_calls,
            )
        )
        response_metadata.append(
            {
                "id": response.get("id"),
                "model": response.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "output_truncated": choice.get("finish_reason") == "length",
                "trajectory_message_index": message_index,
                "trajectory_message_role": message.get("role"),
                "interrupt_type": extra.get("interrupt_type")
                if isinstance(extra, Mapping)
                else None,
                "trajectory_timestamp": extra.get("timestamp")
                if isinstance(extra, Mapping)
                else None,
                "generation_elapsed_msec": extra.get("generation_elapsed_msec")
                if isinstance(extra, Mapping)
                else None,
                "tool_elapsed_msec": extra.get("tool_elapsed_msec")
                if isinstance(extra, Mapping)
                else None,
                "usage": _plain(usage) if isinstance(usage, Mapping) else None,
                "created": response.get("created"),
                "service_tier": response.get("service_tier"),
                "system_fingerprint": response.get("system_fingerprint"),
                "raw_response_bytes": len(compact_json_bytes(response)),
            }
        )

    if not selected_logprobs:
        raise RuntimeError("Mini-SWE native trajectory contains no token logprobs")
    detail: dict[str, Any] = {
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "logprobs": selected_logprobs,
        "extra": {
            # RolloutDetail.extra is a per-turn column store.  Keep every
            # value list-shaped so Harbor's Pydantic serializer can validate
            # the real result artifact without warnings or coercion.
            "telemetry_contract": [TELEMETRY_CONTRACT] * len(turn_statistics),
            "turn_statistics": turn_statistics,
            "response_metadata": response_metadata,
        },
    }
    return [detail]


class SeededMiniSweAgent(MiniSweAgent):
    """Pinned Mini-SWE v2 with task-paired seeds and mandatory logprob capture."""

    def __init__(
        self,
        *args: Any,
        condition_id: str,
        candidate_index: int,
        pool_seed: int = 20260830,
        top_p: float = 0.95,
        max_output_tokens: int = 4096,
        max_agent_steps: int = 20,
        wall_clock_seconds: int = 1800,
        **kwargs: Any,
    ) -> None:
        profile = native_profile(condition_id, candidate_index, pool_seed)
        self._identity = {
            **profile["identity"],
            "harness_id": "mini_swe_native",
            "harness_version": "2.4.6",
        }
        self._candidate_index = candidate_index
        self._pool_seed = pool_seed
        self._rollout_config = {
            "agent": {
                "system_template": profile["system_prompt"],
                "instance_template": profile["instance_prompt"],
                "step_limit": max_agent_steps,
                "wall_time_limit_seconds": wall_clock_seconds,
                "max_consecutive_format_errors": 0,
            },
            "model": {
                "format_error_template": TOOL_FORMAT_REMINDER,
                "model_kwargs": {
                    "temperature": profile["temperature"],
                    "top_p": top_p,
                    "max_tokens": max_output_tokens,
                    "logprobs": True,
                    "top_logprobs": 0,
                    "drop_params": False,
                    "num_retries": 0,
                    "tool_choice": "auto",
                    "extra_body": {
                        "return_token_ids": True,
                        "return_tokens_as_token_ids": True,
                        "include_reasoning": True,
                    },
                },
            },
        }
        self._task_id: str | None = None
        self._sampling_seed: int | None = None
        self._example_enabled = profile["identity"]["model_family"] == "instruct"
        super().__init__(*args, config=self._rollout_config, max_tokens=None, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        """Install mini-swe and verify the actual pinned tool schema in situ."""
        await super().install(environment)
        expected = json.dumps(
            NATIVE_BASH_TOOLS[0], sort_keys=True, separators=(",", ":")
        )
        verification_code = (
            "import json; "
            "from minisweagent.models.utils.actions_toolcall import BASH_TOOL; "
            f"expected=json.loads({expected!r}); "
            "assert BASH_TOOL == expected, "
            "f'mini-swe-agent bash schema drift: {BASH_TOOL!r}'"
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
                'else export PATH="$HOME/.local/bin:$PATH"; fi; '
                'mini_bin="$(command -v mini-swe-agent)"; '
                'mini_python="$(sed -n \'1s/^#!//p\' "$mini_bin")"; '
                f'"$mini_python" -c {shlex.quote(verification_code)}'
            ),
        )
        runtime_source = (ROOT / "olmo_harbor/mini_swe_native_runtime.py").read_text(
            encoding="utf-8"
        )
        install_runtime_code = (
            "import pathlib,site; "
            "target=pathlib.Path(site.getsitepackages()[0])/'evilgenie_native_v2.py'; "
            f"target.write_text({runtime_source!r}, encoding='utf-8')"
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
                'else export PATH="$HOME/.local/bin:$PATH"; fi; '
                'mini_bin="$(command -v mini-swe-agent)"; '
                'mini_python="$(sed -n \'1s/^#!//p\' "$mini_bin")"; '
                f'"$mini_python" -c {shlex.quote(install_runtime_code)}'
            ),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._task_id, visible_instruction = split_internal_task_instruction(
            instruction
        )
        self._sampling_seed = stable_sampling_seed(
            self._pool_seed, self._task_id, self._candidate_index
        )
        config = json.loads(json.dumps(self._rollout_config))
        config["model"]["model_kwargs"]["seed"] = self._sampling_seed
        config["agent"]["agent_class"] = (
            "evilgenie_native_v2.ExampleNativeSubmissionAgent"
            if self._example_enabled
            else "evilgenie_native_v2.NativeSubmissionAgent"
        )
        config["environment"] = {
            "environment_class": "evilgenie_native_v2.NativeSubmissionEnvironment"
        }
        self._config_yaml = yaml.safe_dump(config, sort_keys=False)
        await super().run(visible_instruction, environment, context)

    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                **self._identity,
                "task_id": self._task_id,
                "sampling_seed": self._sampling_seed,
                "native_function_calling": True,
                "tool_call_parser": self._identity.get(
                    "tool_call_parser", "olmo3_strict"
                ),
                "tool_choice": self._identity.get("tool_choice", "auto"),
                "reasoning_parser": self._identity.get(
                    "reasoning_parser",
                    "olmo3" if self._identity.get("model_family") == "think" else None,
                ),
                "telemetry_contract": TELEMETRY_CONTRACT,
                "top_logprobs": 0,
                "top_k_candidate_arrays": 0,
                "top_k_bytes": 0,
                "request_retry_count": 0,
            }
        )
        context.metadata = metadata
        path = self.logs_dir / "mini-swe-agent.trajectory.json"
        if not path.is_file():
            # Preserve an earlier environment or agent exception when no trajectory exists.
            return
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        completed_commands = mini_swe_completed_commands(trajectory)
        evidence = completion_evidence(completed_commands)
        exit_status = str((trajectory.get("info") or {}).get("exit_status") or "")
        termination = stopping_condition(
            completion_detected=evidence["completion_detected"],
            wall_clock_limit=exit_status == "TimeExceeded",
            step_limit=exit_status == "LimitsExceeded",
            exception=exit_status
            not in {"", "Submitted", "TimeExceeded", "LimitsExceeded"},
        )
        metadata.update(
            {
                "completion_detected": evidence["completion_detected"],
                "completion_evidence": evidence,
                "stopping_condition": termination,
                "harness_exit_status": exit_status,
            }
        )
        details = extract_rollout_details(trajectory)
        context.rollout_details = details
        metadata["logprob_turn_count"] = len(details[0]["logprobs"])
        response_metadata = details[0]["extra"]["response_metadata"]
        metadata["generation_times_msec"] = [
            float(item["generation_elapsed_msec"])
            for item in response_metadata
            if isinstance(item.get("generation_elapsed_msec"), (int, float))
        ]
        metadata["tool_times_msec"] = [
            float(item["tool_elapsed_msec"])
            for item in response_metadata
            if isinstance(item.get("tool_elapsed_msec"), (int, float))
        ]
        for message in trajectory.get("messages") or []:
            if not isinstance(message, dict):
                continue
            extra = message.get("extra")
            if not isinstance(extra, dict) or "response" not in extra:
                continue
            response = extra.pop("response")
            response = _plain(response)
            if not isinstance(response, Mapping):
                continue
            choices = response.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else {}
            response_message = (
                choice.get("message") if isinstance(choice, Mapping) else None
            )
            extra["model_response_summary"] = {
                "id": response.get("id"),
                "model": response.get("model"),
                "finish_reason": choice.get("finish_reason")
                if isinstance(choice, Mapping)
                else None,
                "stop_reason": choice.get("stop_reason")
                if isinstance(choice, Mapping)
                else None,
                "content": response_message.get("content")
                if isinstance(response_message, Mapping)
                else None,
                "reasoning_content": (
                    response_message.get("reasoning_content")
                    or response_message.get("reasoning")
                )
                if isinstance(response_message, Mapping)
                else None,
                "tool_calls": response_message.get("tool_calls")
                if isinstance(response_message, Mapping)
                else None,
            }
        temporary_path = path.with_name(path.name + ".tmp")
        temporary_path.write_bytes(compact_json_bytes(trajectory))
        temporary_path.replace(path)
        atif_path = self.logs_dir / "trajectory.json"
        if atif_path.is_file():
            atif = json.loads(atif_path.read_text(encoding="utf-8"))
            for step in atif.get("steps") or []:
                metrics = step.get("metrics") if isinstance(step, dict) else None
                if not isinstance(metrics, dict):
                    continue
                for duplicated_key in (
                    "prompt_token_ids",
                    "completion_token_ids",
                    "logprobs",
                ):
                    metrics.pop(duplicated_key, None)
                metrics["telemetry_reference"] = (
                    "result.json:agent_result.rollout_details"
                )
            atif["telemetry_contract"] = TELEMETRY_CONTRACT
            atif_temporary = atif_path.with_name(atif_path.name + ".tmp")
            atif_temporary.write_bytes(compact_json_bytes(atif))
            atif_temporary.replace(atif_path)
        metadata["native_trajectory_sha256"] = _sha256(path)
        context.metadata = metadata
