"""Terminus-2 loop using OLMo3 native bash tool calls with lossless logprobs."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import litellm
from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import Command
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import LLMBackend, LLMResponse
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step

from settings import NATIVE_BASH_TOOLS, stable_sampling_seed
from olmo_harbor.native_profile import native_profile
from olmo_harbor.native_contract import (
    FORMAT_REMINDER,
    completion_evidence,
    is_completion_command,
    parsed_bash_command,
    split_internal_task_instruction,
    stopping_condition,
)
from olmo_harbor.native_examples import TOOL_FORMAT_REMINDER, example_messages
from olmo_harbor.completed_commands import completed_native_bash_commands
from olmo_harbor.selected_telemetry import (
    TELEMETRY_CONTRACT,
    compact_json_bytes,
    selected_turn_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
BASH_TOOLS = NATIVE_BASH_TOOLS


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


class NativeToolLiteLLM(LiteLLM):
    """Harbor LiteLLM backend retaining selected-token-only telemetry."""

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] = [],
        response_format: dict[str, Any] | None = None,
        logging_path: Path | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del response_format
        kwargs.pop("previous_response_id", None)
        input_message = kwargs.pop(
            "_native_input_message", {"role": "user", "content": prompt}
        )
        messages = [*message_history, input_message]
        completion_kwargs: dict[str, Any] = {
            **self._build_base_kwargs(logging_path),
            "messages": messages,
            "tools": BASH_TOOLS,
            "tool_choice": "auto",
            "reasoning_effort": self._reasoning_effort,
        }
        if self._temperature is not None:
            completion_kwargs["temperature"] = self._temperature
        if self._collect_rollout_details:
            completion_kwargs["logprobs"] = True
            completion_kwargs["top_logprobs"] = 0
            completion_kwargs["extra_body"] = {
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "include_reasoning": True,
            }
        completion_kwargs["num_retries"] = 0
        extra_body = kwargs.pop("extra_body", None)
        if isinstance(extra_body, Mapping):
            completion_kwargs["extra_body"] = {
                **completion_kwargs.get("extra_body", {}),
                **dict(extra_body),
            }
        completion_kwargs.update(kwargs)
        if completion_kwargs.get("top_logprobs") != 0:
            raise ValueError("selected-only telemetry requires top_logprobs=0")
        started = time.perf_counter()
        try:
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as exc:
            self._handle_litellm_error(exc)
        elapsed_msec = (time.perf_counter() - started) * 1000.0

        usage = self._extract_usage_info(response)
        prompt_ids, completion_ids = self._extract_token_ids(response)
        selected_logprobs = self._extract_logprobs(response)
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        # Preserve length-truncated generations as model outcomes. Harbor Terminus otherwise recursively retries them until the trial wall-clock timeout.
        tool_calls = _plain(message.get("tool_calls") or [])
        logprobs = choice.get("logprobs")
        logprob_content = _plain(logprobs.get("content") or []) if logprobs else []
        if not isinstance(logprob_content, list) or not logprob_content:
            raise RuntimeError("native Terminus response contains no token logprobs")
        if any(
            not isinstance(item, Mapping) or item.get("top_logprobs") not in (None, [])
            for item in logprob_content
        ):
            raise RuntimeError(
                "native Terminus response contains forbidden top-k candidates"
            )
        if not isinstance(selected_logprobs, list) or len(selected_logprobs) != len(
            completion_ids or []
        ):
            raise RuntimeError("native Terminus completion telemetry is misaligned")
        if len(selected_logprobs) != len(logprob_content):
            raise RuntimeError("native Terminus selected logprobs do not align")
        provider_extra = self._extract_provider_extra(response) or {}
        extra = {
            **provider_extra,
            "native_tool_calls": tool_calls,
            "telemetry_contract": TELEMETRY_CONTRACT,
            "turn_statistics": selected_turn_statistics(
                selected_logprobs,
                finish_reason=choice.get("finish_reason"),
                stop_reason=choice.get("stop_reason"),
                output_truncated=choice.get("finish_reason") == "length",
                tool_calls=tool_calls,
            ),
            "response_metadata": {
                "id": response.get("id"),
                "model": response.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "output_truncated": choice.get("finish_reason") == "length",
                "usage": _plain(response.get("usage")),
                "created": response.get("created"),
                "service_tier": response.get("service_tier"),
                "system_fingerprint": response.get("system_fingerprint"),
                "request_elapsed_msec": elapsed_msec,
                "raw_response_bytes": len(compact_json_bytes(_plain(response))),
            },
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": choice.get("stop_reason"),
            "output_truncated": choice.get("finish_reason") == "length",
        }
        return LLMResponse(
            content=content,
            reasoning_content=message.get("reasoning_content")
            or message.get("reasoning"),
            model_name=response.get("model"),
            usage=usage,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
            logprobs=selected_logprobs,
            extra=extra,
        )


class NativeToolChat(Chat):
    """Chat history that preserves assistant tool calls and tool results."""

    async def chat(
        self, prompt: str, logging_path: Path | None = None, **kwargs: Any
    ) -> LLMResponse:
        input_message: dict[str, Any] = {"role": "user", "content": prompt}
        if self._messages:
            last = self._messages[-1]
            calls = last.get("tool_calls") if isinstance(last, Mapping) else None
            if isinstance(calls, list) and len(calls) == 1:
                input_message = {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": prompt,
                }
        response = await self._model.call(
            prompt=prompt,
            message_history=self._messages,
            logging_path=logging_path,
            _native_input_message=input_message,
            **kwargs,
        )
        if response.usage is not None:
            self._cumulative_input_tokens += response.usage.prompt_tokens
            self._cumulative_output_tokens += response.usage.completion_tokens
            self._cumulative_cache_tokens += response.usage.cache_tokens
            self._cumulative_cost += response.usage.cost_usd
        self._accumulate_rollout_details(response)
        assistant: dict[str, Any] = {"role": "assistant", "content": response.content}
        tool_calls = (response.extra or {}).get("native_tool_calls")
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        if self._interleaved_thinking and response.reasoning_content:
            assistant["reasoning_content"] = response.reasoning_content
        self._messages.extend([input_message, assistant])
        return response


class SeededNativeTerminus2(Terminus2):
    """Terminus-2 with paired seeds and selected-token-only native telemetry."""

    def __init__(
        self,
        *args: Any,
        condition_id: str,
        candidate_index: int,
        pool_seed: int = 20260830,
        **kwargs: Any,
    ) -> None:
        profile = native_profile(condition_id, candidate_index, pool_seed)
        self._native_system_prompt = profile["system_prompt"]
        self._native_instance_prompt = profile["instance_prompt"]
        self._identity = {
            **profile["identity"],
            "harness_id": "terminus_native",
            "harness_version": "2.0.0+olmo-native1",
        }
        self._candidate_index = candidate_index
        self._pool_seed = pool_seed
        self._task_id: str | None = None
        self._sampling_seed: int | None = None
        self._example_messages = []
        if profile["identity"]["model_family"] == "instruct":
            self._example_messages = example_messages("instruct")
        self._format_reminder = TOOL_FORMAT_REMINDER.replace(
            "{{ error }}", "The response was not parsed as exactly one bash call."
        )
        kwargs["temperature"] = profile["temperature"]
        super().__init__(*args, parser_name="json", **kwargs)

    def _init_llm(
        self,
        llm_backend: LLMBackend | str,
        model_name: str,
        temperature: float | None,
        collect_rollout_details: bool,
        llm_kwargs: dict[str, Any] | None,
        api_base: str | None,
        session_id: str | None,
        max_thinking_tokens: int | None,
        reasoning_effort: str | None,
        model_info: dict[str, Any] | None,
        use_responses_api: bool,
    ) -> NativeToolLiteLLM:
        del max_thinking_tokens, use_responses_api
        backend = (
            llm_backend.value if isinstance(llm_backend, LLMBackend) else llm_backend
        )
        if backend != LLMBackend.LITELLM.value:
            raise ValueError("native Terminus supports only the LiteLLM backend")
        return NativeToolLiteLLM(
            model_name=model_name,
            api_base=api_base,
            temperature=temperature,
            collect_rollout_details=collect_rollout_details,
            session_id=session_id,
            reasoning_effort=reasoning_effort,
            model_info=model_info,
            **dict(llm_kwargs or {}),
        )

    async def _query_llm(
        self,
        chat: Chat,
        prompt: str,
        original_instruction: str = "",
        session: Any = None,
    ) -> LLMResponse:
        """Make one scientific model request; never retry model behavior."""

        del original_instruction, session
        started = time.perf_counter()
        response = await chat.chat(prompt, **self._llm_call_kwargs)
        self._api_request_times.append((time.perf_counter() - started) * 1000.0)
        return response

    def _get_error_response_type(self) -> str:
        """Keep Terminus recovery phrased in native-tool terms, not JSON terms."""

        return "single native bash call"

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        original_instruction: str = "",
        session: Any = None,
    ) -> tuple[list[Command], bool, str, str, str, LLMResponse]:
        response = await self._query_llm(chat, prompt, original_instruction, session)
        calls = (response.extra or {}).get("native_tool_calls") or []
        format_reminder = getattr(self, "_format_reminder", FORMAT_REMINDER)
        if len(calls) != 1:
            return (
                [],
                False,
                f"ERROR: {format_reminder}",
                response.content,
                "",
                response,
            )
        command = parsed_bash_command(calls[0])
        if command is None:
            return (
                [],
                False,
                f"ERROR: {format_reminder}",
                response.content,
                "",
                response,
            )
        command = command.strip()
        is_complete = is_completion_command(command)
        if is_complete:
            self._pending_completion = True
        return (
            [Command(keystrokes=command + "\n", duration_sec=1.0)],
            is_complete,
            "",
            response.content,
            "",
            response,
        )

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        started = time.perf_counter()
        try:
            return await super()._execute_commands(commands, session)
        finally:
            self._tool_times_msec.append((time.perf_counter() - started) * 1000.0)

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
        self._llm_call_kwargs["seed"] = self._sampling_seed
        self._tool_times_msec: list[float] = []
        self._reset_per_run_state()
        self._chat = NativeToolChat(
            self._llm, interleaved_thinking=self._interleaved_thinking
        )
        self._chat._messages.append(
            {"role": "system", "content": self._native_system_prompt}
        )
        self._chat._messages.extend(self._example_messages)
        self._context = context
        initial_prompt = self._native_instance_prompt.replace(
            "{{task}}", visible_instruction
        )
        self._trajectory_steps.append(
            Step(
                step_id=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=initial_prompt,
            )
        )
        run_exception: BaseException | None = None
        try:
            await self._run_agent_loop(
                initial_prompt=initial_prompt,
                chat=self._chat,
                original_instruction=visible_instruction,
            )
        except BaseException as exc:
            run_exception = exc
            raise
        finally:
            context.rollout_details = self._chat.rollout_details
            context.n_input_tokens = self._chat.total_input_tokens
            context.n_output_tokens = self._chat.total_output_tokens
            context.n_cache_tokens = self._chat.total_cache_tokens
            context.cost_usd = (
                self._chat.total_cost if self._chat.total_cost > 0 else None
            )
            self._dump_trajectory()
            trajectory_path = self.logs_dir / "trajectory.json"
            completed_commands = completed_native_bash_commands(
                "terminus_native", self._chat.rollout_details, trajectory_path
            )
            evidence = completion_evidence(completed_commands)
            termination = stopping_condition(
                completion_detected=evidence["completion_detected"],
                step_limit=(
                    run_exception is None
                    and not evidence["completion_detected"]
                    and self._n_episodes >= self._max_episodes
                ),
                exception=run_exception is not None,
            )
            metadata = {
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
                "completion_detected": evidence["completion_detected"],
                "completion_evidence": evidence,
                "stopping_condition": termination,
                "logprob_turn_count": len(
                    self._chat.rollout_details[0].get("logprobs", [])
                )
                if self._chat.rollout_details
                else 0,
                "n_episodes": self._n_episodes,
                "api_request_times_msec": self._api_request_times,
                "generation_times_msec": self._api_request_times,
                "tool_times_msec": self._tool_times_msec,
            }
            context.metadata = metadata
            if not self._chat.rollout_details or metadata["logprob_turn_count"] == 0:
                raise RuntimeError("native Terminus rollout contains no token logprobs")
            if trajectory_path.is_file():
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                for step in trajectory.get("steps") or []:
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
                trajectory["telemetry_contract"] = TELEMETRY_CONTRACT
                temporary_path = trajectory_path.with_name(
                    trajectory_path.name + ".tmp"
                )
                temporary_path.write_bytes(compact_json_bytes(trajectory))
                temporary_path.replace(trajectory_path)
