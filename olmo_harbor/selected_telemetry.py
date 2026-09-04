"""Selected-token-only telemetry primitives shared by both native harnesses."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence


TELEMETRY_CONTRACT = "selected-logprobs-v1"
PROBABILITY_THRESHOLDS = (0.1, 0.01, 0.001)
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


def native_action_events(
    tool_calls: Any,
    *,
    completion_token_count: int,
) -> list[dict[str, Any]]:
    """Describe native actions without retaining any alternate-token payload."""

    events: list[dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return events
    for action_index, call in enumerate(tool_calls):
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if not isinstance(function, Mapping) or function.get("name") != "bash":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        command = arguments.get("command") if isinstance(arguments, Mapping) else None
        events.append(
            {
                "event": "native_action",
                "action_index": action_index,
                "tool_call_id": call.get("id"),
                "tool_name": "bash",
                "command": command if isinstance(command, str) else None,
                # vLLM does not expose a tool-call-only token subspan. Preserve the
                # exact enclosing completion span instead of guessing boundaries.
                "completion_token_start": 0,
                "completion_token_end": completion_token_count,
            }
        )
    return events


def selected_turn_statistics(
    selected_logprobs: Sequence[float],
    *,
    finish_reason: str | None,
    stop_reason: str | None = None,
    output_truncated: bool,
    tool_calls: Any = None,
) -> dict[str, Any]:
    """Calculate the frozen lossless-audit primitives for one model turn."""

    values = [float(value) for value in selected_logprobs]
    if not values:
        raise ValueError("selected-token telemetry cannot summarize an empty turn")
    if any(not math.isfinite(value) or value > 1e-6 for value in values):
        raise ValueError("selected logprobs must be finite and non-positive")
    token_count = len(values)
    total = math.fsum(values)
    total_squares = math.fsum(value * value for value in values)
    mean = total / token_count
    events = native_action_events(
        _plain(tool_calls), completion_token_count=token_count
    )
    return {
        "telemetry_contract": TELEMETRY_CONTRACT,
        "token_count": token_count,
        "selected_logprob_sum": total,
        "selected_logprob_sum_squares": total_squares,
        "selected_logprob_min": min(values),
        "selected_logprob_max": max(values),
        "probability_lt_0_1_count": sum(
            value < math.log(PROBABILITY_THRESHOLDS[0]) for value in values
        ),
        "probability_lt_0_01_count": sum(
            value < math.log(PROBABILITY_THRESHOLDS[1]) for value in values
        ),
        "probability_lt_0_001_count": sum(
            value < math.log(PROBABILITY_THRESHOLDS[2]) for value in values
        ),
        "sequence_log_probability": total,
        "mean_token_log_probability": mean,
        "mean_selected_surprisal_nats": -mean,
        "geometric_mean_selected_probability": math.exp(mean),
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "output_truncated": bool(output_truncated),
        "native_action_count": len(events),
        "event_offsets": events,
    }


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
