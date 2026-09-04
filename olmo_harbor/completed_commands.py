"""Validation helpers for completed native bash exchanges in Harbor traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import json
from olmo_harbor.native_contract import parsed_bash_command


def mini_swe_completed_commands(trajectory: Mapping[str, Any]) -> list[str]:
    """Return calls whose matching role=tool result is present in the trace."""

    pending: dict[str, str] = {}
    completed: list[str] = []
    for message in trajectory.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "assistant" and message.get("tool_calls") is not None:
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                continue
            command = parsed_bash_command(calls[0])
            call_id = calls[0].get("id") if isinstance(calls[0], Mapping) else None
            if command is not None and isinstance(call_id, str) and call_id:
                pending[call_id] = command
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id in pending:
                completed.append(pending.pop(call_id))
    return completed


def _terminus_native_commands(details: Any) -> list[str]:
    commands: list[str] = []
    if not isinstance(details, list):
        return commands
    for detail in details:
        extra = detail.get("extra") if isinstance(detail, Mapping) else None
        turns = extra.get("native_tool_calls") if isinstance(extra, Mapping) else None
        if not isinstance(turns, list):
            continue
        for calls in turns:
            if not isinstance(calls, list) or len(calls) != 1:
                continue
            command = parsed_bash_command(calls[0])
            if command is not None:
                commands.append(command)
    return commands


def _terminus_observed_call_count(trajectory: Mapping[str, Any]) -> int:
    count = 0
    for step in trajectory.get("steps") or []:
        if not isinstance(step, Mapping) or step.get("source") != "agent":
            continue
        calls = step.get("tool_calls")
        observation = step.get("observation")
        results = (
            observation.get("results") if isinstance(observation, Mapping) else None
        )
        bash_calls = (
            [
                call
                for call in calls
                if isinstance(call, Mapping)
                and call.get("function_name") == "bash_command"
            ]
            if isinstance(calls, list)
            else []
        )
        if len(bash_calls) != 1 or not isinstance(results, list) or not results:
            continue
        call_id = bash_calls[0].get("tool_call_id")
        if any(
            isinstance(result, Mapping) and result.get("source_call_id") == call_id
            for result in results
        ):
            count += 1
    return count


def completed_native_bash_commands(
    harness_id: str,
    details: Any,
    trajectory_path: Path,
) -> list[str]:
    """Return semantically valid calls that also have environment observations."""

    if not trajectory_path.is_file():
        return []
    trajectory = json.loads(trajectory_path.read_text())
    if not isinstance(trajectory, Mapping):
        return []
    if harness_id == "mini_swe_native":
        return mini_swe_completed_commands(trajectory)
    if harness_id == "terminus_native":
        commands = _terminus_native_commands(details)
        return commands[: _terminus_observed_call_count(trajectory)]
    return []
