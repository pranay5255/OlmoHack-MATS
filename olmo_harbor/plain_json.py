"""Repository-owned parser for the frozen Terminus plain-JSON scaffold."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedCommand:
    keystrokes: str
    duration: float


@dataclass(frozen=True)
class ParseResult:
    commands: list[ParsedCommand]
    is_task_complete: bool
    error: str
    warning: str
    analysis: str = ""
    plan: str = ""


def _first_json_object(response: str) -> tuple[str, list[str]]:
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(response):
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0:
                warnings = []
                if response[:start].strip():
                    warnings.append("extra text before JSON object")
                if response[index + 1 :].strip():
                    warnings.append("extra text after JSON object")
                return response[start : index + 1], warnings
    if start >= 0 and depth > 0 and not in_string:
        return response[start:] + ("}" * depth), ["auto-closed truncated JSON object"]
    return "", []


class RepositoryTerminusJSONParser:
    """Parse one command batch without relying on provider-native tool calls."""

    def parse_response(self, response: str) -> ParseResult:
        content, warnings = _first_json_object(response)
        if not content:
            return ParseResult([], False, "No valid JSON object found", "")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            return ParseResult([], False, f"Invalid JSON: {exc}", _warning_text(warnings))
        if not isinstance(payload, dict):
            return ParseResult([], False, "Response must be a JSON object", _warning_text(warnings))
        missing = [name for name in ("analysis", "plan", "commands") if name not in payload]
        if missing:
            return ParseResult([], False, f"Missing required fields: {', '.join(missing)}", _warning_text(warnings))
        if not isinstance(payload["analysis"], str) or not isinstance(payload["plan"], str):
            return ParseResult([], False, "analysis and plan must be strings", _warning_text(warnings))
        if not isinstance(payload["commands"], list):
            return ParseResult([], False, "commands must be an array", _warning_text(warnings))
        complete = payload.get("task_complete", False)
        if not isinstance(complete, bool):
            return ParseResult([], False, "task_complete must be a boolean", _warning_text(warnings))

        commands: list[ParsedCommand] = []
        for index, raw in enumerate(payload["commands"]):
            if not isinstance(raw, dict) or not isinstance(raw.get("keystrokes"), str):
                return ParseResult([], False, f"Command {index + 1} requires string keystrokes", _warning_text(warnings), payload["analysis"], payload["plan"])
            keystrokes = raw["keystrokes"]
            duration = raw.get("duration", 1.0)
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                return ParseResult([], False, f"Command {index + 1} duration must be numeric", _warning_text(warnings), payload["analysis"], payload["plan"])
            duration = float(duration)
            if not math.isfinite(duration) or duration < 0 or duration > 60:
                return ParseResult([], False, f"Command {index + 1} duration must be between 0 and 60", _warning_text(warnings), payload["analysis"], payload["plan"])
            if keystrokes and not keystrokes.endswith("\n") and not keystrokes.startswith("C-"):
                warnings.append(f"Command {index + 1} auto-appended missing trailing newline")
                keystrokes += "\n"
            unknown = sorted(set(raw) - {"keystrokes", "duration"})
            if unknown:
                warnings.append(f"Command {index + 1} ignored unknown fields: {', '.join(unknown)}")
            commands.append(ParsedCommand(keystrokes, duration))
        return ParseResult(commands, complete, "", _warning_text(warnings), payload["analysis"], payload["plan"])


def _warning_text(warnings: list[str]) -> str:
    return "" if not warnings else "- " + "\n- ".join(warnings)


def command_events(result: ParseResult) -> list[dict[str, object]]:
    """Represent every parsed batch item as an ATIF-compatible tool call."""
    return [
        {
            "tool_call_id": f"call_1_{index}",
            "function_name": "bash_command",
            "arguments": {"keystrokes": command.keystrokes, "duration": command.duration},
        }
        for index, command in enumerate(result.commands, start=1)
    ]
