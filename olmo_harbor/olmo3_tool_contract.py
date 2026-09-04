"""Strict normalization for the OLMo3 native bash tool-call contract."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Mapping


_WRAPPED = re.compile(
    r"\A\s*<function_calls>\s*(.*?)\s*</function_calls>\s*\Z", re.DOTALL
)
_PYTHONIC = re.compile(
    r"\A\s*bash\s*\(\s*command\s*=\s*(.+?)\s*\)\s*;?\s*\Z", re.DOTALL
)
_FENCED_JSON = re.compile(
    r"\A\s*```json\s*(\{.*\})\s*```\s*\Z", re.DOTALL | re.IGNORECASE
)


def _canonical(command: str) -> str | None:
    if not command.strip():
        return None
    return f"<function_calls>bash(command={json.dumps(command)})</function_calls>"


def _json_command(body: str) -> str | None:
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None

    # This is the only observed alternate format accepted by the experiment.
    # Keeping the accepted shape narrow prevents prose or arbitrary JSON from
    # being reinterpreted as a tool invocation.
    if set(value) != {"function", "parameters"} or value.get("function") != "bash":
        return None
    parameters = value.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != {"command"}:
        return None
    command = parameters.get("command")
    return command if isinstance(command, str) and command.strip() else None


def _pythonic_command(body: str) -> str | None:
    match = _PYTHONIC.fullmatch(body)
    if match is None:
        return None
    try:
        command: Any = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    return command if isinstance(command, str) and command.strip() else None


def _observed_fenced_json_command(output: str) -> str | None:
    """Extract one bash command from exact whole-response Think variants."""

    match = _FENCED_JSON.fullmatch(output)
    if match is None:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    command: Any = None
    if set(value) == {"command"}:
        command = value["command"]
    elif set(value) == {"bash"}:
        command = value["bash"]
    elif set(value) == {"function_calls"}:
        calls = value["function_calls"]
        if (
            isinstance(calls, list)
            and len(calls) == 1
            and isinstance(calls[0], Mapping)
            and set(calls[0]) == {"command"}
        ):
            command = calls[0]["command"]
    return command if isinstance(command, str) and command.strip() else None


def normalize_olmo3_bash_call(output: str) -> str | None:
    """Return one canonical complete call, or None for all other output.

    Accepted final-output forms are deliberately limited to canonical wrapped
    Pythonic syntax, one wrapped function/parameters JSON object, one bare
    Pythonic call with an optional trailing semicolon, or one of the exact
    fenced single-command JSON shapes observed from the pinned Think model.
    The whole string must match; calls mentioned inside reasoning, prose, or
    arbitrary code are never scraped.
    """

    if not isinstance(output, str) or not output.strip():
        return None
    wrapped = _WRAPPED.fullmatch(output)
    body = wrapped.group(1) if wrapped is not None else output
    command = _pythonic_command(body)
    if command is None and wrapped is not None:
        command = _json_command(body)
    if command is None and wrapped is None:
        command = _observed_fenced_json_command(output)
    return _canonical(command) if command is not None else None
