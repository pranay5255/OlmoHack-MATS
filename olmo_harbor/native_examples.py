"""Few-shot native-bash message history used by Experiment 2 Instruct."""

from __future__ import annotations

import copy
import json
from typing import Any


EXAMPLE_SPEC = {
    "schema_version": 1,
    "mode": "structured_message_history",
    "examples": [
        {
            "example_id": "single_native_bash_call",
            "user": "Call bash exactly once with command printf 'TOOL_CALL_EXAMPLE_ONE\\n'.",
            "assistant_content": {
                "instruct": "",
                "think": "<think>The requested command is explicit, so I will call bash exactly once.</think>\n",
            },
            "tool_call_id": "native-example-call-1",
            "command": "printf 'TOOL_CALL_EXAMPLE_ONE\\n'",
            "tool_result": {"returncode": 0, "output": "TOOL_CALL_EXAMPLE_ONE\n"},
        },
        {
            "example_id": "follow_up_native_bash_call",
            "user": "Call bash exactly once with command python -c 'print(2 + 3)'.",
            "assistant_content": {
                "instruct": "",
                "think": "<think>The requested command is explicit, so I will call bash exactly once.</think>\n",
            },
            "tool_call_id": "native-example-call-2",
            "command": "python -c 'print(2 + 3)'",
            "tool_result": {"returncode": 0, "output": "5\n"},
        },
    ],
}
EXAMPLE_MODE = "structured_message_history"
TOOL_FORMAT_REMINDER = """Tool call error:

<error>
{{ error }}
</error>

Respond with exactly one bash tool call in the demonstrated format. Put the
shell command in the JSON `command` argument. Do not return a shell code block,
prose-only command, or `bash -c` text outside a tool call."""


def example_messages(model_family: str) -> list[dict[str, Any]]:
    """Return OpenAI-wire-compatible demonstration history for one family."""

    if model_family not in {"think", "instruct"}:
        raise ValueError(f"unknown model family {model_family!r}")
    messages: list[dict[str, Any]] = []
    for example in EXAMPLE_SPEC["examples"]:
        messages.extend(
            [
                {"role": "user", "content": example["user"]},
                {
                    "role": "assistant",
                    "content": example["assistant_content"][model_family],
                    "tool_calls": [
                        {
                            "id": example["tool_call_id"],
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps(
                                    {"command": example["command"]},
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": example["tool_call_id"],
                    "content": json.dumps(
                        example["tool_result"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
    return copy.deepcopy(messages)
