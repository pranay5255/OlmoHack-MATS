"""Runtime patch injected into the pinned mini-swe-agent 2.4.6 environment.

This module is imported by mini-swe-agent inside the task container.  It moves
submission detection from stdout inspection to the already parsed bash action,
while persisting the matching tool result before emitting the Submitted exit.
"""

from __future__ import annotations

import copy
import json
import time

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError, Submitted


COMPLETION_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FORMAT_REMINDER = "Call bash exactly once with a non-empty command."
TOOL_FORMAT_REMINDER = """Tool call error:

<error>
{{ error }}
</error>

Respond with exactly one bash tool call in the demonstrated format. Put the
shell command in the JSON `command` argument. Do not return a shell code block,
prose-only command, or `bash -c` text outside a tool call."""

# This standalone module is copied into the pinned mini-swe environment. Keep
# these records identical to native_examples.EXAMPLE_SPEC.
EXAMPLE_RECORDS = [
    {
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
        "user": "Call bash exactly once with command python -c 'print(2 + 3)'.",
        "assistant_content": {
            "instruct": "",
            "think": "<think>The requested command is explicit, so I will call bash exactly once.</think>\n",
        },
        "tool_call_id": "native-example-call-2",
        "command": "python -c 'print(2 + 3)'",
        "tool_result": {"returncode": 0, "output": "5\n"},
    },
]


def example_messages(model_family: str) -> list[dict]:
    if model_family not in {"think", "instruct"}:
        raise ValueError(f"unknown model family {model_family!r}")
    messages = []
    for example in EXAMPLE_RECORDS:
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


class _ExampleAugmentedModel:
    """Prepend examples to API input without contaminating real trajectory."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        model_name = str(getattr(getattr(wrapped, "config", None), "model_name", ""))
        lowered = model_name.lower()
        if "think" in lowered:
            self._model_family = "think"
        elif "instruct" in lowered:
            self._model_family = "instruct"
        else:
            raise ValueError(f"cannot infer OLMo model family from {model_name!r}")

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def query(self, messages, **kwargs):
        if not messages or messages[0].get("role") != "system":
            raise ValueError("native examples require a leading system message")
        augmented = [
            copy.deepcopy(messages[0]),
            *example_messages(self._model_family),
            *copy.deepcopy(messages[1:]),
        ]
        return self._wrapped.query(augmented, **kwargs)


class NativeSubmissionEnvironment(LocalEnvironment):
    """Disable mini-SWE's stdout-position submission heuristic."""

    def _check_finished(self, output: dict) -> None:
        return None


class NativeSubmissionAgent(InteractiveAgent):
    """Submit after one exact parsed command and its tool result are persisted."""

    def query(self) -> dict:
        started = time.perf_counter()
        try:
            message = super().query()
        except FormatError as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            for message in getattr(exc, "messages", []):
                message.setdefault("extra", {})["generation_elapsed_msec"] = elapsed
            raise
        message.setdefault("extra", {})["generation_elapsed_msec"] = (
            time.perf_counter() - started
        ) * 1000.0
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        if len(actions) != 1:
            raise FormatError(
                {
                    "role": "user",
                    "content": FORMAT_REMINDER,
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        self._ask_confirmation_or_interrupt([actions[0]["command"]])
        started = time.perf_counter()
        try:
            outputs = [self.env.execute(actions[0])]
        finally:
            message.setdefault("extra", {})["tool_elapsed_msec"] = (
                time.perf_counter() - started
            ) * 1000.0
        observations = self.add_messages(
            *self.model.format_observation_messages(
                message, outputs, self.get_template_vars()
            )
        )
        command = actions[0].get("command")
        if (
            isinstance(command, str)
            and command.strip() == COMPLETION_COMMAND
            and outputs[0].get("returncode") == 0
        ):
            raise Submitted(
                {
                    "role": "exit",
                    "content": "",
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": "",
                        "completion_detected": True,
                    },
                }
            )
        return observations


class ExampleNativeSubmissionAgent(NativeSubmissionAgent):
    """Microgate-only agent that augments model input with two examples."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = _ExampleAugmentedModel(self.model)
