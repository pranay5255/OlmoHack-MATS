"""Pinned OLMo 3 released chat-template assets and byte-level provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final


TEMPLATE_SPECS: Final[dict[str, dict[str, str]]] = {
    "think_sft": {
        "filename": "olmo3-think-sft-released.jinja",
        "sha256": "fb2b8736121ab4001112a9ccfb33c0081230536c861c706bf24b1b390964c1cd",
        "upstream_file": "tokenizer_config.json#chat_template",
    },
    "think_dpo": {
        "filename": "olmo3-think-dpo-released.jinja",
        "sha256": "6d549883b5ed12879e191845c256a30c7dfd4eced0a4f160060a3ea0199d9e3a",
        "upstream_file": "chat_template.jinja",
    },
    "think_rlvr": {
        "filename": "olmo3-think-rlvr-released.jinja",
        "sha256": "6d549883b5ed12879e191845c256a30c7dfd4eced0a4f160060a3ea0199d9e3a",
        "upstream_file": "chat_template.jinja",
    },
    "instruct_sft": {
        "filename": "olmo3-instruct-sft-released.jinja",
        "sha256": "edbb35ef1219ad2273f716feed08f535a868e167be5b5518adc37ab2bcb8afe6",
        "upstream_file": "chat_template.jinja",
    },
    "instruct_dpo": {
        "filename": "olmo3-instruct-dpo-released.jinja",
        "sha256": "edbb35ef1219ad2273f716feed08f535a868e167be5b5518adc37ab2bcb8afe6",
        "upstream_file": "chat_template.jinja",
    },
    "instruct_rlvr": {
        "filename": "olmo3-instruct-rlvr-released.jinja",
        "sha256": "f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b",
        "upstream_file": "chat_template.jinja",
    },
}

RECOMMENDED_SYSTEM_PROMPTS: Final[dict[str, str]] = {
    "think_sft": (
        "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is "
        "December 2024, and your model weights are available at "
        "https://huggingface.co/allenai."
    ),
    "think_dpo": (
        "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your "
        "date cutoff is November 2024, and your model weights are available at "
        "https://huggingface.co/allenai."
    ),
    "think_rlvr": (
        "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your "
        "date cutoff is November 2024, and your model weights are available at "
        "https://huggingface.co/allenai."
    ),
    "instruct_sft": (
        "You are a helpful function-calling AI assistant. You are provided with "
        "function signatures within <functions></functions> XML tags. You may call "
        "one or more functions to assist with the user query. Output any function "
        "calls within <function_calls></function_calls> XML tags. Do not make "
        "assumptions about what values to plug into functions."
    ),
    "instruct_dpo": (
        "You are a helpful function-calling AI assistant. You are provided with "
        "function signatures within <functions></functions> XML tags. You may call "
        "one or more functions to assist with the user query. Output any function "
        "calls within <function_calls></function_calls> XML tags. Do not make "
        "assumptions about what values to plug into functions."
    ),
    "instruct_rlvr": (
        "You are a helpful function-calling AI assistant. You are provided with "
        "function signatures within <functions></functions> XML tags. You may call "
        "one or more functions to assist with the user query. Output any function "
        "calls within <function_calls></function_calls> XML tags. Do not make "
        "assumptions about what values to plug into functions."
    ),
}


def released_template_bytes(path: Path) -> bytes:
    """Return the upstream bytes, removing only the repository packaging newline."""

    packaged = path.read_bytes()
    if not packaged.endswith(b"\n") or packaged.endswith(b"\n\n"):
        raise ValueError(f"released template must have one packaging newline: {path}")
    return packaged[:-1]


def verified_template_bytes(condition_id: str, prompt_root: Path) -> bytes:
    spec = TEMPLATE_SPECS[condition_id]
    path = prompt_root / spec["filename"]
    template = released_template_bytes(path)
    observed = hashlib.sha256(template).hexdigest()
    if observed != spec["sha256"]:
        raise ValueError(
            f"released template hash mismatch for {condition_id}: {observed}"
        )
    return template


def materialize_template(condition_id: str, prompt_root: Path, output: Path) -> Path:
    """Write the exact verified upstream bytes for vLLM without adding a BOS/newline."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(verified_template_bytes(condition_id, prompt_root))
    return output


def compose_recommended_system_prompt(condition_id: str, task_contract: str) -> str:
    """Combine the released checkpoint preamble with the shared coding contract."""

    return f"{RECOMMENDED_SYSTEM_PROMPTS[condition_id]}\n\n{task_contract.strip()}"
