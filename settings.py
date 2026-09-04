"""Pinned checkpoints and sampling seeds shared by the rollout entrypoints."""

from __future__ import annotations

import hashlib
import json

MODEL_SPECS = {
    "think_sft": {
        "model_id": "allenai/Olmo-3-7B-Think-SFT",
        "revision": "6ff857587e040d6d523a3d5f3a56e918f5401d66",
    },
    "think_dpo": {
        "model_id": "allenai/Olmo-3-7B-Think-DPO",
        "revision": "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38",
    },
    "think_rlvr": {
        "model_id": "allenai/Olmo-3-7B-Think",
        "revision": "d97e442d7cc678210054dbcc9b440894d62c89a4",
    },
    "instruct_sft": {
        "model_id": "allenai/Olmo-3-7B-Instruct-SFT",
        "revision": "e1452fc572d51966ff4aaeb25118b891eb93e549",
    },
    "instruct_dpo": {
        "model_id": "allenai/Olmo-3-7B-Instruct-DPO",
        "revision": "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1",
    },
    "instruct_rlvr": {
        "model_id": "allenai/Olmo-3-7B-Instruct",
        "revision": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
    },
}


def stable_sampling_seed(pool_seed: int, task_id: str, candidate_index: int) -> int:
    payload = json.dumps(
        [pool_seed, task_id, candidate_index],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


NATIVE_BASH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    }
]
