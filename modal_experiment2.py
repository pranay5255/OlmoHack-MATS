#!/usr/bin/env python3
"""Modal deployment for three OLMo3 checkpoints per continuously batched H200."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


from settings import MODEL_SPECS as CONDITION_SPECS

MODEL_SPECS = {
    key: (value["model_id"], value["revision"])
    for key, value in CONDITION_SPECS.items()
}

VLLM_VERSION = "0.21.0"
MAX_MODEL_LEN = 65_536
MAX_NUM_SEQS = 4
GPU_MEMORY_UTILIZATION = 0.29
KV_CACHE_MEMORY_BYTES = 13 * 1024**3
TOOL_PARSER_PLUGIN = "/opt/olmo3_strict_tool_parser.py"
THINK_MULTITURN_TEMPLATE = "/opt/olmo3-think-native-multiturn-v3.jinja"
THINK_MULTITURN_TEMPLATE_SHA256 = (
    "8cf820848852adb5bbd4964f4aa16ebac44c4ba0bc9331cef2ba8aef646f542b"
)
PACKAGED_CHAT_TEMPLATES = {
    condition: f"/opt/{condition}-released.jinja"
    for condition in (
        "instruct_sft",
        "instruct_dpo",
        "instruct_rlvr",
    )
}
CHAT_TEMPLATES = {
    condition: f"/tmp/olmo3-released-templates/{condition}.jinja"
    for condition in MODEL_SPECS
}
CHAT_TEMPLATE_SHA256 = {
    "think_sft": "fb2b8736121ab4001112a9ccfb33c0081230536c861c706bf24b1b390964c1cd",
    "think_dpo": "6d549883b5ed12879e191845c256a30c7dfd4eced0a4f160060a3ea0199d9e3a",
    "think_rlvr": "6d549883b5ed12879e191845c256a30c7dfd4eced0a4f160060a3ea0199d9e3a",
    "instruct_sft": "edbb35ef1219ad2273f716feed08f535a868e167be5b5518adc37ab2bcb8afe6",
    "instruct_dpo": "edbb35ef1219ad2273f716feed08f535a868e167be5b5518adc37ab2bcb8afe6",
    "instruct_rlvr": "f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b",
}
GROUPS = {
    "think": ("think_sft", "think_dpo", "think_rlvr"),
    "instruct": ("instruct_sft", "instruct_dpo", "instruct_rlvr"),
}
OLMO3_ROPE_PARAMETERS = {
    "attention_factor": 1.2079441541679836,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 8.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "yarn",
    "rope_theta": 500000,
}


try:
    import modal
except ImportError:
    modal = None


def served_name(condition: str) -> str:
    model_id, revision = MODEL_SPECS[condition]
    return f"{model_id}@{revision}"


def group_model_map(group: str) -> dict[str, int]:
    if group not in GROUPS:
        raise ValueError(f"unknown model group {group!r}")
    return {
        served_name(condition): 8100 + index
        for index, condition in enumerate(GROUPS[group])
    }


def materialize_chat_template(condition: str) -> Path:
    if condition.startswith("think_"):
        template = Path(THINK_MULTITURN_TEMPLATE).read_bytes()
        observed = hashlib.sha256(template).hexdigest()
        if observed != THINK_MULTITURN_TEMPLATE_SHA256:
            raise RuntimeError(
                f"Think multiturn chat template hash mismatch for {condition}: {observed}"
            )
        output = Path(CHAT_TEMPLATES[condition])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(template)
        return output
    packaged = Path(PACKAGED_CHAT_TEMPLATES[condition]).read_bytes()
    if not packaged.endswith(b"\n") or packaged.endswith(b"\n\n"):
        raise RuntimeError(f"invalid packaged template newline for {condition}")
    released = packaged[:-1]
    observed = hashlib.sha256(released).hexdigest()
    if observed != CHAT_TEMPLATE_SHA256[condition]:
        raise RuntimeError(
            f"released chat template hash mismatch for {condition}: {observed}"
        )
    output = Path(CHAT_TEMPLATES[condition])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(released)
    return output


def vllm_command(condition: str, port: int) -> list[str]:
    model_id, revision = MODEL_SPECS[condition]
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model_id,
        "--revision",
        revision,
        "--served-model-name",
        served_name(condition),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--kv-cache-memory-bytes",
        str(KV_CACHE_MEMORY_BYTES),
        "--enable-auto-tool-choice",
        "--tool-parser-plugin",
        TOOL_PARSER_PLUGIN,
        "--tool-call-parser",
        "olmo3" if condition.startswith("think_") else "olmo3_strict",
        "--chat-template",
        CHAT_TEMPLATES[condition],
        "--max-logprobs",
        "0",
        "--logprobs-mode",
        "raw_logprobs",
        "--hf-overrides",
        json.dumps({"rope_parameters": OLMO3_ROPE_PARAMETERS}, separators=(",", ":")),
        "--enforce-eager",
    ]
    if condition.startswith("think_"):
        command.extend(["--reasoning-parser", "olmo3"])
    return command


def wait_for_engine(port: int, process: subprocess.Popen[Any], log_path: Path) -> None:
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(f"vLLM on port {port} exited early:\n{tail}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"vLLM on port {port} did not become healthy")


def wait_for_router(process: subprocess.Popen[Any], log_path: Path) -> None:
    deadline = time.monotonic() + 180
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(f"model router exited early:\n{tail}")
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(1)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    raise TimeoutError(f"model router did not become healthy: {last_error}\n{tail}")


if modal is not None:
    APP_NAME = os.environ.get("OLMO_APP_NAME", "olmohack-mats-experiment2")
    MODEL_GROUP = os.environ.get("OLMO_MODEL_GROUP", "think")
    SERVER_TOKEN = os.environ.get("VLLM_API_KEY", "__MISSING_DEPLOY_TOKEN__")
    if MODEL_GROUP not in GROUPS:
        raise RuntimeError(f"OLMO_MODEL_GROUP must be one of {sorted(GROUPS)}")

    app = modal.App(APP_NAME)
    hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
    vllm_cache = modal.Volume.from_name("olmo3-vllm-cache", create_if_missing=True)
    download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "huggingface-hub>=0.34,<1"
    )
    server_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
        )
        .pip_install(
            f"vllm=={VLLM_VERSION}", "fastapi>=0.115", "uvicorn>=0.34", "httpx>=0.28"
        )
        .add_local_file("olmo_harbor/model_router.py", "/opt/model_router.py")
        .add_local_file(
            "olmo_harbor/olmo3_tool_contract.py", "/opt/olmo3_tool_contract.py"
        )
        .add_local_file("olmo_harbor/olmo3_strict_tool_parser.py", TOOL_PARSER_PLUGIN)
        .add_local_file(
            "harbor/prompts/olmo3-think-native-multiturn-v3.jinja",
            THINK_MULTITURN_TEMPLATE,
        )
    )
    for condition, destination in PACKAGED_CHAT_TEMPLATES.items():
        server_image = server_image.add_local_file(
            f"harbor/prompts/olmo3-{condition.replace('_', '-')}-released.jinja",
            destination,
        )
    server_secret = modal.Secret.from_dict(
        {
            "VLLM_API_KEY": SERVER_TOKEN,
            "OLMO_MODEL_GROUP": MODEL_GROUP,
        }
    )

    @app.function(
        image=download_image,
        cpu=2,
        memory=4096,
        timeout=7200,
        volumes={"/root/.cache/huggingface": hf_cache},
    )
    def download_group(group: str) -> list[dict[str, str]]:
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}")
        from huggingface_hub import snapshot_download

        downloaded = []
        for condition in GROUPS[group]:
            model_id, revision = MODEL_SPECS[condition]
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir="/root/.cache/huggingface",
            )
            downloaded.append(
                {"condition_id": condition, "model_id": model_id, "revision": revision}
            )
        hf_cache.commit()
        return downloaded

    server_decorator = getattr(app, "server", None)
    options: dict[str, Any] = {
        "image": server_image,
        "port": 8000,
        "gpu": "H200",
        "min_containers": 0,
        "max_containers": 1,
        "target_concurrency": 12,
        "scaledown_window": 600,
        "startup_timeout": 1800,
        "exit_grace_period": 30,
        "secrets": [server_secret],
        "volumes": {
            "/root/.cache/huggingface": hf_cache,
            "/root/.cache/vllm": vllm_cache,
        },
    }
    if server_decorator is not None:
        options["unauthenticated"] = True
    else:
        server_decorator = app._experimental_server
        options["proxy_regions"] = ["us-east"]

    @server_decorator(**options)
    class GroupedVLLMServer:
        @modal.enter()
        def start(self) -> None:
            token = os.environ.get("VLLM_API_KEY", "")
            if token == "__MISSING_DEPLOY_TOKEN__" or len(token) < 32:
                raise RuntimeError(
                    "VLLM_API_KEY must be a generated 32+ character token"
                )
            self.processes: list[subprocess.Popen[Any]] = []
            logs: list[tuple[Path, Any]] = []
            model_map = group_model_map(MODEL_GROUP)
            for condition in GROUPS[MODEL_GROUP]:
                materialize_chat_template(condition)
            engine_env = {**os.environ, "PYTHONPATH": "/opt"}
            for condition, port in zip(
                GROUPS[MODEL_GROUP], model_map.values(), strict=True
            ):
                log_path = Path(f"/tmp/vllm-{condition}.log")
                log_handle = log_path.open("wb")
                process = subprocess.Popen(
                    vllm_command(condition, port),
                    env=engine_env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                self.processes.append(process)
                logs.append((log_path, log_handle))
                # vLLM's memory profiler treats allocations from sibling engines as
                # non-vLLM usage. An explicit KV cache plus sequential startup avoids
                # a false "no available cache blocks" failure when three engines share
                # one H200.
                wait_for_engine(port, process, log_path)
            router_env = {
                **os.environ,
                "PYTHONPATH": "/opt",
                "OLMO_MODEL_PORT_MAP": json.dumps(model_map, separators=(",", ":")),
            }
            router_log_path = Path("/tmp/model-router.log")
            router_log = router_log_path.open("wb")
            self.router = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "uvicorn",
                    "model_router:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ],
                env=router_env,
                stdout=router_log,
                stderr=subprocess.STDOUT,
            )
            wait_for_router(self.router, router_log_path)
            self.log_handles = [handle for _, handle in logs] + [router_log]

        @modal.exit()
        def stop(self) -> None:
            processes = [getattr(self, "router", None), *getattr(self, "processes", [])]
            for process in processes:
                if process is not None:
                    process.terminate()
            for process in processes:
                if process is None:
                    continue
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
            for handle in getattr(self, "log_handles", []):
                handle.close()

    @app.local_entrypoint()
    def download(group: str) -> None:
        print(json.dumps(download_group.remote(group), sort_keys=True))

    @app.local_entrypoint()
    def endpoint_url() -> None:
        from modal import Server

        print(Server.from_name(APP_NAME, "GroupedVLLMServer").get_url())
