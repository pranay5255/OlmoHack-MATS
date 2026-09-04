"""Experiment 1: one pinned OLMo checkpoint per A100-80GB."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Mapping


from settings import MODEL_SPECS

VLLM_VERSION = "0.21.0"
MAX_MODEL_LEN = 65_536
MAX_NUM_SEQS = 1

# Transformers 5 represents OLMo3 RoPE settings per attention type, while
# vLLM 0.21.0's OLMo3 loader expects the equivalent flat full-attention dict
# and derives the sliding-attention default from its rope_theta field.
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


if modal is not None:
    APP_NAME = os.environ.get("OLMO_APP_NAME", "olmohack-mats-experiment1")
    SELECTED_CONDITION = os.environ.get("OLMO_CONDITION", "think_sft")
    SERVER_TOKEN = os.environ.get("VLLM_API_KEY", "__MISSING_DEPLOY_TOKEN__")
    if SELECTED_CONDITION not in MODEL_SPECS:
        raise RuntimeError(f"OLMO_CONDITION must be one of {sorted(MODEL_SPECS)}")

    app = modal.App(APP_NAME)
    hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
    vllm_cache = modal.Volume.from_name("olmo3-vllm-cache", create_if_missing=True)

    download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "huggingface-hub>=0.34,<1"
    )
    server_image = modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    ).pip_install(f"vllm=={VLLM_VERSION}")
    vllm_secret = modal.Secret.from_dict(
        {"VLLM_API_KEY": SERVER_TOKEN, "OLMO_CONDITION": SELECTED_CONDITION}
    )

    @app.function(
        image=download_image,
        cpu=2,
        memory=4096,
        timeout=3600,
        volumes={"/root/.cache/huggingface": hf_cache},
    )
    def download_weights(condition: str) -> dict[str, str]:
        if condition not in MODEL_SPECS:
            raise ValueError(f"unsupported condition {condition!r}")
        from huggingface_hub import snapshot_download

        spec = MODEL_SPECS[condition]
        snapshot_download(
            repo_id=spec["model_id"],
            revision=spec["revision"],
            cache_dir="/root/.cache/huggingface",
        )
        hf_cache.commit()
        return {
            "condition_id": condition,
            "model_id": spec["model_id"],
            "revision": spec["revision"],
        }

    server_decorator = getattr(app, "server", None)
    server_options: dict[str, Any] = {
        "image": server_image,
        "port": 8000,
        "gpu": "A100-80GB",
        "min_containers": 0,
        "max_containers": 1,
        "target_concurrency": 1,
        "scaledown_window": 3600,
        "startup_timeout": 1800,
        "exit_grace_period": 25,
        "secrets": [vllm_secret],
        "volumes": {
            "/root/.cache/huggingface": hf_cache,
            "/root/.cache/vllm": vllm_cache,
        },
    }
    if server_decorator is not None:
        # Current Modal Servers require opting out of proxy auth because vLLM's
        # independently generated bearer token is the endpoint authentication.
        server_options["unauthenticated"] = True
    else:
        # Modal 1.3.x exposed the same Server primitive under this provisional
        # name and generated public flash URLs; vLLM still enforces its token.
        server_decorator = app._experimental_server
        server_options["proxy_regions"] = ["us-east"]

    @server_decorator(**server_options)
    class VLLMServer:
        @modal.enter()
        def start(self) -> None:
            token = os.environ.get("VLLM_API_KEY", "")
            if token == "__MISSING_DEPLOY_TOKEN__" or len(token) < 32:
                raise RuntimeError(
                    "a generated VLLM_API_KEY of at least 32 characters is required"
                )
            spec = MODEL_SPECS[SELECTED_CONDITION]
            served_name = f"{spec['model_id']}@{spec['revision']}"
            self.process = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--model",
                    spec["model_id"],
                    "--revision",
                    spec["revision"],
                    "--hf-overrides",
                    json.dumps(
                        {"rope_parameters": OLMO3_ROPE_PARAMETERS},
                        separators=(",", ":"),
                    ),
                    "--served-model-name",
                    served_name,
                    "--dtype",
                    "bfloat16",
                    "--max-model-len",
                    str(MAX_MODEL_LEN),
                    "--max-num-seqs",
                    str(MAX_NUM_SEQS),
                ],
                env=os.environ.copy(),
            )

        @modal.exit()
        def stop(self) -> None:
            if getattr(self, "process", None) is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.kill()

    @app.local_entrypoint()
    def download(condition: str) -> None:
        result = download_weights.remote(condition)
        print(json.dumps(result, sort_keys=True))

    @app.local_entrypoint()
    def endpoint_url() -> None:
        # Modal 1.3.3 ships the synchronized public Server handle in the
        # ``modal.server`` submodule but does not re-export it from ``modal``.
        # Newer clients expose ``modal.Server`` directly, so support both
        # without deriving a URL from workspace/app naming conventions.
        server_type = getattr(modal, "Server", None)
        if server_type is None:
            from modal.server import Server as server_type

        server = server_type.from_name(APP_NAME, "VLLMServer")
        if hasattr(server, "get_url"):
            url = server.get_url()
        else:
            urls = server.get_urls()
            if not isinstance(urls, Mapping) or len(urls) != 1:
                raise RuntimeError(
                    f"expected one deployed Modal Server URL, got {urls!r}"
                )
            url = next(iter(urls.values()))
        if not isinstance(url, str) or not url.startswith("https://"):
            raise RuntimeError(f"invalid deployed Modal Server URL: {url!r}")
        print(url)
