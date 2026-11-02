"""Start and stop a local vLLM server from a YAML profile (the modes' profiles ship in profiles/; the serving
experiments' are in experiments/qwen38/profiles/)."""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from importlib.metadata import version
from pathlib import Path

import httpx
import yaml


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_profile(path: str | Path, overrides: dict | None = None) -> dict:
    """A profile YAML file, or the name of one shipped with the package ("27b", "flash_next")."""
    source = Path(path)
    if source.exists():
        profile = yaml.safe_load(source.read_text())
    else:
        profile = yaml.safe_load(resources.files(__package__).joinpath("profiles", f"{path}.yaml").read_text())
    profile.setdefault("args", {})
    profile["args"].update(overrides or {})
    profile["name"] = profile.get("name") or Path(path).stem
    return profile


class Server:
    def __init__(self, profile: dict, model: str | None = None, log_dir: str | Path = "logs",
                 port: int | None = None, startup_timeout: float = 1800):
        self.profile = profile
        self.model = model or profile["model"]
        self.served_name = profile.get("served_name", "judge")
        self.port = port or free_port()
        self.log_dir = Path(log_dir)
        self.timeout = startup_timeout
        self.process = None
        self.log_path = None
        self.startup_seconds = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def command(self) -> list[str]:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", self.model,
               "--served-model-name", self.served_name, "--host", "127.0.0.1", "--port", str(self.port)]
        for key, value in self.profile["args"].items():
            if value is True:
                cmd.append(f"--{key}")
            elif value is not False and value is not None:
                cmd += [f"--{key}", json.dumps(value) if isinstance(value, dict) else str(value)]
        return cmd

    def start(self) -> Server:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"vllm_{self.profile['name']}_{int(time.time())}.log"
        env = {**os.environ, **{k: str(v) for k, v in self.profile.get("env", {}).items()}}
        log = self.log_path.open("w")
        log.write(" ".join(self.command()) + "\n")
        log.flush()
        self.process = subprocess.Popen(self.command(), env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        started = time.monotonic()
        while True:
            if self.process.poll() is not None:
                tail = self.log_path.read_text(errors="replace")[-4000:]
                raise RuntimeError(f"vLLM exited with code {self.process.returncode}; log {self.log_path}\n{tail}")
            try:
                if httpx.get(f"{self.base_url}/models", timeout=3).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() - started > self.timeout:
                self.stop()
                raise TimeoutError(f"vLLM not ready after {self.timeout:.0f} s; log {self.log_path}")
            time.sleep(3)
        self.startup_seconds = time.monotonic() - started
        print(f"vLLM ({self.profile['name']}) ready in {self.startup_seconds:.0f} s at {self.base_url}", flush=True)
        return self

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self):
        """Stop the server and every process it started (with tensor parallelism, one worker per GPU), so the GPUs'
        memory is free when this returns."""
        if self.process is not None:
            group = self.process.pid   # the server leads its own process group (start_new_session)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    self.process.poll()
                    try:
                        os.killpg(group, 0)   # raises once no process of the group is left
                    except ProcessLookupError:
                        break
                    time.sleep(0.5)
                else:
                    continue
                break
        self.process = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def memory_report(self) -> dict:
        """Model weights, KV cache capacity, graph capture and start-up timings from the startup log. Weight
        loading is fast when the files are already in the OS page cache, and torch.compile when vLLM's compile
        cache hits (`compile_cache_hit`), so a first start on a machine is slower than a restart."""
        text = re.sub(r"\x1b\[[0-9;]*m", "", self.log_path.read_text(errors="replace")) if self.log_path else ""
        patterns = {"model_gib": r"Model loading took ([\d.]+) GiB",
                    "kv_cache_gib": r"Available KV cache memory: (-?[\d.]+) GiB",   # negative: start-up fails
                    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
                    "max_concurrency_at_max_len": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
                    # 0.11 and 0.32 word these lines differently ("GiB and" / "GiB memory and", "torch.compile
                    # takes X s in total" / "(compilation: X s"); both forms match
                    "weights_seconds": r"Model loading took [\d.]+ GiB(?: memory)? and ([\d.]+) seconds",
                    "compile_seconds": r"(?:torch.compile takes|\(compilation:) ([\d.]+) s",
                    "graph_capture_seconds": r"Graph capturing finished in ([\d.]+) secs",
                    "engine_init_seconds": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"}
        out = {}
        for key, pattern in patterns.items():
            hits = re.findall(pattern, text)
            out[key] = float(hits[-1].replace(",", "")) if hits else None
        out["cuda_graphs"] = "Graph capturing finished" in text
        out["compile_cache_hit"] = "Directly load the compiled graph" in text
        out["startup_seconds"] = self.startup_seconds
        out["vllm"] = version("vllm")
        return out


class Replicas:
    """One server per group of GPUs (as many as the profile's tensor-parallel-size), each with its own copy of the
    model and its own API process, started in parallel. The Client spreads requests over `base_url` (a list), so
    this scales out without vLLM's data-parallel mode, whose single API process does the tokenization and image
    preprocessing for every replica."""

    def __init__(self, profile: dict, gpus: list[str], model: str | None = None, log_dir: str | Path = "logs"):
        size = int(profile["args"].get("tensor-parallel-size", 1))
        if not gpus or len(gpus) % size:
            raise ValueError(f"{profile['name']} runs on {size} GPU(s) per server; got GPUs {gpus}")
        groups = [",".join(gpus[i:i + size]) for i in range(0, len(gpus), size)]
        self.profile = {**profile, "replicas": len(groups)}
        self.servers = [Server({**profile, "name": f"{profile['name']}_gpu{group.replace(',', '-')}",
                                "env": {**profile.get("env", {}), "CUDA_VISIBLE_DEVICES": group}}, model, log_dir)
                        for group in groups]
        self.served_name = self.servers[0].served_name

    @property
    def base_url(self) -> list[str]:
        return [s.base_url for s in self.servers]

    def running(self) -> bool:
        return all(s.running() for s in self.servers)

    def memory_report(self) -> dict:
        """The first replica's (all are configured alike), with every replica's start-up time."""
        return {**self.servers[0].memory_report(), "startup_seconds": [s.startup_seconds for s in self.servers]}

    def __enter__(self):
        try:
            with ThreadPoolExecutor(len(self.servers)) as pool:
                list(pool.map(Server.start, self.servers))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        for s in self.servers:
            s.stop()
