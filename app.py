"""Serve the complete FLUX.3 Klein pipeline on one GPU per replica."""

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

MODEL = os.getenv("MODEL", "distilled")
if MODEL not in ("distilled", "base"):
    raise ValueError("MODEL must be distilled or base")
HIGH_RES = os.getenv("HIGH_RES", "0") == "1"
GPU = os.getenv("GPU", "B200").strip().upper()
if GPU not in ("H100", "H100!", "H200", "B200"):
    raise ValueError("Choose one GPU type: H100, H100!, H200 or B200")
WARM = int(os.getenv("WARM", "0"))
REPLICAS = int(os.getenv("REPLICAS", "1"))
IDLE_SECONDS = int(os.getenv("IDLE_SECONDS", "120"))
if not 0 <= WARM <= REPLICAS:
    raise ValueError("Require 0 <= WARM <= REPLICAS")
if not 1 <= REPLICAS <= 16 or not 2 <= IDLE_SECONDS <= 3600:
    raise ValueError("REPLICAS must be 1-16 and IDLE_SECONDS 2-3600")
if HIGH_RES and MODEL != "distilled":
    raise ValueError("HIGH_RES is experimental and currently limited to Distilled text-to-image")
if HIGH_RES and GPU != "B200":
    raise ValueError("HIGH_RES is currently qualified only on B200")
QUALITY = "lossless" if MODEL == "distilled" else "exact"
REQUEST_TIMEOUT = 900 if HIGH_RES else 300
SNAPSHOT = os.getenv("SNAPSHOT", "1") == "1"
APP_NAME = os.getenv("APP_NAME", f"flux3-klein-{MODEL}")
STEPS = 16 if MODEL == "distilled" else 50
WARMUP_TOKENS = 1280
REVISIONS = {
    "distilled": "337337621b8526e120f2275d250242f4a6bedd75",
    "base": "3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86",
}
PUP_REVISION = "3f3edb52b2c55de0d49feab28598b68f7f8e4830"
SGLANG_REVISION = "586355a83bdf609579b3309d4125620351156bb7"
REGISTRY = "lmsysorg/sglang@sha256:b4f09a2ccbcfcc262d49600c3cd7f8581a793ea7d104947257b6a95500953b90"
CHECKPOINT = "/weights/flux-3-image-klein" + ("-base" if MODEL == "base" else "")
SAMPLING = dict(temperature=0.0, top_p=0.95, top_k=20, min_p=0.0, max_tokens=3584, presence_penalty=1.5)
ENGINE_ENV = {"SGLANG_FORCE_FUSED_OP_BACKEND": "torch"}
# Kernel caches the engine fills on its first warmup; the image build runs that warmup once on the
# deployment's GPU type so every fresh boot reads them instead of compiling.
CACHE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "SGLANG_CACHE_DIT_ENABLED": "false",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/image/inductor",
    "TRITON_CACHE_DIR": "/cache/image/triton",
    "CUDA_CACHE_PATH": "/cache/image/cuda",
}
KERNEL_CACHES = ("/cache/image", "/cache/pup", "/root/.cache/sglang")
PUP_CUDA = "/opt/pup/lib/python3.12/site-packages/nvidia/cu13"
# The prompt engine's compiled graph is keyed on its configuration, so the build and the server
# must start it with the same command and environment for the copy in the image to be reused.
PUP_ENV = {
    "CUDA_HOME": PUP_CUDA,
    "LD_LIBRARY_PATH": f"{PUP_CUDA}/lib",
    "VLLM_HOST_IP": "127.0.0.1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_USE_DEEP_GEMM": "0",
    "VLLM_SERVER_DEV_MODE": "1",
    "VLLM_CACHE_ROOT": "/cache/pup/vllm",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/pup/inductor",
    "TRITON_CACHE_DIR": "/cache/pup/triton",
    "CUDA_CACHE_PATH": "/cache/pup/cuda",
}
BUILD_CAPTION = (
    "A still life photograph of two blue ceramic mugs on a weathered oak table beside a tall window, "
    "soft morning light falling across the wood grain and catching the glaze of the mugs, a folded "
    "linen napkin and a small brass spoon beside them, shallow depth of field, muted warm tones, the "
    "garden outside blurred into green and gold, quiet and unhurried, shot on medium format film."
)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def engine_command(model, lora_repo=""):
    """The image engine's command line; the image build and the worker must start it identically."""
    return [
        "sglang",
        "serve",
        "--model-path",
        "/weights/flux-3-image-klein" + ("-base" if model == "base" else ""),
        "--model-variant",
        "fp8r",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--num-gpus",
        "1",
        "--attention-backend",
        "torch_sdpa",
        "--cpu-offload-components",
        "none",
        "--batching-max-size",
        "1",
        "--batching-delay-ms",
        "0",
        "--output-path",
        "",
        "--enable-breakable-cuda-graph",
        str(model == "distilled").lower(),
        "--warmup-mode",
        "server",
        "--warmup-resolutions",
        "1024x1024",
        "--warmup-steps",
        str(16 if model == "distilled" else 50),
        "--warmup-sampling-params",
        json.dumps(
            {
                "quality": "lossless" if model == "distilled" else "exact",
                **({"guidance_scale": 3.5} if model == "base" else {}),
            }
        ),
    ]


def prompt_command(gpu_name):
    """The prompt engine's command line for the assigned GPU."""
    fraction, kv_bytes = prompt_memory_budget(gpu_name)
    return [
        "/opt/pup/bin/vllm",
        "serve",
        "/weights/pup",
        "--host",
        "127.0.0.1",
        "--port",
        "9002",
        "--served-model-name",
        "pup",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "16",
        "--gpu-memory-utilization",
        fraction,
        "--kv-cache-memory-bytes",
        kv_bytes,
        "--limit-mm-per-prompt",
        '{"image":10}',
        "--attention-config",
        '{"backend":"TRITON_ATTN"}',
        "--enable-sleep-mode",
    ]


def assigned_gpu_name():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()


def wait_for_engine(process, port, timeout, name):
    import httpx

    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited; inspect the container logs")
            try:
                if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(1)
    raise TimeoutError(f"{name} exceeded the startup deadline")


def expand_once(tokenizer, system_prompt, request, timeout):
    """One completed prompt expansion from the local prompt engine, returned as the caption."""
    import httpx

    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": request}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    stop_tokens = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    with httpx.Client(timeout=timeout) as client:
        expansion = client.post(
            "http://127.0.0.1:9002/v1/completions",
            json={
                "model": "pup",
                "prompt": tokenizer.encode(text, add_special_tokens=False),
                **SAMPLING,
                "stop_token_ids": stop_tokens,
            },
        ).raise_for_status()
    answer = expansion.json()["choices"][0]
    if answer.get("finish_reason") != "stop" or not answer.get("text", "").strip():
        raise RuntimeError("Prompt expansion did not complete")
    return answer["text"].strip()


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def generate_once(parameters, timeout):
    """One picture from the local engine, checked to be the expected 1024-square JPEG."""
    import base64
    import io

    import httpx
    from PIL import Image

    with httpx.Client(timeout=timeout) as client:
        generated = client.post(
            "http://127.0.0.1:8001/v1/images/generations", json=parameters
        ).raise_for_status()
    raw = base64.b64decode(generated.json()["data"][0]["b64_json"], validate=True)
    with Image.open(io.BytesIO(raw)) as output:
        if output.size != (1024, 1024) or output.format != "JPEG":
            raise RuntimeError("Generation did not return the expected 1024-square JPEG")
        output.load()
    return raw


def compile_kernels(model, lora_repo=""):
    """Start each engine on the build GPU once so its compiled graphs, JIT kernels and autotune tables ship in the image."""
    from transformers import AutoTokenizer

    started = time.monotonic()
    started_at = time.time()
    # The engines run one after the other so the prompt engine's memory profiler sees only itself.
    process = subprocess.Popen(
        prompt_command(assigned_gpu_name()),
        env={**os.environ, **PUP_ENV, "HOME": "/root"},
        start_new_session=True,
    )
    try:
        wait_for_engine(process, 9002, 1500, "prompt engine")
        prompt_ready_s = time.monotonic() - started
        expand_once(
            AutoTokenizer.from_pretrained("/weights/pup", local_files_only=True),
            json.loads(Path("/weights/pup/system_prompts.json").read_text())["t2i"],
            "Two blue mugs on a wooden table by a window",
            120,
        )
    finally:
        stop_process(process)
    process = subprocess.Popen(
        engine_command(model, lora_repo),
        env={**os.environ, **ENGINE_ENV, "HOME": "/root"},
        start_new_session=True,
    )
    try:
        wait_for_engine(process, 8001, 1500, "image engine")
        ready_s = time.monotonic() - started
        generate_once(
            {
                "prompt": BUILD_CAPTION,
                "seed": 42,
                "generator_device": "cpu",
                "num_inference_steps": 16 if model == "distilled" else 50,
                "quality": "lossless" if model == "distilled" else "exact",
                "output_format": "jpeg",
                "response_format": "b64_json",
                "enable_cache_dit": False,
                "enhance_prompt": False,
                "output_compression": 95,
                "size": "1024x1024",
                **({"guidance_scale": 3.5} if model == "base" else {}),
            },
            900,
        )
        generated_s = time.monotonic() - started
    finally:
        stop_process(process)
    written, total = {}, {}
    for root in KERNEL_CACHES:
        for path in Path(root).rglob("*"):
            if not path.is_file():
                continue
            parts = path.relative_to(root).parts
            key = f"{root}/{parts[0]}" if len(parts) > 1 else root
            size = path.stat().st_size
            total[key] = total.get(key, 0) + size
            if path.stat().st_mtime >= started_at - 1:
                written[key] = written.get(key, 0) + size
    if not any(key.startswith("/cache/image/") for key in written) or "/cache/pup/vllm" not in written:
        raise RuntimeError(
            f"The build warmup left a kernel cache empty; the engines' cache paths changed: {written}"
        )
    report = {
        "prompt_engine_ready_s": prompt_ready_s,
        "engine_ready_s": ready_s,
        "first_picture_s": generated_s,
        "written_bytes": written,
        "cache_bytes": total,
    }
    build = json.loads(Path("/build.json").read_text())
    build["kernel_cache"] = report
    Path("/build.json").write_text(json.dumps(build, indent=2))
    print("FLUX3_BUILD_KERNELS " + json.dumps(report), flush=True)


def create_api(runtime):
    from flux3_api import create_api as build_api

    return build_api(runtime)


def serving_policy():
    from flux3_api import ServingPolicy

    if GPU in ("H100", "H100!"):
        return ServingPolicy(
            quality=QUALITY,
            max_side=1536,
            max_output_pixels=1536**2,
            max_references=2,
            max_reference_pixels=1024**2,
        )
    return ServingPolicy(
        quality=QUALITY,
        max_side=6256 if HIGH_RES else 4096,
        max_output_pixels=16_822_272 if HIGH_RES else 4 * 1024**2,
    )


def prompt_memory_budget(gpu_name):
    """GPU fraction and KV cache bytes for the prompt engine on the assigned device.

    The KV cache is sized explicitly because vLLM's profiler would otherwise count the image engine's
    concurrent allocations as its own; the fraction still gates vLLM's free-memory check at start.
    """
    for kind, fraction, kv_gib in (("B200", "0.20", 16), ("H200", "0.25", 14), ("H100", "0.32", 6)):
        if kind in gpu_name:
            return fraction, str(kv_gib << 30)
    raise RuntimeError(f"Unsupported assigned GPU: {gpu_name}")


def patch_engine_source(path, needle, replacement):
    """Edit the pinned engine source at one reviewed line, refusing any other version of the file."""
    source = path.read_text()
    if source.count(needle) != 1:
        raise RuntimeError(f"Pinned SGLang source differs from the reviewed version: {path.name}")
    patched = source.replace(needle, replacement)
    compile(patched, str(path), "exec")
    path.write_text(patched)
    return source, patched


def guard_vae(source):
    """Reject quadratic attention above 4MP if the native efficient path is unavailable."""
    expected = "0bb139ea9d377acca6a859121efc24e86f0095a46976cfbd6043deff4a9ceda4"
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError("Pinned VAE source changed; review its high-resolution fallback")
    call = "nn.functional.scaled_dot_product_attention(q, k, v)"
    if source.count(call) != 2:
        raise RuntimeError("Pinned VAE fallback sites changed")
    guarded = source.replace(call, "_bounded_default_attention(q, k, v)")
    guarded += """

def _bounded_default_attention(q, k, v):
    if q.shape[-2] > 65536:
        raise RuntimeError("Outputs above 4MP require efficient VAE attention on this GPU")
    return nn.functional.scaled_dot_product_attention(q, k, v)
"""
    compile(guarded, "flux3_image_vae.py", "exec")
    return guarded


def download_pup():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "black-forest-labs/flux-3-image-klein-pup",
        revision=PUP_REVISION,
        local_dir="/weights/pup",
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
    )


def prepare_model(model, high_res=False):
    import importlib.metadata

    import sglang
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    repo = "black-forest-labs/flux-3-image-klein" + ("-base" if model == "base" else "")
    checkpoint = Path("/weights") / repo.split("/")[-1]
    snapshot_download(
        repo,
        revision=REVISIONS[model],
        local_dir=checkpoint,
        allow_patterns=[
            "config.json",
            "manifest.json",
            "ae.safetensors",
            "text_encoder/*",
            "variants/fp8r/*",
            "LICENSE.md",
            "NOTICE.md",
        ],
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
    )
    source_root = Path(sglang.__file__).parent
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SGLANG_REVISION:
        raise RuntimeError(f"Unexpected SGLang revision: {revision}")

    # Bound startup work; longer captions use SGLang's exact eager fallback.
    processor = AutoProcessor.from_pretrained(checkpoint / "text_encoder", local_files_only=True)
    config = json.loads((checkpoint / "config.json").read_text())["text_encoder"]
    multiple, maximum = config["pad_multiple"], config["max_tokens"]
    warmup_limit = min(WARMUP_TOKENS, maximum)
    captions = {}
    for length in range(1, maximum + multiple):
        caption = "A blue ceramic mug on a wooden table. " * length
        formatted = processor.apply_chat_template(
            [{"role": "user", "content": caption}], tokenize=False, add_generation_prompt=True
        )
        tokens = processor.tokenizer(formatted, truncation=True, max_length=maximum)["input_ids"]
        bucket = min(((len(tokens) + multiple - 1) // multiple) * multiple, maximum)
        captions.setdefault(bucket, caption)
        if bucket >= warmup_limit:
            break
    expected = {*range(multiple, warmup_limit, multiple), warmup_limit}
    if set(captions) != expected:
        raise RuntimeError(f"Incomplete warmup token buckets: {sorted(captions)}")
    source, patched = patch_engine_source(
        source_root / "multimodal_gen/runtime/warmup_request_builder.py",
        "else (DEFAULT_PLACEHOLDER_PROMPT,)",
        f"else {tuple(captions.values())!r}",
    )
    vae_patch_sha256 = None
    if high_res:
        vae_path = source_root / "multimodal_gen/runtime/models/vaes/flux3_image_vae.py"
        vae_path.write_text(guard_vae(vae_path.read_text()))
        vae_patch_sha256 = hashlib.sha256(vae_path.read_bytes()).hexdigest()
    build = {
        "vae_patch_sha256": vae_patch_sha256,
        "model": model,
        "model_revision": REVISIONS[model],
        "pup_revision": PUP_REVISION,
        "sglang_revision": revision,
        "registry": REGISTRY,
        "text_token_buckets": sorted(captions),
        "text_encoder_max_tokens": maximum,
        "warmup_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "warmup_patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "sglang", "transformers", "pillow")},
    }
    Path("/build.json").write_text(json.dumps(build, indent=2))
    print(json.dumps(build), flush=True)


image = (
    modal.Image.from_registry(REGISTRY)
    .apt_install("g++", "libgl1", "libglib2.0-0", "libnuma1")
    .env({"HF_HUB_OFFLINE": "0", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(download_pup, secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
    .add_local_file(ROOT / "requirements-pup.txt", "/opt/requirements-pup.txt", copy=True)
    .run_commands(
        "python3 -m pip install --no-deps uv==0.9.18",
        "uv venv --python python3 /opt/pup",
        "uv pip install --python /opt/pup/bin/python --index-strategy unsafe-best-match -r /opt/requirements-pup.txt",
        "uv pip install --python /opt/pup/bin/python flashinfer-jit-cache==0.6.8.post1 --extra-index-url https://flashinfer.ai/whl/cu130",
        "ln -s lib /opt/pup/lib/python3.12/site-packages/nvidia/cu13/lib64",
        "ln -s libcudart.so.13 /opt/pup/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so",
    )
    .run_function(
        prepare_model,
        kwargs={"model": MODEL, "high_res": HIGH_RES},
        secrets=[modal.Secret.from_name("huggingface")],
        timeout=1800,
    )
    .env(CACHE_ENV)
    .run_function(
        compile_kernels,
        kwargs={"model": MODEL},
        gpu=GPU,
        cpu=4,
        memory=65536,
        timeout=2400,
    )
    .add_local_file(ROOT / "flux3_api.py", "/root/flux3_api.py", copy=True)
    .add_local_file(ROOT / "app.py", "/root/app.py", copy=True)
    .env(
        {
            "MODEL": MODEL,
            "GPU": GPU,
            "SNAPSHOT": str(int(SNAPSHOT)),
            "APP_NAME": APP_NAME,
            "WARM": str(WARM),
            "REPLICAS": str(REPLICAS),
            "IDLE_SECONDS": str(IDLE_SECONDS),
            "HIGH_RES": str(int(HIGH_RES)),
        }
    )
)
app = modal.App(APP_NAME)


@app.server(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=65536,
    port=8000,
    min_containers=WARM,
    max_containers=REPLICAS,
    target_concurrency=1,
    max_concurrency=4,  # equals the in-flight cap in flux3_api.py, so overflow waits in Modal's queue
    scaledown_window=IDLE_SECONDS,
    startup_timeout=900,
    routing_region="us-east",
    unauthenticated=False,
    # The source is in the image, so an unchanged redeploy keeps its memory snapshot.
    include_source=False,
    enable_memory_snapshot=SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": True} if SNAPSHOT else {},
)
class Server:
    @modal.enter(snap=SNAPSHOT)
    def prepare(self):
        import httpx
        from transformers import AutoProcessor, AutoTokenizer

        self.source_task = os.getenv("MODAL_TASK_ID")
        self.prepared_at = time.time()
        self.processes = {}
        pup_command = prompt_command(assigned_gpu_name())
        # Both engines boot at once; the prompt engine goes first so its memory check sees the GPU
        # before the image engine allocates, and its fraction leaves the image engine room.
        image_command = engine_command(MODEL)
        self.launch("pup", pup_command, {**os.environ, **PUP_ENV})
        self.launch("image", image_command, {**os.environ, **ENGINE_ENV})
        # Tokenizers and prompts load while the engines boot.
        self.tokenizer = AutoTokenizer.from_pretrained("/weights/pup", local_files_only=True)
        self.image_processor = AutoProcessor.from_pretrained(
            Path(CHECKPOINT) / "text_encoder", local_files_only=True
        )
        self.image_text_max_tokens = json.loads((Path(CHECKPOINT) / "config.json").read_text())[
            "text_encoder"
        ]["max_tokens"]
        self.serving_policy = serving_policy()
        self.request_timeout = REQUEST_TIMEOUT
        self.system_prompts = json.loads(Path("/weights/pup/system_prompts.json").read_text())
        self.system_prompt = self.system_prompts["t2i"]
        self.stop_tokens = [self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        self.wait_for("pup", 9002)
        self.wait_for("image", 8001)
        caption = expand_once(
            self.tokenizer, self.system_prompt, "Two blue mugs on a wooden table by a window", 120
        )
        from flux3_api import prepare_generation, validate_caption_tokens

        prepared = prepare_generation({"caption": caption, "seed": 42}, MODEL, self.serving_policy)
        validate_caption_tokens(prepared.text.caption, self.image_processor, self.image_text_max_tokens)
        generate_once(prepared.image_parameters(), 120)
        if SNAPSHOT:
            with httpx.Client(timeout=120) as client:
                client.post(
                    "http://127.0.0.1:9002/sleep", params={"level": 1, "mode": "abort"}
                ).raise_for_status()
                if (
                    not client.get("http://127.0.0.1:9002/is_sleeping")
                    .raise_for_status()
                    .json()["is_sleeping"]
                ):
                    raise RuntimeError("Prompt expander did not sleep before GPU capture")
        self.build = json.loads(Path("/build.json").read_text())
        self.commands = {"pup": pup_command, "image": image_command}
        print(
            "FLUX3_PREPARED " + json.dumps({"source_task_id": self.source_task, "at": time.time()}),
            flush=True,
        )

    def launch(self, name, command, env):
        self.processes[name] = subprocess.Popen(command, env=env, start_new_session=True)

    def wait_for(self, name, port, timeout=None):
        import httpx

        deadline = time.monotonic() + (
            timeout if timeout is not None else max(1, 850 - (time.time() - self.prepared_at))
        )
        with httpx.Client(timeout=5) as client:
            while time.monotonic() < deadline:
                if self.processes[name].poll() is not None:
                    raise RuntimeError(f"{name} exited; inspect the container logs")
                try:
                    if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                time.sleep(1)
        raise TimeoutError(f"{name} exceeded the startup deadline")

    @modal.enter(snap=False)
    def serve(self):
        import httpx
        import uvicorn

        started = time.perf_counter()
        if SNAPSHOT:
            with httpx.Client(timeout=120) as client:
                client.post("http://127.0.0.1:9002/wake_up").raise_for_status()
                if client.get("http://127.0.0.1:9002/is_sleeping").raise_for_status().json()["is_sleeping"]:
                    raise RuntimeError("Prompt expander remained asleep after wake")
        self.wait_for("pup", 9002, timeout=120)
        self.wait_for("image", 8001, timeout=120)
        task_id = os.getenv("MODAL_TASK_ID")
        self.state = {
            "model": MODEL,
            "quality": QUALITY,
            "high_resolution": HIGH_RES,
            "idle_seconds": IDLE_SECONDS,
            "max_replicas": REPLICAS,
            "task_id": task_id,
            "source_task_id": self.source_task,
            "restored": task_id != self.source_task,
            "prepared_at": self.prepared_at,
            "ready_at": time.time(),
            "restore_hook_s": time.perf_counter() - started,
            "region": os.getenv("MODAL_REGION"),
            "gpu": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip(),
            "build": self.build,
            "commands": self.commands,
        }

        async def start_http():
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                self.client = client
                await uvicorn.Server(
                    uvicorn.Config(create_api(self), host="0.0.0.0", port=8000, log_level="warning")
                ).serve()

        self.http = threading.Thread(target=lambda: asyncio.run(start_http()), daemon=True)
        self.http.start()
        print("FLUX3_HTTP_READY " + json.dumps(self.state), flush=True)

    @modal.exit()
    def stop(self):
        for process in getattr(self, "processes", {}).values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)


@app.local_entrypoint()
def main(warm: int = -1):
    """Print the deployed URL; optionally update the minimum warm replica count."""
    server = modal.Server.from_name(APP_NAME, "Server")
    if warm != -1:
        if not 0 <= warm <= REPLICAS:
            raise ValueError("--warm must be between zero and REPLICAS")
        server.update_autoscaler(min_containers=warm)
    print(server.get_url())
