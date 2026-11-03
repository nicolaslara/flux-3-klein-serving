# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 Klein requests and HTTP serving, with bounded GPU admission.

Reference sizing and inline layout reconciliation adapt Black Forest Labs'
pipeline.py, pup.py and bbox.py at bfc8889a4497f846416f253a3972a1941ac698b9:
https://github.com/black-forest-labs/flux-3-image-klein/tree/bfc8889a4497f846416f253a3972a1941ac698b9/src/flux3
ServingPolicy defines application budgets, not universal model limits.
"""

import asyncio
import base64
import binascii
import hashlib
import io
import json
import math
import os
import re
import time
from urllib.parse import quote
from dataclasses import asdict, dataclass
from typing import Literal, Self

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SCALE = 1000
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _detect(prompt: str) -> tuple[str, list[dict]] | None:
    """(prompt, rows) when the prompt already carries boxes on the 0-1000 canvas, else None. Two forms:
    "<caption> [rows]" (the DiT prompt format) and the upsampler's own "ORIGINAL PROMPT: ... LOCKED
    BOUNDING-BOX ANNOTATIONS ...:\\n[rows]" input. A trailing list that is not all box rows (e.g. a
    caption ending in "[SALE]") is not a box table."""
    text, rows = _split(prompt)
    if not rows or not all(
        isinstance(r, dict)
        and isinstance(r.get("id"), str)
        and _ID.fullmatch(r["id"])
        and any(_valid_box(r.get(k)) for k in ("bbox", "src_bbox", "tgt_bbox"))
        for r in rows
    ):
        return None
    locked = re.fullmatch(r"ORIGINAL PROMPT:\n(.*?)\n\nLOCKED BOUNDING-BOX ANNOTATIONS .*", text, re.S)
    return (locked.group(1) if locked else text), rows


def _is_finished(prompt: str) -> bool:
    """An upsampler answer, not a request: its box table is cited in full as <id>s in the caption."""
    found = _detect(prompt)
    return found is not None and all(f"<{row['id']}>" in found[0] for row in found[1])


def _layout_plan(prompt: str, n_images: int) -> tuple[str, str, list[dict] | None]:
    """Choose the pinned layout prompt when an inline box table is present."""
    task = "i2i" if n_images else "t2i"
    if found := _detect(prompt):
        prompt, locked = found
        return f"{task}_bbox", _upsampling_input(prompt, locked), locked
    return task, prompt, None


def _upsampling_input(prompt: str, rows: list[dict]) -> str:
    return (
        f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n"
        "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
        "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
        f"provenance value, null, and coordinate exactly:\n{json.dumps(rows, ensure_ascii=False)}"
    )


def _split(text: str) -> tuple[str, list | None]:
    """caption, rows: the last '[' from which the rest parses as a JSON list. Not a regex: descs may
    contain brackets and escaped quotes."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


def _valid_box(box) -> list | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf")
        for v in box
    ):
        return None
    y1, x1, y2, x2 = box
    # max <= 1 means 0-1 coordinates, not the 0-1000 canvas
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > _SCALE or x2 > _SCALE or max(box) <= 1:
        return None
    return list(box)


def _valid_added(row, caption: str) -> dict | None:
    """A row the upsampler added: kept only if its id is cited as <id> and it is well formed."""
    if not isinstance(row, dict):
        return None
    row_id, desc = row.get("id"), row.get("desc")
    if (
        not isinstance(row_id, str)
        or not _ID.fullmatch(row_id)
        or f"<{row_id}>" not in caption
        or not isinstance(desc, str)
        or not desc.strip()
    ):
        return None
    if (box := _valid_box(row.get("bbox"))) is not None:
        return {"id": row_id, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = _valid_box(src) if src is not None else None
    tgt_box = _valid_box(tgt) if tgt is not None else None
    if (
        not isinstance(origin, str)
        or not re.fullmatch(r"ref_image_\d+|target", origin)
        or (src is not None and src_box is None)
        or (tgt is not None and tgt_box is None)
        or (src_box is None and tgt_box is None)
        or (origin == "target") == (src_box is not None)
    ):
        return None
    return {"id": row_id, "from": origin, "src_bbox": src_box, "tgt_bbox": tgt_box, "desc": desc.strip()}


def _iou(a: list, b: list) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _reconcile(text: str, locked: list[dict]) -> str:
    """The upsampler's answer with the user's boxes re-imposed.

    1. A row re-planning a locked box under its own id (IoU >= 0.9, e.g. balloon_1 for obj_1) takes the
       locked id, in the rows and in the caption's <id> citations.
    2. Locked rows retain identity, geometry and provenance; a generated description may refine them.
    """
    caption, rows = _split(text)
    rows = rows or []
    locked_ids = {r["id"] for r in locked}
    row_ids = {r.get("id") for r in rows if isinstance(r, dict)}
    renamed = {}
    for lock in locked:
        lock_box = lock.get("bbox") or lock.get("tgt_bbox") or lock.get("src_bbox")
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") in renamed or row.get("id") in locked_ids:
                continue
            box = _valid_box(row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox"))
            if box is not None and _iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")

    by_id = {r.get("id"): r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)}
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append(
            {**lock, "desc": desc.strip() if isinstance(desc, str) and desc.strip() else lock.get("desc", "")}
        )
    for row in rows:
        added = _valid_added(row, caption)
        if added is not None and added["id"] not in {r["id"] for r in out}:
            out.append(added)
    return f"{caption} {json.dumps(out, ensure_ascii=False)}"


Model = Literal["distilled", "base"]


@dataclass(frozen=True)
class ServingPolicy:
    quality: Literal["exact", "lossless"] = "exact"
    max_side: int = 4096
    max_output_pixels: int = 4 * 1024**2
    max_steps: int = 100
    max_guidance: float = 20.0
    max_references: int = 10
    max_reference_bytes: int = 10 * 1024**2
    max_reference_pixels: int = 16 * 1024**2
    max_encoded_reference_bytes: int = 40 * 1024**2

    def __post_init__(self):
        if self.quality not in ("exact", "lossless"):
            raise ValueError("Only exact and lossless execution are allowed")
        if min(self.max_side, self.max_output_pixels, self.max_steps) < 1:
            raise ValueError("Serving dimensions and steps must have positive limits")
        if not math.isfinite(self.max_guidance) or self.max_guidance < 0:
            raise ValueError("Serving guidance limit must be finite and nonnegative")
        if not 0 <= self.max_references <= 10:
            raise ValueError("Serving reference count must be between zero and ten")
        if min(self.max_reference_bytes, self.max_reference_pixels, self.max_encoded_reference_bytes) < 1:
            raise ValueError("Serving reference limits must be positive")


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    prompt: str | None = Field(default=None, min_length=1, max_length=65536)
    caption: str | None = Field(default=None, min_length=1, max_length=65536)
    images: list[str] = Field(default_factory=list, max_length=10)
    width: int | None = Field(default=None, strict=True, ge=64, multiple_of=16)
    height: int | None = Field(default=None, strict=True, ge=64, multiple_of=16)
    seed: int = Field(default=0, strict=True, ge=0, le=2**32 - 1)
    steps: int | None = Field(default=None, strict=True, ge=1)
    guidance: float | None = Field(default=None, strict=True, ge=0)
    output_format: Literal["jpeg", "png"] = "jpeg"

    @field_validator("prompt", "caption")
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Text must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def exclusive_inputs(self) -> Self:
        if (self.prompt is None) == (self.caption is None):
            raise ValueError("Supply exactly one of prompt or caption")
        if (self.width is None) != (self.height is None):
            raise ValueError("Supply both width and height, or neither")
        return self


@dataclass(frozen=True)
class PromptPlan:
    caption: str | None
    system_key: str | None
    user_text: str | None
    locked_rows: list[dict] | None
    reference_urls: tuple[str, ...]
    presence_penalty: float

    @property
    def needs_expansion(self) -> bool:
        return self.caption is None

    def finish(self, expanded: str | None = None) -> str:
        if self.caption is not None:
            return self.caption
        if not isinstance(expanded, str) or not expanded.strip():
            raise ValueError("Prompt expansion must finish with a nonempty caption")
        caption = expanded.strip()
        return _reconcile(caption, self.locked_rows) if self.locked_rows else caption

    def messages(self, system_prompts: dict[str, str]) -> list[dict]:
        if not self.needs_expansion:
            raise ValueError("A supplied caption must bypass the prompt upsampler")
        content = self.user_text
        if self.reference_urls:
            content = [{"type": "image_url", "image_url": {"url": url}} for url in self.reference_urls]
            content.append({"type": "text", "text": self.user_text})
        return [
            {"role": "system", "content": system_prompts[self.system_key]},
            {"role": "user", "content": content},
        ]


@dataclass(frozen=True)
class Reference:
    raw: bytes
    size: tuple[int, int]
    conditioning_size: tuple[int, int]
    crop: tuple[int, int, int, int] | None
    pup_size: tuple[int, int] | None
    pup_url: str | None


@dataclass(frozen=True)
class PreparedGeneration:
    request: GenerationRequest
    model: Model
    steps: int
    guidance: float | None
    text: PromptPlan
    references: tuple[Reference, ...]
    quality: Literal["exact", "lossless"]

    def image_parameters(self, expanded: str | None = None) -> dict:
        parameters = {
            "prompt": self.text.finish(expanded),
            "seed": self.request.seed,
            "generator_device": "cpu",
            "num_inference_steps": self.steps,
            "quality": self.quality,
            "output_format": self.request.output_format,
            "response_format": "b64_json",
            "enable_cache_dit": False,
            "enhance_prompt": False,
        }
        if self.request.output_format == "jpeg":
            parameters["output_compression"] = 95
        if self.request.width is not None:
            parameters["size"] = f"{self.request.width}x{self.request.height}"
        elif not self.references:
            parameters["size"] = "1024x1024"
        if self.guidance is not None:
            parameters["guidance_scale"] = self.guidance
        return parameters

    @property
    def media_type(self) -> str:
        return f"image/{self.request.output_format}"

    @property
    def expected_size(self) -> tuple[int, int]:
        if self.request.width is not None:
            return self.request.width, self.request.height
        if not self.references:
            return 1024, 1024
        reference = self.references[0]
        if reference.crop:
            left, top, right, bottom = reference.crop
            return right - left, bottom - top
        return reference.conditioning_size


def _reference_geometry(width: int, height: int, count: int):
    if count > 1:
        if width * height > 1024**2:
            scale = math.sqrt(1024**2 / (width * height))
            width, height = int(width * scale), int(height * scale)
        size = width // 16 * 16, height // 16 * 16
        if min(size) == 0:
            raise ValueError("A reference is too narrow after multi-reference preprocessing")
        return size, None
    scale = min(1.0, math.sqrt(4 * 1024**2 / (width * height)), 4096 / max(width, height))
    while True:
        fit = max(1, round(width * scale)), max(1, round(height * scale))
        canvas = tuple(min(4096, max(64, -(-side // 16) * 16)) for side in fit)
        if canvas[0] * canvas[1] <= 4 * 1024**2:
            break
        scale *= 0.995
    left, top = (canvas[0] - fit[0]) // 2, (canvas[1] - fit[1]) // 2
    return canvas, (left, top, left + fit[0], top + fit[1]) if fit != canvas else None


def pup_size(width: int, height: int, count: int) -> tuple[int, int]:
    if max(width, height) / min(width, height) > 200:
        raise ValueError("Prompt-upsampled references must not exceed 200:1 aspect")
    cap = {1: 1024**2, 2: 724**2}.get(count, 512**2)

    def floor(value):
        return max(32, int(math.floor(value / 32)) * 32)

    w, h = max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)
    if w * h > cap:
        scale = math.sqrt(width * height / cap)
        w, h = floor(width / scale), floor(height / scale)
    if w * h > cap:
        if w == 32:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    return w, h


def _reference(encoded: str, count: int, expand: bool, policy: ServingPolicy) -> Reference:
    if len(encoded) > (policy.max_reference_bytes + 2) // 3 * 4:
        raise ValueError("Reference exceeds the serving byte limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > policy.max_reference_bytes:
            raise ValueError("Reference exceeds the serving byte limit")
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > policy.max_reference_pixels:
                raise ValueError("Reference exceeds the serving pixel limit")
            image = ImageOps.exif_transpose(opened)
            size = image.size
            conditioning, crop = _reference_geometry(*size, count)
            pup_dimensions, pup_url = None, None
            if expand:
                if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
                    rgba = image.convert("RGBA")
                    image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
                image = image.convert("RGB")
                pup_dimensions = pup_size(*size, count)
                if image.size != pup_dimensions:
                    image = image.resize(pup_dimensions, Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=92)
                pup_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("References must be valid base64-encoded images") from exc
    return Reference(raw, size, conditioning, crop, pup_dimensions, pup_url)


def prepare_generation(
    payload: dict | GenerationRequest, model: Model, policy: ServingPolicy | None = None
) -> PreparedGeneration:
    """Validate one self-hosted request and prepare expander/native-engine inputs."""
    policy = policy or ServingPolicy()
    if model not in ("distilled", "base"):
        raise ValueError("Model must be distilled or base")
    request = payload if isinstance(payload, GenerationRequest) else GenerationRequest.model_validate(payload)
    if len(request.images) > policy.max_references:
        raise ValueError(
            "Reference count exceeds this deployment's serving limit; use a B200-only deployment"
        )
    steps = request.steps if request.steps is not None else (16 if model == "distilled" else 50)
    if steps > policy.max_steps:
        raise ValueError("Steps exceed the configured serving limit")
    if model == "distilled" and request.guidance not in (None, 1.0):
        raise ValueError("Distilled samples without CFG; omit guidance or use 1")
    guidance = None if model == "distilled" else (3.5 if request.guidance is None else request.guidance)
    if guidance is not None and guidance > policy.max_guidance:
        raise ValueError("Guidance exceeds the configured serving limit")
    if sum(map(len, request.images)) > policy.max_encoded_reference_bytes:
        raise ValueError("Combined references exceed the serving byte limit")
    direct = request.caption
    if direct is None and _is_finished(request.prompt):
        direct = request.prompt
    references = tuple(
        _reference(value, len(request.images), direct is None, policy) for value in request.images
    )
    if direct is not None:
        text = PromptPlan(direct, None, None, None, (), 0.0)
    else:
        key, user, locked = _layout_plan(request.prompt, len(references))
        if references:
            scaffold = (
                "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
                "User editing instruction: {instruction}\n"
                "Decide the edit, imagine the target image, and output the annotation record."
                if locked is not None
                else "Reference images follow in order (the first image, the second image, ...).\n"
                "User request: {instruction}\nOutput the caption."
            )
            user = scaffold.format(instruction=user)
        text = PromptPlan(
            None, key, user, locked, tuple(r.pup_url for r in references), 0.0 if references else 1.5
        )
    prepared = PreparedGeneration(request, model, steps, guidance, text, references, policy.quality)
    canvas = (
        (request.width, request.height)
        if request.width is not None
        else references[0].conditioning_size
        if references
        else (1024, 1024)
    )
    if max(canvas) > policy.max_side or math.prod(canvas) > policy.max_output_pixels:
        raise ValueError("Output canvas exceeds the configured serving limit")
    if math.prod(canvas) > 4 * 1024**2:
        if model != "distilled" or policy.quality != "lossless":
            raise ValueError("Outputs above four megapixels require guarded Distilled lossless execution")
        if references:
            raise ValueError("Reference edits above four megapixels are not qualified")
    return prepared


def validate_caption_tokens(caption: str, processor, maximum: int = 8192) -> int:
    """Count the native text encoder's formatted input before its truncation path."""
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": caption}], tokenize=False, add_generation_prompt=True
    )
    count = len(processor.tokenizer(formatted, padding=False, truncation=False)["input_ids"])
    if count > maximum:
        raise ValueError(f"Caption needs {count} text-encoder tokens; the package limit is {maximum}")
    return count


def create_api(runtime):
    """Serve one resident PUP/image pair; ambiguous native completion requires restart."""
    api = FastAPI()
    policy = runtime.serving_policy
    active = 0
    quarantined = False
    # SGLang renders one picture at a time; a second admitted request lets its encoding and
    # transfer overlap the next render (+10% throughput, same latency at one in flight).
    image_slot = asyncio.Semaphore(2)
    # High-resolution deployments serialize PUP too, including their smaller requests.
    pipeline_slots = asyncio.Semaphore(1 if policy.max_output_pixels > 4 * 1024**2 else 4)

    def available():
        if quarantined:
            raise HTTPException(503, "Native completion is unknown; restart this worker")

    @api.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Validation inputs can contain prompts or image data, so only return field names.
        fields = sorted({str(error["loc"][-1]) for error in exc.errors()})
        return JSONResponse(
            status_code=422, content={"detail": "Invalid request fields: " + ", ".join(fields)}
        )

    async def native_post(url, **kwargs):
        nonlocal quarantined
        available()
        try:
            response = await runtime.client.post(url, **kwargs)
        except (asyncio.CancelledError, httpx.TransportError):
            # Closing HTTP does not cancel GPU work in either native process, so the engines are in
            # an unknown state: answer this request, then exit so Modal replaces the container.
            quarantined = True
            asyncio.get_running_loop().call_later(2, os._exit, 3)
            raise
        if response.status_code in (400, 422):
            raise HTTPException(422, "Native engine rejected the request or its context length")
        if response.status_code != 200:
            raise HTTPException(502, "Native engine failed")
        return response.json()

    @api.get("/health")
    @api.post("/wake")
    async def health():
        available()
        for name, port in (("pup", 9002), ("image", 8001)):
            if runtime.processes[name].poll() is not None:
                raise HTTPException(503, "Native engine exited")
            try:
                (await runtime.client.get(f"http://127.0.0.1:{port}/health", timeout=5)).raise_for_status()
            except httpx.HTTPError as exc:
                raise HTTPException(503, "Native engine unavailable") from exc
        available()
        return {**runtime.state, "serving_policy": asdict(policy)}

    async def expand(prepared):
        if not prepared.text.needs_expansion:
            return None, 0
        messages = prepared.text.messages(runtime.system_prompts)
        sampling = dict(
            model="pup",
            temperature=0.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            max_tokens=3584,
            presence_penalty=prepared.text.presence_penalty,
            stop_token_ids=runtime.stop_tokens,
        )
        if prepared.references:
            result = await native_post(
                "http://127.0.0.1:9002/v1/chat/completions",
                json={"messages": messages, **sampling, "chat_template_kwargs": {"enable_thinking": False}},
            )
        else:
            text = runtime.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            tokens = runtime.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) + sampling["max_tokens"] > 8192:
                raise HTTPException(422, "Prompt exceeds prompt-expander context")
            result = await native_post(
                "http://127.0.0.1:9002/v1/completions", json={"prompt": tokens, **sampling}
            )
        choice = result["choices"][0]
        caption = choice["message"]["content"] if prepared.references else choice["text"]
        if not isinstance(caption, str) or not caption.strip() or choice.get("finish_reason") != "stop":
            raise HTTPException(502, "Prompt expansion incomplete")
        tokens = result["usage"]["completion_tokens"]
        if type(tokens) is not int or tokens < 0:
            raise HTTPException(502, "Prompt engine returned invalid token usage")
        return caption, tokens

    async def generate_image(prepared, expansion):
        parameters = prepared.image_parameters(expansion)
        if prepared.references:
            result = await native_post(
                "http://127.0.0.1:8001/v1/images/edits",
                data={k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in parameters.items()},
                files=[
                    ("image", (f"ref-{i}.jpg", ref.raw, "image/jpeg"))
                    for i, ref in enumerate(prepared.references)
                ],
            )
        else:
            result = await native_post("http://127.0.0.1:8001/v1/images/generations", json=parameters)
        data = base64.b64decode(result["data"][0]["b64_json"], validate=True)
        await asyncio.to_thread(verify_output, data, prepared)
        return data

    def verify_output(data, prepared):
        with Image.open(io.BytesIO(data)) as output:
            if (
                output.size != prepared.expected_size
                or output.format.lower() != prepared.request.output_format
            ):
                raise HTTPException(502, "Image engine returned unexpected dimensions or format")
            output.load()

    @api.post("/generate")
    async def generate(body: GenerationRequest, request: Request):
        nonlocal active
        available()
        if active >= 4:
            raise HTTPException(429, "Four requests already in flight")
        deadline = asyncio.get_running_loop().time() + getattr(runtime, "request_timeout", 300)
        try:
            # Reference decoding and resizing are CPU work; keep them off the event loop.
            prepared = await asyncio.to_thread(prepare_generation, body, runtime.state["model"], policy)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        active += 1
        try:
            async with asyncio.timeout_at(deadline), pipeline_slots:
                available()
                if await request.is_disconnected():
                    raise HTTPException(499, "Disconnected before generation")
                started = time.perf_counter()
                expansion, pup_tokens = await expand(prepared)
                pup_ms = (time.perf_counter() - started) * 1000 if prepared.text.needs_expansion else 0.0
                caption = prepared.text.finish(expansion)
                try:
                    validate_caption_tokens(caption, runtime.image_processor, runtime.image_text_max_tokens)
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
                queued = time.perf_counter()
                async with image_slot:
                    available()
                    if await request.is_disconnected():
                        raise HTTPException(499, "Disconnected before image generation")
                    admitted = time.perf_counter()
                    data = await generate_image(prepared, expansion)
                    image_ms = (time.perf_counter() - admitted) * 1000
            return Response(
                data,
                media_type=prepared.media_type,
                headers={
                    "X-Task-Id": runtime.state["task_id"],
                    "X-Seed": str(body.seed),
                    "X-Caption-Sha256": hashlib.sha256(caption.encode()).hexdigest(),
                    "X-Caption": quote(caption),
                    "X-Pup-Output-Tokens": str(pup_tokens),
                    "X-Pup-Ms": str(pup_ms),
                    "X-Image-Queue-Ms": str((admitted - queued) * 1000),
                    "X-Image-Ms": str(image_ms),
                },
            )
        except TimeoutError as exc:
            raise HTTPException(504, "Generation deadline exceeded") from exc
        except httpx.TransportError as exc:
            raise HTTPException(503, "Native completion is unknown; restart this worker") from exc
        except (ValueError, KeyError, IndexError, TypeError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(502, "Native engine returned an invalid result") from exc
        finally:
            active -= 1

    return api
