"""Minimal Tinker sampler for Qwen/Qwen3.6-35B-A3B base (no adapter). Shared by E8/E10/E11/E14 small-scale runs."""
import io, math, os, threading
import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
_tok = AutoTokenizer.from_pretrained(MODEL)
_client = None
_lock = threading.Lock()
USAGE = {"prefill": 0, "sample": 0, "calls": 0}


def client():
    global _client
    with _lock:
        if _client is None:
            _client = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
    return _client


def tok():
    return _tok


def _img_chunk(data: bytes):
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    fmt = (im.format or "PNG").lower()
    if fmt not in ("png", "jpeg"):
        buf = io.BytesIO(); im.convert("RGB").save(buf, format="PNG"); data, fmt = buf.getvalue(), "png"
    q = 90
    while len(data) > 2_000_000 and q >= 50:  # Tinker asset limit is 2 MiB: re-encode same pixels as JPEG, lowering quality
        buf = io.BytesIO(); im.convert("RGB").save(buf, format="JPEG", quality=q); data, fmt = buf.getvalue(), "jpeg"; q -= 10
    w, h = im.size
    # Qwen2VL smart_resize: factor = patch(16)*merge(2) = 32; min 65536 px, max 16777216 px.
    f, mn, mx = 32, 65536, 16777216
    hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
    if hb * wb > mx:
        b = math.sqrt(h * w / mx); hb, wb = math.floor(h / b / f) * f, math.floor(w / b / f) * f
    elif hb * wb < mn:
        b = math.sqrt(mn / (h * w)); hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
    return types.ImageChunk(data=data, format=fmt, expected_tokens=(hb // f) * (wb // f))


def build_input(messages, tools=None, images=None, enable_thinking=False):
    """messages use OpenAI-ish dicts; images: list of bytes, placed where {"type":"image"} parts occur."""
    text = _tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=enable_thinking)
    images = list(images or [])
    marker = "<|image_pad|>"
    parts = text.split(marker)
    assert len(parts) - 1 == len(images), (len(parts) - 1, len(images))
    chunks = []
    for i, p in enumerate(parts):
        if p:
            chunks.append(types.EncodedTextChunk(tokens=_tok.encode(p, add_special_tokens=False)))
        if i < len(images):
            chunks.append(_img_chunk(images[i]))
    return types.ModelInput(chunks=chunks), text


def sample(messages, max_tokens, tools=None, images=None, temperature=0.0, enable_thinking=False, stop=None):
    mi, text = build_input(messages, tools, images, enable_thinking)
    sp = types.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                              stop=stop or ["<|im_end|>"])
    res = client().sample(prompt=mi, num_samples=1, sampling_params=sp).result()
    seq = res.sequences[0]
    out = _tok.decode(seq.tokens, skip_special_tokens=False)
    for s in ("<|im_end|>", "<|endoftext|>"):
        out = out.split(s)[0]
    with _lock:
        USAGE["prefill"] += mi.length; USAGE["sample"] += len(seq.tokens); USAGE["calls"] += 1
    return {"text": out, "prompt_tokens": mi.length, "completion_tokens": len(seq.tokens),
            "stop_reason": getattr(seq, "stop_reason", None)}
