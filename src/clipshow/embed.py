"""Local embedding models (lazy-loaded, CPU by default)."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config


@lru_cache(maxsize=1)
def device() -> str:
    """CLIPSHOW_DEVICE: 'cpu', 'cuda', or 'auto' (the GPU if there is one; ROCm builds also report as "cuda").

    The web app defaults to the CPU: a single query embeds in milliseconds there, and doesn't queue behind
    indexing work on the GPU. The worker defaults to auto.
    """
    import os

    import torch

    want = os.environ.get("CLIPSHOW_DEVICE", "auto")
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


@lru_cache(maxsize=1)
def _text_model():
    from sentence_transformers import SentenceTransformer

    import os

    import torch

    if os.environ.get("CLIPSHOW_TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["CLIPSHOW_TORCH_THREADS"]))
    return SentenceTransformer(config.TEXT_MODEL, device=device())


@lru_cache(maxsize=1)
def _siglip():
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.set_grad_enabled(False)
    dtype = torch.float16 if device() == "cuda" else torch.float32
    model = AutoModel.from_pretrained(config.VISUAL_MODEL, dtype=dtype).to(device()).eval()
    proc = AutoProcessor.from_pretrained(config.VISUAL_MODEL)
    return model, proc


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def embed_texts(texts: list[str]) -> np.ndarray:
    return _text_model().encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


def embed_query_text(q: str) -> np.ndarray:
    return embed_texts([config.TEXT_QUERY_PREFIX + q])[0]


def _features(out):
    # Newer transformers return a model output object rather than a tensor.
    return out if hasattr(out, "numpy") else out.pooler_output


def embed_images(images: list[Path | bytes], batch: int = 64) -> np.ndarray:
    """Images as file paths or encoded bytes (e.g. JPEG)."""
    import io

    from PIL import Image

    model, proc = _siglip()
    outs = []
    for i in range(0, len(images), batch):
        imgs = [Image.open(io.BytesIO(p) if isinstance(p, bytes) else p).convert("RGB") for p in images[i : i + batch]]
        inputs = proc(images=imgs, return_tensors="pt").to(device(), model.dtype)
        outs.append(_features(model.get_image_features(**inputs)).float().cpu().numpy())
    return _norm(np.concatenate(outs)) if outs else np.zeros((0, 768), dtype=np.float32)


def embed_moment_frames(frame_groups: list[list[Path | bytes]]) -> np.ndarray:
    """Mean-pooled SigLIP embedding per moment."""
    flat = [p for g in frame_groups for p in g]
    vecs = embed_images(flat)
    out, k = [], 0
    for g in frame_groups:
        out.append(vecs[k : k + len(g)].mean(axis=0))
        k += len(g)
    return _norm(np.stack(out))


def embed_query_visual(q: str) -> np.ndarray:
    model, proc = _siglip()
    inputs = proc(text=[f"a still from a TV show of {q}"], padding="max_length", return_tensors="pt").to(device())
    return _norm(_features(model.get_text_features(**inputs)).float().cpu().numpy())[0]
