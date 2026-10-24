"""Gemini Flash (via OpenRouter) watches each moment's clip (video + audio) and describes the reaction."""

import base64
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from . import config, embed, store
from .render import make_clip
from .store import Moment

PROMPT = """This is a short clip from a TV episode, with audio. It is a candidate "reaction moment" \
someone might want to send as a reaction GIF.

Transcript of the line(s) around it: {context}

Describe it so it can be found by a search like "sarcastic eye roll" or "awkward silence after bad news". \
Focus on facial expressions, body language, tone of voice (sarcastic, deadpan, yelling, laughing), \
and the social meaning of the moment. Do not guess character names unless the dialogue states them."""


class Description(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict json_schema needs additionalProperties: false

    reaction: str = Field(description='One sentence on what makes this a reaction, e.g. "stunned disbelief at being dumped"')
    emotion: str = Field(description="Comma-separated emotions shown")
    action: str = Field(description="Visible actions, facial expressions, gestures")
    tone: str = Field(description='How the line is delivered, or "none" if silent')
    reaction_gif_worthy: bool

    def as_text(self) -> str:
        return f"{self.reaction} Emotion: {self.emotion}. Action: {self.action}. Tone: {self.tone}."


class OpenRouter:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        self.http = httpx.Client(
            timeout=120,
            headers={"Authorization": f"Bearer {config.openrouter_key()}", "X-Title": "Clip Show"},
        )
        self.cost = 0.0
        self._lock = threading.Lock()

    def complete(self, content: list[dict], schema: dict) -> str:
        resp = self.http.post(self.URL, json={
            "model": config.GEMINI_MODEL,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "description", "strict": True, "schema": schema}},
            "reasoning": {"effort": config.GEMINI_REASONING},
            "usage": {"include": True},
        })
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(body["error"])
        with self._lock:
            self.cost += float(body.get("usage", {}).get("cost") or 0.0)
        return body["choices"][0]["message"]["content"]


def _client():
    return OpenRouter()


def describe_moment(client: OpenRouter, m: Moment) -> Description:
    with tempfile.TemporaryDirectory() as tmp:
        clip = make_clip(m, Path(tmp) / "c.mp4", m.start, m.end, height=360, subs=False)
        data = base64.b64encode(clip.read_bytes()).decode()
    text = client.complete(
        [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{data}"}},
            {"type": "text", "text": PROMPT.format(context=m.context or "(none)")},
        ],
        Description.model_json_schema(),
    )
    return Description.model_validate_json(text)


def _with_retries(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** (i + 1))


def describe_pending(conn, episode_ids: list[int] | None = None, limit: int | None = None, workers: int = 8, log=print) -> int:
    """Describes every moment without a description yet. Resumable: commits as it goes."""
    q = store.MOMENT_SELECT + " WHERE m.description IS NULL"
    args: list = []
    if episode_ids:
        q += f" AND m.episode_id IN ({','.join('?' * len(episode_ids))})"
        args += episode_ids
    q += " ORDER BY m.episode_id, m.start"
    if limit:
        q += f" LIMIT {int(limit)}"
    moments = [store.row_to_moment(r) for r in conn.execute(q, args)]
    if not moments:
        return 0

    client = _client()
    done, failed = 0, 0
    pending: list[tuple[int, str]] = []

    def flush():
        if not pending:
            return
        ids = [p[0] for p in pending]
        vecs = embed.embed_texts([p[1] for p in pending])
        for mid, text in pending:
            store.set_description(conn, mid, text)
        store.put_vectors(conn, "desc", ids, vecs)
        conn.commit()
        pending.clear()

    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(_with_retries, lambda m=m: describe_moment(client, m)): m for m in moments}
        for fut in as_completed(futs):
            m = futs[fut]
            try:
                pending.append((m.id, fut.result().as_text()))
                done += 1
            except Exception as e:  # noqa: BLE001 - log and keep going; the moment stays pending
                failed += 1
                log(f"  describe failed for moment {m.id} ({m.label} {m.start:.1f}s): {e}")
            if len(pending) >= 32:
                flush()
                log(f"  described {done}/{len(moments)}  (${client.cost:.3f} so far)")
    flush()
    log(f"  described {done}, failed {failed}, cost ${client.cost:.4f}" + (f" (${client.cost / done:.5f}/moment)" if done else ""))
    return done
