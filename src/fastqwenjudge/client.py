"""Async client for any OpenAI-compatible chat endpoint (vLLM, SGLang, hosted APIs).

All judge requests go through `Client.chat_json`: output constrained to a JSON schema (greedy unless the judge sets
its own sampling), and a retry loop that tells the model what went wrong instead of resending the same request. A
global limit bounds the number of requests in flight; the server does its own batching on top of that. Requests
waiting for a slot are admitted in arrival order, or by the rank of the image they belong to (see `Slots` and
`request_rank`). Given several servers of the same model, each request goes to the one most likely to have its
prompt's beginning cached (see `PrefixRouter`), or to the one with the fewest requests in flight.
"""
from __future__ import annotations

import asyncio
import contextvars
import copy
import heapq
import itertools
import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI


@dataclass
class Reply:
    answer: Any
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    latency: float
    retries: list[dict] = field(default_factory=list)
    logprobs: list[dict] | None = None   # the answer's tokens with their top alternatives, when asked for


@dataclass
class RequestLog:
    stage: str
    start: float
    latency: float
    prompt_tokens: int
    completion_tokens: int
    ok: bool
    rank: int = 0
    server: int = 0


# Rank of the image a request belongs to, lower is more urgent. The runner sets it in each image's task; every
# request made from that task inherits it.
request_rank: contextvars.ContextVar[int] = contextvars.ContextVar("request_rank", default=0)


class Slots:
    """A counting semaphore that admits waiters by rank, lowest first and in arrival order among equal ranks.
    With all ranks equal it is a plain FIFO semaphore."""

    def __init__(self, n: int):
        self.free = n
        self.waiters: list[tuple[int, int, asyncio.Future]] = []
        self.arrival = itertools.count()

    async def acquire(self, rank: int = 0):
        if self.free and not self.waiters:
            self.free -= 1
            return
        granted = asyncio.get_running_loop().create_future()
        heapq.heappush(self.waiters, (rank, next(self.arrival), granted))
        try:
            await granted
        except asyncio.CancelledError:
            if granted.done() and not granted.cancelled():
                self.release()   # granted and cancelled in the same tick: hand the slot on
            raise

    def release(self):
        while self.waiters:
            *_, granted = heapq.heappop(self.waiters)
            if not granted.done():   # waiters cancelled while queued are skipped
                granted.set_result(None)
                return
        self.free += 1


def prefix_chain(messages: list[dict], chunk: int = 256) -> list[int]:
    """Hashes of a prompt's growing prefixes, one per `chunk` characters of text and one per image, in prompt order.
    Two prompts share the first k hashes when they share their beginning up to the k-th piece."""
    chain, h = [], 0
    for message in messages:
        content = message["content"]
        h = hash((h, message["role"]))
        for part in [{"type": "text", "text": content}] if isinstance(content, str) else content:
            if part["type"] == "text":
                pieces = [part["text"][i:i + chunk] for i in range(0, len(part["text"]), chunk)]
            else:
                pieces = [part["image_url"]["url"]]
            for piece in pieces:
                h = hash((h, piece))
                chain.append(h)
    return chain


class PrefixRouter:
    """Cache-aware routing over servers that each keep their own prefix cache. A request goes to the server that was
    sent the longest beginning of its prompt, ties to the one with the fewest in flight, so requests that share a
    long prefix (images judged against one checklist, a retry after its first attempt) land where that prefix is
    likely still cached. A server that already has more than `load_factor` times its share of the requests in
    flight is passed over (consistent hashing with bounded loads), so a popular prefix cannot pile onto one server.

    The record of what each server holds is approximate: the last `capacity` prompt pieces sent to it, oldest
    forgotten first, the way its cache evicts. A wrong guess only costs the cache hit, never the answer."""

    def __init__(self, servers: int, capacity: int = 8192, load_factor: float = 1.25):
        self.sent = [OrderedDict() for _ in range(servers)]
        self.capacity = capacity
        self.load_factor = load_factor

    def pick(self, messages: list[dict], in_flight: list[int]) -> int:
        chain = prefix_chain(messages)
        # at least one server is always under the cap: all of them at it would add up to more than are in flight
        cap = math.ceil(self.load_factor * (sum(in_flight) + 1) / len(in_flight))

        def score(server: int) -> tuple[int, int]:
            matched = next((i for i, h in enumerate(chain) if h not in self.sent[server]), len(chain))
            return -matched, in_flight[server]
        server = min((s for s in range(len(in_flight)) if in_flight[s] < cap), key=score)
        sent = self.sent[server]
        for h in chain:
            sent[h] = None
            sent.move_to_end(h)
        while len(sent) > self.capacity:
            sent.popitem(last=False)
        return server


class Client:
    def __init__(self, base_url: str | list[str], model: str, *, concurrency: int = 128, api_key: str = "EMPTY",
                 timeout: float = 600, retries: int = 3, seed: int | None = 0, temperature: float = 0.0,
                 admission: str = "fifo", server_priority: bool = False, routing: str = "least_outstanding"):
        """`admission="rank"` gives a free slot to the waiting request with the lowest `request_rank`, so images
        already in progress finish before new ones start; "fifo" ignores ranks. `server_priority` also sends the
        rank as vLLM's request priority (the server must run with --scheduling-policy priority). With several
        servers, `routing` is "least_outstanding" (fewest requests in flight) or "prefix" (`PrefixRouter`, for servers
        that keep a prefix cache; the bundled profiles turn it off, and every request shares the system prompt, so
        prefix routing would only pile requests onto one server up to its load bound)."""
        if admission not in ("fifo", "rank"):
            raise ValueError(f"admission must be 'fifo' or 'rank', not {admission!r}")
        if routing not in ("prefix", "least_outstanding"):
            raise ValueError(f"routing must be 'prefix' or 'least_outstanding', not {routing!r}")
        urls = [base_url] if isinstance(base_url, str) else base_url
        self.apis = [AsyncOpenAI(base_url=url, api_key=api_key, timeout=timeout, max_retries=0) for url in urls]
        self.in_flight = [0] * len(self.apis)
        self.router = PrefixRouter(len(self.apis)) if routing == "prefix" and len(self.apis) > 1 else None
        self.model = model
        self.limit = Slots(concurrency)
        self.admission = admission
        self.server_priority = server_priority
        self.retries = retries
        self.seed = seed
        self.temperature = temperature
        self.log: list[RequestLog] = []

    # Close clients explicitly. An unclosed client that is garbage-collected while a newer one is busy can leave
    # some of the newer client's requests hanging until they time out.
    async def close(self):
        for api in self.apis:
            await api.close()

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def _create(self, stage: str, **payload) -> dict:
        rank = request_rank.get()
        if self.server_priority:
            payload["extra_body"] = {**payload.get("extra_body", {}), "priority": rank}
        await self.limit.acquire(rank if self.admission == "rank" else 0)
        # the count changes before the first await, so concurrent picks see it
        if self.router:
            server = self.router.pick(payload["messages"], self.in_flight)
        else:
            server = min(range(len(self.apis)), key=self.in_flight.__getitem__)
        self.in_flight[server] += 1
        try:
            start = time.monotonic()
            try:
                response = await self.apis[server].chat.completions.create(model=self.model, **payload)
            except Exception:
                self.log.append(RequestLog(stage, start, time.monotonic() - start, 0, 0, False, rank, server))
                raise
            raw = response.model_dump(mode="json", exclude_none=True)
            usage = raw.get("usage") or {}
            self.log.append(RequestLog(stage, start, time.monotonic() - start, usage.get("prompt_tokens", 0),
                                       usage.get("completion_tokens", 0), True, rank, server))
            return raw
        finally:
            self.in_flight[server] -= 1
            self.limit.release()

    async def chat_json(self, stage: str, system: str, content: list[dict], schema: dict, *, max_tokens: int,
                        validate: Callable[[dict], Any] = lambda a: a, field_name: str | None = None,
                        count: int | None = None, temperature: float | None = None,
                        extra_body: dict | None = None, top_logprobs: int | None = None) -> Reply:
        """One schema-constrained request. If `field_name`/`count` are given, the schema also pins the length of
        that array, so the model returns exactly one entry per requirement. `temperature` overrides the client's
        (a judge that samples, such as a thinking model, sets its own); `extra_body` adds server-specific request
        fields (sampling settings, chat-template switches). With `top_logprobs`, the reply carries each answer
        token's log-probability and that many alternatives (vLLM: the model's raw probabilities, before the JSON
        grammar and the sampling settings)."""
        schema = copy.deepcopy(schema)
        if field_name and count:
            schema["properties"][field_name].update(minItems=count, maxItems=count)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        payload = {"messages": messages, "temperature": self.temperature if temperature is None else temperature,
                   "max_tokens": max_tokens, "response_format": {
                       "type": "json_schema", "json_schema": {"name": stage, "strict": True, "schema": schema}}}
        if self.seed is not None:   # vLLM samples requests that carry their own seed one at a time
            payload["seed"] = self.seed
        if extra_body:
            payload["extra_body"] = extra_body
        if top_logprobs:
            payload.update(logprobs=True, top_logprobs=top_logprobs)
        retries, started, tokens = [], time.monotonic(), [0, 0]
        for attempt in range(1, self.retries + 1):
            finish = None
            try:
                raw = await self._create(stage, **payload)
                usage = raw.get("usage") or {}
                tokens[0] += usage.get("prompt_tokens", 0)
                tokens[1] += usage.get("completion_tokens", 0)
                choice = raw["choices"][0]
                finish = choice.get("finish_reason")
                if finish != "stop":
                    raise ValueError(f"Incomplete response: {finish}")
                answer = validate(json.loads(choice["message"]["content"]))
                return Reply(answer, attempt, tokens[0], tokens[1], time.monotonic() - started, retries,
                             (choice.get("logprobs") or {}).get("content"))
            except Exception as error:
                last = error
                note = {"attempt": attempt, "error": str(error)[:1000], "finish_reason": finish}
                if attempt < self.retries:
                    correction = feedback(error, finish, field_name, count)
                    if correction:
                        messages.append({"role": "user", "content": correction})
                        note["correction"] = correction
                    await asyncio.sleep(min(2 ** attempt, 20))
                retries.append(note)
        raise RuntimeError(f"{stage}: {type(last).__name__}: {str(last)[:800]}")


def feedback(error: Exception, finish: str | None, field_name: str | None, count: int | None) -> str | None:
    if finish == "length":
        # A plain retry tends to repeat the same long enumeration and get cut off again.
        return (f"Your response reached the output limit before completing JSON. Return complete JSON with "
                f"exactly {count} {field_name}. Keep each evidence field to one short sentence of at most 40 "
                "words, avoid repeated descriptions, and retain every required field and observation.")
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return ("Your response failed validation: " + str(error)[:300] +
                ". Return a corrected complete JSON response. Do not repeat the invalid construction.")
    return None
