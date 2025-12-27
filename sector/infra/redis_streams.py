#!/usr/bin/env python3
"""
Lightweight Redis Streams helper for stateless workers.

Provides a thin wrapper around redis.asyncio for:
- appending frames/events to an append-only stream
- reading frames (optionally via consumer groups)
- storing/loading the latest snapshot in a hash
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, List, Optional

from redis import asyncio as aioredis
from redis.exceptions import ResponseError


DEFAULT_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_EVENT_STREAM = os.environ.get("EVENT_STREAM", "sector:events")
DEFAULT_ORDER_STREAM = os.environ.get("ORDER_STREAM", "sector:orders")
DEFAULT_SNAPSHOT_KEY = os.environ.get("SNAPSHOT_KEY", "sector:snapshot")


class RedisStreams:
    def __init__(
        self,
        url: str | None = None,
        event_stream: str | None = None,
        order_stream: str | None = None,
        snapshot_key: str | None = None,
    ) -> None:
        self.url = url or DEFAULT_REDIS_URL
        self.event_stream = event_stream or DEFAULT_EVENT_STREAM
        self.order_stream = order_stream or DEFAULT_ORDER_STREAM
        self.snapshot_key = snapshot_key or DEFAULT_SNAPSHOT_KEY
        # decode_responses=True so we deal with str, not bytes
        self._redis = aioredis.from_url(self.url, decode_responses=True)

    @property
    def client(self):
        return self._redis

    async def close(self) -> None:
        await self._redis.close()

    # --- Consumer group helpers ---
    async def ensure_consumer_group(self, stream: str, group: str) -> None:
        """
        Create the consumer group if it does not already exist.
        """
        try:
            await self._redis.xgroup_create(stream, group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                return
            raise

    async def read_orders(
        self,
        group: str,
        consumer: str,
        count: int = 20,
        block_ms: int = 1000,
    ) -> list[tuple[str, dict]]:
        """
        Read orders from the orders stream via consumer group semantics.
        Returns a list of (message_id, payload_dict).
        """
        entries = await self._redis.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={self.order_stream: ">"},
            count=count,
            block=block_ms,
        )
        if not entries:
            return []
        out: list[tuple[str, dict]] = []
        for _, messages in entries:
            for message_id, fields in messages:
                payload_raw = fields.get("data")
                payload = json.loads(payload_raw) if payload_raw else {}
                out.append((message_id, payload))
        return out

    async def ack_orders(self, ids: Iterable[str], group: str) -> None:
        if not ids:
            return
        await self._redis.xack(self.order_stream, group, *ids)

    # --- Event / frame helpers ---
    async def append_event(self, payload: dict, maxlen: Optional[int] = None) -> str:
        """
        Append a payload to the event stream. Uses field name 'data' to store JSON.
        """
        args: dict[str, Any] = {"data": json.dumps(payload)}
        return await self._redis.xadd(
            name=self.event_stream,
            fields=args,
            maxlen=maxlen,
            approximate=True,
        )

    async def read_events(
        self,
        last_id: str = "$",
        count: int = 50,
        block_ms: int | None = None,
    ) -> list[tuple[str, dict]]:
        """
        Read events after last_id (inclusive semantics handled by caller).
        Use last_id="$" to block for new entries.
        """
        entries = await self._redis.xread(
            streams={self.event_stream: last_id},
            count=count,
            block=block_ms,
        )
        if not entries:
            return []
        out: list[tuple[str, dict]] = []
        for _, messages in entries:
            for message_id, fields in messages:
                payload_raw = fields.get("data")
                payload = json.loads(payload_raw) if payload_raw else {}
                out.append((message_id, payload))
        return out

    async def tail_events(self, count: int = 50) -> list[dict]:
        """
        Fetch the most recent events (reverse range).
        """
        entries = await self._redis.xrevrange(self.event_stream, max="+", min="-", count=count)
        out: list[dict] = []
        for _, fields in entries:
            payload_raw = fields.get("data")
            payload = json.loads(payload_raw) if payload_raw else {}
            out.append(payload)
        out.reverse()
        return out

    # --- Snapshot helpers ---
    async def save_snapshot(self, snapshot: dict) -> None:
        """
        Store the latest snapshot as a hash. The 'data' field holds JSON.
        """
        mapping = {"data": json.dumps(snapshot), "tick": str(snapshot.get("tick", 0))}
        await self._redis.hset(self.snapshot_key, mapping=mapping)

    async def load_snapshot(self) -> Optional[dict]:
        data = await self._redis.hget(self.snapshot_key, "data")
        if not data:
            return None
        return json.loads(data)
