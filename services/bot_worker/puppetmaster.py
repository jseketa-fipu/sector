#!/usr/bin/env python3
"""
Sector bot worker that reads snapshots from Redis, generates AI orders,
and submits them through the /orders API endpoint.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Set

import redis

from sector.puppet import generate_orders_from_snapshot
from sector.models.redis_config import REDIS_SETTINGS


@dataclass(frozen=True)
class BotConfig:
    redis_url: str
    snapshot_key: str
    event_stream: str
    universe_key: str
    human_factions_key: str
    faction_player_prefix: str
    bot_api_url: str
    bot_api_token: str
    poll_interval: float
    max_orders_per_tick: int
    event_block_ms: int
    api_service_host: str | None
    api_service_port: str | None


def _load_config() -> BotConfig:
    return BotConfig(
        redis_url=str(REDIS_SETTINGS.redis_url),
        snapshot_key=REDIS_SETTINGS.snapshot_key,
        event_stream=REDIS_SETTINGS.event_stream,
        universe_key=os.environ.get("UNIVERSE_KEY", "sector:universe_id"),
        human_factions_key=os.environ.get(
            "HUMAN_FACTIONS_KEY", "sector:human_factions"
        ),
        faction_player_prefix=os.environ.get(
            "FACTION_PLAYER_PREFIX", "sector:faction:player"
        ),
        bot_api_url=os.environ.get("BOT_API_URL", "").rstrip("/"),
        bot_api_token=os.environ.get("BOT_API_TOKEN", ""),
        poll_interval=float(os.environ.get("BOT_POLL_INTERVAL", "1.0")),
        max_orders_per_tick=int(os.environ.get("BOT_MAX_ORDERS", "200")),
        event_block_ms=int(os.environ.get("BOT_EVENT_BLOCK_MS", "1000")),
        api_service_host=os.environ.get("API_SERVICE_HOST"),
        api_service_port=os.environ.get("API_SERVICE_PORT"),
    )


_CONFIG = _load_config()
_ACTIVE_API_URL: str | None = None


def _load_snapshot(client: redis.Redis, config: BotConfig) -> Optional[dict]:
    data = client.hget(config.snapshot_key, "data")
    if not data:
        return None
    return json.loads(data)


def _wait_for_tick(client: redis.Redis, config: BotConfig, last_id: str) -> str:
    entries = client.xread(
        {config.event_stream: last_id}, count=1, block=config.event_block_ms
    )
    if not entries:
        return last_id
    _, messages = entries[0]
    if not messages:
        return last_id
    msg_id, _ = messages[0]
    return msg_id


def _get_human_factions(client: redis.Redis, config: BotConfig) -> Set[str]:
    universe_id = client.get(config.universe_key) or "1"
    key = f"{config.human_factions_key}:{universe_id}"
    return set(client.smembers(key) or [])


def _get_universe_id(client: redis.Redis, config: BotConfig) -> str:
    return str(client.get(config.universe_key) or "1")


def _get_claimed_factions(
    client: redis.Redis, config: BotConfig, universe_id: str, known_factions: Set[str]
) -> Set[str]:
    claimed = _get_human_factions(client, config)
    if not known_factions:
        return claimed
    for fid in known_factions:
        if fid in claimed:
            continue
        key = f"{config.faction_player_prefix}:{universe_id}:{fid}"
        if client.get(key):
            claimed.add(fid)
    return claimed


def _api_candidates() -> List[str]:
    candidates: List[str] = []
    if _CONFIG.bot_api_url:
        candidates.append(_CONFIG.bot_api_url)
    if _CONFIG.api_service_host and _CONFIG.api_service_port:
        candidates.append(
            f"http://{_CONFIG.api_service_host}:{_CONFIG.api_service_port}"
        )
    candidates.append("http://api:8000")
    candidates.append("http://api.sector-sim.svc.cluster.local:8000")
    return list(dict.fromkeys(candidates))


def _resolve_api_url() -> str | None:
    for base in _api_candidates():
        req = urllib.request.Request(
            url=f"{base}/health",
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=3):
                return base
        except Exception:
            continue
    return None


def _api_ready() -> bool:
    global _ACTIVE_API_URL
    if _ACTIVE_API_URL:
        return True
    _ACTIVE_API_URL = _resolve_api_url()
    return _ACTIVE_API_URL is not None


def _post_orders(orders: List[dict]) -> None:
    if not orders:
        return
    if not _ACTIVE_API_URL:
        return
    payload = json.dumps({"orders": orders}).encode("utf-8")
    req = urllib.request.Request(
        url=f"{_ACTIVE_API_URL}/orders",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Bot-Token": _CONFIG.bot_api_token,
        },
    )
    with urllib.request.urlopen(req, timeout=5):
        return


def main() -> None:
    global _ACTIVE_API_URL
    if not _CONFIG.bot_api_token:
        raise SystemExit("BOT_API_TOKEN is required for sector bot worker.")
    client = redis.from_url(_CONFIG.redis_url, decode_responses=True)
    last_tick = -1
    last_api_warn = 0.0
    last_event_id = "0-0"
    last_universe_id: Optional[str] = None
    api_hint = _CONFIG.bot_api_url or "auto"
    print(f"[sector-bot] starting poll={_CONFIG.poll_interval}s api={api_hint}")
    while True:
        last_event_id = _wait_for_tick(client, _CONFIG, last_event_id)
        universe_id = _get_universe_id(client, _CONFIG)
        if last_universe_id is None:
            last_universe_id = universe_id
        elif universe_id != last_universe_id:
            last_universe_id = universe_id
            last_tick = -1
            last_event_id = "0-0"
        snapshot = _load_snapshot(client, _CONFIG)
        if not snapshot:
            time.sleep(_CONFIG.poll_interval)
            continue
        known_factions = {
            f.get("id") for f in snapshot.get("factions", []) if isinstance(f, dict)
        }
        tick = int(snapshot.get("tick", 0))
        if tick < last_tick:
            last_tick = -1
            last_event_id = "0-0"
        if tick <= last_tick:
            time.sleep(_CONFIG.poll_interval)
            continue
        human_factions = _get_claimed_factions(
            client, _CONFIG, universe_id, known_factions
        )
        orders = generate_orders_from_snapshot(snapshot)
        filtered = [
            {
                "faction": o.faction,
                "origin_id": o.origin_id,
                "target_id": o.target_id,
                "reason": o.reason,
            }
            for o in orders
            if o.faction not in human_factions
        ][:_CONFIG.max_orders_per_tick]
        if not _api_ready():
            now = time.time()
            if now - last_api_warn > 15:
                print("[sector-bot] API unavailable; skipping orders")
                last_api_warn = now
            time.sleep(_CONFIG.poll_interval)
            continue
        try:
            _post_orders(filtered)
        except Exception as exc:
            _ACTIVE_API_URL = None
            print(f"[sector-bot] failed to post orders: {exc}")
        last_tick = tick
        time.sleep(_CONFIG.poll_interval)


if __name__ == "__main__":
    main()
