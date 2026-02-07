#!/usr/bin/env python3
"""
Simulation worker using the rich sector world/bot logic.

This worker owns the in-memory world, optionally consumes external orders from Redis,
advances the world, and publishes snapshots/events back to Redis. A Redis lease
ensures only one worker advances a given sim at a time.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

from sector.puppet import reset_factions
from sector.models import SIM_CONFIG
from sector.infra.redis_streams import RedisStreams
from sector.state_utils import snapshot_from_world
from sector.world import Fleet, Order, World, advance_world, create_sector, FACTION_NAMES


@dataclass(frozen=True)
class WorkerConfig:
    event_maxlen: int
    snapshot_every: int
    order_block_ms: int
    lease_key: str
    lease_ttl_ms: int
    session_key_prefix: str
    address_session_prefix: str
    player_faction_prefix: str
    faction_player_prefix: str
    worker_id: str
    universe_key: str
    human_factions_key: str
    restart_key: str
    reset_universe_on_start: bool
    order_group: str
    order_consumer: str
    wait_for_frontend: bool
    frontend_health_url: str


def _load_config() -> WorkerConfig:
    return WorkerConfig(
        event_maxlen=int(os.environ.get("EVENT_STREAM_MAXLEN", 5000)),
        snapshot_every=int(os.environ.get("SNAPSHOT_EVERY", 1)),
        order_block_ms=int(
            os.environ.get(
                "ORDER_BLOCK_MS", SIM_CONFIG.simulation_modifiers.order_block_ms
            )
        ),
        lease_key=os.environ.get("LEASE_KEY", "sector:lease"),
        lease_ttl_ms=int(
            os.environ.get("LEASE_TTL_MS", SIM_CONFIG.lease_ttl_ms)
        ),
        session_key_prefix=os.environ.get("SESSION_KEY_PREFIX", "sector:session"),
        address_session_prefix=os.environ.get(
            "ADDRESS_SESSION_PREFIX", "sector:session:addr"
        ),
        player_faction_prefix=os.environ.get(
            "PLAYER_FACTION_PREFIX", "sector:player:faction"
        ),
        faction_player_prefix=os.environ.get(
            "FACTION_PLAYER_PREFIX", "sector:faction:player"
        ),
        worker_id=os.environ.get("WORKER_ID", socket.gethostname()),
        universe_key=os.environ.get("UNIVERSE_KEY", "sector:universe_id"),
        human_factions_key=os.environ.get("HUMAN_FACTIONS_KEY", "sector:human_factions"),
        restart_key=os.environ.get("RESTART_KEY", "sector:restart"),
        reset_universe_on_start=os.environ.get("RESET_UNIVERSE_ON_START", "true")
        .lower()
        == "true",
        order_group=os.environ.get("ORDER_GROUP", "sim"),
        order_consumer=os.environ.get("ORDER_CONSUMER", f"sim-{os.getpid()}"),
        wait_for_frontend=os.environ.get("WAIT_FOR_FRONTEND", "").lower() == "true",
        frontend_health_url=os.environ.get(
            "FRONTEND_HEALTH_URL", "http://viz:9000/"
        ),
    )


_CONFIG = _load_config()

TICK_DELAY: float = SIM_CONFIG.simulation_modifiers.tick_delay


@dataclass
class RouteState:
    faction: str
    path: List[int]
    current_index: int = 0
    pending_hop: bool = False
    last_sent_tick: Optional[int] = None
    fleet_id: Optional[int] = None
    reason: Optional[str] = None


class SimulationWorker:
    def __init__(self) -> None:
        self.streams = RedisStreams()
        self.world: World = create_sector()
        self.consumer_group = _CONFIG.order_group
        self.consumer_name = _CONFIG.order_consumer
        self._stop = asyncio.Event()
        self.lease_key = _CONFIG.lease_key
        self.lease_ttl_ms = _CONFIG.lease_ttl_ms
        self.worker_id = _CONFIG.worker_id
        self._last_frame: dict | None = None
        self._revoked_factions: set[str] = set()
        self._routes: list[RouteState] = []

    def _find_path(self, start_id: int, end_id: int) -> Optional[List[int]]:
        if start_id == end_id:
            return [start_id]
        if start_id not in self.world.systems or end_id not in self.world.systems:
            return None
        queue = deque([start_id])
        visited = {start_id}
        prev: dict[int, int] = {}
        while queue:
            current = queue.popleft()
            for neigh in self.world.systems[current].neighbors:
                if neigh in visited:
                    continue
                visited.add(neigh)
                prev[neigh] = current
                if neigh == end_id:
                    path = [end_id]
                    back = current
                    while True:
                        path.append(back)
                        if back == start_id:
                            break
                        back = prev[back]
                    return list(reversed(path))
                queue.append(neigh)
        return None

    def _has_idle_fleet(self, faction: str, system_id: int) -> bool:
        for fl in self.world.fleets.values():
            if (
                fl.owner == faction
                and fl.eta == 0
                and fl.system_id == system_id
                and fl.strength > 0
            ):
                return True
        return False

    def _get_idle_fleet_by_id(
        self, faction: str, system_id: int, fleet_id: int
    ) -> Optional[Fleet]:
        fl = self.world.fleets.get(fleet_id)
        if not fl:
            return None
        if (
            fl.owner != faction
            or fl.eta != 0
            or fl.system_id != system_id
            or fl.strength <= 0
        ):
            return None
        return fl

    def _normalize_external_orders(self, orders: List[Order]) -> List[Order]:
        direct: List[Order] = []
        for order in orders:
            origin = self.world.systems.get(order.origin_id)
            target = self.world.systems.get(order.target_id)
            if not origin or not target:
                continue
            if order.fleet_id is not None:
                if not self._get_idle_fleet_by_id(
                    order.faction, order.origin_id, order.fleet_id
                ):
                    continue
            if order.target_id in origin.neighbors:
                direct.append(order)
                continue
            path = self._find_path(order.origin_id, order.target_id)
            if not path or len(path) < 2:
                continue
            if order.fleet_id is None:
                self._routes = [
                    r
                    for r in self._routes
                    if not (
                        r.faction == order.faction
                        and r.path[0] == path[0]
                        and r.path[-1] == path[-1]
                        and r.fleet_id is None
                    )
                ]
            self._routes.append(
                RouteState(
                    faction=order.faction,
                    path=path,
                    fleet_id=order.fleet_id,
                    reason=order.reason,
                )
            )
        return direct

    def _route_orders(self) -> List[Order]:
        if not self._routes:
            return []
        tick = self.world.tick
        next_routes: list[RouteState] = []
        orders: List[Order] = []
        for route in self._routes:
            if route.current_index >= len(route.path) - 1:
                continue
            from_id = route.path[route.current_index]
            to_id = route.path[route.current_index + 1]
            if route.fleet_id is not None and route.fleet_id not in self.world.fleets:
                continue
            if route.pending_hop:
                if route.fleet_id is not None:
                    if self._get_idle_fleet_by_id(route.faction, to_id, route.fleet_id):
                        route.current_index += 1
                        route.pending_hop = False
                else:
                    if self._has_idle_fleet(route.faction, to_id):
                        route.current_index += 1
                        route.pending_hop = False
                next_routes.append(route)
                continue
            if route.last_sent_tick == tick:
                next_routes.append(route)
                continue
            if route.fleet_id is not None:
                if not self._get_idle_fleet_by_id(route.faction, from_id, route.fleet_id):
                    next_routes.append(route)
                    continue
            else:
                if not self._has_idle_fleet(route.faction, from_id):
                    next_routes.append(route)
                    continue
            orders.append(
                Order(
                    faction=route.faction,
                    origin_id=from_id,
                    target_id=to_id,
                    reason=route.reason or "human-route",
                    fleet_id=route.fleet_id,
                )
            )
            route.pending_hop = True
            route.last_sent_tick = tick
            next_routes.append(route)
        self._routes = [
            r for r in next_routes if r.current_index < len(r.path) - 1
        ]
        return orders

    async def setup(self) -> None:
        await self.streams.ensure_consumer_group(
            self.streams.order_stream, self.consumer_group
        )
        if not await self._acquire_lease():
            raise RuntimeError("lease already held; refusing to start")
        if _CONFIG.reset_universe_on_start:
            await self.streams.client.incr(_CONFIG.universe_key)
        await self._ensure_universe_id()
        print(f"[sim-worker] lease acquired key={self.lease_key} holder={self.worker_id}")

    async def _ensure_universe_id(self) -> str:
        current = await self.streams.client.get(_CONFIG.universe_key)
        if current:
            return current
        await self.streams.client.setnx(_CONFIG.universe_key, "1")
        return await self.streams.client.get(_CONFIG.universe_key) or "1"

    async def _get_human_factions(self) -> set[str]:
        universe_id = await self._ensure_universe_id()
        factions = await self.streams.client.smembers(
            f"{_CONFIG.human_factions_key}:{universe_id}"
        )
        return set(factions or [])

    async def _ensure_initial_snapshot(self) -> None:
        if self._last_frame is not None:
            return
        frame = snapshot_from_world(self.world, tick_delay=TICK_DELAY, include_ai_state=False)
        await self.streams.save_snapshot(frame)
        await self.streams.append_event(
            {"type": "snapshot", "data": frame}, maxlen=_CONFIG.event_maxlen
        )
        self._last_frame = frame

    async def _get_external_orders(self) -> Tuple[List[Order], List[str]]:
        """
        Pull orders from Redis stream. Returns (orders, message_ids_to_ack).
        """
        raw = await self.streams.read_orders(
            group=self.consumer_group,
            consumer=self.consumer_name,
            count=50,
            block_ms=_CONFIG.order_block_ms,
        )
        if not raw:
            return [], []

        orders: List[Order] = []
        ids: List[str] = []
        for msg_id, payload in raw:
            maybe_orders = payload.get("orders", [])
            for item in maybe_orders:
                try:
                    orders.append(Order(**item))
                except Exception:
                    # Skip malformed entries; ack to avoid blocking the stream
                    continue
            ids.append(msg_id)
        return orders, ids

    async def _ack_orders(self, ids: List[str]) -> None:
        if ids:
            await self.streams.ack_orders(ids, self.consumer_group)

    async def tick_once(self) -> None:
        ext_orders: List[Order] = []
        to_ack: List[str] = []
        ext_orders, to_ack = await self._get_external_orders()

        human_factions = await self._get_human_factions()
        if human_factions:
            ext_orders = [
                o
                for o in ext_orders
                if not (o.source == "bot" and o.faction in human_factions)
            ]

        ext_orders = self._normalize_external_orders(ext_orders)
        route_orders = self._route_orders()
        orders = ext_orders + route_orders
        advance_world(self.world, orders)

        frame = snapshot_from_world(
            self.world, tick_delay=TICK_DELAY, include_ai_state=False
        )
        frame["orders_applied"] = len(orders)
        event_payload = self._build_event(frame)
        await self.streams.append_event(event_payload, maxlen=_CONFIG.event_maxlen)
        if self.world.tick % _CONFIG.snapshot_every == 0:
            await self.streams.save_snapshot(frame)
        self._last_frame = frame

        if to_ack:
            await self._ack_orders(to_ack)

        await self._invalidate_eliminated_humans()

        restart_winner = self._detect_winner()
        if restart_winner is not None:
            await self._handle_restart(restart_winner)

    def _detect_winner(self) -> str | None:
        """
        Detect if a single faction has effectively won and we should restart.
        Mirrors the logic from the original main loop.
        """
        system_owners = {sys.owner for sys in self.world.systems.values() if sys.owner}
        fleet_owners = {fl.owner for fl in self.world.fleets.values() if fl.strength > 0}
        active = system_owners | fleet_owners

        # Immediate win if one faction owns all systems
        if len(system_owners) == 1:
            dominant = next(iter(system_owners))
            other_owners = {
                fl.owner
                for fl in self.world.fleets.values()
                if fl.owner != dominant and fl.strength > 0
            }
            if not other_owners:
                return dominant
            other_strength = sum(
                fl.strength
                for fl in self.world.fleets.values()
                if fl.owner != dominant and fl.strength > 0
            )
            if other_strength < 2.0:
                return dominant

        # If only one active faction remains (systems or fleets), declare winner
        if len(active) == 1:
            return next(iter(active))

        return None

    async def _invalidate_eliminated_humans(self) -> None:
        universe_id = await self._ensure_universe_id()
        human_factions = await self._get_human_factions()
        if not human_factions:
            return
        system_owners = {sys.owner for sys in self.world.systems.values() if sys.owner}
        fleet_owners = {fl.owner for fl in self.world.fleets.values() if fl.strength > 0}
        active = system_owners | fleet_owners
        eliminated = {f for f in human_factions if f not in active}
        if not eliminated:
            return
        for faction in eliminated:
            if faction in self._revoked_factions:
                continue
            player_addr = await self.streams.client.get(
                f"{_CONFIG.faction_player_prefix}:{universe_id}:{faction}"
            )
            if player_addr:
                await self.streams.client.delete(
                    f"{_CONFIG.player_faction_prefix}:{universe_id}:{player_addr}"
                )
                await self.streams.client.delete(
                    f"{_CONFIG.faction_player_prefix}:{universe_id}:{faction}"
                )
            await self.streams.client.srem(
                f"{_CONFIG.human_factions_key}:{universe_id}", faction
            )
            self._revoked_factions.add(faction)

    async def _handle_restart(
        self,
        winner: str | None,
        universe_id: str | None = None,
        seed: Optional[str] = None,
    ) -> None:
        """
        Emit a run_end event, then reset factions and world to start a new run.
        """
        winner_label = FACTION_NAMES.get(winner, winner) if winner else "none"
        await self.streams.append_event(
            {
                "type": "run_end",
                "winner": winner,
                "winner_label": winner_label,
                "tick": self.world.tick,
            },
            maxlen=_CONFIG.event_maxlen,
        )
        reset_factions()
        self.world = create_sector(seed=seed) if seed is not None else create_sector()
        if universe_id is not None:
            await self.streams.client.set(_CONFIG.universe_key, universe_id)
        else:
            await self.streams.client.incr(_CONFIG.universe_key)
        self._revoked_factions.clear()
        self._routes.clear()
        # Save a fresh snapshot to kick off the new run
        frame = snapshot_from_world(self.world, tick_delay=TICK_DELAY, include_ai_state=False)
        await self.streams.save_snapshot(frame)
        self._last_frame = frame
        # Emit a fresh snapshot event so clients can resync immediately.
        await self.streams.append_event(
            {"type": "snapshot", "data": frame}, maxlen=_CONFIG.event_maxlen
        )

    def _build_event(self, frame: dict) -> dict:
        """
        Build a delta event relative to the previous frame to reduce payload size.
        If no previous frame exists, emit a full snapshot event.
        """
        if self._last_frame is None:
            return {"type": "snapshot", "data": frame}
        delta = self._compute_delta(self._last_frame, frame)
        delta["type"] = "delta"
        return delta

    def _compute_delta(self, prev: dict, curr: dict) -> dict:
        """
        Compute a simple delta between two frames.
        Includes changed systems (id + changed fields), changed fleets (full),
        removed fleet ids, and updated highlights/events/tick.
        """
        delta: dict = {"tick": curr.get("tick"), "tick_delay_ms": curr.get("tick_delay_ms")}
        if curr.get("generator_seed") != prev.get("generator_seed"):
            delta["generator_seed"] = curr.get("generator_seed")

        # Systems diff
        prev_sys = {s["id"]: s for s in prev.get("systems", [])}
        changed_systems = []
        for sys in curr.get("systems", []):
            sid = sys.get("id")
            old = prev_sys.get(sid, {})
            changes = {k: v for k, v in sys.items() if old.get(k) != v}
            if changes:
                changes["id"] = sid
                changed_systems.append(changes)
        if changed_systems:
            delta["changed_systems"] = changed_systems

        # Fleets diff
        prev_f = {f["id"]: f for f in prev.get("fleets", [])}
        curr_f = {f["id"]: f for f in curr.get("fleets", [])}
        removed = [fid for fid in prev_f.keys() if fid not in curr_f]
        changed_fleets = []
        for fid, fl in curr_f.items():
            old = prev_f.get(fid)
            if old is None or any(old.get(k) != fl.get(k) for k in ("owner", "system_id", "strength", "enroute_from", "enroute_to", "eta")):
                changed_fleets.append(fl)
        if removed:
            delta["removed_fleets"] = removed
        if changed_fleets:
            delta["changed_fleets"] = changed_fleets

        # Highlights/events/factions for UI
        delta["highlight_ids"] = curr.get("highlight_ids", [])
        delta["events"] = curr.get("events", [])
        delta["factions"] = curr.get("factions", [])
        return delta

    async def _acquire_lease(self) -> bool:
        return bool(
            await self.streams.client.set(
                name=self.lease_key,
                value=self.worker_id,
                nx=True,
                px=self.lease_ttl_ms,
            )
        )

    async def _renew_lease(self) -> bool:
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('PEXPIRE', KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        res = await self.streams.client.eval(
            script, 1, self.lease_key, self.worker_id, self.lease_ttl_ms
        )
        return res == 1

    async def _release_lease(self) -> None:
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        else
            return 0
        end
        """
        await self.streams.client.eval(script, 1, self.lease_key, self.worker_id)

    async def run(self) -> None:
        try:
            await self.setup()
        except RuntimeError as exc:
            print(f"[sim-worker] {exc}")
            return

        print(
            f"[sim-worker] starting loop worker_id={self.worker_id} "
            f"tick_delay={TICK_DELAY}s lease_key={self.lease_key}"
        )
        try:
            while not self._stop.is_set():
                # Renew early in the loop to keep TTL fresh.
                if not await self._renew_lease():
                    # If renewal fails, attempt to reacquire immediately.
                    reacquired = await self._acquire_lease()
                    if reacquired:
                        print("[sim-worker] lease reacquired after early lapse")
                    else:
                        print("[sim-worker] lost lease (pre-tick); stopping")
                        break

                restart_payload = await self.streams.client.get(_CONFIG.restart_key)
                if restart_payload:
                    await self.streams.client.delete(_CONFIG.restart_key)
                    restart_seed = None
                    restart_universe = None
                    try:
                        parsed = json.loads(restart_payload)
                        if isinstance(parsed, dict):
                            restart_universe = parsed.get("universe_id")
                            restart_seed = parsed.get("seed")
                        else:
                            restart_universe = str(parsed)
                    except (json.JSONDecodeError, TypeError):
                        restart_universe = restart_payload
                    if restart_universe is not None:
                        restart_universe = str(restart_universe)
                    await self._handle_restart(
                        None,
                        universe_id=restart_universe,
                        seed=restart_seed,
                    )
                    await asyncio.sleep(0.5)
                    continue

                human_factions = await self._get_human_factions()
                if not human_factions and self.world.tick == 0:
                    await self._ensure_initial_snapshot()
                    await asyncio.sleep(1)
                    continue

                try:
                    await self.tick_once()
                except Exception as exc:  # pragma: no cover - background safety
                    print(f"[sim-worker] error during tick: {exc}")

                ok = await self._renew_lease()
                if not ok:
                    # Retry once after a short delay before giving up to avoid transient stalls.
                    await asyncio.sleep(TICK_DELAY)
                    ok = await self._renew_lease()
                if not ok:
                    # Try to reacquire in case the lease simply expired without a holder.
                    reacquired = await self._acquire_lease()
                    if reacquired:
                        print("[sim-worker] lease reacquired after lapse")
                        ok = True
                if not ok:
                    print("[sim-worker] lost lease after retry; stopping")
                    break

                await asyncio.sleep(TICK_DELAY)
        finally:
            try:
                await self._release_lease()
            except Exception:
                pass
            try:
                await self.streams.close()
                # Explicitly close the underlying pool to avoid asyncio warnings on shutdown.
                await self.streams.client.connection_pool.disconnect()  # type: ignore[attr-defined]
            except Exception:
                pass
            print("[sim-worker] stopping loop")

    def stop(self) -> None:
        self._stop.set()


async def main() -> None:
    worker = SimulationWorker()

    # Optional gate: wait for a frontend to be ready before starting the loop.
    wait_flag = _CONFIG.wait_for_frontend
    frontend_url = _CONFIG.frontend_health_url
    if wait_flag:
        import http.client
        import time
        parsed = None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(frontend_url)
        except Exception:
            parsed = None
        print(f"[sim-worker] WAIT_FOR_FRONTEND enabled; waiting on {frontend_url}")
        while True:
            try:
                if parsed is None:
                    raise RuntimeError("invalid FRONTEND_HEALTH_URL")
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=2)
                path = parsed.path or "/"
                conn.request("GET", path)
                resp = conn.getresponse()
                if resp.status < 500:
                    print("[sim-worker] frontend reachable; starting simulation")
                    break
            except Exception as exc:
                print(f"[sim-worker] frontend not ready: {exc}; retrying in 2s")
                time.sleep(2)
            finally:
                try:
                    conn.close()  # type: ignore[attr-defined]
                except Exception:
                    pass

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.stop)

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
