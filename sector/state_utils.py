#!/usr/bin/env python3
"""
Helpers for building public-facing snapshots from the in-memory world.
"""
from __future__ import annotations

from typing import Dict, List

from sector.puppet import FACTION_COLORS, get_ai_debug_state
from sector.world import FACTION_NAMES, World


def compute_siege_state(world: World) -> dict[int, dict]:
    """
    Build a per-system map describing whether it's under siege and by whom.
    A system is considered besieged if:
    - it has an owner and any idle fleet from another faction is present, or
    - it is currently being occupied (occupation_faction set), or
    - it is unowned and multiple factions have fleets present (contested neutral).
    """
    fleets_at: dict[int, list] = {}
    for fl in world.fleets.values():
        if fl.eta != 0 or fl.system_id is None or fl.strength <= 0:
            continue
        fleets_at.setdefault(fl.system_id, []).append(fl)

    siege_state: dict[int, dict] = {}
    for sys_id, sys in world.systems.items():
        present = fleets_at.get(sys_id, [])
        enemy_by_owner: dict[str, float] = {}
        occupants: set[str] = set()
        for fl in present:
            occupants.add(fl.owner)
            if sys.owner is None or fl.owner != sys.owner:
                enemy_by_owner[fl.owner] = (
                    enemy_by_owner.get(fl.owner, 0.0) + fl.strength
                )

        enemy_strength = sum(enemy_by_owner.values())
        siege_owner = None
        if enemy_by_owner:
            siege_owner = max(enemy_by_owner.items(), key=lambda kv: kv[1])[0]

        contested_neutral = sys.owner is None and len(occupants) > 1
        is_besieged = (
            bool(sys.occupation_faction)
            or (sys.owner is not None and enemy_strength > 0)
            or contested_neutral
        )

        siege_state[sys_id] = {
            "is_besieged": is_besieged,
            "siege_owner": siege_owner,
            "siege_strength": enemy_strength,
        }
    return siege_state


def snapshot_from_world(world: World, tick_delay: float, include_ai_state: bool = False) -> dict:
    """
    Generate a snapshot payload suitable for API/WebSocket consumers.
    """
    siege_state = compute_siege_state(world)
    ai_state = get_ai_debug_state(world) if include_ai_state else None

    systems_payload = []
    for sys in world.systems.values():
        systems_payload.append(
            {
                "id": sys.id,
                "x": sys.x,
                "y": sys.y,
                "owner": sys.owner,
                "value": sys.value,
                "kind": sys.kind,
                "heat": sys.heat,
                "stability": sys.stability,
                "unrest": sys.unrest,
                "reclaim_cooldown": sys.reclaim_cooldown,
                "occupation_faction": sys.occupation_faction,
                "occupation_progress": sys.occupation_progress,
                "is_besieged": siege_state.get(sys.id, {}).get("is_besieged", False),
                "siege_owner": siege_state.get(sys.id, {}).get("siege_owner"),
                "siege_strength": siege_state.get(sys.id, {}).get("siege_strength", 0.0),
            }
        )

    fleets_payload = []
    for fl in world.fleets.values():
        fleets_payload.append(
            {
                "id": fl.id,
                "owner": fl.owner,
                "system_id": fl.system_id,
                "strength": fl.strength,
                "enroute_from": fl.enroute_from,
                "enroute_to": fl.enroute_to,
                "eta": fl.eta,
            }
        )

    history_tail = [
        {
            "tick": ev.tick,
            "kind": ev.kind,
            "systems": ev.systems,
            "factions": ev.factions,
            "text": ev.text,
        }
        for ev in world.history[-40:]
    ]

    snapshot = {
        "tick_delay": tick_delay,
        "tick_delay_ms": int(tick_delay * 1000),
        "tick": world.tick,
        "generator_seed": world.generator_seed,
        "systems": systems_payload,
        "lanes": [[a, b] for (a, b) in world.lanes],
        "fleets": fleets_payload,
        "events": world.events[-30:],
        "highlight_ids": world.last_event_systems,
        "ai_state": ai_state,
        "factions": [
            {
                "id": fid,
                "name": name,
                "color": FACTION_COLORS.get(fid, "#ffffff"),
            }
            for fid, name in FACTION_NAMES.items()
        ],
        "history_tail": history_tail,
    }
    return snapshot
