#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Any

def load_factions(sim_cfg: Dict[str, Any]) -> Dict[str, dict]:
    """
    Build the faction configuration from sim_config.
    Priority:
    1) If "use_faction_count" is true, ignore explicit map and generate faction_count factions.
    2) Else if "factions" map is provided and non-empty, use it as-is.
    3) Else, if "faction_count" is provided, generate that many generic factions.
    4) Fallback to five default factions if nothing specified.
    """
    force_count = str(sim_cfg.get("use_faction_count", False)).lower() == "true"
    explicit = sim_cfg.get("factions", {}) or {}
    if explicit and not force_count:
        return dict(explicit)

    count = int(sim_cfg.get("faction_count", 5))
    name_list = sim_cfg.get("faction_name_list") or sim_cfg.get("faction_names") or []
    available_names = list(name_list) if isinstance(name_list, list) else []
    palette = [
        "#ff5555",
        "#f1fa8c",
        "#8be9fd",
        "#a6e22e",
        "#ff79c6",
        "#7fffd4",
        "#50fa7b",
        "#bd93f9",
        "#ffb86c",
        "#c0c0c0",
    ]
    factions: Dict[str, dict] = {}
    for idx in range(count):
        fid = f"F{idx+1}"
        name = available_names.pop(0) if available_names else f"Faction {idx+1}"
        factions[fid] = {
            "name": name,
            "base_attempts": sim_cfg.get("attempts_per_tick", 4),
            "personality": {
                "aggression": 0.5,
                "expansionism": 0.5,
                "greed": 0.5,
                "caution": 0.5,
            },
            "color": palette[idx % len(palette)],
        }
    return factions
