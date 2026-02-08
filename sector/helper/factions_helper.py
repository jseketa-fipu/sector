#!/usr/bin/env python3
from __future__ import annotations

from itertools import cycle
from typing import Any

from sector.models.sim_config import SimulationSettings, FactionConfig


def _dump_faction(cfg: FactionConfig) -> dict[str, Any]:
    return cfg.model_dump()


def load_factions(sim_cfg: SimulationSettings) -> dict[str, dict[str, Any]]:
    """
    Build the faction configuration from the validated SimulationSettings model.
    Priority:
    1) If use_faction_count is true, generate faction_count factions using
       faction_name_list and templates from the configured factions.
    2) Else if factions map is provided and non-empty, use it as-is.
    """
    explicit = sim_cfg.factions
    if explicit and not sim_cfg.use_faction_count:
        return_faction_dictionary = {
            faction_id: _dump_faction(config) for faction_id, config in explicit.items()
        }
        return return_faction_dictionary

    count = sim_cfg.simulation_modifiers.faction_count
    names = list(sim_cfg.faction_name_list)
    if len(names) < count:
        raise ValueError(
            "faction_name_list must contain at least faction_count names when use_faction_count is true"
        )
    if not explicit:
        raise ValueError(
            "factions must be provided as templates when use_faction_count is true"
        )

    templates = cycle(explicit.values())
    factions: dict[str, dict[str, Any]] = {}
    for idx in range(count):
        faction_index = idx + 1
        faction_id = f"F{faction_index}"

        template = next(templates)
        config = _dump_faction(template)

        name = names[idx]
        config["name"] = name

        factions[faction_id] = config

    return factions
