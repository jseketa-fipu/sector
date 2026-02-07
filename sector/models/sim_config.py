import json
from pathlib import Path
from pydantic import BaseModel, PositiveInt, PositiveFloat
from pydantic import Field  # type: ignore
from typing import Annotated, List, Dict

# # NOTE: Each service imports this module in its own process. The loaded config
# # lives in-process only; it is not shared across services or persisted.


class BotPersonality(BaseModel):
    aggression: Annotated[float, Field(ge=0, le=1)]
    expansionism: Annotated[float, Field(ge=0, le=1)]
    greed: Annotated[float, Field(ge=0, le=1)]
    caution: Annotated[float, Field(ge=0, le=1)]


class FactionConfig(BaseModel):
    name: Annotated[str, Field(min_length=1)]
    base_attempts: Annotated[int, Field(ge=0)]
    personality: BotPersonality


class OverextensionModifiers(BaseModel):
    threshold: PositiveInt
    risk_per_extra_system: PositiveFloat
    economy_penalty_per_extra_system: PositiveFloat
    colonizer_order_penalty_per_extra_system: PositiveFloat
    colonizer_reserve_per_system: PositiveFloat


class EconomyModifiers(BaseModel):
    maturity_ticks: PositiveInt
    maturity_minimum_factor: PositiveFloat


class LeagueModifiers(BaseModel):
    tech_bonus: PositiveFloat
    militia_strength: PositiveFloat


class SimulationModifiers(BaseModel):
    number_of_systems: PositiveInt
    lanes_per_system: Annotated[PositiveInt, Field(ge=2)]
    minimum_system_distance: PositiveFloat
    maximum_lane_length: PositiveFloat
    maximum_placement_attempts: PositiveInt
    faction_count: PositiveInt
    attempts_per_tick: PositiveInt
    tick_delay: PositiveFloat
    order_block_ms: PositiveInt
    minimum_launch_strength: PositiveFloat
    damage_factor: PositiveFloat
    frontline_origin_bonus: PositiveFloat
    goal_path_bonus: PositiveFloat
    stagnation_ticks: PositiveInt
    max_reach_hops: PositiveInt
    rally_min_strength: PositiveFloat
    rally_max_orders: PositiveInt
    max_dynamic_factions: PositiveInt


class BattleModifiers(BaseModel):
    capture_ticks: PositiveInt
    minimum_capture_strength: PositiveFloat
    garrison_maximum_strength: PositiveFloat
    garrison_regeneration_per_tick: PositiveFloat
    auto_win_ratio: PositiveFloat
    defender_minimum_remaining: PositiveFloat
    defender_garrison_soak_factor: PositiveFloat


class WorldTuning(BaseModel):
    rebellion_stability_threshold: PositiveFloat
    rebellion_chance: PositiveFloat
    new_faction_chance: PositiveFloat
    upkeep_idle: PositiveFloat
    travel_ticks: PositiveInt
    build_rate: PositiveFloat
    build_cost: PositiveFloat
    new_fleet_strength: PositiveFloat
    repair_rate: PositiveFloat
    league_faction_id: str


class SimulationSettings(BaseModel):
    simulation_modifiers: SimulationModifiers
    faction_name_list: Annotated[List[str], Field(default_factory=List)]
    use_faction_count: bool
    sector_seed: int | None

    lease_ttl_ms: PositiveInt

    world_tuning: WorldTuning
    overextension_modifiers: OverextensionModifiers
    economy_modifiers: EconomyModifiers
    league_modifiers: LeagueModifiers
    battle_modifiers: BattleModifiers

    colonizer_reserve_strength: PositiveFloat
    colonizer_strength_per_order: PositiveFloat
    special_systems: Dict[str, int]
    factions: Dict[str, FactionConfig]

    @classmethod
    def load_json(cls, path: str | Path) -> "SimulationSettings":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


_BASE_DIR = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _BASE_DIR / "config" / "sim_config.json"

SIM_CONFIG = SimulationSettings.model_validate_json(
    _CONFIG_PATH.read_text(encoding="utf-8")
)
