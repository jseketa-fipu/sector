from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass
class StarSystem:
    id: int
    x: float  # normalized [0, 1]
    y: float  # normalized [0, 1]
    owner: Optional[str] = None  # faction id or None
    neighbors: List[int] = field(default_factory=list)
    value: int = 1  # resource / importance, 1–10
    stability: float = 1.0  # 0..1, low = rebellion likely
    kind: str = "normal"  # "normal" | "relic" | "forge" | "hive"
    heat: float = 0.0  # recent conflict, for visuals
    unrest: float = 0.0  # builds up on frequent ownership changes
    reclaim_cooldown: int = 0  # ticks before auto-claim can occur after rebellion
    occupation_faction: Optional[str] = None  # who is occupying
    occupation_progress: int = 0  # ticks of continuous occupation
    garrison: float = 0.0  # passive defense that regrows when owned
    econ_maturity: int = 0  # ticks since owned (caps at ECON_MATURITY_TICKS)


@dataclass
class Fleet:
    id: int
    owner: str
    system_id: Optional[int]  # None when enroute
    strength: float
    max_strength: float
    experience: float = 0.0
    enroute_from: Optional[int] = None
    enroute_to: Optional[int] = None
    eta: int = 0  # ticks remaining to arrival; 0 means idle at system


@dataclass
class HistoricalEvent:
    tick: int
    kind: str  # "expansion", "fleet_battle_win", "rebellion_*", ...
    systems: List[int]
    factions: List[str]
    text: str


@dataclass
class World:
    tick: int
    systems: Dict[int, StarSystem]
    lanes: List[Tuple[int, int]]
    events: List[str]
    last_event_systems: List[int] = field(default_factory=list)
    history: List[HistoricalEvent] = field(default_factory=list)
    generator_seed: Optional[int] = None

    # Fleets & production
    fleets: Dict[int, Fleet] = field(default_factory=dict)
    next_fleet_id: int = 0
    faction_build_stock: Dict[str, float] = field(default_factory=dict)


@dataclass
class Order:
    """A generic order issued by a faction for this tick."""

    faction: str  # e.g. "E", "P", "T"
    origin_id: int  # system id issuing the order
    target_id: int  # neighbor system id being targeted
    reason: Optional[str] = None  # why this move was chosen (for history logging)
    source: Optional[str] = None  # "human" | "bot" (set by API)
    fleet_id: Optional[int] = None  # specific fleet id to move (optional)


@dataclass
class TickSummary:
    """Aggregated outcome of a single tick, for AI adaptation."""

    battles_won: Dict[str, int]
    battles_lost: Dict[str, int]
    expansions: Dict[str, int]
