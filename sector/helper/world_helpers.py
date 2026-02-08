import hashlib
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
import random

# Delaunay method + Edge pruning for placement generation
from scipy.spatial import Delaunay  # type: ignore

# simulation config import
from sector.models import SIM_CONFIG
from sector.models import StarSystem, World, Fleet

from .factions_helper import load_factions

# simulation config variable mapping
LEAGUE_FACTION_ID = SIM_CONFIG.world_tuning.league_faction_id
LEAGUE_TECH_BONUS = SIM_CONFIG.league_modifiers.tech_bonus
# World-level config from JSON
NUM_SYSTEMS: int = SIM_CONFIG.simulation_modifiers.number_of_systems
LANES_PER_SYSTEM: int = SIM_CONFIG.simulation_modifiers.lanes_per_system
MIN_SYSTEM_DISTANCE: float = SIM_CONFIG.simulation_modifiers.minimum_system_distance
MAX_PLACEMENT_ATTEMPTS: int = SIM_CONFIG.simulation_modifiers.maximum_placement_attempts
MAX_LANE_LENGTH: float = SIM_CONFIG.simulation_modifiers.maximum_lane_length
# Optional deterministic seed for sector generation
RAW_SECTOR_SEED = SIM_CONFIG.sector_seed
# economic variables
ECON_MATURITY_TICKS = SIM_CONFIG.economy_modifiers.maturity_ticks
ECON_MATURITY_MIN_FACTOR = SIM_CONFIG.economy_modifiers.maturity_minimum_factor
# Faction IDs and names from JSON
FACTION_CONFIG: Dict[str, dict[str, Any]] = load_factions(SIM_CONFIG)
FACTION_NAMES: Dict[str, str] = {
    faction_id: faction_config["name"]
    for faction_id, faction_config in FACTION_CONFIG.items()
}
# max passive planetary garrison
GARRISON_MAX = SIM_CONFIG.battle_modifiers.garrison_maximum_strength
# regen per tick
GARRISON_REGEN = SIM_CONFIG.battle_modifiers.garrison_regeneration_per_tick
# Specials config
SPECIAL_CFG = SIM_CONFIG.special_systems
NUM_RELIC = SPECIAL_CFG["relic"]
NUM_FORGE = SPECIAL_CFG["forge"]
NUM_HIVE = SPECIAL_CFG["hive"]
# overextension
OVEREXTENSION_THRESHOLD = SIM_CONFIG.overextension_modifiers.threshold
ECON_OVEREXT_PENALTY_PER_EXTRA = (
    SIM_CONFIG.overextension_modifiers.economy_penalty_per_extra_system
)


def tech_multiplier(faction: Optional[str]) -> float:
    if faction == LEAGUE_FACTION_ID:
        return 1.0 + LEAGUE_TECH_BONUS
    return 1.0


# ---------- Sector generation ----------

SEED_BITS = 48
SEED_MASK = (1 << SEED_BITS) - 1


def normalize_seed(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned, 0) & SEED_MASK
        except ValueError:
            digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            return int(digest, 16) & SEED_MASK
    if isinstance(value, (bytes, bytearray)):
        digest = hashlib.sha256(value).hexdigest()
        return int(digest, 16) & SEED_MASK
    try:
        return int(value) & SEED_MASK  # type: ignore[arg-type]
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return int(digest, 16) & SEED_MASK


def create_sector(num_systems: int = NUM_SYSTEMS, seed: Optional[int] = None) -> World:
    """
    Generate a new sector. When a seed (or sim_config['sector_seed']) is provided,
    the map layout will be deterministic.
    """
    effective_seed = normalize_seed(seed if seed is not None else RAW_SECTOR_SEED)
    if effective_seed is None:
        effective_seed = random.SystemRandom().randrange(1 << SEED_BITS)
    rng = random.Random(effective_seed)
    systems: Dict[int, StarSystem] = {}

    # Random positions & values (avoid tight clustering)
    min_dist2 = MIN_SYSTEM_DISTANCE * MIN_SYSTEM_DISTANCE
    positions: List[Tuple[float, float]] = []
    for i in range(num_systems):
        x = rng.random()
        y = rng.random()
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            if all((x - px) ** 2 + (y - py) ** 2 >= min_dist2 for px, py in positions):
                break
            x = rng.random()
            y = rng.random()
        positions.append((x, y))
        value = rng.randint(1, 10)
        systems[i] = StarSystem(
            id=i, x=x, y=y, value=value, econ_maturity=ECON_MATURITY_TICKS
        )

    # Lanes: build a connected graph with a hard max degree per system
    lane_set: set[tuple[int, int]] | None = set()
    ids = list(systems.keys())
    max_degree = LANES_PER_SYSTEM
    if num_systems > 2 and max_degree < 2:
        raise ValueError("LANES_PER_SYSTEM must be >= 2 to keep the graph connected.")

    def dist2(a: StarSystem, b: StarSystem) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return dx * dx + dy * dy

    def _delaunay_edges() -> list[tuple[float, int, int]]:
        if len(ids) < 3:
            edges: list[tuple[float, int, int]] = []
            for i in ids:
                for j in ids:
                    if i >= j:
                        continue
                    edges.append((dist2(systems[i], systems[j]), i, j))
            edges.sort(key=lambda t: t[0])
            return edges

        points = np.array([(systems[i].x, systems[i].y) for i in ids])
        tri = Delaunay(points)  # type: ignore
        edge_set: set[tuple[int, int]] = set()
        for simplex in tri.simplices:  # type: ignore
            a, b, c = simplex  # type: ignore
            for u, v in ((a, b), (b, c), (c, a)):  # type: ignore
                i = ids[u]  # type: ignore
                j = ids[v]  # type: ignore
                edge = (i, j) if i < j else (j, i)  # type: ignore
                edge_set.add(edge)  # type: ignore
        edges: list[tuple[float, int, int]] = []
        for a, b in edge_set:
            edges.append((dist2(systems[a], systems[b]), a, b))
        edges.sort(key=lambda t: t[0])
        return edges

    base_edges = _delaunay_edges()

    def _build_lanes(max_lane_len: float) -> set[tuple[int, int]] | None:
        if not base_edges:
            return None
        max_lane_dist2 = max_lane_len * max_lane_len
        edges = [(d2, a, b) for (d2, a, b) in base_edges if d2 <= max_lane_dist2]
        if not edges:
            return None

        parent = list(range(num_systems))
        rank = [0] * num_systems
        degree = [0] * num_systems
        lanes: set[tuple[int, int]] = set()

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        # Build a connected backbone (spanning tree) respecting max degree
        for _, a, b in edges:
            if degree[a] >= max_degree or degree[b] >= max_degree:
                continue
            if find(a) != find(b):
                lanes.add((a, b))
                degree[a] += 1
                degree[b] += 1
                union(a, b)

        roots = {find(i) for i in ids}
        if len(roots) > 1:
            return None

        return lanes

    max_lane_len = MAX_LANE_LENGTH
    max_allowed = 1.5  # slightly above unit square diagonal (~1.414)
    lane_set = None
    while max_lane_len <= max_allowed + 1e-9:
        lane_set = _build_lanes(max_lane_len)
        if lane_set is not None:
            break
        max_lane_len *= 1.25

    if lane_set is None:
        raise ValueError(
            "Unable to build a connected graph with the current "
            "LANES_PER_SYSTEM and maximum_lane_length constraints."
        )

    print(f"[sector] max lane length used: {max_lane_len:.3f}")

    for a, b in lane_set:
        systems[a].neighbors.append(b)
        systems[b].neighbors.append(a)

    # Assign special system types
    available = ids.copy()
    rng.shuffle(available)

    def pop_n(n: int) -> List[int]:
        chosen = available[:n]
        del available[:n]
        return chosen

    for sid in pop_n(min(NUM_RELIC, len(available))):
        systems[sid].kind = "relic"
    for sid in pop_n(min(NUM_FORGE, len(available))):
        systems[sid].kind = "forge"
    for sid in pop_n(min(NUM_HIVE, len(available))):
        systems[sid].kind = "hive"

    # Starting systems for each (initial) faction
    faction_ids = list(FACTION_NAMES.keys())
    if len(faction_ids) > len(ids):
        raise ValueError("More factions defined than systems in sector.")
    starting_ids = rng.sample(ids, k=len(faction_ids))
    for faction, sid in zip(faction_ids, starting_ids):
        systems[sid].owner = faction

    world = World(
        tick=0,
        systems=systems,
        lanes=list(lane_set),
        events=[],
        last_event_systems=[],
        history=[],
        generator_seed=effective_seed,
    )

    # Initialize build stock
    world.faction_build_stock = {fid: 0.0 for fid in faction_ids}
    # Initialize garrisons
    for sys in systems.values():
        sys.garrison = GARRISON_MAX

    # Starting fleets: one per faction at its home system
    def spawn_fleet(owner: str, system_id: int, strength: float):
        fid = world.next_fleet_id
        world.next_fleet_id += 1
        world.fleets[fid] = Fleet(
            id=fid,
            owner=owner,
            system_id=system_id,
            strength=strength,
            max_strength=strength,
            enroute_from=None,
            enroute_to=None,
            eta=0,
        )

    for faction, sid in zip(faction_ids, starting_ids):
        spawn_fleet(faction, sid, strength=10.0)

    return world


# ---------- Economic power helper ----------


def system_econ_value(sys: StarSystem) -> float:
    """Effective economic contribution of a single system."""
    base = float(sys.value)
    # Special worlds are worth more economically
    if sys.kind == "relic":
        base *= 1.8
    elif sys.kind == "forge":
        base *= 1.4
    elif sys.kind == "hive":
        base *= 1.2
    # Stability matters: unstable worlds don't contribute fully
    stability_factor = 0.5 + 0.5 * sys.stability
    maturity = min(1.0, sys.econ_maturity / float(max(1, ECON_MATURITY_TICKS)))
    maturity_factor = (
        ECON_MATURITY_MIN_FACTOR + (1.0 - ECON_MATURITY_MIN_FACTOR) * maturity
    )
    return base * stability_factor * maturity_factor


def compute_econ_power(world: World, faction_ids: List[str]) -> Dict[str, float]:
    econ_power = {f: 1.0 for f in faction_ids}  # baseline so nobody is 0
    sizes: Dict[str, int] = {f: 0 for f in faction_ids}
    for sys in world.systems.values():
        owner = sys.owner
        if owner in econ_power:
            sizes.setdefault(owner, 0)
            sizes[owner] += 1
            econ_power[owner] += system_econ_value(sys)

    for fid, size in sizes.items():
        extra = max(0, size - OVEREXTENSION_THRESHOLD)
        if extra > 0:
            penalty = 1.0 / (1.0 + ECON_OVEREXT_PENALTY_PER_EXTRA * extra)
            econ_power[fid] *= penalty
    return econ_power
