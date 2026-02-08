#!/usr/bin/env python3
from __future__ import annotations


import random
from typing import Dict, List, Optional, Any

from sector.helper.world_helpers import (
    FACTION_NAMES,
    LEAGUE_FACTION_ID,
    GARRISON_MAX,
    GARRISON_REGEN,
    OVEREXTENSION_THRESHOLD,
    ECON_MATURITY_TICKS,
    create_sector,
    tech_multiplier,
    compute_econ_power,
)


from sector.models import SIM_CONFIG
from sector.models import StarSystem, Fleet, HistoricalEvent, World, Order, TickSummary


# Rebellion tuning
REBELLION_STABILITY_THRESHOLD = (
    SIM_CONFIG.world_tuning.rebellion_stability_threshold
)  # only VERY unstable worlds can revolt
REBELLION_CHANCE = (
    SIM_CONFIG.world_tuning.rebellion_chance
)  # chance when under threshold
NEW_FACTION_CHANCE = (
    SIM_CONFIG.world_tuning.new_faction_chance
)  # fraction of rebellions that spawn a new faction
UPKEEP_IDLE = SIM_CONFIG.world_tuning.upkeep_idle  # per-tick idle fleet upkeep (drain)

OVEREXTENSION_RISK_PER_EXTRA = SIM_CONFIG.overextension_modifiers.risk_per_extra_system


LEAGUE_MILITIA_STRENGTH = SIM_CONFIG.league_modifiers.militia_strength
CAPTURE_TICKS = (
    SIM_CONFIG.battle_modifiers.capture_ticks
)  # ticks required to flip undefended system
MIN_CAPTURE_STRENGTH = SIM_CONFIG.battle_modifiers.minimum_capture_strength
# min idle strength to capture


# Fleet travel config
TRAVEL_TICKS = (
    SIM_CONFIG.world_tuning.travel_ticks
)  # movement duration between neighboring systems
BUILD_RATE = SIM_CONFIG.world_tuning.build_rate  # econ -> build stock per tick
BUILD_COST = SIM_CONFIG.world_tuning.build_cost  # stock needed for a small fleet
NEW_FLEET_STRENGTH = SIM_CONFIG.world_tuning.new_fleet_strength
REPAIR_RATE = SIM_CONFIG.world_tuning.repair_rate
DAMAGE_FACTOR = SIM_CONFIG.simulation_modifiers.damage_factor
MIN_LAUNCH_STRENGTH = SIM_CONFIG.simulation_modifiers.minimum_launch_strength
AUTO_WIN_RATIO = SIM_CONFIG.battle_modifiers.auto_win_ratio
DEFENDER_MIN_REMAINING = SIM_CONFIG.battle_modifiers.defender_minimum_remaining
DEFENDER_GARRISON_SOAK_FACTOR = (
    SIM_CONFIG.battle_modifiers.defender_garrison_soak_factor
)

# create the sector

# ---------- Simulation (fleets, orders, rebellions, sprawl) ----------


def advance_world(world: World, orders: List[Order]) -> TickSummary:
    """
    Advance the world by one tick by applying a list of Orders.
    The 'AI' deciding these orders lives in puppet.py.
    Returns a TickSummary so bots can adapt.
    """
    world.tick += 1
    events: List[str] = []
    history = world.history
    highlight_ids = set()

    faction_ids = list(FACTION_NAMES.keys())

    # Counters for summary (can grow with new factions via setdefault)
    wins: Dict[str, int] = {f: 0 for f in faction_ids}
    losses: Dict[str, int] = {f: 0 for f in faction_ids}
    expansions: Dict[str, int] = {f: 0 for f in faction_ids}

    owned_by: Dict[str, List[StarSystem]] = {f: [] for f in faction_ids}
    for sys in world.systems.values():
        if sys.owner in owned_by:
            owned_by[sys.owner].append(sys)

    # Economic power snapshot at the start of the tick
    econ_power = compute_econ_power(world, faction_ids)

    # Ensure build stock for any new factions
    for f in faction_ids:
        world.faction_build_stock.setdefault(f, 0.0)

    def describe_system(sys: StarSystem) -> str:
        if sys.kind == "relic":
            return f"relic world #{sys.id}"
        if sys.kind == "forge":
            return f"forge world #{sys.id}"
        if sys.kind == "hive":
            return f"hive world #{sys.id}"
        return f"system #{sys.id}"

    def log_event(
        kind: str, systems_ids: List[int], factions_ids: List[str], text: str
    ):
        events.append(text)
        history.append(
            HistoricalEvent(
                tick=world.tick,
                kind=kind,
                systems=systems_ids,
                factions=factions_ids,
                text=text,
            )
        )

    def check_rebellion(sys: StarSystem):
        """
        If stability is very low, the system may rebel:
        - either becomes independent (owner None)
        - or (rarely) spawns / joins a rebel faction (splinter state).
        """
        from sector.puppet import (
            register_new_faction,
            ensure_league_faction,
        )  # late import to avoid cyclic import

        if sys.owner is None:
            return

        old_owner = sys.owner

        # Only consider VERY unstable systems (affected by unrest)
        effective_stability = sys.stability - 0.2 * sys.unrest
        if effective_stability > REBELLION_STABILITY_THRESHOLD:
            return

        # Only a fraction of such systems actually rebel this tick (higher with unrest/overextension)
        unrest_factor = 1 + sys.unrest
        base_chance = REBELLION_CHANCE * unrest_factor
        owner_size = sum(1 for s in world.systems.values() if s.owner == old_owner)
        if owner_size > OVEREXTENSION_THRESHOLD:
            extra = owner_size - OVEREXTENSION_THRESHOLD
            base_chance *= 1.0 + OVEREXTENSION_RISK_PER_EXTRA * extra
            base_chance = min(base_chance, 0.9)
        if random.random() > base_chance:
            return

        old_owner = sys.owner
        base_label = (
            FACTION_NAMES[old_owner] if old_owner in FACTION_NAMES else "Rebels"
        )

        roll = random.random()
        if roll > NEW_FACTION_CHANCE:
            # Most rebellions just go independent
            ensure_league_faction(world)
            sys.owner = LEAGUE_FACTION_ID
            sys.stability = 0.7
            sys.reclaim_cooldown = 6  # give the league a moment to secure
            sys.occupation_faction = None
            sys.occupation_progress = 0
            sys.garrison = min(GARRISON_MAX, GARRISON_MAX + 1.0)
            sys.econ_maturity = 0
            # spawn a militia fleet with tech edge
            fid = world.next_fleet_id
            world.next_fleet_id += 1
            world.fleets[fid] = Fleet(
                id=fid,
                owner=LEAGUE_FACTION_ID,
                system_id=sys.id,
                strength=LEAGUE_MILITIA_STRENGTH,
                max_strength=LEAGUE_MILITIA_STRENGTH,
                enroute_from=None,
                enroute_to=None,
                eta=0,
            )
            world.faction_build_stock.setdefault(LEAGUE_FACTION_ID, 0.0)
            text = (
                f"t={world.tick}: Rebellion in {describe_system(sys)}! "
                f"{FACTION_NAMES[old_owner]} lost control; the world joined the {FACTION_NAMES[LEAGUE_FACTION_ID]}."
            )
            log_event(
                "rebellion_independence", [sys.id], [old_owner, LEAGUE_FACTION_ID], text
            )
        else:
            # A minority actually form / join a rebel faction
            new_fid = register_new_faction(old_owner, base_label)
            sys.owner = new_fid
            sys.stability = 0.8
            sys.reclaim_cooldown = 3
            sys.occupation_faction = None
            sys.occupation_progress = 0
            sys.garrison = GARRISON_MAX
            sys.econ_maturity = 0
            # give the new rebels an initial defensive fleet
            fid = world.next_fleet_id
            world.next_fleet_id += 1
            world.fleets[fid] = Fleet(
                id=fid,
                owner=new_fid,
                system_id=sys.id,
                strength=8.0,
                max_strength=8.0,
                enroute_from=None,
                enroute_to=None,
                eta=0,
            )
            # seed some build stock so they can reinforce soon
            world.faction_build_stock.setdefault(new_fid, 0.0)
            world.faction_build_stock[new_fid] += BUILD_COST
            text = (
                f"t={world.tick}: Secession in {describe_system(sys)}! "
                f"A splinter of {FACTION_NAMES[old_owner]} joined the rebel faction "
                f"{FACTION_NAMES[new_fid]}."
            )
            expansions.setdefault(new_fid, 0)
            expansions[new_fid] += 1
            log_event("rebellion_secession", [sys.id], [old_owner, new_fid], text)

        highlight_ids.add(sys.id)
        losses.setdefault(old_owner, 0)
        losses[old_owner] += 1

    # Handle in-flight fleets (arrivals happen before new orders)
    arrivals: List[int] = []
    for fid, fl in world.fleets.items():
        if fl.eta > 0:
            fl.eta -= 1
            if fl.eta <= 0:
                arrivals.append(fid)

    def resolve_system_conflict(
        dest_id: int,
        attacker_faction: str,
        fleets_here: List[Fleet],
    ):
        target_sys = world.systems[dest_id]
        old_owner = target_sys.owner

        defenders = [fl for fl in fleets_here if fl.owner != attacker_faction]
        defender_factions = {fl.owner for fl in defenders}

        def _complete_capture(instantly: bool = False):
            target_sys.owner = attacker_faction
            target_sys.occupation_faction = None
            target_sys.occupation_progress = 0
            target_sys.unrest = min(1.0, target_sys.unrest + 0.25)
            target_sys.stability = max(
                0.0, target_sys.stability - 0.05 - 0.1 * target_sys.unrest
            )
            target_sys.heat = min(3.0, target_sys.heat + 1.5)
            target_sys.garrison = min(
                GARRISON_MAX, target_sys.garrison + 0.5 * GARRISON_MAX
            )
            target_sys.econ_maturity = 0

            expansions.setdefault(attacker_faction, 0)
            expansions[attacker_faction] += 1
            if old_owner:
                losses.setdefault(old_owner, 0)
                losses[old_owner] += 1
                wins.setdefault(attacker_faction, 0)
                wins[attacker_faction] += 1

            text = (
                f"t={world.tick}: Fleet of {FACTION_NAMES[attacker_faction]} secured "
                f"{describe_system(target_sys)} {'via rapid occupation' if instantly else 'after occupation.'}"
            )
            log_event(
                "fleet_capture_no_defense", [target_sys.id], [attacker_faction], text
            )
            highlight_ids.add(target_sys.id)
            check_rebellion(target_sys)

        # No defenders
        if not defenders:
            # reset occupation if returning owner
            if old_owner == attacker_faction:
                target_sys.occupation_faction = None
                target_sys.occupation_progress = 0
                return

            # tally idle attacking strength at this system
            attackers_here = [fl for fl in fleets_here if fl.owner == attacker_faction]
            total_attack = sum(fl.strength for fl in attackers_here)

            # Garrison soaks damage first
            if target_sys.garrison > 0:
                absorbed = min(target_sys.garrison, total_attack)
                target_sys.garrison -= absorbed
                total_attack -= absorbed
                if total_attack <= 0:
                    target_sys.occupation_faction = None
                    target_sys.occupation_progress = 0
                    return

            # If there's no garrison left, flip immediately with any remaining attackers
            if target_sys.garrison <= 0 and total_attack >= MIN_CAPTURE_STRENGTH:
                _complete_capture(instantly=True)
                return

            if total_attack >= MIN_CAPTURE_STRENGTH:
                if target_sys.occupation_faction == attacker_faction:
                    target_sys.occupation_progress += 1
                else:
                    target_sys.occupation_faction = attacker_faction
                    target_sys.occupation_progress = 1
            else:
                target_sys.occupation_faction = None
                target_sys.occupation_progress = 0
                return

            if target_sys.occupation_progress >= CAPTURE_TICKS:
                target_sys.owner = attacker_faction
                target_sys.occupation_faction = None
                target_sys.occupation_progress = 0
                target_sys.unrest = min(1.0, target_sys.unrest + 0.25)
                target_sys.stability = max(
                    0.0, target_sys.stability - 0.05 - 0.1 * target_sys.unrest
                )
                target_sys.heat = min(3.0, target_sys.heat + 1.5)
                target_sys.garrison = min(
                    GARRISON_MAX, target_sys.garrison + 0.5 * GARRISON_MAX
                )
                target_sys.econ_maturity = 0

                _complete_capture()
            return

        # Attack/defend with local fleets
        attackers_here = [fl for fl in fleets_here if fl.owner == attacker_faction]
        base_attack = sum(fl.strength for fl in attackers_here)
        def_owner = None
        if defenders:
            defender_owners = {fl.owner for fl in defenders}
            if target_sys.owner in defender_owners:
                def_owner = target_sys.owner
            else:
                def_owner = defenders[0].owner
        base_defense = sum(fl.strength for fl in defenders)
        if base_attack <= 0 or base_defense <= 0:
            return

        attack_power = base_attack * tech_multiplier(attacker_faction)
        defense_power = base_defense * tech_multiplier(def_owner)
        target_sys.occupation_faction = None
        target_sys.occupation_progress = 0

        target_sys.stability = max(0.0, target_sys.stability - 0.15)
        target_sys.heat = min(3.0, target_sys.heat + 2.0)

        base = 0.5
        p_attack_wins = base
        garrison_defense = 0.0
        if def_owner is not None and target_sys.owner == def_owner:
            garrison_defense = target_sys.garrison * tech_multiplier(def_owner)
        effective_defense = defense_power + garrison_defense
        if attack_power >= effective_defense * AUTO_WIN_RATIO:
            attacker_wins = True
        elif effective_defense >= attack_power * AUTO_WIN_RATIO:
            attacker_wins = False
        else:
            local_adv = (attack_power - effective_defense) / (
                attack_power + effective_defense
            )
            defender_home_bonus = (
                0.05 if def_owner is not None and target_sys.owner == def_owner else 0.0
            )
            p_attack_wins = base + 0.3 * local_adv - defender_home_bonus
            if target_sys.reclaim_cooldown > 0:
                p_attack_wins -= 0.15  # defending rebels get a brief edge
            p_attack_wins = max(0.2, min(0.8, p_attack_wins))
            attacker_wins = random.random() < p_attack_wins

        if attacker_wins:
            total_losers = base_defense
            remaining = max(1.0, base_attack - DAMAGE_FACTOR * total_losers)
            target_sys.garrison = 0.0  # spent in defense

            for fl in defenders:
                fl.strength = 0.0

            if attackers_here:
                main = max(attackers_here, key=lambda fl: fl.strength)
                main.strength = remaining
                main.max_strength = max(main.max_strength, remaining)
                for fl in attackers_here:
                    if fl is not main:
                        fl.strength = 0.0

            old_owner = target_sys.owner
            if old_owner != attacker_faction:
                target_sys.owner = attacker_faction
                target_sys.unrest = min(1.0, target_sys.unrest + 0.25)
                target_sys.stability = max(
                    0.0, target_sys.stability - 0.05 - 0.1 * target_sys.unrest
                )
                target_sys.econ_maturity = 0
                expansions.setdefault(attacker_faction, 0)
                expansions[attacker_faction] += 1
                if old_owner:
                    losses.setdefault(old_owner, 0)
                    losses[old_owner] += 1

            wins.setdefault(attacker_faction, 0)
            wins[attacker_faction] += 1

            text = (
                f"t={world.tick}: Fleet of {FACTION_NAMES[attacker_faction]} won a battle in "
                f"{describe_system(target_sys)} (local odds {p_attack_wins:.2f}) and took control."
            )
            log_event(
                "fleet_battle_win",
                [target_sys.id],
                [attacker_faction] + list(defender_factions),
                text,
            )
            highlight_ids.add(target_sys.id)
        else:
            total_losers = base_attack
            effective_attack = total_losers
            if (
                def_owner is not None
                and target_sys.owner == def_owner
                and target_sys.garrison > 0
            ):
                garrison_soak = target_sys.garrison * DEFENDER_GARRISON_SOAK_FACTOR
                effective_attack = max(0.0, effective_attack - garrison_soak)
            remaining = max(
                DEFENDER_MIN_REMAINING, base_defense - DAMAGE_FACTOR * effective_attack
            )
            target_sys.garrison = max(0.0, target_sys.garrison - base_attack)

            for fl in attackers_here:
                fl.strength = 0.0

            defenders_alive = [fl for fl in defenders if fl.strength > 0]
            if defenders_alive:
                main = max(defenders_alive, key=lambda fl: fl.strength)
                main.strength = remaining
                main.max_strength = max(main.max_strength, remaining)
                for fl in defenders_alive:
                    if fl is not main:
                        fl.strength = 0.0

            for df in defender_factions:
                wins.setdefault(df, 0)
                wins[df] += 1

            losses.setdefault(attacker_faction, 0)
            losses[attacker_faction] += 1

            text = (
                f"t={world.tick}: Fleet of {FACTION_NAMES[attacker_faction]} failed to take "
                f"{describe_system(target_sys)} (local odds {p_attack_wins:.2f})."
            )
            log_event(
                "fleet_battle_lose",
                [target_sys.id],
                [attacker_faction] + list(defender_factions),
                text,
            )
            highlight_ids.add(target_sys.id)

        check_rebellion(target_sys)

    def resolve_arrival(fleet: Fleet):
        # Put fleet at destination
        dest_id = fleet.enroute_to
        if dest_id is None:
            return
        fleet.system_id = dest_id
        fleet.enroute_from = None
        fleet.enroute_to = None
        fleet.eta = 0

        fleets_at_or_arriving = [
            fl
            for fl in world.fleets.values()
            if fl.strength > 0
            and fl.eta == 0
            and (fl.system_id == dest_id or fl.enroute_to == dest_id)
        ]
        resolve_system_conflict(dest_id, fleet.owner, fleets_at_or_arriving)

    for fid in arrivals:
        if fid in world.fleets:
            fl = world.fleets[fid]
            resolve_arrival(fl)

    # Failsafe: if multiple factions still share a system, resolve a local battle.
    contested_systems = {}
    for fl in world.fleets.values():
        if fl.eta != 0 or fl.system_id is None or fl.strength <= 0:
            continue
        contested_systems.setdefault(fl.system_id, []).append(fl)
    for sys_id, fleets_here in contested_systems.items():
        owners = {fl.owner for fl in fleets_here}
        if len(owners) <= 1:
            continue
        strength_by_owner: Dict[str, float] = {}
        for fl in fleets_here:
            strength_by_owner.setdefault(fl.owner, 0.0)
            strength_by_owner[fl.owner] += fl.strength
        attacker = max(strength_by_owner.items(), key=lambda kv: kv[1])[0]
        resolve_system_conflict(sys_id, attacker, fleets_here)

    # Safety: if a system has fleets present and no owner, auto-resolve control
    # (unless in reclaim cooldown after rebellion).
    for sys in world.systems.values():
        # decay occupation if no occupying fleets remain
        if sys.occupation_faction:
            occupying_present = any(
                fl.owner == sys.occupation_faction
                and fl.system_id == sys.id
                and fl.strength > 0
                and fl.eta == 0
                for fl in world.fleets.values()
            )
            if not occupying_present:
                sys.occupation_faction = None
                sys.occupation_progress = 0

        if sys.owner is not None or sys.reclaim_cooldown > 0:
            continue
        fleets_here = [
            fl
            for fl in world.fleets.values()
            if fl.system_id == sys.id and fl.strength > 0 and fl.eta == 0
        ]
        if not fleets_here:
            continue

        # Pick faction with highest total strength; if tie, leave contested.
        strength_by_owner: Dict[str, float] = {}
        for fl in fleets_here:
            strength_by_owner.setdefault(fl.owner, 0.0)
            strength_by_owner[fl.owner] += fl.strength

        if not strength_by_owner:
            continue

        sorted_strength = sorted(
            strength_by_owner.items(), key=lambda kv: kv[1], reverse=True
        )
        if len(sorted_strength) > 1 and sorted_strength[0][1] == sorted_strength[1][1]:
            # contested tie; do nothing this tick
            continue

        owner = sorted_strength[0][0]
        sys.owner = owner
        sys.econ_maturity = 0
        sys.unrest = min(1.0, sys.unrest + 0.15)
        sys.stability = max(0.0, sys.stability - 0.05 - 0.05 * sys.unrest)
        sys.reclaim_cooldown = 0
        sys.garrison = min(GARRISON_MAX, sys.garrison + 0.5 * GARRISON_MAX)
        expansions.setdefault(owner, 0)
        expansions[owner] += 1
        text = (
            f"t={world.tick}: {FACTION_NAMES[owner]} claimed empty {describe_system(sys)} "
            f"after securing orbit."
        )
        log_event("auto_claim", [sys.id], [owner], text)
        highlight_ids.add(sys.id)

    # --- Fleet building based on economy ---
    for f in faction_ids:
        stock = world.faction_build_stock[f]
        stock += econ_power[f] * BUILD_RATE
        world.faction_build_stock[f] = stock

        while world.faction_build_stock[f] >= BUILD_COST:
            world.faction_build_stock[f] -= BUILD_COST
            owned = [s.id for s in world.systems.values() if s.owner == f]
            if not owned:
                break
            sid = random.choice(owned)
            fid = world.next_fleet_id
            world.next_fleet_id += 1
            world.fleets[fid] = Fleet(
                id=fid,
                owner=f,
                system_id=sid,
                strength=NEW_FLEET_STRENGTH,
                max_strength=NEW_FLEET_STRENGTH,
                enroute_from=None,
                enroute_to=None,
                eta=0,
            )

    # Apply orders using fleets (set them in motion)
    valid_orders: list[Order] = []
    for order in orders:
        attacker = order.faction
        if attacker not in FACTION_NAMES:
            continue
        if order.origin_id not in world.systems or order.target_id not in world.systems:
            continue
        origin_sys = world.systems[order.origin_id]
        target_sys = world.systems[order.target_id]
        # must be neighbors
        if target_sys.id not in origin_sys.neighbors:
            continue
        valid_orders.append(order)

    orders_by_origin: dict[tuple[str, int], list[Order]] = {}
    for order in valid_orders:
        key = (order.faction, order.origin_id)
        orders_by_origin.setdefault(key, []).append(order)

    def _idle_fleets_at(faction: str, system_id: int) -> list[Fleet]:
        return [
            fl
            for fl in world.fleets.values()
            if fl.owner == faction
            and fl.system_id == system_id
            and fl.strength > 0
            and fl.eta == 0
        ]

    for (attacker, origin_id), origin_orders in orders_by_origin.items():
        idle = _idle_fleets_at(attacker, origin_id)
        if not idle:
            continue
        origin_sys = world.systems[origin_id]
        total_strength = sum(fl.strength for fl in idle)

        def _take_fleet(order: Order, pool: list[Fleet]) -> Optional[Fleet]:
            if order.fleet_id is not None:
                for idx, fl in enumerate(pool):
                    if fl.id == order.fleet_id:
                        return pool.pop(idx)
                return None
            if not pool:
                return None
            best_idx = max(range(len(pool)), key=lambda idx: pool[idx].strength)
            return pool.pop(best_idx)

        if len(origin_orders) == 1:
            order = origin_orders[0]
            fleet = _take_fleet(order, idle)
            if not fleet:
                continue
            target_sys = world.systems[order.target_id]
            move_reason = getattr(order, "reason", None)
            text = (
                f"t={world.tick}: {FACTION_NAMES[attacker]} launched a fleet "
                f"from system #{origin_sys.id} to system #{target_sys.id} "
                f"(strength {fleet.strength:.1f})."
            )
            if move_reason:
                text += f" Reason: {move_reason}."
            log_event("fleet_move", [origin_sys.id, target_sys.id], [attacker], text)
            fleet.enroute_from = origin_sys.id
            fleet.enroute_to = target_sys.id
            fleet.system_id = None
            fleet.eta = TRAVEL_TICKS
            continue

        idle_sorted = [
            fl for fl in sorted(idle, key=lambda fl: fl.strength, reverse=True)
        ]
        orders_to_run = origin_orders[:]
        for order in orders_to_run:
            if not idle_sorted:
                break
            target_sys = world.systems[order.target_id]
            fleet = _take_fleet(order, idle_sorted)
            if not fleet:
                continue
            if order.fleet_id is None and fleet.strength < MIN_LAUNCH_STRENGTH:
                continue
            move_reason = getattr(order, "reason", None)
            text = (
                f"t={world.tick}: {FACTION_NAMES[attacker]} launched a fleet "
                f"from system #{origin_sys.id} to system #{target_sys.id} "
                f"(strength {fleet.strength:.1f})."
            )
            if move_reason:
                text += f" Reason: {move_reason}."
            log_event("fleet_move", [origin_sys.id, target_sys.id], [attacker], text)
            fleet.enroute_from = origin_sys.id
            fleet.enroute_to = target_sys.id
            fleet.system_id = None
            fleet.eta = TRAVEL_TICKS

    # Remove dead fleets and apply small auto-repair
    to_delete = []
    for fid, fl in world.fleets.items():
        if fl.strength <= 0.0:
            to_delete.append(fid)
            continue

        # upkeep drain to discourage hoarding idle fleets
        if fl.eta == 0:
            fl.strength *= 1.0 - UPKEEP_IDLE

        if fl.strength <= 0.0:
            to_delete.append(fid)
            continue

        # only repair when idle
        if fl.eta == 0:
            fl.strength = min(fl.max_strength, fl.strength * (1.0 + REPAIR_RATE))

    for fid in to_delete:
        del world.fleets[fid]

    # Merge idle fleets of the same owner at the same system to avoid tiny strays
    merged: Dict[tuple[str, int], Fleet] = {}
    to_remove: List[int] = []
    for fid, fl in world.fleets.items():
        if fl.eta != 0 or fl.system_id is None or fl.strength <= 0:
            continue
        key = (fl.owner, fl.system_id)
        if key not in merged:
            merged[key] = fl
            continue
        primary = merged[key]
        primary.strength += fl.strength
        primary.max_strength = max(
            primary.max_strength, fl.max_strength, primary.strength
        )
        to_remove.append(fid)
    for fid in to_remove:
        del world.fleets[fid]

    # Empire size snapshot after this tick (for sprawl / instability)
    systems_per_faction: Dict[str, int] = {}
    for sys in world.systems.values():
        if sys.owner is not None:
            systems_per_faction.setdefault(sys.owner, 0)
            systems_per_faction[sys.owner] += 1

    # Global updates: heat decay & stability recovery + empire sprawl penalty
    for sys in world.systems.values():
        sys.heat *= 0.85
        if sys.heat < 0.05:
            sys.heat = 0.0

        # unrest naturally cools off
        sys.unrest *= 0.9

        # reclaim cooldown ticks down
        if sys.reclaim_cooldown > 0:
            sys.reclaim_cooldown -= 1
        # garrison decay/regeneration
        if sys.owner is None:
            sys.garrison = 0.0
            sys.econ_maturity = 0
        else:
            sys.garrison = min(GARRISON_MAX, sys.garrison + GARRISON_REGEN)
            sys.econ_maturity = min(ECON_MATURITY_TICKS, sys.econ_maturity + 1)

        # Base natural recovery
        sys.stability = min(1.0, sys.stability + 0.01)

        # Empire sprawl: big empires are harder to control
        if sys.owner in systems_per_faction:
            size = systems_per_faction[sys.owner]
            # Only kick in if you have more than 10 systems
            if size > 10:
                extra_decay = 0.002 * (size - 10)
                sys.stability = max(0.0, sys.stability - extra_decay)

    world.events.extend(events)
    max_events = 80
    if len(world.events) > max_events:
        world.events = world.events[-max_events:]

    # Cap history so it doesn't grow unbounded
    # Keep full run history (no cap) so completed runs can be dumped intact

    world.last_event_systems = sorted(highlight_ids)

    return TickSummary(
        battles_won=wins,
        battles_lost=losses,
        expansions=expansions,
    )
