import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

import jwt
import redis
import uvicorn
from eth_account import Account
from eth_account.messages import encode_defunct
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sector.infra.redis_streams import RedisStreams
from sector.world import FACTION_NAMES


@dataclass(frozen=True)
class ApiConfig:
    redis_url: str | None
    cors_allow_origins: str
    jwt_secret: str
    jwt_ttl_seconds: int
    bot_api_token: str
    nonce_ttl_seconds: int
    session_key_prefix: str
    address_session_prefix: str
    nonce_key_prefix: str
    universe_key: str
    human_factions_key: str
    player_faction_prefix: str
    faction_player_prefix: str
    restart_key: str
    bots_only_key: str
    port: int


def _load_config() -> ApiConfig:
    return ApiConfig(
        redis_url=os.environ.get("REDIS_URL"),
        cors_allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*"),
        jwt_secret=os.environ.get("JWT_SECRET", "dev-secret"),
        jwt_ttl_seconds=int(os.environ.get("JWT_TTL_SECONDS", "86400")),
        bot_api_token=os.environ.get("BOT_API_TOKEN", ""),
        nonce_ttl_seconds=int(os.environ.get("AUTH_NONCE_TTL_SECONDS", "300")),
        session_key_prefix=os.environ.get("SESSION_KEY_PREFIX", "sector:session"),
        address_session_prefix=os.environ.get(
            "ADDRESS_SESSION_PREFIX", "sector:session:addr"
        ),
        nonce_key_prefix=os.environ.get("NONCE_KEY_PREFIX", "sector:auth:nonce"),
        universe_key=os.environ.get("UNIVERSE_KEY", "sector:universe_id"),
        human_factions_key=os.environ.get(
            "HUMAN_FACTIONS_KEY", "sector:human_factions"
        ),
        player_faction_prefix=os.environ.get(
            "PLAYER_FACTION_PREFIX", "sector:player:faction"
        ),
        faction_player_prefix=os.environ.get(
            "FACTION_PLAYER_PREFIX", "sector:faction:player"
        ),
        restart_key=os.environ.get("RESTART_KEY", "sector:restart"),
        bots_only_key=os.environ.get("BOTS_ONLY_KEY", "sector:bots_only"),
        port=int(os.environ.get("PORT", "8000")),
    )


_CONFIG = _load_config()

# Redis streams helper (async). We reuse the sector-backend stream/key defaults.
streams = RedisStreams(url=_CONFIG.redis_url)

app = FastAPI(title="Sector API", version="0.1.0")

cors_origins = _CONFIG.cors_allow_origins
origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class PlayerContext(TypedDict):
    address: str
    claims: dict[str, Any]


def _normalize_address(address: str) -> str:
    addr = address.strip().lower()
    if not addr.startswith("0x") or len(addr) != 42:
        raise HTTPException(status_code=400, detail="invalid wallet address")
    return addr


def _login_message(address: str, nonce: str) -> str:
    return f"Sector login\nAddress: {address}\nNonce: {nonce}"


async def _get_player_faction(universe_id: str, address: str) -> str | None:
    return await streams.client.get(
        f"{_CONFIG.player_faction_prefix}:{universe_id}:{address}"
    )


async def _clear_faction_claims(universe_id: str) -> None:
    faction_pattern = f"{_CONFIG.faction_player_prefix}:{universe_id}:*"
    player_pattern = f"{_CONFIG.player_faction_prefix}:{universe_id}:*"
    keys: list[str] = []
    async for key in streams.client.scan_iter(match=faction_pattern):
        keys.append(key)
    async for key in streams.client.scan_iter(match=player_pattern):
        keys.append(key)
    if keys:
        await streams.client.delete(*keys)
    await streams.client.delete(f"{_CONFIG.human_factions_key}:{universe_id}")


async def _filter_bot_orders(
    universe_id: str, orders: List["OrderIn"]
) -> List["OrderIn"]:
    claimed = set(
        await streams.client.smembers(f"{_CONFIG.human_factions_key}:{universe_id}")
        or []
    )
    orders = [order for order in orders if order.faction not in claimed]
    if not orders:
        return []
    factions = sorted({order.faction for order in orders})
    keys = [
        f"{_CONFIG.faction_player_prefix}:{universe_id}:{fid}" for fid in factions
    ]
    owners = await streams.client.mget(keys)
    claimed_by_map = {fid for fid, owner in zip(factions, owners) if owner}
    return [order for order in orders if order.faction not in claimed_by_map]


async def _get_universe_id() -> str:
    current = await streams.client.get(_CONFIG.universe_key)
    if current:
        return current
    await streams.client.setnx(_CONFIG.universe_key, "1")
    return await streams.client.get(_CONFIG.universe_key) or "1"


async def _get_current_player(
    authorization: str | None = Header(None),
) -> PlayerContext:
    if not authorization:
        raise HTTPException(status_code=401, detail="missing authorization")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="invalid authorization")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims: dict[str, Any] = jwt.decode(  # type: ignore[reportUnknownMemberType]
            token, _CONFIG.jwt_secret, algorithms=["HS256"]
        )
    except jwt.PyJWTError as exc:  # type: ignore[reportUnknownMemberType]
        raise HTTPException(status_code=401, detail="invalid token") from exc
    addr = claims.get("sub")
    jti = claims.get("jti")
    if not addr or not jti:
        raise HTTPException(status_code=401, detail="invalid token claims")
    stored = await streams.client.get(f"{_CONFIG.session_key_prefix}:{jti}")
    if not stored or stored.lower() != addr.lower():
        raise HTTPException(status_code=401, detail="session expired")
    return {"address": addr.lower(), "claims": claims}


async def _get_current_player_optional(request: Request) -> PlayerContext | None:
    authorization = request.headers.get("authorization")
    if not authorization:
        return None
    return await _get_current_player(authorization=authorization)


def _is_bot_request(request: Request) -> bool:
    if not _CONFIG.bot_api_token:
        return False
    token = request.headers.get("x-bot-token", "")
    return bool(token) and secrets.compare_digest(token, _CONFIG.bot_api_token)


class OrderIn(BaseModel):
    """Move/attack order used by the sector sim."""

    faction: str = Field(..., description="Faction id, e.g. 'E'")
    origin_id: int = Field(..., description="Source system id")
    target_id: int = Field(..., description="Neighbor/target system id")
    reason: str | None = Field(None, description="Optional reason for debugging")
    source: str | None = Field(None, description="Order source (server assigned)")
    fleet_id: int | None = Field(None, description="Optional fleet id to move")


class OrdersPayload(BaseModel):
    """Payload for posting one or more orders to the worker."""

    orders: List[OrderIn]


class NonceRequest(BaseModel):
    address: str = Field(..., description="Wallet address (0x...)")


class VerifyRequest(BaseModel):
    address: str = Field(..., description="Wallet address (0x...)")
    signature: str = Field(..., description="Signature of the login message")


class ClaimRequest(BaseModel):
    faction: str = Field(..., description="Faction id to claim, e.g. 'F1'")


@app.get("/health")
async def health() -> Dict[str, str]:
    try:
        await streams.client.ping()
    except redis.exceptions.RedisError as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=503, detail=f"redis error: {exc}") from exc
    return {"status": "ok"}


@app.post("/auth/nonce")
async def auth_nonce(payload: NonceRequest) -> Dict[str, str]:
    address = _normalize_address(payload.address)
    nonce = secrets.token_urlsafe(16)
    await streams.client.set(
        f"{_CONFIG.nonce_key_prefix}:{address}",
        nonce,
        ex=_CONFIG.nonce_ttl_seconds,
    )
    return {"nonce": nonce, "message": _login_message(address, nonce)}


@app.post("/auth/verify")
async def auth_verify(payload: VerifyRequest) -> Dict[str, str]:
    address = _normalize_address(payload.address)
    nonce = await streams.client.get(f"{_CONFIG.nonce_key_prefix}:{address}")
    if not nonce:
        raise HTTPException(status_code=400, detail="nonce expired")
    message = _login_message(address, nonce)
    try:
        recovered = Account.recover_message(
            encode_defunct(text=message), signature=payload.signature
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid signature") from exc
    if recovered.lower() != address.lower():
        raise HTTPException(status_code=401, detail="signature mismatch")
    await streams.client.delete(f"{_CONFIG.nonce_key_prefix}:{address}")
    now = int(time.time())
    jti = uuid.uuid4().hex
    claims = {
        "sub": address,
        "iat": now,
        "exp": now + _CONFIG.jwt_ttl_seconds,
        "jti": jti,
    }
    token = jwt.encode(claims, _CONFIG.jwt_secret, algorithm="HS256")
    await streams.client.set(
        f"{_CONFIG.session_key_prefix}:{jti}",
        address,
        ex=_CONFIG.jwt_ttl_seconds,
    )
    session_set_key = f"{_CONFIG.address_session_prefix}:{address}"
    await streams.client.sadd(session_set_key, jti)
    await streams.client.expire(session_set_key, _CONFIG.jwt_ttl_seconds)
    return {"token": token}


@app.get("/me")
async def me(player: dict = Depends(_get_current_player)) -> Dict[str, str | None]:
    universe_id = await _get_universe_id()
    addr = player["address"]
    faction = await _get_player_faction(universe_id, addr)
    return {"address": addr, "faction": faction, "universe_id": universe_id}


@app.get("/factions")
async def factions() -> Dict[str, Any]:
    universe_id = await _get_universe_id()
    claimed = await streams.client.smembers(
        f"{_CONFIG.human_factions_key}:{universe_id}"
    )
    out = []
    for fid, name in FACTION_NAMES.items():
        out.append({"id": fid, "name": name, "claimed": fid in claimed})
    return {"universe_id": universe_id, "factions": out}


@app.post("/factions/claim")
async def claim_faction(
    payload: ClaimRequest, player: dict = Depends(_get_current_player)
) -> Dict[str, str]:
    universe_id = await _get_universe_id()
    addr = player["address"]
    faction = payload.faction
    if faction not in FACTION_NAMES:
        raise HTTPException(status_code=400, detail="unknown faction")
    player_key = f"{_CONFIG.player_faction_prefix}:{universe_id}:{addr}"
    existing = await streams.client.get(player_key)
    if existing and existing != faction:
        raise HTTPException(status_code=409, detail="player already claimed a faction")
    faction_key = f"{_CONFIG.faction_player_prefix}:{universe_id}:{faction}"
    if not await streams.client.setnx(faction_key, addr):
        raise HTTPException(status_code=409, detail="faction already claimed")
    await streams.client.set(player_key, faction)
    await streams.client.sadd(f"{_CONFIG.human_factions_key}:{universe_id}", faction)
    return {"address": addr, "faction": faction}


@app.get("/snapshot")
async def snapshot() -> Dict[str, Any]:
    """
    Return the latest world snapshot saved by the simulation worker.
    """
    snap = await streams.load_snapshot()
    if not snap:
        raise HTTPException(status_code=404, detail="no snapshot yet")
    return snap


@app.get("/events")
async def events(after: str = "0-0", count: int = 100) -> List[Dict[str, Any]]:
    """
    Stream events after a given stream id. Events are published as JSON under the
    'data' field by the simulation worker.
    """
    entries = await streams.read_events(last_id=after, count=count, block_ms=None)
    out: List[Dict[str, Any]] = []
    for entry_id, payload in entries:
        item = dict(payload)
        item["id"] = entry_id
        out.append(item)
    return out


@app.post("/orders")
async def post_orders(
    payload: OrdersPayload,
    request: Request,
    player: dict | None = Depends(_get_current_player_optional),
) -> Dict[str, str]:
    """
    Append one or more orders to the Redis orders stream.
    The simulation worker consumes them.
    """
    is_bot = _is_bot_request(request)
    if not is_bot:
        if not player:
            raise HTTPException(status_code=401, detail="missing authorization")
        universe_id = await _get_universe_id()
        addr = player["address"]
        faction = await _get_player_faction(universe_id, addr)
        if not faction:
            raise HTTPException(status_code=403, detail="faction not claimed")
        for order in payload.orders:
            if order.faction != faction:
                raise HTTPException(status_code=403, detail="order faction mismatch")
    else:
        universe_id = await _get_universe_id()
        orders = await _filter_bot_orders(universe_id, payload.orders)
        if not orders:
            return {"id": "skipped"}
        payload.orders = orders
    order_source = "bot" if is_bot else "human"
    raw = {
        "orders": [
            {**order.model_dump(exclude_none=True), "source": order_source}
            for order in payload.orders
        ]
    }
    msg_id = await streams.client.xadd(
        name=streams.order_stream,
        fields={"data": json.dumps(raw)},
        maxlen=5000,
        approximate=True,
    )
    return {"id": msg_id}


@app.post("/admin/restart")
async def restart_sim(
    seed: Optional[str] = None,
    player: dict = Depends(_get_current_player),
) -> Dict[str, str]:
    """
    Trigger a simulation restart. Requires authentication.
    """
    new_universe = await streams.client.incr(_CONFIG.universe_key)
    if seed:
        payload = json.dumps({"universe_id": str(new_universe), "seed": seed})
        await streams.client.set(_CONFIG.restart_key, payload, ex=30)
    else:
        await streams.client.set(_CONFIG.restart_key, str(new_universe), ex=30)
    return {"status": "queued", "universe_id": str(new_universe)}


@app.post("/admin/bots-only")
async def bots_only(
    player: dict = Depends(_get_current_player),
) -> Dict[str, str]:
    """
    Clear any human faction claims so the run is bots-only.
    """
    universe_id = await _get_universe_id()
    await _clear_faction_claims(universe_id)
    await streams.client.set(f"{_CONFIG.bots_only_key}:{universe_id}", "1")
    return {"status": "ok", "universe_id": universe_id}


if __name__ == "__main__":
    uvicorn.run(
        "services.api.main:app", host="0.0.0.0", port=_CONFIG.port, reload=False
    )
