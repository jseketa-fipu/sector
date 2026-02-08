import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MinterConfig(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)

    port: int = Field(default=9100, alias="PORT")
    api_key: str = Field(default="", alias="MINTER_API_KEY")


_CONFIG = MinterConfig()

app = FastAPI(title="Sector NFT Minter", version="0.1.0")


class MintRequest(BaseModel):
    universe_id: str
    winner: str
    winner_label: Optional[str] = None
    tick: Optional[int] = None
    address: str
    faction: str
    timestamp: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class MintResponse(BaseModel):
    mint_id: str
    status: str
    received_at: int
    request: MintRequest


def _require_api_key(auth_header: Optional[str]) -> None:
    if not _CONFIG.api_key:
        return
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing authorization")
    token = auth_header.split(" ", 1)[1].strip()
    if token != _CONFIG.api_key:
        raise HTTPException(status_code=401, detail="invalid authorization")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/mint", response_model=MintResponse)
async def mint(
    payload: MintRequest,
    authorization: Optional[str] = Header(None),
) -> MintResponse:
    _require_api_key(authorization)
    mint_id = uuid.uuid4().hex
    received_at = int(time.time())
    return MintResponse(
        mint_id=mint_id,
        status="ok",
        received_at=received_at,
        request=payload,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.nft_minter.main:app", host="0.0.0.0", port=_CONFIG.port)
