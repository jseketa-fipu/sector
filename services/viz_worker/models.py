from dataclasses import dataclass


@dataclass(frozen=True)
class VizConfig:
    redis_url: str | None
    api_url: str
    port: int
