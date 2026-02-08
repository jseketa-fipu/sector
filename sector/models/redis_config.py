from pydantic import RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    redis_url: RedisDsn
    event_stream: str
    order_stream: str
    snapshot_key: str


REDIS_SETTINGS = RedisSettings()
