from pydantic_settings import BaseSettings
from pydantic import HttpUrl, RedisDsn
from pydantic import Field  # type: ignore
from typing import Annotated


class VisualizationWorkerSettings(BaseSettings):
    redis_url: Annotated[
        RedisDsn,
        Field(validation_alias="REDIS_URL"),
    ]
    api_url: Annotated[HttpUrl, Field(validation_alias="API_URL")]
    port: Annotated[int, Field(validation_alias="PORT")]
