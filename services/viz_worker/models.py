from pydantic_settings import BaseSettings
from pydantic import AnyUrl
from pydantic import Field  # type: ignore
from typing import Annotated


class VisualizationWorkerSettings(BaseSettings):
    redis_url: Annotated[
        AnyUrl,
        Field(validation_alias="REDIS_URL"),
    ]
    api_url: Annotated[AnyUrl, Field(validation_alias="API_URL")]
    port: Annotated[int, Field(validation_alias="PORT")]
