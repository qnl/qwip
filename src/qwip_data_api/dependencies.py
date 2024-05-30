from typing import Annotated, Generator

import sqlalchemy as sa
from fastapi import Depends
from pydantic import BaseModel
from pydantic_extra_types.pendulum_dt import DateTime
from sqlalchemy.orm import Session

from qwip_data_api.settings import settings

engine = sa.create_engine(str(settings.SQLALCHEMY_DATABASE_URI))


def db_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


SessionDepends = Annotated[Session, Depends(db_session)]


class DatasetQueryParameters(BaseModel):
    start_time: DateTime | None = None
    end_time: DateTime | None = None
    user: str | None = None
    host: str | None = None
    sample_id: str | None = None
    cooldown_id: str | None = None
    comments: str | None = None
    limit: int = 50
    offset: int = 0


DatasetQueryDepends = Annotated[DatasetQueryParameters, Depends(DatasetQueryParameters)]
