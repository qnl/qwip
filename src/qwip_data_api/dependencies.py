from typing import Annotated, Generator

import sqlalchemy as sa
from fastapi import Depends
from sqlalchemy.orm import Session

from qwip_data_api.settings import settings

engine = sa.create_engine(str(settings.SQLALCHEMY_DATABASE_URI))


def db_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


SessionDepends = Annotated[Session, Depends(db_session)]
