from typing import Annotated

import sqlalchemy as sa
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from loguru import logger

from qwip.config import ConfigDB, Database, DoltDB
from qwip.config.schema import ConfigSchema
from qwip_api.settings import settings

security = HTTPBasic()


def get_database(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    db_name: str | None = None,
    schema: str | None = None,
) -> Database:
    match schema:
        case "ConfigSchema":
            cls = ConfigDB
            kwargs = dict(schema=ConfigSchema)
        case _:
            cls = DoltDB
            kwargs = dict()

    db = cls.from_parameters(
        host=settings.DOLT_SERVER,
        username=credentials.username,
        password=credentials.password,
        database=db_name,
        **kwargs,
    )

    try:
        db.connect(test=True)
        logger.debug(f"Connected to database with url: {db.url}")
        yield db

    except sa.exc.OperationalError as e:
        logger.info(f"Could not connect to database: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    finally:
        db.disconnect()


def authenticate_user(db: Annotated[Database | None, Depends(get_database)]) -> str:
    return db.username
