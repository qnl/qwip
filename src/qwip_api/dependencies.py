from typing import Annotated

import sqlalchemy as sa

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from loguru import logger

from qwip_api.settings import settings
from qwip.config import DoltDB, Database

security = HTTPBasic()

def get_database(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> Database:
    db = DoltDB.from_parameters(
        host=settings.DOLT_SERVER,
        username=credentials.username,
        password=credentials.password
    )

    try:
        db.connect(test=True)
        return db

    except sa.exc.OperationalError as e:
        logger.info(f"Could not connect to database: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED
        )

def authenticate_user(db: Annotated[Database | None, Depends(get_database)]) -> str:
    return db.username