from typing import Annotated

from fastapi import APIRouter, Depends
from loguru import logger

import qwip
from qwip.config import Database
from qwip_api.dependencies import get_database

router = APIRouter(prefix="/config")


@router.get("/")
def read_all_databases(db: Annotated[Database, Depends(get_database)]) -> list[str]:
    """Get a list of all databases accessible by the user."""

    return db.get_databases()


@router.get("/{db_name}")
def read_config(
    db: Annotated[Database, Depends(get_database)], schema: str, path: str = "/"
):
    return qwip.converter.unstructure(db.config.todict())
