from typing import Annotated

from fastapi import APIRouter, Depends
from loguru import logger
from qwip.config import Database

from qwip_api.dependencies import get_database

router = APIRouter(prefix="/database")

@router.get("/")
def read_databases(
    db: Annotated[Database, Depends(get_database)]
) -> list[str]:
    """Get a list of all databases accessible by the user."""

    return db.get_databases()
