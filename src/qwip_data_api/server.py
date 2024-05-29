import sys
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from uuid6 import UUID
from pydantic import BaseModel
from pendulum import DateTime

import qwip
from qwip.data.models import Dataset
from qwip_data_api.dependencies import SessionDepends
from qwip_data_api.settings import settings

logger.add(sys.stderr)

api_v1 = APIRouter(prefix="/api/v1")


def add_equals(stmt, **kwargs):
    for col, value in kwargs.items():
        stmt = stmt.where(getattr(Dataset, col) == value)
    return stmt   

def add_substring_search(stmt, **kwargs):
    for col, value in kwargs.items():
        search_value = f"%{value}%" if "%" not in value else value
        stmt = stmt.where(getattr(Dataset, col).ilike(search_value))

    return stmt

        
    

@api_v1.get("/datasets/", response_model=None)
async def get_datasets(
    session: SessionDepends,
    time_start: DateTime | None = None, #fastapi doesn't like this 
    time_end: DateTime | None = None, 
    user: str | None= None, 
    sample_id: str | None = None, # should this be turned into a list? 
    cooldown_id: str | None = None,
    comments: str | None = None,
    limit: int | None = None
    ) -> dict:

    stmt = sa.select(Dataset)

    exact = dict(sample_id=sample_id, cooldown_id=cooldown_id)
    substring = dict(user=user, comments=comments)

    stmt = add_equals(stmt, **{c: v for c, v in exact.items() if v is not None})
    stmt = add_substring_search(stmt, **{c: v for c, v in substring.items() if v is not None})
    
    if time_start:
        stmt = stmt.where(Dataset.timestamp >= time_start)
    if time_end:
        stmt = stmt.where(Dataset.timestamp <= time_end)

    datasets = session.scalars(stmt).all()
    return qwip.converter.unstructure(datasets)

@api_v1.get("/datasets/{dataset_id}")
async def get_dataset_by_id(
    dataset_id: Annotated[str, Path(title="Dataset UUID")], 
    session: SessionDepends
) -> dict:
    dataset_id = UUID(dataset_id)

    stmt = sa.select(Dataset).where(Dataset.id == dataset_id)
    dataset = session.scalars(stmt).one_or_none()

    ## Should probably return a pydantic model instead
    return qwip.converter.unstructure(dataset)


logger.info(f"Using the following environment settings: {settings}")

app = FastAPI(
    title="QWiP Data API",
)
app.include_router(api_v1)


# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
