import sys
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from uuid6 import UUID

import qwip
from qwip.data.models import Dataset
from qwip_data_api.dependencies import DatasetQueryDepends, SessionDepends
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
    query: DatasetQueryDepends,
) -> dict:

    stmt = sa.select(Dataset).options(sa.orm.lazyload(Dataset._assets))

    stmt = stmt.order_by(Dataset.id.desc())

    exact = dict(
        sample_id=query.sample_id, cooldown_id=query.cooldown_id, host=query.host
    )
    substring = dict(user=query.user, comments=query.comments)

    stmt = add_equals(stmt, **{c: v for c, v in exact.items() if v is not None})
    stmt = add_substring_search(
        stmt, **{c: v for c, v in substring.items() if v is not None}
    )

    if query.start_time:
        stmt = stmt.where(Dataset.timestamp >= query.start_time)
    if query.end_time:
        stmt = stmt.where(Dataset.timestamp <= query.end_time)

    stmt = stmt.limit(query.limit).offset(query.offset)

    datasets = session.scalars(stmt).all()
    return qwip.converter.unstructure(datasets)


@api_v1.get("/datasets/{dataset_id}")
async def get_dataset_by_id(
    dataset_id: Annotated[str, Path(title="Dataset UUID")], session: SessionDepends
) -> dict:
    err_msg = f'Dataset "{dataset_id}" does not exist.'

    try:
        dataset_id = UUID(dataset_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, err_msg) from e

    stmt = sa.select(Dataset).where(Dataset.id == dataset_id)
    dataset = session.scalars(stmt).one_or_none()

    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, err_msg)

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
