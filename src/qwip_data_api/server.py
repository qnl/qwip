import sys
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from uuid6 import UUID

import qwip
from qwip.data.models import Dataset
from qwip_data_api.dependencies import SessionDepends
from qwip_data_api.settings import settings

logger.add(sys.stderr)

api_v1 = APIRouter(prefix="/api/v1")


@api_v1.get("/datasets/")
async def get_datasets(session: SessionDepends) -> list[dict]:
    # Add query parameters to filter by database columns
    # Implement pagination
    ...


@api_v1.get("/datasets/{dataset_id}")
async def get_dataset_by_id(
    dataset_id: Annotated[str, Path(title="Dataset UUID")], session: SessionDepends
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
