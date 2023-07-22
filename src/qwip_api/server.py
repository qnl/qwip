import sys
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from qwip_api.api import api_router
from qwip_api.dependencies import authenticate_user
from qwip_api.events import lifespan
from qwip_api.settings import settings

logger.add(sys.stdout, level="INFO")
app = FastAPI(title="QWiP", lifespan=lifespan)

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

app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {"version": "23.07.1"}


logger.info(f"Using the following environment settings: {settings}.")
