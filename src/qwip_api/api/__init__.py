from fastapi import APIRouter

from qwip_api.api import config, dolt

api_router = APIRouter()
api_router.include_router(config.router, tags=["database"])
api_router.include_router(dolt.router, tags=["version-control"])
