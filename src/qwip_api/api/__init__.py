from fastapi import APIRouter
from qwip_api.api import database

api_router = APIRouter()
api_router.include_router(database.router, tags=["database"])
