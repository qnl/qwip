from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    
    on_startup()
    yield
    on_shutdown()

def on_startup():
    logger.info("Starting server...")

def on_shutdown():
    logger.info("Shutting down server...")
