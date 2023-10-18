import shutil
import sys
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from qwip_dataserver.settings import Settings, get_settings

logger.add(sys.stderr)


def get_root_path(settings: Annotated[Settings, Depends(get_settings)]) -> Path:
    """Returns the root path where files are stored."""

    return settings.DATASERVER_ROOT


api_v1 = APIRouter(prefix="/api/v1")


@api_v1.get("/folder/{file_path:path}")
async def list_directory(
    root: Annotated[Path, Depends(get_root_path)], file_path: str = ""
):
    """List directory with the given path.

    Args:
        root: The root data server path, where all data files are stored.
        file_path: The relative path to the directory to list.
    """
    directory = root / file_path

    if not directory.exists() or not directory.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid directory."
        )

    contents = [p.relative_to(directory).as_posix() for p in directory.iterdir()]
    return dict(
        path="/" + file_path.rstrip("/"), num_children=len(contents), contents=contents
    )


@api_v1.post("/folder/{file_path:path}")
async def make_directory(
    root: Annotated[Path, Depends(get_root_path)], file_path: str = ""
):
    """Create a directory with the given path.

    Args:
        root: The root data server path, where all data files are stored.
        file_path: The relative path to the directory to list.
    """
    directory = root / file_path

    if directory.exists() and not directory.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot create directory."
        )

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception(
            f"Error creating directory '{directory.as_posix()}'", exception=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error when creating directory.",
        )

    return dict(path=directory.relative_to(root).as_posix())


@api_v1.post("/upload/{file_path:path}")
def upload_files(
    root: Annotated[Path, Depends(get_root_path)],
    uploads: Annotated[list[UploadFile], File(description="File upload.")],
    file_path: str = "",
):
    """Saves the uploaded files to the root data server folder.

    Filenames are taken from the upload file metadata.

    Args:
        root: The root data server path, where all data files are stored.
        uploads: A list of file uploads to save to the server.
        file_path: The relative path to the parent directory to save the files.
    """
    directory = root / file_path

    if not directory.exists() or not directory.is_dir():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    addresses = []
    for file in uploads:
        try:
            path = directory / file.filename
            with open(directory / path, "wb") as dest:
                shutil.copyfileobj(file.file, dest)
            address = (Path("/static") / file_path / file.filename).as_posix()
            addresses.append(dict(address=address, size=path.stat().st_size))
        except Exception as e:
            addresses.append(dict(error=str(e)))
        finally:
            file.file.close()

    return dict(uploads=addresses)


settings = get_settings()

logger.info(f"Using the following environment settings: {settings}")

app = FastAPI(
    title="QWiP Data Server",
)
app.include_router(api_v1)
app.mount("/static", StaticFiles(directory=settings.DATASERVER_ROOT), "static")


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
