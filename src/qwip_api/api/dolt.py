from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

import qwip
from qwip.config import Database
from qwip_api.dependencies import get_database

router = APIRouter(prefix="/dolt")


@router.get("/{db_name}/status")
def read_status(db: Annotated[Database, Depends(get_database)]):
    return qwip.converter.unstructure(db.status())


@router.get("/{db_name}/log")
def read_log(db: Annotated[Database, Depends(get_database)]):
    return qwip.converter.unstructure(db.log())


@router.get("/{db_name}/head")
def read_head(db: Annotated[Database, Depends(get_database)]):
    return qwip.converter.unstructure(db.current_branch())


@router.get("/{db_name}/branch")
def read_all_branches(db: Annotated[Database, Depends(get_database)]):
    return qwip.converter.unstructure(db.get_all_branches())


@router.post("/{db_name}/branch/", status_code=status.HTTP_201_CREATED)
def create_branch(
    db: Annotated[Database, Depends(get_database)],
    branch_name: str,
    checkout: bool = True,
):
    try:
        if checkout:
            db.checkout(branch_name, new_branch=True)
        else:
            db.branch(branch_name, action="create")
    except sa.exc.OperationalError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Branch '{branch_name}' already exists.",
        )

    return qwip.converter.unstructure(db.get_branch(branch_name))


@router.get("/{db_name}/branch/{branch_name}")
def read_branch(db: Annotated[Database, Depends(get_database)], branch_name: str):
    branch = db.get_branch(branch_name)

    if branch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Branch '{branch_name}' does not exist.",
        )

    return qwip.converter.unstructure(branch)


@router.delete("/{db_name}/branch/{branch_name}")
def delete_branch(
    db: Annotated[Database, Depends(get_database)],
    branch_name: str,
    force: bool = False,
):
    branch = db.get_branch(branch_name)

    if branch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Branch '{branch_name}' does not exist.",
        )

    try:
        db.branch(branch_name, action="delete", force=force)
    except sa.exc.OperationalError as e:
        logger.error(f"Unable to delete branch {branch_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Unable to delete branch {branch_name}.",
        )

    return "Success"
