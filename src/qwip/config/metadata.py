from sqlalchemy import MetaData
from sqlalchemy.orm import registry

CONFIGDB_METADATA = MetaData()
CONFIGDB_REGISTRY = registry(metadata=CONFIGDB_METADATA)
