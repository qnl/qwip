from sqlalchemy import MetaData
from sqlalchemy.orm import registry

QWIP_DB_METADATA = MetaData()
QWIP_DB_REGISTRY = registry(metadata=QWIP_DB_METADATA)
