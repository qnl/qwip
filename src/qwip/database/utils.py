from pathlib import Path

import pendulum
from sqlalchemy import types
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement
from uuid6 import UUID

JSONTypes = dict | list | bool | float | int | str | None


class utc_timestamp(FunctionElement):
    """Gets the current timestamp in UTC.

    Based on SQLAlchemy documentation [example](https://docs.sqlalchemy.org/en/20/core/compiler.html#utc-timestamp-function).
    """

    type = types.DateTime()
    inherit_cache = True


@compiles(utc_timestamp, "mysql")
def mysql_utc_timestamp(element, compiler, **kwargs):
    return "UTC_TIMESTAMP()"


@compiles(utc_timestamp, "sqlite")
def sqlite_utc_timestamp(element, compiler, **kwargs):
    return "DATETIME('now')"


class GUID(types.TypeDecorator):
    impl = types.BINARY(16)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return value.bytes

    def process_result_value(self, value, dialect):
        return UUID(bytes=value)


class PendulumDateTime(types.TypeDecorator):
    impl = types.DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        match dialect.name:
            case "sqlite":
                # sqlite requires datetime not string
                return value
            case "mysql":
                # pendulum 3.0 default repr is no longer ISO8601 compatible.
                return value.isoformat()
            case _:
                return value

    def process_result_value(self, value, dialect):
        return pendulum.instance(value).in_tz("local")


class FilePath(types.TypeDecorator):
    impl = types.String
    cache_ok = True

    def process_result_value(self, value, dialect):
        return Path(value)
