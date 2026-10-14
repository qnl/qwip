from __future__ import annotations

import sys
import types
from unittest.mock import patch

from qwip_explorer.connections import close_datastore, connect_datastore


class FakeStorage:
    def __init__(self, value):
        self.value = value
        self.closed = False

    def close(self):
        self.closed = True


class FakeDatastore:
    made = []

    def __init__(self, storage):
        self.storage = storage
        self.session = None
        self.disconnected = False

    @classmethod
    def from_parameters(cls, **kwargs):
        result = cls(kwargs["storage"])
        cls.made.append(result)
        return result

    def connect(self):
        self.session = object()

    def disconnect(self):
        self.disconnected = True
        self.session = None


class FakeHTTPStorageBackend:
    @classmethod
    def from_url(cls, url):
        return FakeStorage(url)


class FakeLocalStorageBackend:
    def __init__(self, directory):
        self.directory = directory


def test_datastore_connections_are_not_shared_between_browser_sessions():
    fake_data = types.ModuleType("qwip.data")
    fake_data.Datastore = FakeDatastore
    fake_data.HTTPStorageBackend = FakeHTTPStorageBackend
    fake_data.LocalStorageBackend = FakeLocalStorageBackend
    FakeDatastore.made.clear()

    with patch.dict(sys.modules, {"qwip.data": fake_data}):
        first = connect_datastore("db.example", "catalog", "alice", "one", "HTTP", "http://assets")
        second = connect_datastore("db.example", "catalog", "bob", "two", "HTTP", "http://assets")

    assert first is not second
    assert len(FakeDatastore.made) == 2
    close_datastore(first)
    assert first.disconnected
    assert first.storage.closed
    assert not second.disconnected
