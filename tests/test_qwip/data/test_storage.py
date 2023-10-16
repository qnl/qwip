from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from qwip.data.storage import HTTPStorageBackend
from qwip.flatdict import FlatDict
from qwip_dataserver.settings import get_settings


class TestHTTPStorageBackend:
    @pytest.fixture(scope="class")
    def settings(self, tmp_path_factory):
        settings = get_settings()
        root = settings.DATASERVER_ROOT = tmp_path_factory.mktemp("pytest")

        file_structure = FlatDict(
            {
                "empty": {},
                "README.md": b"This is the test directory structure",
                "data-folder": {
                    "file.txt": b"A text file with some strings",
                    "number.bin": (1234567890).to_bytes(8, "big"),
                },
            }
        )

        ## Prepopulate dataserver directory.
        for path, contents in file_structure.flatitems():
            if isinstance(contents, dict):
                folder = Path(path)
            else:
                folder = Path(path).parent

            (root / folder).mkdir(exist_ok=True, parents=True)
            if not isinstance(contents, dict):
                with open(root / path, "wb") as f:
                    f.write(contents)

        yield settings

    @pytest.fixture(scope="class")
    def client(self, settings, dataserver):
        if dataserver:
            client = httpx.Client(base_url=dataserver)
        else:
            from qwip_dataserver.server import app

            client = TestClient(app)

        with client:
            yield client

    @pytest.fixture(scope="class")
    def storage(self, client, request):
        if client.app is None:
            for marker in request.node.iter_markers():
                if marker.name == "skip_dataserver":
                    pytest.skip("Skipping test on live dataserver.")

        yield HTTPStorageBackend(client=client)

    @pytest.mark.parametrize(
        "folder,address,expect",
        [
            ("", "", ("/", "")),
            ("/", "", ("/", "")),
            ("", "", ("/", "")),
            ("/folder", "", ("/folder/", "")),
            ("/a", "/a/b/c", ("/a/a/b/", "c")),
            ("/folder", "./file", ("/folder/", "file")),
            ("/folder/subfolder", "../file", ("/folder/", "file")),
            ("", "/folder/file.txt", ("/folder/", "file.txt")),
        ],
    )
    def test_parse_url(self, storage, folder, address, expect):
        assert storage.parse_url(folder, address) == expect

    @pytest.mark.skip_dataserver
    @pytest.mark.parametrize(
        "folder,expect",
        [
            ("/non-existent", httpx.HTTPStatusError),
            (
                "/",
                dict(
                    path="/",
                    num_children=3,
                    contents=["data-folder", "empty", "README.md"],
                ),
            ),
            (
                "/empty",
                dict(
                    path="/empty",
                    num_children=0,
                    contents=[],
                ),
            ),
            (
                "/data-folder/",
                dict(
                    path="/data-folder",
                    num_children=2,
                    contents=["file.txt", "number.bin"],
                ),
            ),
            (
                "/../..",
                dict(
                    path="/",
                    num_children=3,
                    contents=["data-folder", "empty", "README.md"],
                ),
            ),
        ],
    )
    def test_list_directory(self, storage, folder, expect):
        if isinstance(expect, type) and issubclass(expect, Exception):
            with pytest.raises(expect):
                storage.list_directory(folder)
        else:
            assert storage.list_directory(folder) == expect

    @pytest.mark.skip_dataserver
    @pytest.mark.parametrize(
        "folder", ["/pytest", "/pytest/folder1/folder2/folder3", "/", ""]
    )
    def test_make_directory(self, storage, settings, folder):
        storage.make_directory(folder)
        assert (settings.DATASERVER_ROOT / folder.lstrip("./")).exists()

    @pytest.mark.skip_dataserver
    @pytest.mark.parametrize(
        "address,data,expect",
        [
            ("text.txt", b"Binary text.", "/static/save_buffer/text.txt"),
            (
                "/subfolder/text.txt",
                b"Binary text.",
                "/static/save_buffer/subfolder/text.txt",
            ),
            ("exists", b"", None),
        ],
    )
    def test_save_buffer(self, storage, settings, address, data, expect):
        (settings.DATASERVER_ROOT / "save_buffer/exists").mkdir(
            parents=True, exist_ok=True
        )

        download_address = storage.save_buffer(address, data, folder="/save_buffer")
        assert download_address == expect
        if download_address:
            assert (
                settings.DATASERVER_ROOT / "save_buffer" / address.lstrip("./")
            ).read_bytes() == data

    @pytest.mark.skip_dataserver
    @pytest.mark.parametrize(
        "address,expect",
        [
            ("README.md", b"This is the test directory structure"),
            ("data-folder/file.txt", b"A text file with some strings"),
            ("/data-folder/number.bin", (1234567890).to_bytes(8, "big")),
            ("empty", httpx.HTTPStatusError),
        ],
    )
    def test_load_buffer(self, storage, address, expect):
        if isinstance(expect, type) and issubclass(expect, Exception):
            with pytest.raises(expect):
                downloaded = storage.load_buffer(address)

        else:
            downloaded = storage.load_buffer(address).read()

            assert downloaded == expect

    @pytest.mark.parametrize("address,data", [("text.txt", b"ASCII bytes.")])
    def test_round_trip(self, storage, address, data):
        download_address = storage.save_buffer(address, data, folder="pytest-roundtrip")
        downloaded = storage.load_buffer(download_address).read()

        assert data == downloaded
