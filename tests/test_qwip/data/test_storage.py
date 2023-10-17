from pathlib import Path

import httpx
import pytest

import qwip
from qwip.data.storage import HTTPStorageBackend, StorageBackend
from qwip.flatdict import FlatDict


class TestHTTPStorageBackend:
    @pytest.fixture(scope="class")
    def root(self, settings):
        root = settings.DATASERVER_ROOT
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

        yield root

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
    def test_list_directory(self, storage, folder, expect, root):
        if isinstance(expect, type) and issubclass(expect, Exception):
            with pytest.raises(expect):
                storage.list_directory(folder)
        else:
            assert storage.list_directory(folder) == expect

    @pytest.mark.skip_dataserver
    @pytest.mark.parametrize(
        "folder", ["/pytest", "/pytest/folder1/folder2/folder3", "/", ""]
    )
    def test_make_directory(self, storage, root, folder):
        storage.make_directory(folder)
        assert (root / folder.lstrip("./")).exists()

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
    def test_save_buffer(self, storage, root, address, data, expect):
        (root / "save_buffer/exists").mkdir(parents=True, exist_ok=True)

        download_address = storage.save_buffer(address, data, folder="/save_buffer")
        assert download_address == expect
        if download_address:
            assert (root / "save_buffer" / address.lstrip("./")).read_bytes() == data

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
    def test_load_buffer(self, storage, address, expect, root):
        if isinstance(expect, type) and issubclass(expect, Exception):
            with pytest.raises(expect):
                downloaded = storage.load_buffer(address)

        else:
            downloaded = storage.load_buffer(address).read()

            assert downloaded == expect

    @pytest.mark.parametrize("address,data", [("text.txt", b"ASCII bytes.")])
    def test_data_round_trip(self, storage, address, data):
        download_address = storage.save_buffer(address, data, folder="pytest-roundtrip")
        downloaded = storage.load_buffer(download_address).read()

        assert data == downloaded

    def test_unstructure(self, storage):
        unstructured = qwip.converter.unstructure(storage)

        assert unstructured == dict(
            client=dict(base_url=str(storage.client.base_url)),
            __class__="HTTPStorageBackend",
        )

    def test_structure(self):
        structured = qwip.converter.structure(
            dict(
                client=dict(base_url="http://dataserver"),
                __class__="HTTPStorageBackend",
            ),
            StorageBackend,
        )

        assert structured.client.base_url == httpx.URL("http://dataserver")
        assert isinstance(structured, HTTPStorageBackend)

    def test_structure_round_trip(self, storage):
        unstructured = qwip.converter.unstructure(storage)
        structured = qwip.converter.structure(unstructured, StorageBackend)

        assert structured.client.base_url == storage.client.base_url
        assert type(structured) == type(storage)
