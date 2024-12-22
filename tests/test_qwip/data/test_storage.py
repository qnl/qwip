import platform
from io import BytesIO
from pathlib import Path

import httpx
import pytest

import qwip
from qwip.data.storage import HTTPStorageBackend, LocalStorageBackend, StorageBackend
from qwip.flatdict import FlatDict


class TestStorageBackend:
    @pytest.mark.parametrize(
        "folder,address,expect",
        [
            ("", "", ("/", "")),
            ("/", "", ("/", "")),
            ("", "", ("/", "")),
            ("folder", "", ("/folder/", "")),
            ("/folder", "", ("/folder/", "")),
            ("/a", "/a/b/c", ("/a/a/b/", "c")),
            ("/folder", "./file", ("/folder/", "file")),
            ("/folder/subfolder", "../file", ("/folder/", "file")),
            ("", "/folder/file.txt", ("/folder/", "file.txt")),
        ],
    )
    def test_parse_url(self, storage, folder, address, expect):
        assert storage.parse_url(folder, address) == expect

    @pytest.mark.parametrize("address,data", [("text.txt", b"ASCII bytes.")])
    def test_data_round_trip(self, storage, address, data):
        download_address = storage.save_buffer(
            address, BytesIO(data), folder="pytest-roundtrip"
        )
        downloaded = storage.load_buffer(download_address).read()

        assert data == downloaded

        storage.remove_directory("pytest-roundtrip")

    def test_structure_round_trip(self, storage):
        unstructured = qwip.converter.unstructure(storage)
        structured = qwip.converter.structure(unstructured, StorageBackend)

        assert structured == storage
        assert type(structured) == type(storage)


class TestHTTPStorageBackend(TestStorageBackend):
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

    def test_equal(self):
        s1 = HTTPStorageBackend(client=httpx.Client(base_url="https://url"))
        s2 = HTTPStorageBackend(client=httpx.Client(base_url="https://url"))

        assert s1 == s2

    @pytest.mark.skip_dataserver
    @pytest.mark.parametrize(
        "folder,expect",
        [
            ("/non-existent", FileNotFoundError),
            (
                "/",
                ["data-folder", "empty", "README.md"],
            ),
            (
                "/empty",
                [],
            ),
            (
                "/data-folder/",
                ["file.txt", "number.bin"],
            ),
            (
                "/../..",
                ["data-folder", "empty", "README.md"],
            ),
        ],
    )
    def test_list_directory(self, storage, folder, expect, root):
        if isinstance(expect, type) and issubclass(expect, Exception):
            with pytest.raises(expect):
                storage.list_directory(folder)
        else:
            assert set(storage.list_directory(folder)) >= set(expect)

    @pytest.mark.parametrize(
        "folder", ["/pytest", "/pytest/folder1/folder2/folder3", "/", ""]
    )
    def test_make_directory(self, storage, root, folder):
        storage.make_directory(folder)

        if hasattr(storage.client, "app"):
            assert (root / folder.lstrip("./")).exists()

        assert isinstance(storage.list_directory(folder), list)

    def test_delete_folder(self, storage, root):
        folder = "/pytest-delete-folder"
        storage.make_directory(folder + "/a/b/c/")
        storage.make_directory(folder + "/a/b1/c/")
        storage.remove_directory(folder)

        if hasattr(storage.client, "app"):
            assert (root / folder.lstrip("./")).exists() is False

        with pytest.raises(FileNotFoundError):
            storage.list_directory(folder)

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


class TestLocalStorageBackend(TestStorageBackend):
    @pytest.fixture(scope="function")
    def storage(self, tmp_path_factory):
        directory = tmp_path_factory.mktemp("pytest")
        return LocalStorageBackend(directory=directory)

    @pytest.mark.parametrize(
        "folder,expect",
        [
            ("/", "/"),
            ("", "/"),
            ("folder", "/folder"),
            ("/folder1/folder2/", "/folder1/folder2"),
        ],
    )
    def test_make_directory(self, storage, folder, expect):
        address = storage.make_directory(folder)
        assert (storage.directory / folder.lstrip("/")).exists()
        assert address == expect

    def test_make_directory_exists(self, storage):
        with open(storage.directory / "file", "w") as f:
            f.write("")

        with pytest.raises(FileExistsError):
            storage.make_directory("file")

    def test_list_directory(self, storage):
        assert storage.list_directory() == []

        for address in ["folder", "nested/folder"]:
            storage.make_directory(address)

        assert storage.list_directory() == ["folder", "nested"]

    @pytest.mark.parametrize(
        "address,data,expect",
        [
            ("text.txt", b"Binary text.", "/save_buffer/text.txt"),
            (
                "/subfolder/text.txt",
                b"Binary text.",
                "/save_buffer/subfolder/text.txt",
            ),
            ("exists", b"", None),
        ],
    )
    def test_save_buffer(self, storage, address, data, expect):
        (storage.directory / "save_buffer/exists").mkdir(parents=True, exist_ok=True)

        download_address = storage.save_buffer(
            address, BytesIO(data), folder="/save_buffer"
        )
        assert download_address == expect
        if download_address:
            assert (
                storage.directory / "save_buffer" / address.lstrip("./")
            ).read_bytes() == data

    @pytest.mark.parametrize(
        "address,data",
        [
            ("README.md", b"This is the test directory structure"),
            ("data-folder/file.txt", b"A text file with some strings"),
            ("/data-folder/number.bin", (1234567890).to_bytes(8, "big")),
        ],
    )
    def test_load_buffer(self, storage, address, data):
        storage.save_buffer(address, BytesIO(data))
        downloaded = storage.load_buffer(address).read()

        assert downloaded == data

    def test_unstructure(self, storage):
        unstructured = qwip.converter.unstructure(storage)

        assert unstructured == dict(
            directory=str(storage.directory),
            hostname=platform.node(),
            __class__="LocalStorageBackend",
        )
