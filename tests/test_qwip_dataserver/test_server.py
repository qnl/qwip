from pathlib import Path

import pytest
from fastapi import status


@pytest.fixture
def txt_file(tmp_path):
    filename = tmp_path / "file.txt"
    with open(filename, "w") as f:
        f.write("Quantum Nanoelectronics Laboratory")
        f.flush()
        yield filename


def test_static(client, txt_file):
    url = ("/static" / Path(*txt_file.parts[-2:])).as_posix()
    r = client.get(url)

    expected = txt_file.read_bytes()

    assert r.content == expected


@pytest.mark.parametrize(
    "files",
    [
        [],
        ["file1.txt", "file2.cfg", "file3.png"],
    ],
)
def test_list_directory(client, tmp_path, files):
    for f in files:
        (tmp_path / f).write_text("data")

    url = "/api/v1/folder/" + tmp_path.name
    r = client.get(url)

    assert r.status_code == status.HTTP_200_OK
    assert r.json() == {
        "path": "/" + tmp_path.name,
        "num_children": len(files),
        "contents": files,
    }


def test_list_directory_does_not_exist(client, tmp_path):
    url = f"/api/v1/folder/{tmp_path.name}/does_not_exist/"
    r = client.get(url)

    assert r.status_code == status.HTTP_400_BAD_REQUEST
    assert r.json() == {"detail": "Invalid directory."}


def test_list_directory_file(client, tmp_path):
    (tmp_path / "not_a_directory").write_text("I'm a file not a directory!")

    url = f"/api/v1/folder/{tmp_path.name}/not_a_directory"
    r = client.get(url)

    assert r.status_code == status.HTTP_400_BAD_REQUEST
    assert r.json() == {"detail": "Invalid directory."}


def test_make_directory(client, tmp_path):
    url = f"/api/v1/folder/{tmp_path.name}/new_folder"
    r = client.post(url)

    assert r.status_code == status.HTTP_200_OK
    assert r.json() == {"path": f"/{tmp_path.name}/new_folder"}


def test_make_directory_file_exists(client, tmp_path):
    (tmp_path / "file").write_text("I'm a file not a directory!")

    url = f"/api/v1/folder/{tmp_path.name}/file"
    r = client.post(url)
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    assert r.json() == {"detail": "Cannot create directory."}


def test_remove_directory(client, tmp_path):
    (tmp_path / "subdirectory").mkdir()
    (tmp_path / "subdirectory/file.txt").write_text("Some text in a file.")
    (tmp_path / "subdirectory/file1.txt").write_text("Some more text in a file.")

    url = f"/api/v1/folder/{tmp_path.name}/subdirectory"
    r = client.delete(url)
    assert r.status_code == status.HTTP_200_OK
    assert r.json() == {"path": f"/{tmp_path.name}/subdirectory"}
    assert (tmp_path / "subdirectory").exists() is False


def test_remove_file(client, tmp_path):
    (tmp_path / "file.txt").write_text("Some text in a file.")

    url = f"/api/v1/folder/{tmp_path.name}/file.txt"
    r = client.delete(url)
    assert r.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.parametrize(
    "url", ["/api/v1/folder/", "/api/v1/folder//", "/api/v1/folder/\\"]
)
def test_remove_directory_root(client, url):
    r = client.delete(url)

    assert r.status_code == status.HTTP_400_BAD_REQUEST
    assert r.json() == {"detail": "Cannot remove root directory."}


def test_remove_directory_nonexistent(client, tmp_path):
    url = f"/api/v1/folder/{tmp_path.name}/not_a_directory"

    r = client.delete(url)
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    assert r.json() == {"detail": "Invalid directory."}


@pytest.mark.parametrize(
    "uploads",
    [
        [("empty.txt", b"")],
        [("text.txt", b"Some text here.")],
        [("number.bin", (1234567890).to_bytes(8, "big"))],
        [
            ("file1.txt", b"Chapter 1"),
            ("file2.txt", b"Chapter 2"),
            ("file3.txt", b"Chapter 3"),
        ],
    ],
)
def test_upload_files(client, tmp_path, uploads):
    url = f"/api/v1/upload/{tmp_path.name}"
    r = client.post(url, files=[("uploads", f) for f in uploads])

    paths = []
    for filename, contents in uploads:
        paths.append(
            dict(address=f"/static/{tmp_path.name}/{filename}", size=len(contents))
        )
        assert (tmp_path / filename).read_bytes() == contents

    assert r.status_code == status.HTTP_200_OK
    assert r.json() == {"uploads": paths}


@pytest.mark.parametrize(
    "uploads",
    [
        [("empty.txt", b"")],
        [("text.txt", b"Some text here.")],
        [("number.bin", (1234567890).to_bytes(8, "big"))],
        [
            ("file1.txt", b"Chapter 1"),
            ("file2.txt", b"Chapter 2"),
            ("file3.txt", b"Chapter 3"),
        ],
    ],
)
def test_upload_round_trip(client, tmp_path, uploads):
    url = f"/api/v1/upload/{tmp_path.name}"
    r = client.post(url, files=[("uploads", f) for f in uploads])

    for upload, (_, content) in zip(r.json()["uploads"], uploads):
        match upload:
            case {"address": address, **kwargs}:
                r = client.get(address)
                assert r.content == content
