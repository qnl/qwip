import io
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any

import httpx
from loguru import logger
from typing_extensions import Self

from qwip.attrs import qdefine


class StorageBackend(metaclass=ABCMeta):
    __slots__ = ()

    def save(self, address: str, asset: Any, **kwargs):
        ...

    def load(self, address: str, cls: type, **kwargs):
        ...

    @abstractmethod
    def save_buffer(
        self, address: str, stream: io.BufferedReader, folder: str = "/", **kwargs
    ) -> str:
        ...

    def save_buffers(
        self, streams: dict[str, io.BufferedReader], folder: str = "/", **kwargs
    ) -> list[str]:
        for address, stream in streams.items():
            self.save_stream(address, stream, folder=folder, **kwargs)

    @abstractmethod
    def load_buffer(self, address: str, **kwargs) -> SpooledTemporaryFile:
        ...


@qdefine
class HTTPStorageBackend(StorageBackend):
    client: httpx.Client
    read_buffer: int = 1024 * 1024

    @classmethod
    def from_url(cls, url: str, **kwargs) -> Self:
        client = httpx.Client(base_url=url)
        return cls(client=client, **kwargs)

    @staticmethod
    def parse_url(folder: str, address: str) -> tuple[str, str]:
        """Normalizes address, filename pairs so the filename contains no slashes.

        Args:
            folder: The beginning part of the address.
            address: A filename to append to address, possibly containing additional
                folders.

        Returns:
            An (folder, address) pair such that all folders in filename are added to
            the address instead.
        """
        # pathlib handles // differently when it is at the beginning of a path
        # but strips extra / otherwise. Using /// to avoid this difference in behavior.
        url = Path(folder + "///" + address).resolve()
        filename = Path(address)

        base, name = (url.parent, url.name) if filename.name else (url, "")
        base = base.relative_to(url.anchor).as_posix().lstrip(".")

        if not base.startswith("/"):
            base = "/" + base
        if not base.endswith("/"):
            base = base + "/"

        return base, name

    def make_directory(self, folder: str = "/"):
        folder, _ = self.parse_url(folder, "")
        url = "/api/v1/folder" + folder

        response = self.client.post(url)
        response.raise_for_status()

        match response.json():
            case {"path": folder}:
                return folder
            case body:
                raise ValueError(f"Unknown response from server: {body}")

    def list_directory(self, folder: str = "/"):
        folder, _ = self.parse_url(folder, "")
        url = "/api/v1/folder" + folder

        response = self.client.get(url)
        response.raise_for_status()

        return response.json()

    def save_buffer(
        self, address: str, stream: io.BufferedReader, folder: str = "/", **kwargs
    ) -> str | None:
        folder, filename = self.parse_url(folder, address)

        if not filename:
            raise ValueError(f"Filename cannot be empty, got '{filename}'")

        self.make_directory(folder)
        url = "/api/v1/upload" + folder
        files = [("uploads", (filename, stream))]

        response = self.client.post(url, files=files)
        response.raise_for_status()

        match response.json()["uploads"][0]:
            case {"error": error}:
                logger.warning(f"Unable to save file: {error}")
                return None
            case {"address": download_address, **kwargs}:
                return download_address

    def save_buffers(
        self, streams: dict[str, io.BufferedReader], folder: str = "/", **kwargs
    ) -> list[str]:
        folders = set()
        filenames = []

        for address in streams:
            fd, fn = self.parse_url(folder, address)
            folders.add(fd)
            filenames.append(fn)

        if len(folders) > 1:
            # Must send post requests one at a time.
            return super().save_buffers(streams, folder=folder, **kwargs)

        url = "/api/v1/upload" + next(iter(folders))

        files = [
            ("uploads", (filename, stream))
            for filename, stream in zip(filenames, streams.values())
        ]

        response = self.client.post(url, files=files)

        response.raise_for_status()
        download_addresses = []

        for upload in response.json()["uploads"]:
            match upload:
                case {"error": error}:
                    logger.warning(f"Unable to save file: {error}")
                    download_addresses.append(None)
                case {"address": address, **kwargs}:
                    download_addresses.append(address)
        return download_addresses

    def load_buffer(
        self, address: str, folder: str = "/", **kwargs
    ) -> SpooledTemporaryFile:
        folder, filename = self.parse_url(folder, address)

        if not folder.startswith("/static"):
            folder = "/static" + folder
        url = folder + filename

        with self.client.stream("GET", url) as response:
            response.raise_for_status()

            tmp = SpooledTemporaryFile(max_size=self.read_buffer)
            for chunk in response.iter_bytes():
                tmp.write(chunk)

            tmp.seek(0)

        return tmp

    def close(self):
        self.client.close()


@qdefine
class LocalStorageBackend(StorageBackend):
    directory: Path = Path()

    def save_buffer(
        self, address: str, stream: io.BufferedReader, folder: str = "/", **kwargs
    ) -> str:
        ...

    def load_buffer(self, address: str, **kwargs) -> SpooledTemporaryFile:
        ...
