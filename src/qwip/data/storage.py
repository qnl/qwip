import io
import platform
import shutil
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Self

import httpx
from attrs import field
from loguru import logger

import qwip
from qwip._cattr import make_attrs_structure_fn, make_attrs_unstructure_fn
from qwip.attrs import qfrozen
from qwip.data.serializers import Serializer, get_serializer

REGISTERED_STORAGE_BACKENDS: dict[str, type["StorageBackend"]] = dict()


def register_storage_backend(cls: type["StorageBackend"]) -> type["StorageBackend"]:
    if not issubclass(cls, StorageBackend):
        raise TypeError(
            f"Registered storage backends must subclass {StorageBackend}, got {cls}."
        )

    REGISTERED_STORAGE_BACKENDS[cls.__name__] = cls
    return cls


@qfrozen
class StorageBackend(metaclass=ABCMeta):
    read_buffer: int = 1024 * 1024

    @staticmethod
    def parse_url(folder: str, address: str) -> tuple[str, str]:
        """Normalizes address, filename pairs so the filename contains no slashes.

        Args:
            folder: The beginning part of the address.
            address: A filename to append to address, possibly containing additional
                folders.

        Returns:
            An (folder, address) pair such that all folders in address are added to
            the folder instead.
        """
        # pathlib handles // differently when it is at the beginning of a path
        # but strips extra / otherwise. Using //// to avoid this difference in behavior.
        url = Path("////" + folder + "////" + address).resolve()
        filename = Path(address)

        base, name = (url.parent, url.name) if filename.name else (url, "")
        base = base.relative_to(url.anchor).as_posix().lstrip(".")

        if not base.startswith("/"):
            base = "/" + base
        if not base.endswith("/"):
            base = base + "/"

        return base, name

    def save(
        self,
        address: str,
        obj: Any,
        serializer: str | Serializer = "default",
        **kwargs,
    ):
        """Saves a single object.

        Args:
            address: The address specifying where in the storage backend to save
                the object.
            obj: The object to save. This will be serialized and then saved to the
                storage backend.
            serializer: The serializer to use. If a string is given, a serializer will
                be looked from the list of registered serializers.
            **kwargs: Additional keyword arguments are passed to the serializer.

        Return:
            The location of the saved object, or `None` if it failed to save.
        """
        if not isinstance(serializer, Serializer):
            serializer = get_serializer(key=serializer, obj=obj)

        stream = serializer.to_stream(obj, **kwargs)
        return self.save_buffer(address, stream)

    def load(
        self, address: str, serializer: str | Serializer = "default", **kwargs
    ) -> Any:
        """Loads a single object.

        Args:
            address: The address specifying where in the storage backend to locate
                the object.
            serializer: The serializer to use. If a string is given, a serializer will
                be looked from the list of registered serializers.
            **kwargs: Remaining keyword arguments are passed to the serializer.

        Returns:
            The loaded object.
        """
        if not isinstance(serializer, Serializer):
            serializer = get_serializer(key=serializer)

        stream = self.load_buffer(address)
        return serializer.from_stream(stream, **kwargs)

    @abstractmethod
    def save_buffer(
        self, address: str, stream: io.BufferedReader, folder: str = "/"
    ) -> str: ...

    def save_buffers(
        self, streams: dict[str, io.BufferedReader], folder: str = "/"
    ) -> list[str]:
        for address, stream in streams.items():
            self.save_stream(address, stream, folder=folder)

    @abstractmethod
    def load_buffer(self, address: str) -> SpooledTemporaryFile: ...


@register_storage_backend
@qfrozen
class HTTPStorageBackend(StorageBackend):
    client: httpx.Client = field(eq=lambda client: client.base_url)

    @classmethod
    def from_url(cls, url: str, **kwargs) -> Self:
        client = httpx.Client(base_url=url)
        return cls(client=client, **kwargs)

    def make_directory(self, folder: str = "/"):
        """Create a directory in the storage backend.

        If the specified folder already exists, the state of the dataserver will
        remain unchanged.

        Args:
            folder: The path of the folder to create. All parent directories are also
                created as needed.

        Return:
            The created folder path.
        """
        folder, _ = self.parse_url(folder, "")
        url = "/api/v1/folder" + folder

        response = self.client.post(url)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise FileExistsError() from e

        match response.json():
            case {"path": folder}:
                return folder
            case body:
                raise ValueError(f"Unknown response from server: {body}")

    def remove_directory(self, folder: str = "/"):
        """Remove a directory in the storage backend.

        All subfolders and files in the directory are recursively removed. The root
        folder cannot be deleted.

        Args:
            folder: The path of the folder to delete. All child directories and files
                are also removed.

        Return:
            The path of the deleted folder.
        """
        folder, _ = self.parse_url(folder, "")
        url = "/api/v1/folder" + folder

        response = self.client.delete(url)

        match response.status_code:
            case httpx.codes.OK:
                ...
            case httpx.codes.UNAUTHORIZED:
                raise PermissionError("Cannot remove root folder")
            case _:
                raise FileNotFoundError(f"Folder '{folder}' does not exist")

        match response.json():
            case {"path": folder}:
                return folder
            case body:
                raise ValueError(f"Unknown response from server: {body}")

    def list_directory(self, folder: str = "/"):
        """Get contents of a directory.

        Args:
            folder: The folder to list.

        Returns:
            A list of the folder contents.
        """
        folder, _ = self.parse_url(folder, "")
        url = "/api/v1/folder" + folder

        response = self.client.get(url)

        match response.status_code:
            case httpx.codes.OK:
                ...
            case httpx.codes.BAD_REQUEST:
                raise FileNotFoundError(f"Folder '{folder}' does not exist")
            case _:
                response.raise_for_status()

        return response.json()["contents"]

    def save_buffer(
        self, address: str, stream: io.BufferedReader, folder: str = "/"
    ) -> str | None:
        """Saves the contents of a file-like object to the storage backend.

        Args:
            address: The address of the new file to create.
            folder: A subfolder that the address should be referenced to.

        Returns:
            The address of the file on the storage backend or `None`, if it failed to
            save.
        """
        folder, filename = self.parse_url(folder, address)

        if not filename:
            raise ValueError(f"Filename cannot be empty, got '{filename}'")

        self.make_directory(folder)
        url = "/api/v1/upload" + folder
        files = [("uploads", (filename, stream))]

        response = self.client.post(url, files=files)

        try:
            stream.close()
        except AttributeError:
            ...

        match response.status_code:
            case httpx.codes.OK:
                ...
            case httpx.codes.BAD_REQUEST:
                raise FileNotFoundError(f"Folder '{folder}' is not a valid directory.")
            case _:
                response.raise_for_status()

        match response.json()["uploads"][0]:
            case {"error": error}:
                logger.warning(f"Unable to save file: {error}")
                return None
            case {"address": download_address, **kwargs}:
                return download_address

    def save_buffers(
        self, streams: dict[str, io.BufferedReader], folder: str = "/"
    ) -> list[str | None]:
        """Saves multiple files with a single http request.

        Args:
            streams: A mapping of addresses to file-like objects that can be read.
            folder: A subfolder that the addresses should be referenced to.

        Returns:
            A list of addresses for the files that were uploaded. If a given file
            fails to save, the address will be `None` instead.
        """
        folders = set()
        filenames = []

        for address in streams:
            fd, fn = self.parse_url(folder, address)
            folders.add(fd)
            filenames.append(fn)

        if len(folders) > 1:
            # Must send post requests one at a time.
            return super().save_buffers(streams, folder=folder)

        url = "/api/v1/upload" + next(iter(folders))

        files = [
            ("uploads", (filename, stream))
            for filename, stream in zip(filenames, streams.values())
        ]

        response = self.client.post(url, files=files)

        for stream in streams.values():
            try:
                stream.close()
            except AttributeError:
                ...

        match response.status_code:
            case httpx.codes.OK:
                ...
            case httpx.codes.BAD_REQUEST:
                raise FileNotFoundError(f"Folder '{folder}' is not a valid directory.")
            case _:
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

    def load_buffer(self, address: str, folder: str = "/") -> SpooledTemporaryFile:
        """Loads the contents of a file from the storage backend.

        The file contents are loaded to a `SpooledTemporaryFile` so that large files
        are not completely read into memory. The file is deleted from disk and memory
        once it is closed.

        Args:
            address: The address of the file to load.
            folder: A subfolder that the address should be referenced to.

        Returns:
            A temporary file that can be read from.
        """
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


@register_storage_backend
@qfrozen
class LocalStorageBackend(StorageBackend):
    hostname: str = field(default=platform.node(), metadata=dict(serialize=True))
    directory: Path = field(factory=lambda: Path(".").absolute())

    def make_directory(self, folder: str = "/"):
        """Create a directory in the storage backend.

        If the specified folder already exists, the state of the dataserver will
        remain unchanged.

        Args:
            folder: The path of the folder to create. All parent directories are also
                created as needed.

        Return:
            The created folder path.
        """

        folder, _ = self.parse_url(folder, "")
        new_directory = self.directory / folder.lstrip("/")

        try:
            new_directory.mkdir(parents=True, exist_ok=True)
            if folder != "/":
                folder = folder.rstrip("/")
            return folder
        except Exception as e:
            logger.exception(
                f"Error creating directory '{new_directory.as_posix()}'", exception=e
            )
            raise e

    def remove_directory(self, folder: str = "/"):
        folder, _ = self.parse_url(folder, "")

        directory = self.directory / folder.lstrip("/")

        if directory == self.directory:
            raise PermissionError("Cannot remove root directory")

        if not directory.exists():
            raise FileNotFoundError(f"Folder '{folder}' does not exist.")

        try:
            shutil.rmtree(directory)
        except Exception as e:
            logger.exception(
                f"Error removing directory '{directory.as_posix()}'", exception=e
            )

        return "/" + directory.relative_to(self.directory).as_posix()

    def list_directory(self, folder: str = "/"):
        """Get contents of a directory.

        Args:
            folder: The folder to list.

        Returns:
            A list of the folder contents.
        """
        folder, _ = self.parse_url(folder, "")
        directory = self.directory / folder.lstrip("/")

        contents = [
            p.relative_to(self.directory).as_posix() for p in directory.iterdir()
        ]

        return contents

    def save_buffer(
        self, address: str, stream: io.BufferedReader, folder: str = "/"
    ) -> str | None:
        folder, filename = self.parse_url(folder, address)

        if not filename:
            raise ValueError(f"Filename cannot be empty, got '{filename}'")

        self.make_directory(folder)
        full_path = self.directory / folder.lstrip("/") / filename

        try:
            with open(full_path, "wb") as dest:
                shutil.copyfileobj(stream, dest)

            address = folder + filename
        except Exception:
            address = None
        finally:
            stream.close()

        return address

    def load_buffer(self, address: str, folder: str = "/") -> SpooledTemporaryFile:
        folder, filename = self.parse_url(folder, address)

        full_path = self.directory / folder.lstrip("/") / filename

        return open(full_path, "rb")


# ========== httpx.Client converters ========== #

CLIENT_CACHE = {}


def httpx_client_structure_fn(val, cls):
    if isinstance(val, cls):
        return val

    key = tuple(val.items())
    if key in CLIENT_CACHE:
        return CLIENT_CACHE[key]

    client = CLIENT_CACHE[key] = httpx.Client(**val)
    return client


def httpx_client_unstructure_fn(obj):
    return dict(base_url=str(obj.base_url))


qwip.converter.register_structure_hook(httpx.Client, httpx_client_structure_fn)
qwip.converter.register_unstructure_hook(httpx.Client, httpx_client_unstructure_fn)

# ========== StorageBackend converters ========== #


def make_storage_backend_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)

    def structure_fn(val, cls):
        if isinstance(val, cls):
            return val

        subclass = REGISTERED_STORAGE_BACKENDS.get(
            val.get("__class__"),
        )

        if subclass is None:
            logger.warning(f"No registered backend found. Structuring {val} as {cls}.")
            return structure_attrs(val, cls)

        return qwip.converter.structure(val, subclass)

    return structure_fn


def make_storage_backend_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), "__class__": type(obj).__name__}

    return unstructure_fn


qwip.converter.register_structure_hook_factory(
    lambda cls: cls is StorageBackend, make_storage_backend_structure_fn
)

qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, StorageBackend), make_storage_backend_unstructure_fn
)

__all__ = ["HTTPStorageBackend", "LocalStorageBackend", "StorageBackend"]
