import io
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any

import httpx
from loguru import logger
from typing_extensions import Self

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
    ) -> str:
        ...

    def save_buffers(
        self, streams: dict[str, io.BufferedReader], folder: str = "/"
    ) -> list[str]:
        for address, stream in streams.items():
            self.save_stream(address, stream, folder=folder)

    @abstractmethod
    def load_buffer(self, address: str) -> SpooledTemporaryFile:
        ...


@register_storage_backend
@qfrozen
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
        url = Path("///" + folder + "///" + address).resolve()
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
        self, address: str, stream: io.BufferedReader, folder: str = "/"
    ) -> str | None:
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

        response.raise_for_status()

        match response.json()["uploads"][0]:
            case {"error": error}:
                logger.warning(f"Unable to save file: {error}")
                return None
            case {"address": download_address, **kwargs}:
                return download_address

    def save_buffers(
        self, streams: dict[str, io.BufferedReader], folder: str = "/"
    ) -> list[str]:
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
        self, address: str, folder: str = "/"
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


@register_storage_backend
@qfrozen
class LocalStorageBackend(StorageBackend):
    directory: Path = Path()

    def save_buffer(
        self, address: str, stream: io.BufferedReader, folder: str = "/"
    ) -> str:
        ...

    def load_buffer(self, address: str, **kwargs) -> SpooledTemporaryFile:
        ...


# ========== httpx.Client converters ========== #


def httpx_client_structure_fn(val, cls):
    if isinstance(val, cls):
        return val

    return httpx.Client(**val)


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
