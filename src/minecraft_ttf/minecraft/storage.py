import abc
import datetime
import pathlib
import typing
import zipfile

import requests


class Storage(abc.ABC):
    @abc.abstractmethod
    def get_entries(self, prefix: pathlib.PurePath) -> list[pathlib.PurePath]: pass
    @abc.abstractmethod
    def read(self, entry: pathlib.PurePath) -> bytes: pass
    @abc.abstractmethod
    def exists(self, entry: pathlib.PurePath) -> bool: pass
    @abc.abstractmethod
    def modified_time(self, entry: pathlib.PurePath) -> datetime.datetime | None: pass

class FilesystemStorage(Storage):
    def __init__(self, root: pathlib.Path):
        self.root = root
    
    @typing.override
    def get_entries(self, prefix: pathlib.PurePath) -> list[pathlib.PurePath]:
        result: list[pathlib.PurePath] = []
        start = self.root / prefix
        for root, _dirs, files in start.walk():
            rel = root.relative_to(start)
            result.extend([prefix / rel / x for x in files])
        return result

    @typing.override
    def read(self, entry: pathlib.PurePath) -> bytes:
        return (self.root / entry).read_bytes()

    @typing.override
    def exists(self, entry: pathlib.PurePath) -> bool:
        return (self.root / entry).exists()

    @typing.override
    def modified_time(self, entry: pathlib.PurePath) -> datetime.datetime:
        stats = (self.root / entry).stat()
        time = datetime.datetime.fromtimestamp(stats.st_mtime, tz=datetime.UTC)
        return time

class ZipStorage(Storage):
    def __init__(self, zip: zipfile.ZipFile):
        self.zip = zip
        self.names = zip.namelist()

    @typing.override
    def get_entries(self, prefix: pathlib.PurePath) -> list[pathlib.PurePath]:
        result: list[pathlib.PurePath] = []
        prefix_str = prefix.as_posix() + '/'
        for entry in self.names:
            if entry.startswith(prefix_str):
                result.append(pathlib.PurePath(entry))
        return result

    @typing.override
    def read(self, entry: pathlib.PurePath) -> bytes:
        return self.zip.read(entry.as_posix())

    @typing.override
    def exists(self, entry: pathlib.PurePath) -> bool:
        return entry.as_posix() in self.names

    @typing.override
    def modified_time(self, entry: pathlib.PurePath) -> datetime.datetime:
        stats = self.zip.getinfo(entry.as_posix())
        time = zip_time(stats.date_time)
        return time

def zip_time(jartime: tuple) -> datetime.datetime:
     y, m, d, h, mm, s = jartime
     return datetime.datetime(y, m, d, h, mm, s, 0, tzinfo=datetime.UTC)

class StackStorage(Storage):
    def __init__(self, stack: list[Storage]):
        self.stack = stack

    @typing.override
    def get_entries(self, prefix: pathlib.PurePath) -> list[pathlib.PurePath]:
        result: list[pathlib.PurePath] = []
        for member in self.stack:
            sub_results = member.get_entries(prefix)
            result.extend(sub_results)
        unique = list(dict.fromkeys(result))
        return unique

    @typing.override
    def read(self, entry: pathlib.PurePath) -> bytes:
        for member in self.stack:
            if member.exists(entry):
                return member.read(entry)
        raise ValueError(entry)

    @typing.override
    def exists(self, entry: pathlib.PurePath) -> bool:
        return any(x.exists(entry) for x in self.stack)

    @typing.override
    def modified_time(self, entry: pathlib.PurePath) -> datetime.datetime | None:
        for member in self.stack:
            if member.exists(entry):
                return member.modified_time(entry)
        raise ValueError(entry)

def get_storage(location: pathlib.Path) -> Storage | None:
    if location.is_dir():
        return FilesystemStorage(location)
    if location.is_file():
        zip = zipfile.ZipFile(location, 'r')
        return ZipStorage(zip)
    return None

AssetEntry = typing.TypedDict('AssetEntry', {
    'hash': str
})

AssetSource = typing.TypedDict('AssetSource', {
    'objects': dict[str, AssetEntry],
})

class AssetStorage(Storage):
    def __init__(self, server: str, prefix: pathlib.PurePath, source: AssetSource, cache: pathlib.Path):
        self.server = server
        self.prefix = prefix
        self.source = source
        self.cache = cache

    @typing.override
    def get_entries(self, prefix: pathlib.PurePath) -> list[pathlib.PurePath]:
        modified = prefix.relative_to(self.prefix).as_posix()
        return [pathlib.PurePath(x) for x in self.source['objects'] if x.startswith(modified)]
          
    @typing.override
    def read(self, entry: pathlib.PurePath) -> bytes:
        modified = entry.relative_to(self.prefix).as_posix()
        hash = self.source['objects'][modified]['hash']
        cached = self.cache / hash
        if cached.exists():
            return cached.read_bytes()
        url = f'{self.server}/{hash[:2]}/{hash}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.content
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        return data

    @typing.override
    def exists(self, entry: pathlib.PurePath) -> bool:
        modified = entry.relative_to(self.prefix).as_posix()
        return modified in self.source['objects']
       
    @typing.override
    def modified_time(self, entry: pathlib.PurePath) -> None:
        return None
