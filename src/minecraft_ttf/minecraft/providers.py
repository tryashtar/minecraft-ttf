import dataclasses
import datetime
import io
import json
import pathlib
import typing

import PIL.Image

from minecraft_ttf.minecraft.storage import StackStorage, Storage
from minecraft_ttf.minecraft.versions import MinecraftVersion


@dataclasses.dataclass
class BitmapProvider:
    height: int
    ascent: int
    image: PIL.Image.Image
    chars: list[str]
    modified_date: datetime.datetime | None

@dataclasses.dataclass
class SpaceProvider:
    spaces: dict[str, int]
    modified_date: datetime.datetime | None

Provider = BitmapProvider | SpaceProvider

def get_providers(store: Storage, version: MinecraftVersion, identifier: str) -> list[Provider] | None:
    if version.supports_providers:
        entry = identifier_to_entry(identifier, 'font', 'json')
        if not store.exists(entry):
            return None
        providers = load_providers(store, entry)
    else:
        providers = []
        assert version.entry_map is not None
        if identifier not in version.entry_map:
            return None
        img_entry = version.entry_map[identifier]
        img_data = read_image(store, img_entry)
        if version.hardcoded_chars is not None:
            chars = version.hardcoded_chars
            date = img_data.modified_date
        else:
            assert version.lookup_chars
            font_data = read_font_txt(store, pathlib.PurePath('font.txt'))
            empty = '\u0000' * 16
            chars: list[str] = [
                empty,
                empty,
                *font_data.data,
                empty,
                empty,
                empty,
                empty,
                empty
            ]
            date = date_max([img_data.modified_date, font_data.modified_date])
        bitmap = BitmapProvider(height=8, ascent=7, image=img_data.data, chars=chars, sizes=None, modified_date=date)
        providers.append(bitmap)
    if version.hardcoded_spaces is not None:
        providers.insert(0, SpaceProvider(version.hardcoded_spaces, modified_date=None))
    return providers

JsonBitmapProvider = typing.TypedDict('JsonBitmapProvider', {
    'type': typing.Literal['bitmap'],
    'file': str,
    'height': typing.NotRequired[int],
    'ascent': int,
    'chars': list[str],
})

JsonSpaceProvider = typing.TypedDict('JsonSpaceProvider', {
    'type': typing.Literal['space'],
    'advances': dict[str, int],
})

JsonReferenceProvider = typing.TypedDict('JsonReferenceProvider', {
    'type': typing.Literal['reference'],
    'id': str,
})

JsonProvider = JsonBitmapProvider | JsonSpaceProvider | JsonReferenceProvider

def read_image(store: Storage, entry: pathlib.PurePath) -> tuple[PIL.Image.Image, datetime.datetime]:
@dataclasses.dataclass
class ReadEntry[T]:
    data: T
    modified_date: datetime.datetime | None

def read_image(store: Storage, entry: pathlib.PurePath) -> ReadEntry[PIL.Image.Image]:
    data = store.read(entry)
    img = PIL.Image.open(io.BytesIO(data))
    return ReadEntry(img, store.modified_time(entry))

def read_json(store: Storage, entry: pathlib.PurePath) -> ReadEntry:
    text = store.read(entry)
    data = json.loads(text)
    return ReadEntry(data, store.modified_time(entry))

def read_font_definition(store: Storage, entry: pathlib.PurePath) -> ReadEntry[list[JsonProvider]]:
    if not isinstance(store, StackStorage):
        data: ReadEntry[JsonRootProvider] = read_json(store, entry)
        return ReadEntry(data.data['providers'], data.modified_date)
    result: list[JsonProvider] = []
    members: list[ReadEntry[JsonRootProvider]] = [read_json(x, entry) for x in store.stack if x.exists(entry)]
    date = date_max([x.modified_date for x in members])
    for member in members:
        result.extend(member.data['providers'])
    return ReadEntry(result, date)

def read_font_txt(store: Storage, entry: pathlib.PurePath) -> ReadEntry[list[str]]:
    text = store.read(entry).decode('utf-8')
    lines: list[str] = [x for x in text.split('\n') if not x.startswith('#')]
    return ReadEntry(lines, store.modified_time(entry))

def identifier_to_entry(identifier: str, kind: str, suffix: str | None) -> pathlib.PurePath:
    if ':' not in identifier:
        namespace = 'minecraft'
        rest = identifier
    else:
        namespace, rest = identifier.split(':',  maxsplit=1)
    path = f'assets/{namespace}/{kind}/{rest}'
    if suffix is not None:
        path += f'.{suffix}'
    return pathlib.PurePath(path)

def date_max(dates: list[datetime.datetime | None]) -> datetime.datetime | None:
    result: datetime.datetime | None = None
    for entry in dates:
        if entry is not None:
            if result is None:
                result = entry
            else:
                result = max(result, entry)
    return result

def convert_providers(store: Storage, providers: list[JsonProvider], modified_date: datetime.datetime | None) -> list[Provider]:
   result: list[Provider] = []
   for provider in providers:
       match provider['type']:
           case 'bitmap':
               img_entry = identifier_to_entry(provider['file'], kind='textures', suffix=None)
               img_data = read_image(store, img_entry)
               full = BitmapProvider(
                   height=provider.get('height', 8),
                   ascent=provider['ascent'],
                   image=img_data.data,
                   chars=provider['chars'],
                   modified_date=max(modified_date, img_date)
                   modified_date=date_max([modified_date, img_data.modified_date])
               )
               result.append(full)
           case 'space':
               full = SpaceProvider(
                   spaces=provider['advances'],
                   modified_date=modified_date
               )
               result.append(full)
           case 'reference':
               entry = identifier_to_entry(provider['id'], kind='font', suffix='json')
               font_data = read_font_definition(store, entry)
               converted = convert_providers(store, font_data.data, date_max([modified_date, font_data.modified_date]))
               result.extend(converted)
   return result

def load_providers(store: Storage, entry: pathlib.PurePath) -> list[Provider]:
    font_data = read_font_definition(store, entry)
    return convert_providers(store, font_data.data, font_data.modified_date)
