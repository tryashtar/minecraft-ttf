import dataclasses
import datetime
import io
import json
import pathlib
import typing

import PIL.Image

import storage
import versions


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

def get_providers(store: storage.Storage, version: versions.MinecraftVersion, identifier: str) -> list[Provider] | None:
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
        img, img_date = read_image(store, img_entry)
        if version.hardcoded_chars is not None:
            chars = version.hardcoded_chars
            date = img_date
        else:
            assert version.lookup_chars
            font, font_date = read_font_txt(store, pathlib.PurePath('font.txt'))
            empty = '\u0000' * 16
            chars: list[str] = [
                empty,
                empty,
                *font,
                empty,
                empty,
                empty,
                empty,
                empty
            ]
            date = max(img_date, font_date)
        bitmap = BitmapProvider(height=8, ascent=7, image=img, chars=chars, modified_date=date)
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

def read_image(store: storage.Storage, entry: pathlib.PurePath) -> tuple[PIL.Image.Image, datetime.datetime]:
    data = store.read(entry)
    img = PIL.Image.open(io.BytesIO(data))
    return (img, store.modified_time(entry))

def read_json(store: storage.Storage, entry: pathlib.PurePath) -> tuple[dict[str, typing.Any], datetime.datetime]:
    text = store.read(entry)
    data = json.loads(text)
    return (data, store.modified_time(entry))

def read_font_txt(store: storage.Storage, entry: pathlib.PurePath) -> tuple[list[str], datetime.datetime]:
    text = store.read(entry).decode('utf-8')
    lines: list[str] = [x for x in text.split('\n') if not x.startswith('#')]
    return (lines, store.modified_time(entry))

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

def convert_providers(store: storage.Storage, providers: list[JsonProvider], modified_date: datetime.datetime) -> list[Provider]:
   result: list[Provider] = []
   for provider in providers:
       match provider['type']:
           case 'bitmap':
               img_entry = identifier_to_entry(provider['file'], kind='textures', suffix=None)
               img, img_date = read_image(store, img_entry)
               full = BitmapProvider(
                   height=provider.get('height', 8),
                   ascent=provider['ascent'],
                   image=img,
                   chars=provider['chars'],
                   modified_date=max(modified_date, img_date)
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
               data, date = read_json(store, entry)
               entries: list[JsonProvider] = data['providers']
               converted = convert_providers(store, entries, max(modified_date, date))
               result.extend(converted)
   return result

def load_providers(store: storage.Storage, entry: pathlib.PurePath) -> list[Provider]:
    data, date = read_json(store, entry)
    entries: list[JsonProvider] = data['providers']
    return convert_providers(store, entries, date)
