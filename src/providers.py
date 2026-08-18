import dataclasses
import datetime
import io
import json
import typing
import zipfile

import PIL.Image

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

def get_providers(jar: zipfile.ZipFile, version: versions.MinecraftVersion, identifier: str) -> list[Provider] | None:
    names = jar.namelist()
    if version.supports_providers:
        entry = identifier_to_entry(identifier, 'font', 'json')
        if entry not in names:
            return None
        providers = load_providers(jar, entry)
    else:
        providers = []
        assert version.entry_map is not None
        if identifier not in version.entry_map:
            return None
        img_entry = version.entry_map[identifier]
        img, img_date = read_image(jar, img_entry)
        if version.hardcoded_chars is not None:
            chars = version.hardcoded_chars
            date = img_date
        else:
            assert version.lookup_chars
            font, font_date = read_font_txt(jar, 'font.txt')
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

def date_time(jartime: tuple) -> datetime.datetime:
    y, m, d, h, mm, s = jartime
    return datetime.datetime(y, m, d, h, mm, s, 0, tzinfo=datetime.UTC)

def read_image(jar: zipfile.ZipFile, entry: str) -> tuple[PIL.Image.Image, datetime.datetime]:
    data = jar.read(entry)
    img = PIL.Image.open(io.BytesIO(data))
    date = jar.getinfo(entry).date_time
    return (img, date_time(date))

def read_json(jar: zipfile.ZipFile, entry: str) -> tuple[dict[str, typing.Any], datetime.datetime]:
    text = jar.read(entry)
    data = json.loads(text)
    date = jar.getinfo(entry).date_time
    return (data, date_time(date))

def read_font_txt(jar: zipfile.ZipFile, entry: str) -> tuple[list[str], datetime.datetime]:
    text = jar.read(entry).decode('utf-8')
    lines: list[str] = [x for x in text.split('\n') if not x.startswith('#')]
    date = jar.getinfo(entry).date_time
    return (lines, date_time(date))

def identifier_to_entry(identifier: str, kind: str, suffix: str | None) -> str:
    if ':' not in identifier:
        namespace = 'minecraft'
        rest = identifier
    else:
        namespace, rest = identifier.split(':',  maxsplit=1)
    path = f'assets/{namespace}/{kind}/{rest}'
    if suffix is not None:
        path += f'.{suffix}'
    return path

def convert_providers(jar: zipfile.ZipFile, providers: list[JsonProvider], modified_date: datetime.datetime) -> list[Provider]:
   result: list[Provider] = []
   for provider in providers:
       match provider['type']:
           case 'bitmap':
               img_entry = identifier_to_entry(provider['file'], kind='textures', suffix=None)
               img, img_date = read_image(jar, img_entry)
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
               data, date = read_json(jar, entry)
               entries: list[JsonProvider] = data['providers']
               converted = convert_providers(jar, entries, max(modified_date, date))
               result.extend(converted)
   return result

def load_providers(jar: zipfile.ZipFile, entry: str) -> list[Provider]:
    data, date = read_json(jar, entry)
    entries: list[JsonProvider] = data['providers']
    return convert_providers(jar, entries, date)
