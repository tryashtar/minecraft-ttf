import dataclasses
import datetime
import io
import json
import pathlib
import typing
import zipfile

import bitarray
import PIL.Image

from minecraft_ttf.bitmap import Bitmap, bitmap_from_image
from minecraft_ttf.minecraft.storage import StackStorage, Storage, zip_time

# TTFs have a modified date field
# we can track this by giving each provider a modified date
# it's the latest modified date of any file it references
# then the TTF gets the latest modified date across all its providers

# associate characters with parts of an image
# used for the modern 'bitmap' provider, as well as any version that uses an image for fonts
@dataclasses.dataclass
class CharImage:
    image: PIL.Image.Image
    bitmap: Bitmap
 
@dataclasses.dataclass
class ImageProvider:
    height: int
    ascent: int
    has_color: bool
    # the portion of the image, already cropped to its final size
    chars: dict[str, CharImage]
    modified_date: datetime.datetime | None

# associate characters with a one-bit-per-pixel bitmap
# used for the modern 'unihex' provider
@dataclasses.dataclass
class BitmapProvider:
    height: int
    ascent: int
    # the bitmap, already cropped to its final size
    chars: dict[str, Bitmap]
    modified_date: datetime.datetime | None

@dataclasses.dataclass
class SpaceProvider:
    spaces: dict[str, float]
    modified_date: datetime.datetime | None

Provider = BitmapProvider | ImageProvider | SpaceProvider

FontFilter = typing.TypedDict('FontFilter', {
    'uniform': typing.NotRequired[bool],
    'jp': typing.NotRequired[bool],
})

JsonBitmapProvider = typing.TypedDict('JsonBitmapProvider', {
    'type': typing.Literal['bitmap'],
    'file': str,
    'height': typing.NotRequired[int],
    'ascent': int,
    'chars': list[str],
    'filter': typing.NotRequired[FontFilter],
})

JsonSpaceProvider = typing.TypedDict('JsonSpaceProvider', {
    'type': typing.Literal['space'],
    'advances': dict[str, float],
    'filter': typing.NotRequired[FontFilter],
})

JsonReferenceProvider = typing.TypedDict('JsonReferenceProvider', {
    'type': typing.Literal['reference'],
    'id': str,
    'filter': typing.NotRequired[FontFilter],
})

JsonLegacyUnicodeProvider = typing.TypedDict('JsonLegacyUnicodeProvider', {
    'type': typing.Literal['legacy_unicode'],
    'sizes': str,
    'template': str,
    'filter': typing.NotRequired[FontFilter],
})

UnihexSizeOverride = typing.TypedDict('UnihexSizeOverride', {
    'from': str,
    'to': str,
    'left': int,
    'right': int,
})

JsonUnihexProvider = typing.TypedDict('JsonUnihexProvider', {
    'type': typing.Literal['unihex'],
    'hex_file': str,
    'size_overrides': typing.NotRequired[list[UnihexSizeOverride]],
    'filter': typing.NotRequired[FontFilter],
})

JsonProvider = JsonBitmapProvider | JsonSpaceProvider | JsonReferenceProvider | JsonLegacyUnicodeProvider | JsonUnihexProvider

JsonRootProvider = typing.TypedDict('JsonRootProvider', {
    'providers': list[JsonProvider]
})

type ImageColorCheck = typing.Callable[[PIL.Image.Image], bool]

# read JSON providers from the game, find their referenced assets, and load them into our providers
def convert_providers(store: Storage, providers: list[JsonProvider], modified_date: datetime.datetime | None, color: ImageColorCheck) -> list[Provider]:
    result: list[Provider] = []
    for provider in providers:
        match provider['type']:
            case 'bitmap':
                img_entry = identifier_to_entry(provider['file'], kind='textures', suffix=None)
                img_data = read_image(store, img_entry)
                full = ImageProvider(
                    height = provider.get('height', 8),
                    ascent = provider['ascent'],
                    has_color = color(img_data.data),
                    chars = image_grid(img_data.data, filter_nul(provider['chars']), None),
                    modified_date = date_max([modified_date, img_data.modified_date])
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
                converted = convert_providers(store, font_data.data, date_max([modified_date, font_data.modified_date]), color)
                result.extend(converted)
            case 'legacy_unicode':
                size_entry = identifier_to_entry(provider['sizes'], kind=None, suffix=None)
                size_date = store.modified_time(size_entry)
                size_bytes = store.read(size_entry)
                sheet_entries = [identifier_to_entry(provider['template'].replace('%s', f'{sheet_id:02x}'), kind='textures', suffix=None) for sheet_id in range(0xff + 1)]
                converted = legacy_unicode(store, sheet_entries, size_bytes, date_max([modified_date, size_date]), color)
                result.extend(converted)
            case 'unihex':
                hex_entry = identifier_to_entry(provider['hex_file'], kind=None, suffix=None)
                dates = [store.modified_time(hex_entry)]
                hex_bytes = store.read(hex_entry)
                hex_list: list[str] = []
                with zipfile.ZipFile(io.BytesIO(hex_bytes), 'r') as zip:
                    for entry in zip.namelist():
                        if entry.endswith('.hex'):
                            stats = zip.getinfo(entry)
                            dates.append(zip_time(stats.date_time))
                            hex_list.extend(zip.read(entry).decode('utf-8').split('\n'))
                chars: dict[str, Bitmap] = {}
                for entry in hex_list:
                    if len(entry) > 0:
                        char, bitmap = unihex(entry)                        
                        left, right = char_size(char, bitmap, provider.get('size_overrides', []))
                        chars[char] = bitmap.resized((left, 0, right, bitmap.height))
                full = BitmapProvider(height=8, ascent=7, chars=chars, modified_date=date_max(dates))
                result.append(full)
    return result

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

def identifier_to_entry(identifier: str, kind: str | None, suffix: str | None) -> pathlib.PurePath:
    if ':' not in identifier:
        namespace = 'minecraft'
        rest = identifier
    else:
        namespace, rest = identifier.split(':',  maxsplit=1)
    path = f'assets/{namespace}'
    if kind is not None:
        path = f'{path}/{kind}'
    path = f'{path}/{rest}'
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

def image_grid(image: PIL.Image.Image, grid: list[list[str | None]], sizes: dict[str, tuple[int, int]] | None) -> dict[str, CharImage]:
    result: dict[str, CharImage] = {}
    glyph_width = image.width // len(grid[0])
    glyph_height = image.height // len(grid)
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char is None:
                continue
            gx1 = x * glyph_width
            gy1 = y * glyph_height
            gx2 = (x + 1) * glyph_width
            gy2 = (y + 1) * glyph_height
            dimensions = (gx1, gy1, gx2, gy2)
            glyph = image.crop(dimensions)
            base_mask = bitmap_from_image(glyph)
            if sizes is not None:
                left, right = sizes[char]
            else:
                left = 0
                box = base_mask.content_box()
                if box is None:
                    right = 0
                else:
                    _left, _top, right, _bottom = box
            resize_box = (left, 0, right, base_mask.height)
            cropped_mask = base_mask.resized(resize_box)
            cropped_glyph = image.crop(resize_box)
            result[char] = CharImage(cropped_glyph, cropped_mask)
    return result

def filter_nul(chars: list[str]) -> list[list[str | None]]:
    return [[None if y == '\u0000' else y for y in x] for x in chars]

def char_size(char: str, bitmap: Bitmap, overrides: list[UnihexSizeOverride]) -> tuple[int, int]:
    for entry in overrides:
        if char >= entry['from'] and char <= entry['to']:
            return (entry['left'], entry['right'] + 1)
    box = bitmap.content_box()
    if box is None:
        return (0, 0)
    left, _top, right, _bottom = box
    return (left, right)

def unihex(line: str) -> tuple[str, Bitmap]:
    char_code, art = line.split(':', maxsplit=1)
    char = chr(int(char_code, 16))
    bits = bytes.fromhex(art)
    if len(art) == 64:
        size = (16, 16)
    else:
        size = (8, 16)
    bitmap = Bitmap(size)
    bitmap.bits = bitarray.bitarray(bits)
    return (char, bitmap)

def legacy_unicode(store: Storage, sheets: list[pathlib.PurePath], size_bytes: bytes, modified_date: datetime.datetime | None, color: ImageColorCheck) -> list[ImageProvider]:
    result: list[ImageProvider] = []
    for sheet_id, sheet_entry in enumerate(sheets):
        if store.exists(sheet_entry):
            char_offset = sheet_id * 256
            chars = [[chr(x) if size_bytes[x] > 0 and x < 0xffff else None for x in range(char_offset + y * 16, char_offset + y * 16 + 16)] for y in range(16)]
            size_range = size_bytes[char_offset:char_offset + 256]
            sizes = {chr(char_offset + i): (x >> 4 & 0xf, x & 0xf + 1) for i, x in enumerate(size_range) if x > 0}
            img_data = read_image(store, sheet_entry)
            full = ImageProvider(
                height = 8,
                ascent = 7,
                has_color = color(img_data.data),
                chars = image_grid(img_data.data, chars, sizes),
                modified_date = date_max([modified_date, img_data.modified_date])
            )
            result.append(full)
    return result
