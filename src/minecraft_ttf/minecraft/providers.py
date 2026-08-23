import dataclasses
import datetime
import enum
import io
import pathlib
import typing
import zipfile

import bitarray
import json5
import PIL.Image

from minecraft_ttf.bitmap import Bitmap, bitmap_from_image
from minecraft_ttf.minecraft.storage import StackStorage, Storage, zip_time


# TTFs have a modified date field
# we can track the latest modified date of all referenced files
# then the TTF gets the latest modified date across all its providers
class ModifiedTimes:
    oldest: datetime.datetime | None
    newest: datetime.datetime | None

    def __init__(self):
        self.oldest = None
        self.newest = None

    def update(self, time: datetime.datetime | None):
        if time is not None:
            if self.newest is None or time > self.newest:
                self.newest = time
            if self.oldest is None or time < self.oldest:
                self.oldest = time

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

# associate characters with a one-bit-per-pixel bitmap
# used for the modern 'unihex' provider
@dataclasses.dataclass
class BitmapProvider:
    height: int
    ascent: int
    # the bitmap, already cropped to its final size
    chars: dict[str, Bitmap]

@dataclasses.dataclass
class SpaceProvider:
    spaces: dict[str, float]

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

ResolvedJsonProvider = JsonBitmapProvider | JsonSpaceProvider | JsonLegacyUnicodeProvider | JsonUnihexProvider

JsonProvider = ResolvedJsonProvider | JsonReferenceProvider

JsonRootProvider = typing.TypedDict('JsonRootProvider', {
    'providers': list[JsonProvider]
})

class ProviderSupport(enum.Enum):
    NONE = 0
    ONLY_UNICODE_WHEN_FORCED = 1
    SWITCH_FONT_WHEN_FORCED = 2
    FULL = 3

@dataclasses.dataclass
class ProviderOptions:
    image_color_predicate: typing.Callable[[PIL.Image.Image], bool]
    all_char_predicate: typing.Callable[[str], bool]
    unifont_char_predicate: typing.Callable[[str], bool]
    option_uniform: bool
    option_jp: bool

def load_providers(identifier: str, store: Storage, options: ProviderOptions, support: ProviderSupport, times: ModifiedTimes) -> list[Provider] | None:
    filtered = get_filtered_providers(identifier, store, options, support)
    if filtered is None:
        return None
    result: list[Provider] = []
    for provider in filtered:
        times.update(provider.modified_date)
        loaded = convert_provider(store, provider.data, options, times)
        result.extend(loaded)
    return result

# read JSON providers from the game, find their referenced assets, and load them into our providers
def convert_provider(store: Storage, provider: JsonProvider, options: ProviderOptions, times: ModifiedTimes) -> list[Provider]:
    assert provider['type'] != 'reference'
    match provider['type']:
        case 'bitmap':
            img_entry = identifier_to_entry(provider['file'], kind='textures', suffix=None)
            img_data = read_image(store, img_entry)
            times.update(img_data.modified_date)
            full = ImageProvider(
                height = provider.get('height', 8),
                ascent = provider['ascent'],
                has_color = options.image_color_predicate(img_data.data),
                chars = image_grid(img_data.data, filter_nul(provider['chars']), None, options.all_char_predicate),
            )
            return [full]
        case 'space':
            full = SpaceProvider(
                spaces=provider['advances'],
            )
            return [full]
        case 'legacy_unicode':
            size_entry = identifier_to_entry(provider['sizes'], kind=None, suffix=None)
            size_date = store.modified_time(size_entry)
            times.update(size_date)
            size_bytes = store.read(size_entry)
            sheet_entries = [identifier_to_entry(provider['template'].replace('%s', f'{sheet_id:02x}'), kind='textures', suffix=None) for sheet_id in range(0xff + 1)]
            converted = legacy_unicode(store, sheet_entries, size_bytes, options, times)
            return [x for x in converted]
        case 'unihex':
            hex_entry = identifier_to_entry(provider['hex_file'], kind=None, suffix=None)
            hex_date = store.modified_time(hex_entry)
            times.update(hex_date)
            hex_bytes = store.read(hex_entry)
            hex_list: list[str] = []
            with zipfile.ZipFile(io.BytesIO(hex_bytes), 'r') as zip:
                for entry in zip.namelist():
                    if entry.endswith('.hex'):
                        stats = zip.getinfo(entry)
                        times.update(zip_time(stats.date_time))
                        hex_list.extend(zip.read(entry).decode('utf-8').splitlines())
            chars: dict[str, Bitmap] = {}
            for entry in hex_list:
                if len(entry) > 0:
                    char, bitmap = unihex(entry)
                    if options.all_char_predicate(char) and options.unifont_char_predicate(char):
                        left, right = char_size(char, bitmap, provider.get('size_overrides', []))
                        chars[char] = bitmap.resized((left, 0, right, bitmap.height))
            full = BitmapProvider(height=8, ascent=7, chars=chars)
            return [full]

@dataclasses.dataclass
class ReadEntry[T]:
    data: T
    modified_date: datetime.datetime | None

def get_filtered_providers(identifier: str, store: Storage, options: ProviderOptions, support: ProviderSupport) -> list[ReadEntry[ResolvedJsonProvider]] | None:
    if support == ProviderSupport.NONE:
        return None
    if support == ProviderSupport.SWITCH_FONT_WHEN_FORCED and identifier == 'minecraft:default':
        identifier = 'minecraft:uniform'
    resolved = get_resolved_providers(identifier, store)
    if resolved is None:
        return None
    filtered: list[ReadEntry[ResolvedJsonProvider]] = []
    for data in resolved:
        provider = data.data
        if support == ProviderSupport.FULL and 'filter' in provider:
            if 'uniform' in provider['filter'] and provider['filter']['uniform'] != options.option_uniform:
                continue
            if 'jp' in provider['filter'] and provider['filter']['jp'] != options.option_jp:
                continue
        if support == ProviderSupport.ONLY_UNICODE_WHEN_FORCED and identifier == 'minecraft:default' and provider['type'] != 'legacy_unicode':
            continue
        filtered.append(data)
    return filtered

def get_resolved_providers(identifier: str, store: Storage) -> list[ReadEntry[ResolvedJsonProvider]] | None:
    entry = identifier_to_entry(identifier, 'font', 'json')
    if not store.exists(entry):
        return None
    font_data = read_font_definition(store, entry)
    i = 0
    while i < len(font_data):
        provider = font_data[i].data
        if provider['type'] != 'reference':
            i += 1
            continue
        data = resolve_reference(provider['id'], store, provider.get('filter'))
        font_data[i:i + 1] = data
    return typing.cast(list[ReadEntry[ResolvedJsonProvider]], font_data)

def resolve_reference(identifier: str, store: Storage, filter: FontFilter | None) -> list[ReadEntry[JsonProvider]]:
    entry = identifier_to_entry(identifier, kind='font', suffix='json')
    font_data = read_font_definition(store, entry)
    if filter is None:
        return font_data
    result: list[ReadEntry[JsonProvider]] = []
    for entry in font_data:
        merged = merge_filters(filter, entry.data.get('filter', {}))
        if merged is not None:
            entry.data['filter'] = merged
            result.append(entry)
    return result

def merge_filters(filter1: FontFilter, filter2: FontFilter) -> FontFilter | None:
    result: dict[str, typing.Any] = {}
    for k, v in filter2.items():
        result[k] = v
    for k, v in filter1.items():
        if k in result and result[k] != v:
            return None
        result[k] = v
    return typing.cast(FontFilter, result)

def read_image(store: Storage, entry: pathlib.PurePath) -> ReadEntry[PIL.Image.Image]:
    data = store.read(entry)
    img = PIL.Image.open(io.BytesIO(data))
    return ReadEntry(img, store.modified_time(entry))

def read_json(store: Storage, entry: pathlib.PurePath) -> ReadEntry:
    text = store.read(entry).decode('utf-8')
    # due to MC-278459 in some versions, we need to parse JSON leniently
    data = json5.loads(text)
    return ReadEntry(data, store.modified_time(entry))

def read_font_definition(store: Storage, entry: pathlib.PurePath) -> list[ReadEntry[JsonProvider]]:
    stack_sources: list[Storage] = [store]
    if isinstance(store, StackStorage):
        stack_sources = store.stack
    result: list[ReadEntry[JsonProvider]] = []
    for member in stack_sources:
        if member.exists(entry):
            data: ReadEntry[JsonRootProvider] = read_json(member, entry)
            for provider in data.data['providers']:
                result.append(ReadEntry(provider, data.modified_date))
    return result

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

def image_grid(image: PIL.Image.Image, grid: list[list[str | None]], sizes: dict[str, tuple[int, int]] | None, include: typing.Callable[[str], bool]) -> dict[str, CharImage]:
    result: dict[str, CharImage] = {}
    glyph_width = image.width // len(grid[0])
    glyph_height = image.height // len(grid)
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char is None or not include(char):
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
            cropped_glyph = glyph.crop(resize_box)
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
    elif len(art) == 32:
        size = (8, 16)
    else:
        raise ValueError(len(art))
    bitmap = Bitmap(size)
    bitmap.bits = bitarray.bitarray(bits)
    return (char, bitmap)

def legacy_unicode(store: Storage, sheets: list[pathlib.PurePath], size_bytes: bytes, options: ProviderOptions, times: ModifiedTimes) -> list[ImageProvider]:
    result: list[ImageProvider] = []
    for sheet_id, sheet_entry in enumerate(sheets):
        if store.exists(sheet_entry):
            char_offset = sheet_id * 256
            chars = [[chr(x) if size_bytes[x] > 0 else None for x in range(char_offset + y * 16, char_offset + y * 16 + 16)] for y in range(16)]
            size_range = size_bytes[char_offset:char_offset + 256]
            sizes = {chr(char_offset + i): ((x >> 4) & 0xf, (x & 0xf) + 1) for i, x in enumerate(size_range) if x > 0}
            img_data = read_image(store, sheet_entry)
            times.update(img_data.modified_date)
            full = ImageProvider(
                height = 8,
                ascent = 7,
                has_color = options.image_color_predicate(img_data.data),
                chars = image_grid(img_data.data, chars, sizes, lambda x: options.all_char_predicate(x) and options.unifont_char_predicate(x)),
            )
            result.append(full)
    return result
