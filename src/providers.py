import dataclasses
import datetime
import io
import json
import typing
import zipfile

import PIL.Image


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

MinecraftVersion = typing.Literal['22w11a+', '1.13-pre7+', '13w42b+', '13w24a+', 'b1.1+', 'b1.0+', 'a1.2.2+', 'a1.0.9+', 'c0.0.17a+', 'c0.0.11a+']

def detect_version(jar: zipfile.ZipFile) -> MinecraftVersion | None:
    names = jar.namelist()
    rp: int | None = None
    if 'version.json' in names:
        text = jar.read('version.json')
        data = json.loads(text)
        if 'pack_version' in data:
            rp = data['pack_version'].get('resource')
            if rp is None:
                rp = data['pack_version'].get('resource_major')
    if rp is not None and rp >= 9:
        return '22w11a+'
    if 'assets/minecraft/font/default.json' in names:
        return '1.13-pre7+'
    if 'font.txt' not in names and 'assets/minecraft/textures/font/ascii.png' in names:
        # this wrongly includes 13w42a (which removed font.txt)
        # but no easy way to distinguish the two versions
        return '13w42b+'
    if 'pack.mcmeta' in names:
        return '13w24a+'
    if 'font.txt' in names:
        return 'b1.1+'
    if 'lang/en_US.lang' in names:
        return 'b1.0+'
    if 'font/default.png' in names:
        return 'a1.2.2+'
    if 'jar/mob/cow.png' in names:
        # this wrongly includes a1.0.8 (which added cow.png)
        # but no easy way to distinguish the two versions
        return 'a1.0.9+'
    if 'default.png' in names:
        return 'c0.0.17a+'
    if 'default.gif' in names:
        return 'c0.0.11a+'
    return None

def get_providers(jar: zipfile.ZipFile, version: MinecraftVersion, identifier: str) -> list[Provider] | None:
    entry = identifier_to_entry(identifier, 'font', 'json')
    space_provider = SpaceProvider({' ': 4, '\u200c': 0}, modified_date=None)
    match version:
        case '22w11a+':
            # everything is here
            return load_providers(jar, entry)
        case '1.13-pre7+':
            # space provider is missing
            names = jar.namelist()
            if entry not in names:
                return None
            providers = load_providers(jar, entry)
            providers.insert(0, space_provider)
            return providers
        case '13w42b+':
            # hardcoded list of characters
            chars: list[str] = [
                '\u00c0\u00c1\u00c2\u00c8\u00ca\u00cb\u00cd\u00d3\u00d4\u00d5\u00da\u00df\u00e3\u00f5\u011f\u0130',
                '\u0131\u0152\u0153\u015e\u015f\u0174\u0175\u017e\u0207\u0000\u0000\u0000\u0000\u0000\u0000\u0000',
                ' !"#$%&\'()*+,-./',
                '0123456789:;<=>?',
                '@ABCDEFGHIJKLMNO',
                'PQRSTUVWXYZ[\\]^_',
                '`abcdefghijklmno',
                'pqrstuvwxyz{|}~\u0000',
                '\u00c7\u00fc\u00e9\u00e2\u00e4\u00e0\u00e5\u00e7\u00ea\u00eb\u00e8\u00ef\u00ee\u00ec\u00c4\u00c5',
                '\u00c9\u00e6\u00c6\u00f4\u00f6\u00f2\u00fb\u00f9\u00ff\u00d6\u00dc\u00f8\u00a3\u00d8\u00d7\u0192',
                '\u00e1\u00ed\u00f3\u00fa\u00f1\u00d1\u00aa\u00ba\u00bf\u00ae\u00ac\u00bd\u00bc\u00a1\u00ab\u00bb',
                '\u2591\u2592\u2593\u2502\u2524\u2561\u2562\u2556\u2555\u2563\u2551\u2557\u255d\u255c\u255b\u2510',
                '\u2514\u2534\u252c\u251c\u2500\u253c\u255e\u255f\u255a\u2554\u2569\u2566\u2560\u2550\u256c\u2567',
                '\u2568\u2564\u2565\u2559\u2558\u2552\u2553\u256b\u256a\u2518\u250c\u2588\u2584\u258c\u2590\u2580',
                '\u03b1\u03b2\u0393\u03c0\u03a3\u03c3\u03bc\u03c4\u03a6\u0398\u03a9\u03b4\u221e\u2205\u2208\u2229',
                '\u2261\u00b1\u2265\u2264\u2320\u2321\u00f7\u2248\u00b0\u2219\u00b7\u221a\u207f\u00b2\u25a0\u0000',
            ]
            default = 'assets/minecraft/textures/font/ascii.png'
            ench = 'assets/minecraft/textures/font/ascii_sga.png'
            if identifier == 'minecraft:default':
                img, date = read_image(jar, default)
            elif identifier == 'minecraft:alt':
                img, date = read_image(jar, ench)
            else:
                return None
            bitmap_provider = BitmapProvider(height=8, ascent=7, image=img, chars=chars, modified_date=date)
            return [space_provider, bitmap_provider]
        case '13w24a+' | 'b1.1+':
            # characters come from font.txt
            font, date = read_font_txt(jar, 'font.txt')
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
            if version == '13w24a+':
                default = 'assets/minecraft/textures/font/ascii.png'
                ench = 'assets/minecraft/textures/font/ascii_sga.png'
            else:
                default = 'font/default.png'
                ench = 'font/alternate.png'
            if identifier == 'minecraft:default':
                img, img_date = read_image(jar, default)
            elif identifier == 'minecraft:alt':
                img, img_date = read_image(jar, ench)
            else:
                return None
            bitmap_provider = BitmapProvider(height=8, ascent=7, image=img, chars=chars, modified_date=max(date, img_date))
            return [space_provider, bitmap_provider]
        case 'b1.0+':
            # hardcoded list of characters (that does not make sense)
            chars: list[str] = [
                ' !"#$%&\'()*+,-./',
                '0123456789:;<=>?',
                '@ABCDEFGHIJKLMNO',
                'PQRSTUVWXYZ[\\]^_',
                '\'abcdefghijklmno',
                'pqrstuvwxyz{|}~\u00e2',
                '\u0152\u201a\u00c3\u2021\u00c3\u00bc\u00c3\u00a9\u00c3\u00a2\u00c3\u00a4\u00c3\u00a0\u00c3\u00a5',
                '\u00c3\u00a7\u00c3\u00aa\u00c3\u00ab\u00c3\u00a8\u00c3\u00af\u00c3\u00ae\u00c3\u00ac\u00c3\u201e',
                '\u00c3\u2026\u00c3\u2030\u00c3\u00a6\u00c3\u2020\u00c3\u00b4\u00c3\u00b6\u00c3\u00b2\u00c3\u00bb',
                '\u00c3\u00b9\u00c3\u00bf\u00c3\u2013\u00c3\u0153\u00c3\u00b8\u00c2\u00a3\u00c3\u02dc\u00c3\u2014',
                '\u00c6\u2019\u00c3\u00a1\u00c3\u00ad\u00c3\u00b3\u00c3\u00ba\u00c3\u00b1\u00c3\u2018\u00c2\u00aa',
                '\u00c2\u00ba\u00c2\u00bf\u00c2\u00ae\u00c2\u00ac\u00c2\u00bd\u00c2\u00bc\u00c2\u00a1\u00c2\u00ab',
                '\u00c2\u00bb' + ('\u0000' * 14)
            ]
            empty = '\u0000' * 16
            full_chars: list[str] = [
               empty,
               empty,
               *chars,
               empty,
               empty,
               empty,
            ]
            if identifier == 'minecraft:default':
                img, img_date = read_image(jar, 'font/default.png')
            else:
                return None
            bitmap_provider = BitmapProvider(height=8, ascent=7, image=img, chars=full_chars, modified_date=img_date)
            return [space_provider, bitmap_provider]
        case 'a1.2.2+' | 'a1.0.9+':
            # hardcoded list of characters
            chars: list[str] = [
                ' !"#$%&\'()*+,-./',
                '0123456789:;<=>?',
                '@ABCDEFGHIJKLMNO',
                'PQRSTUVWXYZ[\\]^_',
                '\'abcdefghijklmno',
                'pqrstuvwxyz{|}~\u2302',
                '\u00c7\u00fc\u00e9\u00e2\u00e4\u00e0\u00e5\u00e7\u00ea\u00eb\u00e8\u00ef\u00ee\u00ec\u00c4\u00c5',
                '\u00c9\u00e6\u00c6\u00f4\u00f6\u00f2\u00fb\u00f9\u00ff\u00d6\u00dc\u00f8\u00a3\u00d8\u00d7\u0192',
                '\u00e1\u00ed\u00f3\u00fa\u00f1\u00d1\u00aa\u00ba\u00bf\u00ae\u00ac\u00bd\u00bc\u00a1\u00ab\u00bb',
            ]
            empty = '\u0000' * 16
            full_chars: list[str] = [
               empty,
               empty,
               *chars,
               empty,
               empty,
               empty,
               empty,
               empty,
            ]
            if version == 'a1.2.2+':
                default = 'font/default.png'
            else:
                default = 'default.png'
            if identifier == 'minecraft:default':
                img, img_date = read_image(jar, default)
            else:
                return None
            bitmap_provider = BitmapProvider(height=8, ascent=7, image=img, chars=full_chars, modified_date=img_date)
            return [space_provider, bitmap_provider]
        case 'c0.0.17a+' | 'c0.0.11a+':
            # all characters on sheet
            chars: list[str] = [
                ''.join(chr(x) for x in range(y * 16, (y + 1) * 16)) for y in range(16)
            ]
            if version == 'c0.0.17a+':
                default = 'default.png'
            else:
                default = 'default.gif'
            if identifier == 'minecraft:default':
                img, img_date = read_image(jar, default)
            else:
                return None
            bitmap_provider = BitmapProvider(height=8, ascent=7, image=img, chars=chars, modified_date=img_date)
            return [space_provider, bitmap_provider]

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
