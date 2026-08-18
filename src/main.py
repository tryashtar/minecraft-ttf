import argparse
import dataclasses
import datetime
import hashlib
import json
import pathlib
import typing
import zipfile

import fontTools.fontBuilder
import requests

import providers
import storage
import versions
from minecraft_ttf.bitmap import Bitmap, bitmap_from_image
from minecraft_ttf.font import CharInfo, FontInfo, FontPositions, make_font, vectorize


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True, help='Which operation to perform')
    parser.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('cache'), help='Folder for cache files')
    vanilla = commands.add_parser('vanilla', help='Generate TTF fonts from the vanilla game jar')
    vanilla_action = vanilla.add_subparsers(dest='action', required=True, help='Which operation to perform')    
    vanilla_generate = vanilla_action.add_parser('generate', help='Generate TTF fonts from a specific version')
    vanilla_history = vanilla_action.add_parser('history', help='Generate all unique TTF fonts across Minecraft\'s history')
    vanilla_history.add_argument('--start', type=str, help='First version to scan')
    vanilla_history.add_argument('--end', type=str, help='Last version to scan')
    vanilla_generate.add_argument('version', type=str, help='Name of the Minecraft version to download, or "latest"')
    vanilla.add_argument('--identifiers', type=str, nargs='*', default=typing.get_args(DEFAULT_IDENTIFIERS), choices=typing.get_args(DEFAULT_IDENTIFIERS), help='Identifiers of the font definitions to generate fonts from')
    pack = commands.add_parser('pack', help='List or generate fonts from a resource pack')
    pack_action = pack.add_subparsers(dest='action', required=True, help='Which operation to perform')
    pack_list = pack_action.add_parser('list', help='List available font identifiers')
    pack_generate = pack_action.add_parser('generate', help='Generate one TTF font from a resource pack')
    for entry in (pack_generate, pack_list):
        entry.add_argument('version', type=str, help='Name of the Minecraft version the resource pack is targeting, or "latest"')
        entry.add_argument('location', type=pathlib.Path, help='Path to the resource pack folder or zip file')
    for entry in (vanilla, pack_generate):
        entry.add_argument('--output', type=pathlib.Path, default=pathlib.Path('out'), help='Folder to save generated fonts to')
        entry.add_argument('--styles', type=str, nargs='*', default=['regular'], choices=typing.get_args(STYLES), help='Styles to generate')
    pack_generate.add_argument('identifier', type=str, help='Identifier of the font definition to use, e.g. "minecraft:default"')
    pack_generate.add_argument('name', type=str, help='Display name for the generated TTF font')
    args = parser.parse_args()
    match args.command:
        case 'vanilla':
            identifiers = set(args.identifiers)
            styles = set(args.styles)
            match args.action:
                case 'generate':
                    main_vanilla_generate(args.version, identifiers, styles, args.output, args.cache)
                case 'history':
                    main_vanilla_history(args.start, args.end, identifiers, styles, args.output, args.cache)
        case 'pack':
            match args.action:
                case 'generate':
                    styles = set(args.styles)
                    main_pack_generate(args.version, args.location, args.identifier, args.name, styles, args.output, args.cache)
                case 'list':
                    main_pack_list(args.version, args.location, args.cache)

STYLES = typing.Literal['regular', 'italic', 'bold', 'bold_italic']
DEFAULT_IDENTIFIERS = typing.Literal['minecraft:default', 'minecraft:alt', 'minecraft:illageralt']

# TTF metadata includes a creation date
# this information isn't in the jar, so we have to provide it ourselves
def default_font_info(identifier: DEFAULT_IDENTIFIERS) -> tuple[str, datetime.datetime]:
    cache: dict[DEFAULT_IDENTIFIERS, tuple[str, datetime.datetime]] = {
        'minecraft:default': ('Default', datetime.datetime.fromisoformat('2009-05-16T16:52:00Z')),
        'minecraft:alt': ('Enchanting', datetime.datetime.fromisoformat('2011-10-06T00:00:00Z')),
        'minecraft:illageralt': ('Illager Runes', datetime.datetime.fromisoformat('2021-09-15T16:04:30Z')),
    }
    return cache[identifier]

def style_info(style: STYLES) -> tuple[str, bool, bool]:
    cache: dict[STYLES, tuple[str, bool, bool]] = {
        'regular': ('Regular', False, False),
        'italic': ('Italic', False, True),
        'bold': ('Bold', True, False),
        'bold_italic': ('Bold Italic', True, True),
    }
    return cache[style]

def main_vanilla_generate(version_id: str, identifiers: set[DEFAULT_IDENTIFIERS], styles: set[STYLES], output: pathlib.Path, cache: pathlib.Path):
    result = jar_and_version(version_id, cache)
    if result is None:
        return
    jar, version_data, version = result
    print(f'Detected font capabilities: {version.name}')
    aglfn = get_aglfn(cache)
    print(f'Converting fonts from Minecraft {version_data['id']}')
    with jar:
        store = storage.ZipStorage(jar)
        for identifier in identifiers:
            name, date = default_font_info(identifier)
            vanilla_try_font(name, identifier, version, store, date, styles, aglfn, output)
    print('Done!')

def font_digest(font: dict[str, CharInfo]) -> str:
    alg = hashlib.sha1()
    for char, data in font.items():
        string = f'{char}_{data.width}_{data.height}'
        alg.update(string.encode('utf-8'))
        if data.path is not None:
            alg.update(data.path.coordinates.array.tobytes())
    return alg.hexdigest()

def main_vanilla_history(start: str | None, end: str | None, identifiers: set[DEFAULT_IDENTIFIERS], styles: set[STYLES], output: pathlib.Path, cache: pathlib.Path):
    manifest = get_manifest(cache)
    aglfn = get_aglfn(cache)
    seen_hashes: set[str] = set()
    reached_start = start is None
    for version_data in reversed(manifest['versions']):
        if not reached_start and start is not None and version_data['id'] == start:
            reached_start = True
        if not reached_start:
            continue
        jar_path = get_jar(version_data['id'], version_data['url'], cache)
        jar = zipfile.ZipFile(jar_path, 'r')
        version = versions.detect_version(jar)
        if version is None:
            continue
        with jar:
            store = storage.ZipStorage(jar)
            for identifier in identifiers:
                provider_list = providers.get_providers(store, version, identifier)
                if provider_list is None:
                    continue
                name, date = default_font_info(identifier)
                fonts = create_fonts(provider_list, version.supports_providers, date, styles)
                for style, font in fonts.fonts.items():
                    hash = font_digest(font)
                    if hash not in seen_hashes:
                        seen_hashes.add(hash)
                        print(f'{version_data['id']} ({version.name}): {identifier} ({style}) changed')
                        full_name = 'Minecraft ' + name
                        ttf = finalize_font(full_name, style, font, fonts.font_em, fonts.created_date, fonts.modified_date, aglfn)
                        dest = output / f'{identifier.split(':')[1]}-{style}-{version_data['id']}.ttf'
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        ttf.save(dest)
        if end is not None and version_data['id'] == end:
            break

def vanilla_try_font(
   name: str,
   identifier: str,
   version: versions.MinecraftVersion,
   store: storage.Storage,
   created_date: datetime.datetime,
   styles: set[STYLES],
   aglfn: dict[str, str],
   out: pathlib.Path
):
    provider_list = providers.get_providers(store, version, identifier)
    if provider_list is None:
        return
    print(f'Converting {name}')
    fonts = create_fonts(provider_list, version.supports_providers, created_date, styles)
    for style, font in fonts.fonts.items():
        full_name = 'Minecraft ' + name
        ttf = finalize_font(full_name, style, font, fonts.font_em, fonts.created_date, fonts.modified_date, aglfn)
        ttf_name = full_name.replace(' ', '') + '-' + style_info(style)[0].replace(' ', '')
        dest = out / f'{ttf_name}.ttf'
        dest.parent.mkdir(parents=True, exist_ok=True)
        ttf.save(dest)

def jar_and_version(version_id: str, cache: pathlib.Path) -> tuple[zipfile.ZipFile, dict, versions.MinecraftVersion] | None:
    manifest = get_manifest(cache)
    if version_id == 'latest':
        version_data = get_version(manifest, manifest['latest']['snapshot'])
    else:
        version_data = get_version(manifest, version_id)
    if version_data is None:
        print(f'Version {version_id} not found in manifest')
        return None
    jar_path = get_jar(version_data['id'], version_data['url'], cache)
    zip = zipfile.ZipFile(jar_path, 'r')
    version = versions.detect_version(zip)
    if version is None:
        print('Unable to determine capabilities of jar!')
        return None
    return (zip, version_data, version)

def main_pack_list(version_id: str, location: pathlib.Path, cache: pathlib.Path):
    pack_storage = storage.get_storage(location)
    if pack_storage is None:
        print(f'No resource pack at {location}')
        return
    result = jar_and_version(version_id, cache)
    if result is None:
        return
    jar, version_data, version = result
    print(f'Available font identifiers in resource pack {location.name} on Minecraft {version_data['id']}:')
    with jar:
        jar_storage = storage.ZipStorage(jar)
        store = storage.StackStorage([jar_storage, pack_storage])
        identifiers = available_identifiers(version, store)
    for identifier in identifiers:
        print(identifier)

def available_identifiers(version: versions.MinecraftVersion, store: storage.Storage) -> list[str]:
    if version.supports_providers:
        identifiers: list[str] = []
        assets = store.get_entries(pathlib.PurePath('assets'))
        for asset in assets:
            parts = asset.parts
            if len(parts) > 3 and parts[2] == 'font' and asset.suffix == '.json':
                namespace = parts[1]
                body = list(parts[3:])
                body[-1] = asset.stem
                identifier = namespace + ':' + '/'.join(body)
                identifiers.append(identifier)
        return identifiers
    else:
        assert version.entry_map is not None
        return list(version.entry_map.keys())

def main_pack_generate(version_id: str, location: pathlib.Path, identifier: str, name: str, styles: set[STYLES], out: pathlib.Path, cache: pathlib.Path):
    pack_storage = storage.get_storage(location)
    if pack_storage is None:
        print(f'No resource pack at {location}')
        return
    result = jar_and_version(version_id, cache)
    if result is None:
        return
    jar, version_data, version = result
    aglfn = get_aglfn(cache)
    print(f'Converting font {identifier} from resource pack {location.name} on Minecraft {version_data['id']}')
    with jar:
        jar_storage = storage.ZipStorage(jar)
        store = storage.StackStorage([jar_storage, pack_storage])
        provider_list = providers.get_providers(store, version, identifier)
        if provider_list is None:
            print(f'No font with ID {identifier} in jar or resource pack')
            return
        created_date = min(x.modified_date for x in provider_list if x.modified_date is not None)
        fonts = create_fonts(provider_list, version.supports_providers, created_date, styles)
        for style, font in fonts.fonts.items():
            ttf = finalize_font(name, style, font, fonts.font_em, fonts.created_date, fonts.modified_date, aglfn)
            ttf_name = name.replace(' ', '') + '-' + style_info(style)[0].replace(' ', '')
            dest = out / f'{ttf_name}.ttf'
            dest.parent.mkdir(parents=True, exist_ok=True)
            ttf.save(dest)

def get_jar(version_id: str, meta_url: str, cache: pathlib.Path) -> pathlib.Path:
    cached_path = cache / f'minecraft-{version_id}.jar'
    if not cached_path.exists():
        print(f'Downloading Minecraft jar {version_id}')
        response = requests.get(meta_url)
        data = response.json()
        client_jar = data['downloads']['client']['url']
        response = requests.get(client_jar)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'wb') as f:
            f.writelines(response.iter_content(chunk_size=16 * 1024))
    return cached_path

def get_version(manifest: dict, version_id: str) -> dict | None:
    for version in manifest['versions']:
        if version['id'] == version_id:
            return version
    return None

def get_manifest(cache: pathlib.Path) -> dict:
    cached_path = cache / 'manifest.json'
    try:
        with open(cached_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('Downloading version manifest')
        manifest_url = 'https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'
        response = requests.get(manifest_url)
        data = response.json()
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    return data

# The Adobe Glyph List For New Fonts tells us what names to use for the glyphs that characters are mapped to
def get_aglfn(cache: pathlib.Path) -> dict[str, str]:
    cached_path = cache / 'aglfn.txt'
    if not cached_path.exists():
        print('Downloading Adobe AGLFN')
        response = requests.get('https://raw.githubusercontent.com/adobe-type-tools/agl-aglfn/refs/heads/master/aglfn.txt')
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'wb') as f:
            f.writelines(response.iter_content(chunk_size=16 * 1024))
    aglfn_map = {}
    with open(cached_path, 'r', encoding='utf-8') as aglfn:
        for line in aglfn:
            if line.startswith('#') or line.isspace() or len(line) == 0:
                continue
            unihex, name, _uniname = line.split(';')
            uninum = int(unihex, 16)
            codepoint = chr(uninum)
            aglfn_map[codepoint] = name
    return aglfn_map

@dataclasses.dataclass
class CreatedFontInfo:
    fonts: dict[STYLES, dict[str, CharInfo]]
    font_em: int
    created_date: datetime.datetime
    modified_date: datetime.datetime

def create_fonts(
    provider_list: list[providers.Provider],
    has_missing_glyph: bool,
    created_date: datetime.datetime,
    styles: set[STYLES],
) -> CreatedFontInfo:
    modified_date = created_date
    seen_chars: set[str] = set()
    fonts: dict[STYLES, dict[str, CharInfo]] = {}
    for style in styles:
       fonts[style] = {}
    chatbox_height = 12
    font_em = 1200
    pixel_scale = font_em / chatbox_height
    # font textures have a color depth of 1 bit, so they are just 2D bitmasks
    # this lets us leverage some efficient operations provided by pygame
    def add_bitmap_glyph(char: str, mask: Bitmap, height: int, ascent: int):
        m_width, m_height = mask.get_size()
        seen_chars.add(char)
        # bold characters are created by overlapping two copies of the texture
        bold_mask = Bitmap((m_width + 1, m_height))
        bold_mask.draw(mask, (0, 0))
        bold_mask.draw(mask, (1, 0))
        scale = height / m_height * pixel_scale
        offset = (0, 0) if height == 0 else (0, (height - ascent) / height * m_height)
        italic_offset = (0, 0) if height == 0 else (-6 / height, (height - ascent) / height * m_height)
        add_width = 0 if height == 0 else m_height / height
        if 'regular' in styles:
            (path, (w, h)) = vectorize(mask, scale, offset)
            fonts['regular'][char] = CharInfo(width = (w + add_width) * scale, height = h * scale, path = path)
        if 'italic' in styles:
            (italic_path, (iw, ih)) = vectorize(mask, scale, italic_offset, italic=True)
            fonts['italic'][char] = CharInfo(width = (iw + add_width) * scale, height = ih * scale, path = italic_path)
        if 'bold' in styles:
            (bold_path, (bw, bh)) = vectorize(bold_mask, scale, offset)
            fonts['bold'][char] = CharInfo(width = (bw + add_width) * scale, height = bh * scale, path = bold_path)
        if 'bold_italic' in styles:
            (bold_italic_path, (biw, bih)) = vectorize(bold_mask, scale, italic_offset, italic=True)
            fonts['bold_italic'][char] = CharInfo(width = (biw + add_width) * scale, height = bih * scale, path = bold_italic_path)
    mw, mh = (5, 8)
    missing = Bitmap((mw, mh))
    if has_missing_glyph:
        for y in range(mh):
            for x in range(mw):
                if x == 0 or y == 0 or x == mw - 1 or y == mh - 1:
                    missing.set_at((x, y), True)
    add_bitmap_glyph('.notdef', missing, 8, 8)
    for provider in provider_list:
        if provider.modified_date is not None:
            modified_date = max(modified_date, provider.modified_date)
        if isinstance(provider, providers.SpaceProvider):
            for char, width in provider.spaces.items():
                if char in seen_chars:
                    continue
                width = max(0, width)
                seen_chars.add(char)
                if 'regular' in styles:
                    fonts['regular'][char] = CharInfo(width = width * pixel_scale, height = 0, path = None)
                if 'italic' in styles:
                    fonts['italic'][char] = CharInfo(width = width * pixel_scale, height = 0, path = None)
                if 'bold' in styles:
                    fonts['bold'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, path = None)
                if 'bold_italic' in styles:
                    fonts['bold_italic'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, path = None)
        elif isinstance(provider, providers.BitmapProvider):
            glyph_width = provider.image.width // len(provider.chars[0])
            glyph_height = provider.image.height // len(provider.chars)
            for y, row in enumerate(provider.chars):
                for x, char in enumerate(row):
                    if char == '\u0000':
                        continue
                    if char in seen_chars:
                        continue
                    dimensions = (x * glyph_width, y * glyph_height, (x + 1) * glyph_width, (y + 1) * glyph_height)
                    if provider.ascent > -16384 and provider.height > 0:
                        glyph = provider.image.crop(dimensions).convert('RGBA')
                        mask = bitmap_from_image(glyph)
                        p_height = provider.height
                        p_ascent = provider.ascent
                    else:
                        mask = Bitmap((dimensions[2], dimensions[3]))
                        p_height = 0
                        p_ascent = 0
                    add_bitmap_glyph(char, mask, p_height, p_ascent)
    return CreatedFontInfo(fonts, font_em, created_date, modified_date)

def finalize_font(
    full_name: str,
    style: STYLES,
    font: dict[str, CharInfo],
    font_em: int,
    created_date: datetime.datetime,
    modified_date: datetime.datetime,
    aglfn: dict[str, str]
) -> fontTools.fontBuilder.FontBuilder:
    stylename, bold, italic = style_info(style)
    info = FontInfo(
        name = full_name,
        style = stylename,
        bold = bold,
        italic = italic,
        copyright = 'Copyright (c) 2009 Mojang AB',
        sample = 'and the universe said I love you',
        em = font_em,
        created = created_date,
        modified = modified_date,
        version = 'Version 1.000'
    )
    positions = FontPositions(
        ascent = 9 / 12,
        descent = 2 / 12,
        sCapHeight = 7 / 12,
        sxHeight = 5 / 12,
        yStrikeoutPosition = 4 / 12,
        yStrikeoutSize = 1 / 12,
        underlinePosition = -1 / 12,
        underlineThickness = 1 / 12,
        italicAngle = -14.05598
    )
    result = make_font(info, positions, font, aglfn)
    return result

if __name__ == '__main__':
    main()
