import argparse
import datetime
import json
import pathlib
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
    commands = parser.add_subparsers(dest='command', required=True)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('out'))
    parser.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('cache'))
    vanilla = commands.add_parser('vanilla')
    vanilla.add_argument('version', type=str)
    pack = commands.add_parser('pack')
    pack.add_argument('version', type=str)
    pack.add_argument('location', type=pathlib.Path)
    pack.add_argument('identifier', type=str)
    pack.add_argument('name', type=str)
    args = parser.parse_args()
    match args.command:
        case 'vanilla':
            main_vanilla(args.version, args.output, args.cache)
        case 'pack':
            main_pack(args.version, args.location, args.identifier, args.name, args.output, args.cache)

def main_vanilla(version_id: str, output: pathlib.Path, cache: pathlib.Path):
    result = jar_and_version(version_id, cache)
    if result is None:
        return
    jar, version_data, version = result
    print(f'Detected font capabilities: {version.name}')
    aglfn = get_aglfn(cache)
    print(f'Converting fonts from Minecraft {version_data['id']}')
    with jar:
        store = storage.ZipStorage(jar)
        # TTF metadata includes a creation date
        # this information isn't in the jar, so we have to provide it ourselves
        vanilla_try_font('Default', 'minecraft:default', version, store, datetime.datetime.fromisoformat('2009-05-16T16:52:00Z'), aglfn, output)
        vanilla_try_font('Enchanting', 'minecraft:alt', version, store, datetime.datetime.fromisoformat('2011-10-06T00:00:00Z'), aglfn, output)
        vanilla_try_font('Illager Runes', 'minecraft:illageralt', version, store, datetime.datetime.fromisoformat('2021-09-15T16:04:30Z'), aglfn, output)
    print('Done!')

def vanilla_try_font(
   name: str,
   identifier: str,
   version: versions.MinecraftVersion,
   store: storage.Storage,
   created_date: datetime.datetime,
   aglfn: dict[str, str],
   out: pathlib.Path
):
    provider_list = providers.get_providers(store, version, identifier)
    if provider_list is None:
        return
    print(f'Converting {name}')
    fonts = convert_font(name, provider_list, version.supports_providers, created_date, aglfn)
    for style, font in fonts.items():
        full_name = 'Minecraft ' + name
        ttf_name = full_name.replace(' ', '') + '-' + style.replace(' ', '')
        dest = out / f'{ttf_name}.ttf'
        dest.parent.mkdir(parents=True, exist_ok=True)
        font.save(dest)

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

def main_pack(version_id: str, location: pathlib.Path, identifier: str, name: str, out: pathlib.Path, cache: pathlib.Path):
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
        fonts = convert_font(name, provider_list, version.supports_providers, created_date, aglfn)
        for style, font in fonts.items():
            ttf_name = name.replace(' ', '') + '-' + style.replace(' ', '')
            dest = out / f'{ttf_name}.ttf'
            dest.parent.mkdir(parents=True, exist_ok=True)
            font.save(dest)

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

def convert_font(
    name: str,
    provider_list: list[providers.Provider],
    has_missing_glyph: bool,
    created_date: datetime.datetime,
    aglfn: dict[str, str]
) -> dict[str, fontTools.fontBuilder.FontBuilder]:
    modified_date = created_date
    seen_chars: set[str] = set()
    fonts: dict[str, dict[str, CharInfo]] = {'Regular': {}, 'Bold': {}, 'Italic': {}, 'Bold Italic': {}}
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
        (path, (w, h)) = vectorize(mask, scale, offset)
        (italic_path, (iw, ih)) = vectorize(mask, scale, italic_offset, italic=True)
        (bold_path, (bw, bh)) = vectorize(bold_mask, scale, offset)
        (bold_italic_path, (biw, bih)) = vectorize(bold_mask, scale, italic_offset, italic=True)
        add_width = 0 if height == 0 else m_height / height
        fonts['Regular'][char] = CharInfo(width = (w + add_width) * scale, height = h * scale, path = path)
        fonts['Italic'][char] = CharInfo(width = (iw + add_width) * scale, height = ih * scale, path = italic_path)
        fonts['Bold'][char] = CharInfo(width = (bw + add_width) * scale, height = bh * scale, path = bold_path)
        fonts['Bold Italic'][char] = CharInfo(width = (biw + add_width) * scale, height = bih * scale, path = bold_italic_path)
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
                fonts['Regular'][char] = CharInfo(width = width * pixel_scale, height = 0, path = None)
                fonts['Italic'][char] = CharInfo(width = width * pixel_scale, height = 0, path = None)
                fonts['Bold'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, path = None)
                fonts['Bold Italic'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, path = None)
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
    final_fonts: dict[str, fontTools.fontBuilder.FontBuilder] = {}
    for style, data in fonts.items():
        full_name = 'Minecraft ' + name
        info = FontInfo(
            name = full_name,
            style = style,
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
        font = make_font(info, positions, data, aglfn)
        final_fonts[style] = font
    return final_fonts

if __name__ == '__main__':
    main()
