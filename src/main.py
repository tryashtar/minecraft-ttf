import argparse
import datetime
import math
import pathlib
import typing

import PIL.Image

from cache import (
    get_aglfn,
    get_manifest,
    jar_info,
    version_from_data,
)
from minecraft_ttf.bitmap import get_palette_rgba
from minecraft_ttf.font import GlyphInfo
from minecraft_ttf.minecraft.font import (
    STYLE,
    create_font_family,
    finalize_font,
    style_info,
)
from minecraft_ttf.minecraft.providers import ImageColorCheck
from minecraft_ttf.minecraft.storage import (
    StackStorage,
    Storage,
    get_storage,
)
from minecraft_ttf.minecraft.versions import (
    VANILLA_FONT_ID,
    MinecraftVersion,
    get_providers,
)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True, help='Which operation to perform')
    vanilla = commands.add_parser('vanilla', help='Generate TTF fonts from the vanilla game jar')
    vanilla_action = vanilla.add_subparsers(dest='action', required=True, help='Which operation to perform')    
    vanilla_generate = vanilla_action.add_parser('generate', help='Generate TTF fonts from a specific version')
    vanilla_generate.add_argument('version', type=str, help='Name of the Minecraft version to download, or "latest"')
    vanilla_history = vanilla_action.add_parser('history', help='Generate all unique TTF fonts across Minecraft\'s history')
    vanilla_history.add_argument('--start', type=str, help='First version to scan')
    vanilla_history.add_argument('--end', type=str, help='Last version to scan')
    pack = commands.add_parser('pack', help='List or generate fonts from a resource pack')
    pack_action = pack.add_subparsers(dest='action', required=True, help='Which operation to perform')
    pack_list = pack_action.add_parser('list', help='List available font identifiers')
    pack_generate = pack_action.add_parser('generate', help='Generate one TTF font from a resource pack')
    for entry in (pack_generate, pack_list):
        entry.add_argument('version', type=str, help='Name of the Minecraft version the resource pack is targeting, or "latest"')
        entry.add_argument('location', type=pathlib.Path, help='Path to the resource pack folder or zip file')
    pack_generate.add_argument('identifier', type=str, help='Identifier of the font definition to use, e.g. "minecraft:default"')
    pack_generate.add_argument('name', type=str, help='Display name for the generated TTF font')
    for entry in (vanilla_generate, vanilla_history):
        entry.add_argument('--identifiers', type=str, nargs='*', default=typing.get_args(VANILLA_FONT_ID), choices=typing.get_args(VANILLA_FONT_ID), help='Identifiers of the font definitions to generate fonts from')
    for entry in (vanilla_generate, vanilla_history, pack_generate):
        entry.add_argument('--styles', type=str, nargs='*', default=typing.get_args(STYLE), choices=typing.get_args(STYLE), help='Styles to generate')
        entry.add_argument('--color', type=str, default='auto', choices=['never', 'always', 'auto'], help='When to include color for characters that come from images (auto = only if any part of the image is not solid white)')
    for entry in (vanilla_generate, vanilla_history, pack_generate):
        entry.add_argument('--output', type=pathlib.Path, default=pathlib.Path('out'), help='Folder to save the generated fonts in')
    for entry in (vanilla_generate, vanilla_history, pack_generate, pack_list):
        entry.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('cache'), help='Folder for cache files')
    def color_check(color: str) -> ImageColorCheck:
        return {
            'never': lambda _: False,
            'always': lambda _: True,
            'auto': image_has_color
        }[color]
    args = parser.parse_args()
    match args.command:
        case 'vanilla':
            identifiers = set(args.identifiers)
            styles = set(args.styles)
            match args.action:
                case 'generate':
                    main_vanilla_generate(args.version, identifiers, color_check(args.color), styles, args.output, args.cache)
                case 'history':
                    main_vanilla_history((args.start, args.end), identifiers, color_check(args.color), styles, args.output, args.cache)
        case 'pack':
            match args.action:
                case 'generate':
                    styles = set(args.styles)
                    main_pack_generate(args.version, args.location, args.identifier, args.name, color_check(args.color), styles, args.output, args.cache)
                case 'list':
                    main_pack_list(args.version, args.location, args.cache)

def main_vanilla_generate(version_id: str, identifiers: set[VANILLA_FONT_ID], color: ImageColorCheck, styles: set[STYLE], output: pathlib.Path, cache: pathlib.Path):
    info = jar_info(version_id, cache)
    if info is None:
        return
    print(f'Detected font capabilities: {info.version.name}')
    aglfn = get_aglfn(cache)
    print(f'Converting fonts from Minecraft {info.manifest['id']}')
    store = StackStorage([info.jar_storage, info.asset_storage])
    for identifier in identifiers:
        vanilla_try_font(identifier, info.version, store, color, styles, aglfn, output)
    print('Done!')

def vanilla_try_font(
   identifier: VANILLA_FONT_ID,
   version: MinecraftVersion,
   store: Storage,
   color: ImageColorCheck,
   styles: set[STYLE],
   aglfn: dict[str, str],
   out: pathlib.Path
):
    provider_list = get_providers(version, store, identifier, color)
    if provider_list is None:
        print(f'No providers found for {identifier}')
        return
    name, created_date = default_font_info(identifier)
    print(f'Converting {name}')
    family = create_font_family(provider_list, version.supports_providers, created_date, styles)
    print_fontsize_info(family.scales)
    for style, font in family.fonts.items():
        full_name = 'Minecraft ' + name
        ttf = finalize_font(full_name, style, font.chars, font.other_glyphs, family.font_em, family.created_date, family.modified_date, aglfn)
        ttf_name = full_name.replace(' ', '') + '-' + style_info(style)[0].replace(' ', '')
        dest = out / f'{ttf_name}.ttf'
        dest.parent.mkdir(parents=True, exist_ok=True)
        ttf.save(dest)

def main_vanilla_history(version_range: tuple[str | None, str | None], identifiers: set[VANILLA_FONT_ID], color: ImageColorCheck, styles: set[STYLE], output: pathlib.Path, cache: pathlib.Path):
    manifest = get_manifest(cache)
    aglfn = get_aglfn(cache)
    seen_fonts: set[frozenset[frozenset[tuple[str, GlyphInfo]]]] = set()
    start, end = version_range
    reached_start = start is None
    for version_data in reversed(manifest['versions']):
        if not reached_start and start is not None and version_data['id'] == start:
            reached_start = True
        if not reached_start:
            continue
        info = version_from_data(version_data, output)
        if info is None:
            continue
        store = StackStorage([info.jar_storage, info.asset_storage])
        for identifier in identifiers:
            provider_list = get_providers(info.version, store, identifier, color)
            if provider_list is None:
                continue
            name, date = default_font_info(identifier)
            family = create_font_family(provider_list, info.version.supports_providers, date, styles)
            for style, font in family.fonts.items():
                entry = frozenset([frozenset(font.chars.items()), frozenset(font.other_glyphs.items())])
                if entry not in seen_fonts:
                    seen_fonts.add(entry)
                    print(f'{version_data['id']} ({info.version.name}): {identifier} ({style}) changed')
                    full_name = 'Minecraft ' + name
                    ttf = finalize_font(full_name, style, font.chars, font.other_glyphs, family.font_em, family.created_date, family.modified_date, aglfn)
                    dest = output / f'{identifier.split(':')[1]}-{style}-{version_data['id']}.ttf'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    ttf.save(dest)
        if end is not None and version_data['id'] == end:
            break

def main_pack_list(version_id: str, location: pathlib.Path, cache: pathlib.Path):
    pack_storage = get_storage(location)
    if pack_storage is None:
        print(f'No resource pack at {location}')
        return
    info = jar_info(version_id, cache)
    if info is None:
        return
    print(f'Available font identifiers in resource pack {location.name} on Minecraft {info.manifest['id']}:')
    store = StackStorage([info.jar_storage, info.asset_storage, pack_storage])
    identifiers = available_identifiers(info.version, store)
    for identifier in identifiers:
        print(identifier)

def available_identifiers(version: MinecraftVersion, store: Storage) -> list[str]:
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

def main_pack_generate(version_id: str, location: pathlib.Path, identifier: str, name: str, color: ImageColorCheck, styles: set[STYLE], out: pathlib.Path, cache: pathlib.Path):
    pack_storage = get_storage(location)
    if pack_storage is None:
        print(f'No resource pack at {location}')
        return
    info = jar_info(version_id, cache)
    if info is None:
        return
    aglfn = get_aglfn(cache)
    print(f'Converting font {identifier} from resource pack {location.name} on Minecraft {info.manifest['id']}')
    store = StackStorage([info.jar_storage, info.asset_storage, pack_storage])
    provider_list = get_providers(info.version, store, identifier, color)
    if provider_list is None:
        print(f'No font with ID {identifier} in jar or resource pack')
        return
    created_date = min(x.modified_date for x in provider_list if x.modified_date is not None)
    family = create_font_family(provider_list, info.version.supports_providers, created_date, styles)
    print_fontsize_info(family.scales)
    for style, font in family.fonts.items():
        ttf = finalize_font(name, style, font.chars, font.other_glyphs, family.font_em, family.created_date, family.modified_date, aglfn)
        ttf_name = name.replace(' ', '') + '-' + style_info(style)[0].replace(' ', '')
        dest = out / f'{ttf_name}.ttf'
        dest.parent.mkdir(parents=True, exist_ok=True)
        ttf.save(dest)

def print_fontsize_info(scales: dict[tuple[int, int], list[str]]):
    point_sizes: dict[int, list[str]] = {}
    for (num, denom), chars in scales.items():
        top = num * 12
        gcd = math.gcd(top, denom)
        point_size = top // gcd
        if point_size not in point_sizes:
            point_sizes[point_size] = []
        point_sizes[point_size].extend(chars)
    for point, chars in point_sizes.items():
        if len(chars) <= 30:
            print(f'{len(chars)} characters in this font ({''.join(chars)}) will look pixel-perfect at font size multiples of {point}')
        else:
            print(f'{len(chars)} characters in this font will look pixel-perfect at font size multiples of {point}')
    lcm = math.lcm(*point_sizes.keys())
    print(f'All characters in this font will look pixel-perfect at font size multiples of {lcm}')

# TTF metadata includes a name and creation date
# this information isn't in the jar, so we have to provide it ourselves
def default_font_info(identifier: VANILLA_FONT_ID) -> tuple[str, datetime.datetime]:
    cache: dict[VANILLA_FONT_ID, tuple[str, datetime.datetime]] = {
        # release time of 0.0.2a, the first version to have a font, according to the wiki: https://minecraft.wiki/w/Java_Edition_Classic_0.0.2a
        'minecraft:default': ('Default', datetime.datetime.fromisoformat('2009-05-16T16:52:00Z')),
        # release time of b1.9-pre3, the first version to include enchanting, according to the wiki: https://minecraft.wiki/w/Java_Edition_Beta_1.9_Prerelease_3
        'minecraft:alt': ('Enchanting', datetime.datetime.fromisoformat('2011-10-06T14:57:00Z')),
        # release time of 21w37a, the first version to include this font, according to the manifest: https://piston-meta.mojang.com/v1/packages/7dfcb7bb54ac9e9b927627ef2a70d922543bb8bf/21w37a.json
        'minecraft:illageralt': ('Illager Runes', datetime.datetime.fromisoformat('2021-09-15T16:04:30Z')),
    }
    return cache[identifier]

def image_has_color(image: PIL.Image.Image) -> bool:
    colors = image.getcolors(maxcolors = 3)
    if colors is None or len(colors) >= 3:
        return True
    palette = image.getpalette()
    transparent_index = image.info.get('transparency')
    for _count, color in colors:
        if image.mode == 'P':
            assert palette is not None
            assert isinstance(color, int)
            r, g, b, a = get_palette_rgba(palette, transparent_index, color)
        elif image.mode == 'RGB':
            assert isinstance(color, tuple)
            r, g, b = color
            a = 255
        elif image.mode == 'RGBA':
            assert isinstance(color, tuple)
            r, g, b, a = color
        else:
            raise ValueError(image.mode)
        if a != 0 and not (r == 255 and g == 255 and b == 255):
            return True
    return False

if __name__ == '__main__':
    main()
