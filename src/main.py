import argparse
import dataclasses
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
    FontFamilyInfo,
    create_font_family,
    finalize_font,
    style_info,
)
from minecraft_ttf.minecraft.providers import (
    ModifiedTimes,
    ProviderOptions,
    ProviderSupport,
)
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
        entry.add_argument('--chars', type=str, default='00000-fffff', help='Ranges of characters to include. Example: "0020-007e,0370-03ff"')
        entry.add_argument('--unifont-chars', type=str, default='', help='Ranges of characters from GNU unifont providers to include. Example: "0000-ffff"')
        entry.add_argument('--option-uniform', default=False, action=argparse.BooleanOptionalAction, help='Act as though the "Force Unicode Font" option was enabled')
        entry.add_argument('--option-jp', default=False, action=argparse.BooleanOptionalAction, help='Act as though the "Japanese Glyph Variants" option was enabled')
        entry.add_argument('--output', type=pathlib.Path, default=pathlib.Path('out'), help='Folder to save the generated fonts in')
    for entry in (vanilla_generate, vanilla_history, pack_generate, pack_list):
        entry.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('cache'), help='Folder for cache files')
    args = parser.parse_args()
    match args.command:
        case 'vanilla':
            identifiers = set(args.identifiers)
            match args.action:
                case 'generate':
                    main_vanilla_generate(args.version, identifiers, generator_options(args), args.cache)
                case 'history':
                    main_vanilla_history((args.start, args.end), identifiers, generator_options(args), args.cache)
        case 'pack':
            match args.action:
                case 'generate':
                    main_pack_generate(args.version, args.location, args.identifier, args.name, generator_options(args), args.cache)
                case 'list':
                    main_pack_list(args.version, args.location, args.cache)

@dataclasses.dataclass
class GeneratorOptions:
    provider: ProviderOptions
    styles: set[STYLE]
    output: pathlib.Path

def generator_options(args: argparse.Namespace) -> GeneratorOptions:
    styles = set(args.styles)
    color = {
        'never': lambda _: False,
        'always': lambda _: True,
        'auto': image_has_color
    }[args.color]
    all_range = char_range(args.chars)
    unifont_range = char_range(args.unifont_chars)
    provider = ProviderOptions(
        image_color_predicate = color,
        all_char_predicate = lambda x: char_in_range(x, all_range),
        unifont_char_predicate = lambda x: char_in_range(x, unifont_range),
        option_uniform = args.option_uniform,
        option_jp = args.option_jp,
    )
    return GeneratorOptions(provider, styles, args.output)

def char_in_range(char: str, range: list[tuple[int, int]]) -> bool:
    int_char = ord(char)
    for start, stop in range:
        if int_char >= start and int_char <= stop:
            return True
    return False

def char_range(input: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    parts = [] if input == '' else input.split(',')
    for part in parts:
        start, stop = part.split('-', maxsplit=1)
        result.append((int(start, 16), int(stop, 16)))
    return result

def main_vanilla_generate(version_id: str, identifiers: set[VANILLA_FONT_ID], options: GeneratorOptions, cache: pathlib.Path):
    info = jar_info(version_id, cache)
    if info is None:
        return
    print(f'Detected font capabilities: {info.version.name}')
    aglfn = get_aglfn(cache)
    print(f'Converting fonts from Minecraft {info.manifest['id']}')
    store = StackStorage([info.jar_storage, info.asset_storage])
    for identifier in identifiers:
        vanilla_try_font(identifier, info.version, store, options, aglfn)

def vanilla_try_font(
    identifier: VANILLA_FONT_ID,
    version: MinecraftVersion,
    store: Storage,
    options: GeneratorOptions,
    aglfn: dict[str, str],
):
    times = ModifiedTimes()
    provider_list = get_providers(version, store, identifier, options.provider, times)
    if provider_list is None:
        print(f'No providers found for {identifier}')
        return
    name, created_date = default_font_info(identifier)
    print(f'Converting {name}')
    family = create_font_family(provider_list, version.providers != ProviderSupport.NONE, options.styles)
    print_family_info(family)
    assert times.newest is not None
    for style, font in family.fonts.items():
        full_name = 'Minecraft ' + name
        ttf = finalize_font(full_name, style, font.chars, font.other_glyphs, family.font_em, created_date, times.newest, aglfn)
        ttf_name = full_name.replace(' ', '') + '-' + style_info(style)[0].replace(' ', '')
        dest = options.output / f'{ttf_name}.ttf'
        dest.parent.mkdir(parents=True, exist_ok=True)
        ttf.save(dest)

def main_vanilla_history(version_range: tuple[str | None, str | None], identifiers: set[VANILLA_FONT_ID], options: GeneratorOptions, cache: pathlib.Path):
    manifest = get_manifest(cache)
    aglfn = get_aglfn(cache)
    start, end = version_range
    unique_chars: dict[VANILLA_FONT_ID, list[dict[str, GlyphInfo]]] = {}
    reached_start = start is None
    for version_data in reversed(manifest['versions']):
        if not reached_start and start is not None and version_data['id'] == start:
            reached_start = True
        if not reached_start:
            continue
        info = version_from_data(version_data, cache)
        if info is None:
            continue
        store = StackStorage([info.jar_storage, info.asset_storage])
        for identifier in identifiers:
            times = ModifiedTimes()
            provider_list = get_providers(info.version, store, identifier, options.provider, times)
            if provider_list is None:
                continue
            name, created_date = default_font_info(identifier)
            assert times.newest is not None
            family = create_font_family(provider_list, info.version.providers != ProviderSupport.NONE, options.styles)
            if len(family.fonts) > 0:
                font = next(iter(family.fonts.values()))
                if identifier not in unique_chars:
                    report = GlyphChangeReport(list(font.chars.keys()), [], [])
                    unique_chars[identifier] = []
                else:
                    report = different_glyphs(unique_chars[identifier][-1], font.chars)
                if len(report.added) > 0 or len(report.removed) > 0 or len(report.changed) > 0:
                    unique_chars[identifier].append(font.chars)
                    print(f'{version_data['id']} ({info.version.name}): {identifier} changed')
                    if len(report.added) > 0:
                        print(f'\tadded {report_characters(report.added)}')
                    if len(report.removed) > 0:
                        print(f'\tremoved {report_characters(report.removed)}')
                    if len(report.changed) > 0:
                        print(f'\tchanged {report_characters(report.changed)}')
                    for style, font in family.fonts.items():
                        full_name = 'Minecraft ' + name
                        ttf = finalize_font(full_name, style, font.chars, font.other_glyphs, family.font_em, created_date, times.newest, aglfn)
                        dest = options.output / identifier.split(':')[1] / f'{len(unique_chars[identifier]):03d}-{version_data['id']}-{style}.ttf'
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        ttf.save(dest)
        if end is not None and version_data['id'] == end:
            break

def report_characters(chars: list[str]) -> str:
    if len(chars) > 60:
        return f'{len(chars)} characters'
    return f'{len(chars)} characters ({''.join(chars)!r})'

@dataclasses.dataclass
class GlyphChangeReport:
    added: list[str]
    removed: list[str]
    changed: list[str]

def different_glyphs(old: dict[str, GlyphInfo], new: dict[str, GlyphInfo]) -> GlyphChangeReport:
    added = [x for x in new if x not in old]
    removed = [x for x in old if x not in new]
    changed: list[str] = []
    for char in [x for x in old if x in new]:
        old_glyph = old[char]
        new_glyph = new[char]
        if old_glyph != new_glyph:
            changed.append(char)
    return GlyphChangeReport(added, removed, changed)

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
    if version.providers != ProviderSupport.NONE:
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

def main_pack_generate(version_id: str, location: pathlib.Path, identifier: str, name: str, options: GeneratorOptions, cache: pathlib.Path):
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
    times = ModifiedTimes()
    provider_list = get_providers(info.version, store, identifier, options.provider, times)
    if provider_list is None:
        print(f'No font with ID {identifier} in jar or resource pack')
        return
    family = create_font_family(provider_list, info.version.providers != ProviderSupport.NONE, options.styles)
    print_family_info(family)
    assert times.oldest is not None
    assert times.newest is not None
    for style, font in family.fonts.items():
        ttf = finalize_font(name, style, font.chars, font.other_glyphs, family.font_em, times.oldest, times.newest, aglfn)
        ttf_name = name.replace(' ', '') + '-' + style_info(style)[0].replace(' ', '')
        dest = options.output / f'{ttf_name}.ttf'
        dest.parent.mkdir(parents=True, exist_ok=True)
        ttf.save(dest)

def print_family_info(family: FontFamilyInfo):
    if len(family.colored) > 0:
        print(f'\t{report_characters(family.colored)} have color')
    point_sizes: dict[int, list[str]] = {}
    for (num, denom), chars in family.scales.items():
        top = num * 12
        gcd = math.gcd(top, denom)
        point_size = top // gcd
        if point_size not in point_sizes:
            point_sizes[point_size] = []
        point_sizes[point_size].extend(chars)
    for point, chars in point_sizes.items():
        print(f'\t{report_characters(chars)} will look pixel-perfect at font size multiples of {point}px')
    lcm = math.lcm(*point_sizes.keys())
    print(f'\tAll characters in this font will look pixel-perfect at font size multiples of {lcm}px')

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
    # sometimes there are multiple fully transparent colors, so look for more than just 2
    colors = image.getcolors(maxcolors = 5)
    if colors is None or len(colors) >= 5:
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
