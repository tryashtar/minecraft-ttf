import dataclasses
import json
import pathlib
import typing
import zipfile

from minecraft_ttf.minecraft.providers import (
    CharImage,
    ImageProvider,
    ModifiedTimes,
    Provider,
    ProviderOptions,
    ProviderSupport,
    ReadEntry,
    SpaceProvider,
    filter_nul,
    image_grid,
    legacy_unicode,
    load_providers,
    normal_advance,
    read_image,
)
from minecraft_ttf.minecraft.storage import Storage

# throughout Minecraft's history, the implementation of fonts has changed
# if we want to generate fonts from older versions, we need to detect how that version handled fonts, and adapt that to modern providers
VANILLA_FONT_ID = typing.Literal['minecraft:default', 'minecraft:alt', 'minecraft:illageralt']

@dataclasses.dataclass
class MinecraftVersion:
    name: str
    providers: ProviderSupport
    asset_mount: pathlib.PurePath
    hardcoded_chars: list[str] | None
    lookup_chars: pathlib.PurePath | None
    hardcoded_spaces: dict[str, float] | None
    hardcoded_unifont: tuple[str, pathlib.PurePath] | None
    uneven_unifont: bool
    entry_map: dict[VANILLA_FONT_ID, pathlib.PurePath] | None

def detect_version(jar: zipfile.ZipFile) -> MinecraftVersion | None:
    names = jar.namelist()
    rp: int | None = None
    if 'version.json' in names:
        text = jar.read('version.json')
        version_data = json.loads(text)
    else:
        version_data = {}
    if 'pack_version' in version_data:
        pack_version = version_data['pack_version']
        if isinstance(pack_version, int):
            rp = pack_version
        else:
            rp = version_data['pack_version'].get('resource')
            if rp is None:
                rp = version_data['pack_version'].get('resource_major')
    if 'world_version' in version_data:
        world_version = version_data['world_version']
    else:
        world_version = None
    if rp is not None and rp >= 89:
        return MinecraftVersion(
            name = '26.3-snapshot-1+',
            asset_mount = pathlib.PurePath('assets'),
            providers = ProviderSupport.FULL,
            hardcoded_chars = None,
            lookup_chars = None,
            hardcoded_spaces = None,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = None,
        )
    if rp is not None and rp >= 88 and world_version is not None and world_version >= 4896:
        return MinecraftVersion(
            name = '26.2-pre-3+',
            asset_mount = pathlib.PurePath('assets'),
            providers = ProviderSupport.FULL,
            hardcoded_chars = None,
            lookup_chars = None,
            hardcoded_spaces = None,
            hardcoded_unifont = None,
            uneven_unifont = False,
            entry_map = None,
        )
    if rp is not None and rp >= 26:
        return MinecraftVersion(
            name = '24w06a+',
            asset_mount = pathlib.PurePath('assets'),
            providers = ProviderSupport.FULL,
            hardcoded_chars = None,
            lookup_chars = None,
            hardcoded_spaces = None,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = None,
        )
    if rp is not None and rp >= 9:
        return MinecraftVersion(
            name = '22w11a+',
            asset_mount = pathlib.PurePath('assets'),
            providers = ProviderSupport.SWITCH_FONT_WHEN_FORCED,
            hardcoded_chars = None,
            lookup_chars = None,
            hardcoded_spaces = None,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = None,
        )
    if rp is not None and rp >= 8 and world_version is not None and world_version >= 2966:
        return MinecraftVersion(
            name = '22w03a+',
            asset_mount = pathlib.PurePath('assets'),
            providers = ProviderSupport.SWITCH_FONT_WHEN_FORCED,
            hardcoded_chars = None,
            lookup_chars = None,
            hardcoded_spaces = {' ': 4.0, '\u200c': 0.0},
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = None,
        )
    if rp is not None and rp >= 5 and world_version is not None and world_version >= 2529:
        return MinecraftVersion(
            name = '20w17a+',
            asset_mount = pathlib.PurePath('assets'),
            providers = ProviderSupport.SWITCH_FONT_WHEN_FORCED,
            hardcoded_chars = None,
            lookup_chars = None,
            hardcoded_spaces = {' ': 4.0, '\u200c': 0.0},
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = None,
        )
    simple_spaces = {' ': 4.0}
    if 'assets/minecraft/font/default.json' in names:
        return MinecraftVersion(
            name = '1.13-pre7+',
            asset_mount = pathlib.PurePath('assets'),
            providers = ProviderSupport.ONLY_UNICODE_WHEN_FORCED,
            hardcoded_chars = None,
            lookup_chars = None,
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = None,
        )
    asset_map: dict[VANILLA_FONT_ID, pathlib.PurePath] = {
        'minecraft:default': pathlib.PurePath('assets/minecraft/textures/font/ascii.png'),
        'minecraft:alt': pathlib.PurePath('assets/minecraft/textures/font/ascii_sga.png'),
    }
    new_unifont = ('assets/minecraft/textures/font/unicode_page_%x.png', pathlib.PurePath('assets/minecraft/font/glyph_sizes.bin'))
    if 'font.txt' not in names and 'assets/minecraft/textures/font/ascii.png' in names:
        # this wrongly includes 13w42a (which removed font.txt)
        # but no easy way to distinguish the two versions
        # it's a shame because 13w42a has unique behavior
        return MinecraftVersion(
            name = '13w42b+',
            asset_mount = pathlib.PurePath('assets'),
            providers = ProviderSupport.NONE,
            hardcoded_chars = [
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
            ],
            lookup_chars = None,
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = new_unifont,
            uneven_unifont = True,
            entry_map = asset_map
        )
    if 'assets/minecraft/textures/font/ascii.png' in names:
        return MinecraftVersion(
            name = '13w24a+',
            asset_mount = pathlib.PurePath('assets'),
            providers = ProviderSupport.NONE,
            hardcoded_chars = None,
            lookup_chars = pathlib.PurePath('font.txt'),
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = new_unifont,
            uneven_unifont = True,
            entry_map = asset_map
        )
    old_map: dict[VANILLA_FONT_ID, pathlib.PurePath] = {
        'minecraft:default': pathlib.PurePath('font/default.png'), 'minecraft:alt': pathlib.PurePath('font/alternate.png')
    }
    if 'font/glyph_sizes.bin' in names:
        return MinecraftVersion(
            name = '11w49a+',
            asset_mount = pathlib.PurePath(),
            providers = ProviderSupport.NONE,
            hardcoded_chars = None,
            lookup_chars = pathlib.PurePath('font.txt'),
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = ('font/glyph_%X.png', pathlib.PurePath('font/glyph_sizes.bin')),
            uneven_unifont = True,
            entry_map = old_map
        )
    if 'font/alternate.png' in names:
        return MinecraftVersion(
            name = 'b1.9-pre3+',
            asset_mount = pathlib.PurePath(),
            providers = ProviderSupport.NONE,
            hardcoded_chars = None,
            lookup_chars = pathlib.PurePath('font.txt'),
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = old_map
        )
    simple_map: dict[VANILLA_FONT_ID, pathlib.PurePath] = {
        'minecraft:default': pathlib.PurePath('font/default.png'),
    }
    if 'font.txt' in names:
        return MinecraftVersion(
            name = 'b1.1+',
            asset_mount = pathlib.PurePath(),
            providers = ProviderSupport.NONE,
            hardcoded_chars = None,
            lookup_chars = pathlib.PurePath('font.txt'),
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = simple_map
        )
    empty = '\u0000' * 16
    if 'lang/en_US.lang' in names:
        return MinecraftVersion(
            name = 'b1.0+',
            asset_mount = pathlib.PurePath(),
            providers = ProviderSupport.NONE,
            # this list of characters makes no sense
            hardcoded_chars = [
                empty,
                empty,
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
                '\u00c2\u00bb' + ('\u0000' * 14),
                empty,
            ],
            lookup_chars = None,
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = simple_map
        )
    alpha_chars = [
        empty,
        empty,
        ' !"#$%&\'()*+,-./',
        '0123456789:;<=>?',
        '@ABCDEFGHIJKLMNO',
        'PQRSTUVWXYZ[\\]^_',
        '\'abcdefghijklmno',
        'pqrstuvwxyz{|}~\u2302',
        '\u00c7\u00fc\u00e9\u00e2\u00e4\u00e0\u00e5\u00e7\u00ea\u00eb\u00e8\u00ef\u00ee\u00ec\u00c4\u00c5',
        '\u00c9\u00e6\u00c6\u00f4\u00f6\u00f2\u00fb\u00f9\u00ff\u00d6\u00dc\u00f8\u00a3\u00d8\u00d7\u0192',
        '\u00e1\u00ed\u00f3\u00fa\u00f1\u00d1\u00aa\u00ba\u00bf\u00ae\u00ac\u00bd\u00bc\u00a1\u00ab\u00bb',
        empty,
        empty,
        empty,
        empty,
        empty,
    ]
    if 'font/default.png' in names:
        return MinecraftVersion(
            name = 'a1.2.2+',
            asset_mount = pathlib.PurePath(),
            providers = ProviderSupport.NONE,
            hardcoded_chars = alpha_chars,
            lookup_chars = None,
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = simple_map
        )
    if 'mob/cow.png' in names:
        # this wrongly includes a1.0.8 (which added cow.png)
        # but no easy way to distinguish the two versions
        return MinecraftVersion(
            name = 'a1.0.9+',
            asset_mount = pathlib.PurePath(),
            providers = ProviderSupport.NONE,
            hardcoded_chars = alpha_chars,
            lookup_chars = None,
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = {'minecraft:default': pathlib.PurePath('default.png')}
        )
    full_chars = [''.join(chr(x) for x in range(y * 16, (y + 1) * 16)) for y in range(16)]
    if 'default.png' in names:
        return MinecraftVersion(
            name = 'c0.0.17a+',
            asset_mount = pathlib.PurePath(),
            providers = ProviderSupport.NONE,
            hardcoded_chars = full_chars,
            lookup_chars = None,
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = {'minecraft:default': pathlib.PurePath('default.png')}
        )
    if 'default.gif' in names:
        return MinecraftVersion(
            name = 'c0.0.11a+',
            asset_mount = pathlib.PurePath(),
            providers = ProviderSupport.NONE,
            hardcoded_chars = full_chars,
            lookup_chars = None,
            hardcoded_spaces = simple_spaces,
            hardcoded_unifont = None,
            uneven_unifont = True,
            entry_map = {'minecraft:default': pathlib.PurePath('default.gif')}
        )
    return None

def get_providers(version: MinecraftVersion, store: Storage, identifier: str, options: ProviderOptions, times: ModifiedTimes) -> list[Provider] | None:
    providers: list[Provider] = []
    loaded = load_providers(identifier, store, options, version.providers, version.uneven_unifont, times)
    if loaded is None and version.providers != ProviderSupport.NONE:
        return None
    if loaded is not None:
        providers.extend(loaded)
    if version.entry_map is not None:
        if identifier not in version.entry_map:
            return None
        if not (identifier == 'minecraft:default' and options.option_uniform):
            img_entry = version.entry_map[identifier]
            img_data = read_image(store, img_entry)
            times.update(img_data.modified_date)
            if version.hardcoded_chars is not None:
                char_grid = version.hardcoded_chars
            else:
                assert version.lookup_chars is not None
                font_data = read_font_txt(store, version.lookup_chars)
                times.update(font_data.modified_date)
                empty = '\u0000' * 16
                char_grid: list[str] = [
                    empty,
                    empty,
                    *font_data.data,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty
                ]
            chars = image_grid(img_data.data, filter_nul(char_grid), None, options.all_char_predicate)
            bitmap = ImageProvider(
                height = 8,
                ascent = 7,
                has_color = options.image_color_predicate(img_data.data),
                chars = {x: CharImage(img, bm, normal_advance(bm, 8), 1) for x, (img, bm) in chars.items()},
            )
            providers.append(bitmap)
    if version.hardcoded_spaces is not None:
        providers.insert(0, SpaceProvider({x: y for x, y in version.hardcoded_spaces.items() if options.all_char_predicate(x)}))
    if version.hardcoded_unifont is not None:
        sheet_template, size_entry = version.hardcoded_unifont
        size_date = store.modified_time(size_entry)
        times.update(size_date)
        size_bytes = store.read(size_entry)
        sheet_entries = [pathlib.PurePath(sheet_template.replace('%x', f'{sheet_id:02x}').replace('%X', f'{sheet_id:02X}')) for sheet_id in range(0xff + 1)]
        converted = legacy_unicode(store, sheet_entries, size_bytes, options, version.uneven_unifont, times)
        providers.extend(converted)
    return providers

def read_font_txt(store: Storage, entry: pathlib.PurePath) -> ReadEntry[list[str]]:
    text = store.read(entry).decode('utf-8')
    lines: list[str] = [x for x in text.splitlines() if not x.startswith('#')]
    return ReadEntry(lines, store.modified_time(entry))
