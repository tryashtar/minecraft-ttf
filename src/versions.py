import dataclasses
import json
import zipfile


@dataclasses.dataclass
class MinecraftVersion:
    name: str
    supports_providers: bool
    hardcoded_chars: list[str] | None
    lookup_chars: bool
    hardcoded_spaces: dict[str, int] | None
    entry_map: dict[str, str] | None

def detect_version(jar: zipfile.ZipFile) -> MinecraftVersion | None:
    names = jar.namelist()
    rp: int | None = None
    if 'version.json' in names:
        text = jar.read('version.json')
        version_data = json.loads(text)
    else:
        version_data = {}
    if 'pack_version' in version_data:
        rp = version_data['pack_version'].get('resource')
        if rp is None:
            rp = version_data['pack_version'].get('resource_major')
    if rp is not None and rp >= 9:
        return MinecraftVersion(
            name='22w11a+',
            supports_providers=True,
            hardcoded_chars=None,
            lookup_chars=False,
            hardcoded_spaces=None,
            entry_map=None,
        )
    if rp is not None and rp >= 8 and 'world_version' in version_data and version_data['world_version'] >= 2966:
        return MinecraftVersion(
            name='22w03a+',
            supports_providers=True,
            hardcoded_chars=None,
            lookup_chars=False,
            hardcoded_spaces={' ': 4, '\u200c': 0},
            entry_map=None,
        )
    simple_spaces = {' ': 4}
    if 'assets/minecraft/font/default.json' in names:
        return MinecraftVersion(
            name='1.13-pre7+',
            supports_providers=True,
            hardcoded_chars=None,
            lookup_chars=False,
            hardcoded_spaces=simple_spaces,
            entry_map=None,
        )
    asset_map = {
       'minecraft:default': 'assets/minecraft/textures/font/ascii.png',
       'minecraft:alt': 'assets/minecraft/textures/font/ascii_sga.png'
    }
    if 'font.txt' not in names and 'assets/minecraft/textures/font/ascii.png' in names:
        # this wrongly includes 13w42a (which removed font.txt)
        # but no easy way to distinguish the two versions
        return MinecraftVersion(
            name='13w42b+',
            supports_providers=False,
            hardcoded_chars=[
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
            lookup_chars=False,
            hardcoded_spaces=simple_spaces,
            entry_map=asset_map
        )
    if 'pack.mcmeta' in names:
        return MinecraftVersion(
            name='13w24a+',
            supports_providers=False,
            hardcoded_chars=None,
            lookup_chars=True,
            hardcoded_spaces=simple_spaces,
            entry_map=asset_map
        )
    if 'font.txt' in names:
        return MinecraftVersion(
            name='b1.1+',
            supports_providers=False,
            hardcoded_chars=None,
            lookup_chars=True,
            hardcoded_spaces=simple_spaces,
            entry_map={'minecraft:default': 'font/default.png', 'minecraft:alt': 'font/alternate.png'}
        )
    empty = '\u0000' * 16
    if 'lang/en_US.lang' in names:
        return MinecraftVersion(
            name='b1.0+',
            supports_providers=False,
            # this list of characters makes no sense
            hardcoded_chars=[
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
                empty,
                empty
            ],
            lookup_chars=False,
            hardcoded_spaces=simple_spaces,
            entry_map={'minecraft:default': 'font/default.png'}
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
            name='a1.2.2+',
            supports_providers=False,
            hardcoded_chars=alpha_chars,
            lookup_chars=False,
            hardcoded_spaces=simple_spaces,
            entry_map={'minecraft:default': 'font/default.png'}
        )
    if 'mob/cow.png' in names:
        # this wrongly includes a1.0.8 (which added cow.png)
        # but no easy way to distinguish the two versions
        return MinecraftVersion(
            name='a1.0.9+',
            supports_providers=False,
            hardcoded_chars=alpha_chars,
            lookup_chars=False,
            hardcoded_spaces=simple_spaces,
            entry_map={'minecraft:default': 'default.png'}
        )
    full_chars = [''.join(chr(x) for x in range(y * 16, (y + 1) * 16)) for y in range(16)]
    if 'default.png' in names:
        return MinecraftVersion(
            name='c0.0.17a+',
            supports_providers=False,
            hardcoded_chars=full_chars,
            lookup_chars=False,
            hardcoded_spaces=simple_spaces,
            entry_map={'minecraft:default': 'default.png'}
        )
    if 'default.gif' in names:
        return MinecraftVersion(
            name='c0.0.11a+',
            supports_providers=False,
            hardcoded_chars=full_chars,
            lookup_chars=False,
            hardcoded_spaces=simple_spaces,
            entry_map={'minecraft:default': 'default.gif'}
        )
    return None
