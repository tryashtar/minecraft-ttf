import dataclasses
import datetime
import typing

import fontTools.fontBuilder
import fontTools.ttLib.tables.C_P_A_L_

from minecraft_ttf.bitmap import Bitmap, bitmaps_from_colors
from minecraft_ttf.font import (
    ColoredLayer,
    FontInfo,
    FontPositions,
    GlyphInfo,
    make_font,
    vectorize,
)
from minecraft_ttf.minecraft.providers import (
    BitmapProvider,
    ImageProvider,
    Provider,
    SpaceProvider,
)

STYLE = typing.Literal['regular', 'italic', 'bold', 'bold_italic']

def style_info(style: STYLE) -> tuple[str, bool, bool]:
    cache: dict[STYLE, tuple[str, bool, bool]] = {
        'regular': ('Regular', False, False),
        'italic': ('Italic', False, True),
        'bold': ('Bold', True, False),
        'bold_italic': ('Bold Italic', True, True),
    }
    return cache[style]

@dataclasses.dataclass
class FontStyleInfo:
    chars: dict[str, GlyphInfo]
    other_glyphs: dict[str, GlyphInfo]

@dataclasses.dataclass
class FontFamilyInfo:
    fonts: dict[STYLE, FontStyleInfo]
    scales: dict[tuple[int, int], list[str]]
    colored: list[str]
    font_em: int

type PICKER = typing.Callable[[FontStyleInfo], dict[str, GlyphInfo]]

def make_bold_mask(mask: Bitmap):
    bold_mask = Bitmap((mask.width + 1, mask.height))
    bold_mask.draw(mask, (0, 0))
    bold_mask.draw(mask, (1, 0))
    return bold_mask

def create_font_family(
    provider_list: list[Provider],
    has_missing_glyph: bool,
    styles: set[STYLE],
) -> FontFamilyInfo:
    seen_chars: set[str] = set()
    scales: dict[tuple[int, int], list[str]] = {}
    colored: list[str] = []
    fonts: dict[STYLE, FontStyleInfo] = {}
    for style in styles:
        fonts[style] = FontStyleInfo(chars={}, other_glyphs={})
    chatbox_height = 12
    font_em = 1200
    pixel_scale = font_em / chatbox_height
    def add_space_glyph(picker: PICKER, char: str, width: float):
        seen_chars.add(char)
        if 'regular' in styles:
            picker(fonts['regular'])[char] = GlyphInfo(width = width * pixel_scale, height = 0, base_layer = None, colored_layers = [])
        if 'italic' in styles:
            picker(fonts['italic'])[char] = GlyphInfo(width = width * pixel_scale, height = 0, base_layer = None, colored_layers = [])
        if 'bold' in styles:
            picker(fonts['bold'])[char] = GlyphInfo(width = (width + 1) * pixel_scale, height = 0, base_layer = None, colored_layers = [])
        if 'bold_italic' in styles:
            picker(fonts['bold_italic'])[char] = GlyphInfo(width = (width + 1) * pixel_scale, height = 0, base_layer = None, colored_layers = [])
    def add_bitmap_glyph(picker: PICKER, char: str, base_layer: Bitmap, colored_layers: list[tuple[Bitmap, fontTools.ttLib.tables.C_P_A_L_.Color]], advances: tuple[float, float], height: int, ascent: int):
        base_size = base_layer.get_size()
        assert all(mask.get_size() == base_size for mask, _ in colored_layers)
        m_height = base_layer.height
        seen_chars.add(char)
        height_ratio = (m_height, height)
        if height_ratio not in scales:
            scales[height_ratio] = []
        scales[height_ratio].append(char)
        step_size = (0, 0) if height == 0 else (0, (height - ascent) / height * m_height)
        italic_step_size = (0, 0) if height == 0 else (-6 / height, (height - ascent) / height * m_height)
        advance, bold_advance = advances
        char_width = advance * pixel_scale
        bold_width = bold_advance * pixel_scale
        char_height = height * pixel_scale
        bold_base_layer = make_bold_mask(base_layer)
        bold_colored_layers = [(make_bold_mask(mask), color) for mask, color in colored_layers]
        def make_char_info(bold: bool, italic: bool):
            scale = height / m_height * pixel_scale
            base_mask = bold_base_layer if bold else base_layer
            colored_masks = bold_colored_layers if bold else colored_layers
            step = italic_step_size if italic else step_size
            base_path = vectorize(base_mask, scale, step, italic=italic)
            colored_paths: list[ColoredLayer] = []
            for mask, color in colored_masks:
                path = vectorize(mask, scale, step, italic=italic)
                if path is not None:
                    colored_paths.append(ColoredLayer(path, color))
            return GlyphInfo(bold_width if bold else char_width, char_height, base_path, colored_paths)
        if 'regular' in styles:
            picker(fonts['regular'])[char] = make_char_info(bold=False, italic=False)
        if 'italic' in styles:
            picker(fonts['italic'])[char] = make_char_info(bold=False, italic=True)
        if 'bold' in styles:
            picker(fonts['bold'])[char] = make_char_info(bold=True, italic=False)
        if 'bold_italic' in styles:
            picker(fonts['bold_italic'])[char] = make_char_info(bold=True, italic=True)
    mw, mh = (5, 8)
    missing = Bitmap((mw, mh))
    if has_missing_glyph:
        for y in range(mh):
            for x in range(mw):
                if x == 0 or y == 0 or x == mw - 1 or y == mh - 1:
                    missing.set_at((x, y), True)
    add_bitmap_glyph(lambda x: x.other_glyphs, '.notdef', missing, [], advances=(6, 7), height=8, ascent=8)
    for provider in provider_list:
        if isinstance(provider, SpaceProvider):
            for char, width in provider.spaces.items():
                if char in seen_chars:
                    continue
                add_space_glyph(lambda x: x.chars, char, max(0, width))
        elif isinstance(provider, BitmapProvider):
            for char, info in provider.chars.items():
                if char in seen_chars:
                    continue
                add_bitmap_glyph(lambda x: x.chars, char, info.bitmap, [], advances=info.advances, height=provider.height, ascent=provider.ascent)
        elif isinstance(provider, ImageProvider):
            for char, info in provider.chars.items():
                if char in seen_chars:
                    continue
                if provider.ascent > -16384 and provider.height > 0:
                    colored_layers: list[tuple[Bitmap, fontTools.ttLib.tables.C_P_A_L_.Color]] = []
                    if provider.has_color:
                        colored.append(char)
                        color_planes = bitmaps_from_colors(info.image)
                        colored_layers.extend([(mask, fontTools.ttLib.tables.C_P_A_L_.Color(r / 255, g /255, b / 255, a / 255)) for mask, (r, g, b, a) in color_planes if a > 10])
                    add_bitmap_glyph(lambda x: x.chars, char, info.bitmap, colored_layers, advances=info.advances, height=provider.height, ascent=provider.ascent)
                else:
                    add_space_glyph(lambda x: x.chars, char, 0)
    return FontFamilyInfo(fonts, scales, colored, font_em)

def finalize_font(
    full_name: str,
    style: STYLE,
    chars: dict[str, GlyphInfo],
    other_glyphs: dict[str, GlyphInfo],
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
    result = make_font(info, positions, chars, other_glyphs, aglfn)
    return result
