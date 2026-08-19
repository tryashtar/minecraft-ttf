import dataclasses
import datetime
import typing

import fontTools.fontBuilder
import fontTools.ttLib.tables._g_l_y_f
import fontTools.ttLib.tables.C_P_A_L_

from minecraft_ttf.bitmap import Bitmap, bitmap_from_image, bitmaps_from_colors
from minecraft_ttf.font import (
    CharInfo,
    FontInfo,
    FontPositions,
    GlyphLayer,
    make_font,
    vectorize,
)
from minecraft_ttf.minecraft.providers import (
    BitmapProvider,
    ImageProvider,
    Provider,
    SpaceProvider,
)

STYLES = typing.Literal['regular', 'italic', 'bold', 'bold_italic']

def style_info(style: STYLES) -> tuple[str, bool, bool]:
    cache: dict[STYLES, tuple[str, bool, bool]] = {
        'regular': ('Regular', False, False),
        'italic': ('Italic', False, True),
        'bold': ('Bold', True, False),
        'bold_italic': ('Bold Italic', True, True),
    }
    return cache[style]

@dataclasses.dataclass
class CreatedFontInfo:
    fonts: dict[STYLES, dict[str, CharInfo]]
    font_em: int
    created_date: datetime.datetime
    modified_date: datetime.datetime

def create_fonts(
    provider_list: list[Provider],
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
    def add_space_glyph(char: str, width: int):
        seen_chars.add(char)
        if 'regular' in styles:
            fonts['regular'][char] = CharInfo(width = width * pixel_scale, height = 0, layers = [])
        if 'italic' in styles:
            fonts['italic'][char] = CharInfo(width = width * pixel_scale, height = 0, layers = [])
        if 'bold' in styles:
            fonts['bold'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, layers = [])
        if 'bold_italic' in styles:
            fonts['bold_italic'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, layers = [])
    def add_bitmap_glyph(char: str, layers: list[tuple[Bitmap, fontTools.ttLib.tables.C_P_A_L_.Color | None]], height: int, ascent: int):
        assert len(layers) > 0
        assert all(mask.get_size() == layers[0][0].get_size() for mask, _ in layers)
        m_width, m_height = layers[0][0].get_size()
        seen_chars.add(char)
        scale = height / m_height * pixel_scale
        step_size = (0, 0) if height == 0 else (0, (height - ascent) / height * m_height)
        italic_step_size = (0, 0) if height == 0 else (-6 / height, (height - ascent) / height * m_height)
        add_width = 0 if height == 0 else m_height / height
        def make_bold_mask(mask: Bitmap):
             bold_mask = Bitmap((m_width + 1, m_height))
             bold_mask.draw(mask, (0, 0))
             bold_mask.draw(mask, (1, 0))
             return bold_mask
        bold_layers = [(make_bold_mask(mask), color) for mask, color in layers]
        def make_char_info(paths: list[tuple[fontTools.ttLib.tables._g_l_y_f.Glyph | None, fontTools.ttLib.tables.C_P_A_L_.Color | None]]) -> CharInfo:
            final: list[GlyphLayer] = []
            for path, color in paths:
                if path is not None:
                    final.append(GlyphLayer(path, color))
            chw = m_width + add_width
            return CharInfo(width = chw * scale, height = m_height * scale, layers = final)
        if 'regular' in styles:
            paths = [(vectorize(mask, scale, step_size), color) for mask, color in layers]
            fonts['regular'][char] = make_char_info(paths)
        if 'italic' in styles:
            paths = [(vectorize(mask, scale, italic_step_size, italic=True), color) for mask, color in layers]
            fonts['italic'][char] = make_char_info(paths)
        if 'bold' in styles:
            paths = [(vectorize(mask, scale, step_size), color) for mask, color in bold_layers]
            fonts['bold'][char] = make_char_info(paths)
        if 'bold_italic' in styles:
            paths = [(vectorize(mask, scale, italic_step_size, italic=True), color) for mask, color in bold_layers]
            fonts['bold_italic'][char] = make_char_info(paths)
    mw, mh = (5, 8)
    missing = Bitmap((mw, mh))
    if has_missing_glyph:
        for y in range(mh):
            for x in range(mw):
                if x == 0 or y == 0 or x == mw - 1 or y == mh - 1:
                    missing.set_at((x, y), True)
    add_bitmap_glyph('.notdef', [(missing, None)], height=8, ascent=8)
    for provider in provider_list:
        if provider.modified_date is not None:
            modified_date = max(modified_date, provider.modified_date)
        if isinstance(provider, SpaceProvider):
            for char, width in provider.spaces.items():
                if char in seen_chars:
                    continue
                add_space_glyph(char, max(0, width))
        elif isinstance(provider, BitmapProvider):
            for char, bitmap in provider.chars.items():
                if char in seen_chars:
                    continue
                add_bitmap_glyph(char, [(bitmap, None)], height=provider.height, ascent=provider.ascent)
        elif isinstance(provider, ImageProvider):
            glyph_width = provider.image.width // len(provider.chars[0])
            glyph_height = provider.image.height // len(provider.chars)
            for y, row in enumerate(provider.chars):
                for x, char in enumerate(row):
                    if char is None:
                        continue
                    if char in seen_chars:
                        continue
                    if provider.ascent > -16384 and provider.height > 0:
                        gx1 = x * glyph_width
                        gy1 = y * glyph_height
                        gx2 = (x + 1) * glyph_width
                        gy2 = (y + 1) * glyph_height
                        dimensions = (gx1, gy1, gx2, gy2)
                        glyph = provider.image.crop(dimensions)
                        layers: list[tuple[Bitmap, fontTools.ttLib.tables.C_P_A_L_.Color | None]] = []
                        full_mask = bitmap_from_image(glyph)
                        if provider.has_color:
                            color_planes = bitmaps_from_colors(glyph)
                            layers.extend([(mask, fontTools.ttLib.tables.C_P_A_L_.Color(r / 255, g /255, b / 255, a / 255)) for mask, (r, g, b, a) in color_planes if a > 10])
                        else:
                            layers.append((full_mask, None))
                        if provider.sizes is not None:
                            left, right = provider.sizes[char]
                        else:
                            left = 0
                            box = full_mask.content_box()
                            if box is None:
                                right = 0
                            else:
                                _left, _top, right, _bottom = box
                        layers = [(mask.resized((left, 0, right, mask.height)), color) for mask, color in layers]
                        add_bitmap_glyph(char, layers, height=provider.height, ascent=provider.ascent)
                    else:
                        add_space_glyph(char, 0)
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
