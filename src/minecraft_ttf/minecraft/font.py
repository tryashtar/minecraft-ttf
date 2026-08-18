import dataclasses
import datetime
import typing

import fontTools.fontBuilder
import fontTools.ttLib.tables._g_l_y_f

from minecraft_ttf.bitmap import Bitmap, bitmap_from_image
from minecraft_ttf.font import CharInfo, FontInfo, FontPositions, make_font, vectorize
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
    # font textures have a color depth of 1 bit, so they are just 2D bitmasks
    # this lets us leverage some efficient operations provided by pygame
    def add_space_glyph(char: str, width: int):
        seen_chars.add(char)
        if 'regular' in styles:
            fonts['regular'][char] = CharInfo(width = width * pixel_scale, height = 0, path = None)
        if 'italic' in styles:
            fonts['italic'][char] = CharInfo(width = width * pixel_scale, height = 0, path = None)
        if 'bold' in styles:
            fonts['bold'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, path = None)
        if 'bold_italic' in styles:
            fonts['bold_italic'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, path = None)
    def add_bitmap_glyph(char: str, mask: Bitmap, height: int, ascent: int, char_size: tuple[int, int] | None):
        m_width, m_height = mask.get_size()
        seen_chars.add(char)
        # bold characters are created by overlapping two copies of the texture
        bold_mask = Bitmap((m_width + 1, m_height))
        bold_mask.draw(mask, (0, 0))
        bold_mask.draw(mask, (1, 0))
        scale = height / m_height * pixel_scale
        step_size = (0, 0) if height == 0 else (0, (height - ascent) / height * m_height)
        italic_step_size = (0, 0) if height == 0 else (-6 / height, (height - ascent) / height * m_height)
        add_width = 0 if height == 0 else m_height / height
        def make_char_info(path: fontTools.ttLib.tables._g_l_y_f.Glyph | None, size: tuple[int, int]) -> CharInfo:
            w, h = size
            if char_size is not None:
                l, r = char_size
                w = r - l + 1
                if w % 2 == 1:
                    w -= 1
            chw = w + add_width
            return CharInfo(width = chw * scale, height = h * scale, path = path)
        if 'regular' in styles:
            path, size = vectorize(mask, scale, step_size)
            fonts['regular'][char] = make_char_info(path, size)
        if 'italic' in styles:
            path, size = vectorize(mask, scale, italic_step_size, italic=True)
            fonts['italic'][char] = make_char_info(path, size)
        if 'bold' in styles:
            path, size = vectorize(bold_mask, scale, step_size)
            fonts['bold'][char] = make_char_info(path, size)
        if 'bold_italic' in styles:
            path, size = vectorize(bold_mask, scale, italic_step_size, italic=True)
            fonts['bold_italic'][char] = make_char_info(path, size)
    mw, mh = (5, 8)
    missing = Bitmap((mw, mh))
    if has_missing_glyph:
        for y in range(mh):
            for x in range(mw):
                if x == 0 or y == 0 or x == mw - 1 or y == mh - 1:
                    missing.set_at((x, y), True)
    add_bitmap_glyph('.notdef', missing, height=8, ascent=8, char_size=None)
    for provider in provider_list:
        if provider.modified_date is not None:
            modified_date = max(modified_date, provider.modified_date)
        if isinstance(provider, SpaceProvider):
            for char, width in provider.spaces.items():
                if char in seen_chars:
                    continue
                add_space_glyph(char, max(0, width))
        elif isinstance(provider, BitmapProvider):
            for char, (bitmap, size) in provider.chars.items():
                if char in seen_chars:
                    continue
                add_bitmap_glyph(char, bitmap, height=provider.height, ascent=provider.ascent, char_size=size)
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
                        if provider.sizes is not None:
                            size = provider.sizes[char]
                            left, right = size
                            gx1 += left
                            gx2 = gx1 + right + 1
                        else:
                            size = None
                        dimensions = (gx1, gy1, gx2, gy2)
                        glyph = provider.image.crop(dimensions).convert('RGBA')
                        mask = bitmap_from_image(glyph)
                        add_bitmap_glyph(char, mask, height=provider.height, ascent=provider.ascent, char_size=size)
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
