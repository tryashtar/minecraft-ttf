import dataclasses
import datetime

import fontTools.fontBuilder
import fontTools.pens.transformPen
import fontTools.pens.ttGlyphPen
import fontTools.ttLib.tables._g_l_y_f
import fontTools.ttLib.tables.C_P_A_L_


@dataclasses.dataclass
class ColoredLayer:
    path: fontTools.pens.ttGlyphPen.Glyph
    color: fontTools.ttLib.tables.C_P_A_L_.Color

    def __eq__(self, other) -> bool:
        if not isinstance(other, ColoredLayer):
            return False
        if not path_eq(self.path, other.path):
            return False
        rgba1 = self.color.red, self.color.green, self.color.blue, self.color.alpha
        rgba2 = other.color.red, other.color.green, other.color.blue, other.color.alpha
        return rgba1 == rgba2

@dataclasses.dataclass
class GlyphInfo:
    width: float
    height: float
    base_layer: fontTools.pens.ttGlyphPen.Glyph | None
    colored_layers: list[ColoredLayer]
    base_offsets: list[tuple[float, float]]
    color_offsets: list[tuple[float, float]]

    def __eq__(self, other) -> bool:
        if not isinstance(other, GlyphInfo):
            return False
        if self.height != other.height:
            return False
        if self.width != other.width:
            return False
        if self.base_layer is None and other.base_layer is not None:
            return False
        if self.base_layer is not None and other.base_layer is None:
            return False
        if self.base_layer is not None and other.base_layer is not None and not path_eq(self.base_layer, other.base_layer):
            return False
        if self.base_offsets != other.base_offsets:
            return False
        if self.color_offsets != other.color_offsets:
            return False
        if len(self.colored_layers) != len(other.colored_layers):
            return False
        for (c1, c2) in zip(self.colored_layers, other.colored_layers):
            if c1 != c2:
                return False
        return True

def empty_glyph(width: float) -> GlyphInfo:
    return GlyphInfo(width, 0, None, [], [(0, 0)], [(0, 0)])

def path_eq(p1: fontTools.pens.ttGlyphPen.Glyph, p2: fontTools.pens.ttGlyphPen.Glyph) -> bool:
    return p1.coordinates.array.tobytes() == p2.coordinates.array.tobytes()

@dataclasses.dataclass
class FontPositions:
    ascent: float
    descent: float
    sCapHeight: float
    sxHeight: float
    yStrikeoutPosition: float
    yStrikeoutSize: float
    underlinePosition: float
    underlineThickness: float
    italicAngle: float

@dataclasses.dataclass
class FontInfo:
    name: str
    style: str
    bold: bool
    italic: bool
    copyright: str
    sample: str
    version: str
    em: int
    created: datetime.datetime
    modified: datetime.datetime

def make_font(info: FontInfo, positions: FontPositions, char_data: dict[str, GlyphInfo], other_glyphs: dict[str, GlyphInfo], aglfn: dict[str, str]) -> fontTools.fontBuilder.FontBuilder:
    nameStrings = {
        'copyright': info.copyright,
        'familyName': info.name,
        'styleName': info.style,
        'uniqueFontIdentifier': info.name.replace(' ', '') + '.' + info.style.replace(' ', ''),
        'fullName': info.name + ' ' + info.style,
        'version': info.version,
        'psName': info.name.replace(' ', '') + info.style.replace(' ', ''),
        'sampleText': info.sample
    }
    empty_glyph = fontTools.ttLib.tables._g_l_y_f.Glyph()
    codepoints: dict[int, str] = {}
    glyph_widths: dict[str, float] = {'.notdef': 0, '.null': 0}
    glyph_paths: dict[str, fontTools.pens.ttGlyphPen.Glyph] = {'.notdef': empty_glyph, '.null': empty_glyph}
    color_palettes: list[fontTools.ttLib.tables.C_P_A_L_.Color] = []
    color_layers: dict[str, list[tuple[str, int]]] = {}
    def import_glyph(name: str, info: GlyphInfo):
        if info.base_offsets == [(0.0, 0.0)]:
            import_base_glyph(name, info.width, info.base_layer)
            import_colored_layers(name, info.width, info.colored_layers, info.color_offsets)
            return
        base_name = f'{name}.base'
        import_base_glyph(base_name, info.width, info.base_layer)
        import_colored_layers(name, info.width, info.colored_layers, info.color_offsets)
        pen = fontTools.pens.ttGlyphPen.TTGlyphPen(glyph_paths)
        for x, y in info.base_offsets:
            pen.addComponent(base_name, (1, 0, 0, 1, x, y))
        further_width = info.width + max(x for x, _ in info.base_offsets)
        glyph_paths[name] = pen.glyph()
        glyph_widths[name] = further_width
    def import_base_glyph(name: str, width: float, base_layer: fontTools.pens.ttGlyphPen.Glyph | None):
        glyph_widths[name] = width
        if base_layer is None:
            glyph_paths[name] = empty_glyph
        else:
            glyph_paths[name] = base_layer
    def import_colored_layers(name: str, width: float, colored_layers: list[ColoredLayer], offsets: list[tuple[float, float]]):
        for offset_index, (x, y) in enumerate(offsets):
            for layer_index, layer in enumerate(colored_layers):
                layer_name = f'{name}.layer{layer_index + 1}' if len(offsets) == 1 else f'{name}.layer{layer_index + 1}.{offset_index + 1}'
                pen = fontTools.pens.ttGlyphPen.TTGlyphPen(None)
                transform_pen = fontTools.pens.transformPen.TransformPen(pen, (1, 0, 0, 1, x, y))
                layer.path.draw(transform_pen, None)
                glyph_paths[layer_name] = pen.glyph()
                glyph_widths[layer_name] = width + x
                try:
                    color_index = color_palettes.index(layer.color)
                except ValueError:
                    color_index = len(color_palettes)
                    color_palettes.append(layer.color)
                if name not in color_layers:
                    color_layers[name] = []
                color_layers[name].append((layer_name, color_index))
    for char, data in char_data.items():
        char_int = ord(char)
        fallback_name = f'uni{char_int:04X}' if char_int <= 0xffff else f'u{char_int:X}'
        glyph_name = aglfn.get(char, fallback_name)
        codepoints[ord(char)] = glyph_name
        import_glyph(glyph_name, data)
    for name, data in other_glyphs.items():
        import_glyph(name, data)
    widest = max(x.width for x in char_data.values())
    tallest = max(x.height for x in char_data.values())
    font = fontTools.fontBuilder.FontBuilder(unitsPerEm = info.em, isTTF = True)
    font.setupGlyphOrder(list(glyph_paths.keys()))
    font.setupCharacterMap(codepoints)
    font.setupGlyf(glyph_paths)
    if len(color_palettes) > 0:
        font.setupCPAL([color_palettes])
        font.setupCOLR(color_layers)
    metrics = {}
    glyphTable = font.font['glyf']
    assert isinstance(glyphTable, fontTools.ttLib.tables._g_l_y_f.table__g_l_y_f)
    for gn, advanceWidth in glyph_widths.items():
        metrics[gn] = (advanceWidth, glyphTable[gn].xMin)
    font.setupHorizontalMetrics(metrics)
    ascent = int(info.em * positions.ascent)
    descent = int(info.em * positions.descent)
    font.setupHorizontalHeader(ascent = ascent, descent = -descent)
    font.setupNameTable(nameStrings)
    fs_selection = 0
    mac_style = 0
    weight = 400
    if info.bold:
        mac_style += 1
        fs_selection += 32
        weight = 700
    if info.italic:
        mac_style += 2
        fs_selection += 1
    if not info.bold and not info.italic:
        fs_selection += 64
    font.setupOS2(
        sTypoAscender = ascent,
        sTypoDescender = -descent,
        usWinAscent = ascent,
        usWinDescent = descent,
        sCapHeight = int(info.em * positions.sCapHeight),
        sxHeight = int(info.em * positions.sxHeight),
        yStrikeoutPosition = int(info.em * positions.yStrikeoutPosition),
        yStrikeoutSize = int(info.em * positions.yStrikeoutSize),
        sTypoLineGap = 0,
        fsSelection = fs_selection,
        achVendID = '',
        usWeightClass = weight
    )
    font.setupPost(
       underlinePosition = int(info.em * positions.underlinePosition),
       underlineThickness = int(info.em * positions.underlineThickness),
       italicAngle = positions.italicAngle if info.italic else 0
    )
    epoch = datetime.datetime.fromisoformat('1904-01-01T00:00:00Z')
    font.setupHead(
        unitsPerEm = info.em,
        xMin = 0,
        xMax = int(widest),
        yMin = -descent,
        yMax = int(tallest),
        created = int((info.created - epoch).total_seconds()),
        modified = int((info.modified - epoch).total_seconds()),
        macStyle = mac_style
    )
    return font
