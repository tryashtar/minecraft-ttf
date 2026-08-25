import dataclasses
import datetime
import enum

import fontTools.fontBuilder
import fontTools.pens.ttGlyphPen
import fontTools.ttLib.tables._g_l_y_f
import fontTools.ttLib.tables.C_P_A_L_

from minecraft_ttf.bitmap import Bitmap, BitmapLabels


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
    offsets: list[tuple[float, float]]

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
        if self.offsets != other.offsets:
            return False
        if len(self.colored_layers) != len(other.colored_layers):
            return False
        for (c1, c2) in zip(self.colored_layers, other.colored_layers):
            if c1 != c2:
                return False
        return True

def empty_glyph(width: float) -> GlyphInfo:
    return GlyphInfo(width, 0, None, [], [(0, 0)])

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
        if info.offsets == [(0.0, 0.0)]:
            import_base_glyph(name, info.width, info.base_layer, info.colored_layers)
            return
        base_name = f'{name}.base'
        import_base_glyph(base_name, info.width, info.base_layer, info.colored_layers)
        pen = fontTools.pens.ttGlyphPen.TTGlyphPen(glyph_paths)
        for x, y in info.offsets:
            pen.addComponent(base_name, (1, 0, 0, 1, x, y))
        glyph_widths[name] = info.width + max(x for x, _ in info.offsets)
        glyph_paths[name] = pen.glyph()
    def import_base_glyph(name: str, width: float, base_layer: fontTools.pens.ttGlyphPen.Glyph | None, colored_layers: list[ColoredLayer]):
        glyph_widths[name] = width
        if base_layer is None:
            glyph_paths[name] = empty_glyph
        else:
            glyph_paths[name] = base_layer
        for i, layer in enumerate(colored_layers):
            layer_name = f'{name}.layer{i + 1}'
            glyph_paths[layer_name] = layer.path
            glyph_widths[layer_name] = width
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

def start_point(mask: Bitmap) -> tuple[int, int]:
    w, h = mask.get_size()
    for y in range(h):
        for x in range(w):
            if mask.get_at((x, y)):
                return (x, y)
    raise ValueError(mask)

def is_set(mask: Bitmap, point: tuple[int, int]) -> bool:
    x, y = point
    if x < 0 or y < 0:
        return False
    w, h = mask.get_size()
    if x >= w or y >= h:
        return False
    return mask.get_at(point)

class OutlineDirection(enum.Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    def add(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        match self:
            case OutlineDirection.RIGHT:
                return (x + 1, y)
            case OutlineDirection.UP:
                return (x, y - 1)
            case OutlineDirection.LEFT:
                return (x - 1, y)
            case OutlineDirection.DOWN:
                return (x, y + 1)

# trace the outline of a connected mask
# returns a list of all corner points that were visited, in order
def outline(mask: Bitmap) -> list[tuple[int, int]]:
    start = start_point(mask)
    facing = OutlineDirection.RIGHT
    pos = start
    result = [pos]
    while True:
        x, y = pos
        top_left = is_set(mask, (x - 1, y - 1))
        top_right = is_set(mask, (x, y - 1))
        bottom_left = is_set(mask, (x - 1, y))
        bottom_right = is_set(mask, (x, y))
        facing = outline_turn((top_left, top_right, bottom_left, bottom_right), facing)
        pos = facing.add(pos)
        result.append(pos)
        if pos == start:
            break
    return result

def outline_turn(corners: tuple[bool, bool, bool, bool], facing: OutlineDirection) -> OutlineDirection:
    top_left, top_right, bottom_left, bottom_right = corners
    if top_left and bottom_right and not top_right and not bottom_left:
        if facing == OutlineDirection.UP:
            return OutlineDirection.LEFT
        return OutlineDirection.RIGHT
    if top_right and bottom_left and not top_left and not bottom_right:
        if facing == OutlineDirection.RIGHT:
            return OutlineDirection.UP
        return OutlineDirection.DOWN
    if top_left and not bottom_left:
        return OutlineDirection.LEFT
    if top_right and not top_left:
        return OutlineDirection.UP
    if bottom_right and not top_right:
        return OutlineDirection.RIGHT
    if bottom_left and not bottom_right:
        return OutlineDirection.DOWN
    raise ValueError(corners)

def neighbor_connected(mask: Bitmap) -> list[Bitmap]:
    w, h = mask.get_size()
    pixels_checked: set[tuple[int, int]] = set()
    result = []
    for y in range(h):
        for x in range(w):
            pos = (x, y)
            if pos in pixels_checked:
                continue
            if not mask.get_at(pos):
                pixels_checked.add(pos)
                continue
            region = Bitmap((w, h))
            pixel_queue: list[tuple[int, int]] = [pos]
            while len(pixel_queue) > 0:
                pixel = pixel_queue.pop()
                px, py = pixel
                if pixel in pixels_checked:
                    continue
                if not is_set(mask, pixel):
                    pixels_checked.add(pixel)
                    continue
                pixels_checked.add(pixel)
                region.set_at(pixel, True)
                pixel_queue.append((px - 1, py))
                pixel_queue.append((px + 1, py))
                pixel_queue.append((px, py - 1))
                pixel_queue.append((px, py + 1))
            result.append(region)
            pixels_checked.add(pos)
    return result

def separate_regions(mask: Bitmap, labels: BitmapLabels) -> tuple[list[Bitmap], list[Bitmap]]:
    filled = labels.connected_components()
    w, h = mask.get_size()
    inverted = Bitmap((w + 2, h + 2))
    inverted.draw(mask, (1, 1))
    inverted.invert()
    big_unfilled = neighbor_connected(inverted)
    unfilled = []
    for big in big_unfilled[1:]:
        fixed = Bitmap((w, h))
        fixed.draw(big, (-1, -1))
        unfilled.append(fixed)
    return (filled, unfilled)

def collinear(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]) -> bool:
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return abs(x1 * y2 - x2 * y1) == 0

class TrackingPen:
    pen: fontTools.pens.ttGlyphPen.TTGlyphPen
    height: int
    scale: float
    italic: bool
    step_size: tuple[float, float]
    current: tuple[int, int]
    next: tuple[int, int] | None

    def __init__(self, height: int, scale: float, italic: bool, step_size: tuple[float, float]):
        self.pen = fontTools.pens.ttGlyphPen.TTGlyphPen(None)
        self.height = height
        self.scale = scale
        self.italic = italic
        self.step_size = step_size
        self.current = (0, 0)
        self.next = None

    def convert_point(self, point: tuple[int, int]) -> tuple[float, float]:
        ox, oy = self.step_size
        x, y = point
        x += ox
        y += oy
        if self.italic:
            x += (self.height - y) / 4
        result = (x * self.scale, (self.height - y) * self.scale)
        return result

    def draw_last(self):
        if self.next is not None:
            converted = self.convert_point(self.next)
            self.pen.lineTo(converted)
            self.current = self.next
            self.next = None

    def move(self, point: tuple[int, int]):
        self.draw_last()
        converted = self.convert_point(point)
        self.pen.moveTo(converted)
        self.current = point
        self.next = None

    def line(self, point: tuple[int, int]):
        if self.next is not None and not collinear(self.current, self.next, point):
            self.draw_last()
        self.next = point

def vectorize(mask: Bitmap, scale: float, step_size: tuple[float, float], italic: bool=False) -> fontTools.ttLib.tables._g_l_y_f.Glyph | None:
    labels = mask.label()
    filled, empty = separate_regions(mask, labels)
    if len(filled) == 0:
        return None
    height = mask.height
    pen = TrackingPen(height, scale, italic, step_size)
    for region in filled:
        outline_points = outline(region)
        pen.move(outline_points[0])
        for point in outline_points[1:]:
            pen.line(point)
        pen.next = None
        pen.pen.closePath()
    for region in empty:
        outline_points = list(reversed(outline(region)))
        pen.move(outline_points[0])
        for point in outline_points[1:]:
            pen.line(point)
        pen.next = None
        pen.pen.closePath()
    return pen.pen.glyph()
