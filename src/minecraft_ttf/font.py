import dataclasses
import datetime

import fontTools.fontBuilder
import fontTools.pens.ttGlyphPen
import fontTools.ttLib.tables._g_l_y_f
import fontTools.ttLib.tables.C_P_A_L_

from minecraft_ttf.bitmap import Bitmap, BitmapLabels


@dataclasses.dataclass
class GlyphLayer:
    path: fontTools.pens.ttGlyphPen.Glyph
    color: fontTools.ttLib.tables.C_P_A_L_.Color | None

@dataclasses.dataclass
class CharInfo:
    width: float
    height: float
    layers: list[GlyphLayer]

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

def make_font(info: FontInfo, positions: FontPositions, char_data: dict[str, CharInfo], aglfn: dict[str, str]) -> fontTools.fontBuilder.FontBuilder:
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
    for char, data in char_data.items():
        if char not in ('.notdef', '.null'):
            glyph_name = aglfn.get(char, f'uni{ord(char):04x}')
            codepoints[ord(char)] = glyph_name
        else:
            glyph_name = char
        glyph_widths[glyph_name] = data.width
        if len(data.layers) == 0:
            glyph_paths[glyph_name] = empty_glyph
        else:
            for i, layer in enumerate(data.layers):
                layer_name = glyph_name if i == 0 else f'{glyph_name}.layer{i}'
                glyph_paths[layer_name] = layer.path
                glyph_widths[layer_name] = data.width
                if layer.color is not None:
                    try:
                        color_index = color_palettes.index(layer.color)
                    except ValueError:
                        color_index = len(color_palettes)
                        color_palettes.append(layer.color)
                    if glyph_name not in color_layers:
                        color_layers[glyph_name] = []
                    color_layers[glyph_name].append((layer_name, color_index))
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
            if mask.get_at((x, y)) == 1:
                return (x, y)
    raise ValueError(mask)

def is_set(mask: Bitmap, point: tuple[int, int]) -> bool:
    x, y = point
    if x < 0 or y < 0:
        return False
    w, h = mask.get_size()
    if x >= w or y >= h:
        return False
    return mask.get_at(point) == 1

# trace the outline of a connected mask
# returns a list of all corner points that were visited, in order
def outline(mask: Bitmap) -> list[tuple[int, int]]:
    start = start_point(mask)
    facing = 'right'
    pos = start
    result = [pos]
    while True:
        x, y = pos
        top_left = is_set(mask, (x - 1, y - 1))
        top_right = is_set(mask, (x, y - 1))
        bottom_left = is_set(mask, (x - 1, y))
        bottom_right = is_set(mask, (x, y))
        if top_left and bottom_right and not top_right and not bottom_left:
            if facing == 'up':
                facing = 'left'
                pos = (x - 1, y)
            else:
                facing = 'right'
                pos = (x + 1, y)
        elif top_right and bottom_left and not top_left and not bottom_right:
            if facing == 'right':
                facing = 'up'
                pos = (x, y - 1)
            else:
                facing = 'down'
                pos = (x, y + 1)
        elif top_left and not bottom_left:
            facing = 'left'
            pos = (x - 1, y)
        elif top_right and not top_left:
            facing = 'up'
            pos = (x, y - 1)
        elif bottom_right and not top_right:
            facing = 'right'
            pos = (x + 1, y)
        elif bottom_left and not bottom_right:
            facing = 'down'
            pos = (x, y + 1)
        result.append(pos)
        if pos == start:
            break
    return result

def neighbor_connected(mask: Bitmap) -> list[Bitmap]:
    w, h = mask.get_size()
    pixels_checked = set()
    result = []
    for y in range(h):
        for x in range(w):
            pos = (x, y)
            if pos not in pixels_checked:
                if mask.get_at(pos) == 1:
                    region = Bitmap((w, h))
                    pixel_queue = [pos]
                    while len(pixel_queue) > 0:
                        pixel = pixel_queue.pop()
                        px, py = pixel
                        if px < 0 or px >= w or py < 0 or py >= h or pixel in pixels_checked or mask.get_at(pixel) != 1:
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
    return abs(x1 * y2 - x2 * y1) < 1e-12

@dataclasses.dataclass
class TrackingPen:
    pen: fontTools.pens.ttGlyphPen.TTGlyphPen
    step_size: tuple[float, float]
    current: tuple[int, int]
    next: tuple[int, int] | None

def draw_last(pen: TrackingPen, italic: bool, height: int, scale: float):
    if pen.next is not None:
        ox, oy = pen.step_size
        x, y = pen.next
        x += ox
        y += oy
        if italic:
            x += (height - y) / 4
        pen.pen.lineTo((x * scale, (height - y) * scale))
        pen.current = pen.next
        pen.next = None

def move_pen(pen: TrackingPen, point: tuple[int, int], italic: bool, height: int, scale: float):
    draw_last(pen, italic, height, scale)
    pen.current = point
    pen.next = None
    ox, oy = pen.step_size
    x, y = point
    x += ox
    y += oy
    if italic:
        x += (height - y) / 4
    pen.pen.moveTo((x * scale, (height - y) * scale))

def line_pen(pen: TrackingPen, point: tuple[int, int], italic: bool, height: int, scale: float):
    if pen.next is not None and not collinear(pen.current, pen.next, point):
        draw_last(pen, italic, height, scale)
    pen.next = point

def vectorize(mask: Bitmap, scale: float, step_size: tuple[float, float], italic: bool=False) -> fontTools.ttLib.tables._g_l_y_f.Glyph | None:
    labels = mask.label()
    filled, empty = separate_regions(mask, labels)
    if len(filled) == 0:
        return None
    _width, height = mask.get_size()
    glyph_pen = fontTools.pens.ttGlyphPen.TTGlyphPen(None)
    pen = TrackingPen(glyph_pen, step_size, (0, 0), None)
    for region in filled:
        outline_points = outline(region)
        move_pen(pen, outline_points[0], italic, height, scale)
        for point in outline_points[1:]:
            line_pen(pen, point, italic, height, scale)
        pen.next = None
        pen.pen.closePath()
    for region in empty:
        outline_points = list(reversed(outline(region)))
        move_pen(pen, outline_points[0], italic, height, scale)
        for point in outline_points[1:]:
            line_pen(pen, point, italic, height, scale)
        pen.next = None
        pen.pen.closePath()
    return pen.pen.glyph()
