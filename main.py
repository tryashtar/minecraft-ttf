import io
import requests
import os
import sys
import json
import datetime
import zipfile
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import PIL.Image
import fontTools.fontBuilder
import fontTools.pens.ttGlyphPen
import fontTools.ttLib.tables._g_l_y_f


def main():
    name = sys.argv[1] if len(sys.argv) >= 2 else None
    version = get_version(name)
    name = version['id']
    meta_url = version['url']
    date = datetime.datetime.fromisoformat(version['releaseTime'])
    if date < datetime.datetime.fromisoformat('2018-07-10T14:21:42+00:00'):
        raise ValueError(f'{name} is too early; versions from before 1.13-pre7 are unsupported')
    cached_path = f'cache/minecraft-{name}.jar'
    if not os.path.exists(cached_path):
        print(f'Downloading minecraft jar {name}...')
        response = requests.get(meta_url)
        data = response.json()
        client_jar = data['downloads']['client']['url']
        response = requests.get(client_jar)
        os.makedirs('cache', exist_ok=True)
        with open(cached_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=16 * 1024):
                f.write(chunk)
    aglfn = get_aglfn()
    print('Converting fonts...')
    with zipfile.ZipFile(cached_path, 'r') as jar:
        convert_font('Default', 'assets/minecraft/font/default.json', jar, datetime.datetime.fromisoformat('2009-05-16T16:52:00Z'), aglfn)
        convert_font('Enchanting', 'assets/minecraft/font/alt.json', jar, datetime.datetime.fromisoformat('2011-10-06T00:00:00Z'), aglfn)
        if date > datetime.datetime.fromisoformat('2021-09-15T16:04:30+00:00'): # 21w37a, when illageralt was added
            convert_font('Illager Runes', 'assets/minecraft/font/illageralt.json', jar, datetime.datetime.fromisoformat('2021-09-15T16:04:30Z'), aglfn)
    print('Done!')
    
def get_version(snapshot_id) -> dict:
    cached_path = 'cache/manifest.json'
    try:
        with open(cached_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('Downloading version manifest...')
        manifest_url = 'https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'
        response = requests.get(manifest_url)
        data = response.json()
        os.makedirs('cache', exist_ok=True)
        with open(cached_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    if not snapshot_id:
        snapshot_id = data['latest']['snapshot']
    for version in data['versions']:
        if version['id'] == snapshot_id:
            return version
    raise ValueError(snapshot_id)

def get_aglfn() -> dict[str, str]:
    cached_path = 'cache/aglfn.txt'
    if not os.path.exists(cached_path):
        print('Downloading Adobe AGLFN...')
        response = requests.get('https://raw.githubusercontent.com/adobe-type-tools/agl-aglfn/refs/heads/master/aglfn.txt')
        os.makedirs('cache', exist_ok=True)
        with open(cached_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=16 * 1024):
                f.write(chunk)
    aglfn_map = {}
    with open(cached_path, 'r', encoding='utf-8') as aglfn:
        for line in aglfn.readlines():
            if line.startswith('#') or line.isspace() or len(line) == 0:
                continue
            unihex, name, _uniname = line.split(';')
            uninum = int(unihex, 16)
            codepoint = chr(uninum)
            aglfn_map[codepoint] = name
    return aglfn_map

def read_json(jar: zipfile.ZipFile, resource: str, kind: str) -> tuple[dict, datetime.datetime]:
    namespace, rest = resource.split(':')
    path = f'assets/{namespace}/{kind}/{rest}.json'
    text = jar.read(path)
    data = json.loads(text)
    date = jar.getinfo(path).date_time
    return (data, date_time(date))

def read_image(jar: zipfile.ZipFile, resource: str) -> tuple[PIL.Image.Image, datetime.datetime]:
    namespace, rest = resource.split(':')
    path = f'assets/{namespace}/textures/{rest}'
    data = jar.read(path)
    img = PIL.Image.open(io.BytesIO(data))
    date = jar.getinfo(path).date_time
    return (img, date_time(date))

def date_time(jartime: tuple) -> datetime.datetime:
    y, m, d, h, mm, s = jartime
    return datetime.datetime(y, m, d, h, mm, s, 0, tzinfo=datetime.timezone.utc)

def convert_font(name: str, entry: str, jar: zipfile.ZipFile, created_date: datetime.datetime, aglfn: dict[str, str]):
    modified_date = date_time(jar.getinfo(entry).date_time)
    text = jar.read(entry)
    data = json.loads(text)
    providers: list[dict] = []
    providers.extend(data['providers'])
    index = 0
    while index < len(providers):
        if providers[index]['type'] == 'reference':
            (reference, date) = read_json(jar, providers[index]['id'], 'font')
            if date > modified_date:
                modified_date = date
            del providers[index]
            providers[index:index] = reference['providers']
        index += 1
    seen_chars = set()
    fonts = {'Regular': {}, 'Bold': {}, 'Italic': {}, 'Bold Italic': {}}
    chatbox_height = 12
    font_em = 1200
    pixel_scale = font_em / chatbox_height
    def add_bitmap_glyph(char: str, mask: pygame.mask.Mask, height: int, ascent: int):
        m_width, m_height = mask.get_size()
        seen_chars.add(char)
        bold_mask = pygame.mask.Mask((m_width + 1, m_height), fill=False)
        bold_mask.draw(mask, (0, 0))
        bold_mask.draw(mask, (1, 0))
        scale = height / m_height * pixel_scale
        offset = (0, (height - ascent) / height * m_height)
        italic_offset = (-6 / height, (height - ascent) / height * m_height)
        (path, (w, h)) = vectorize(mask, scale, offset)
        (italic_path, (iw, ih)) = vectorize(mask, scale, italic_offset, italic=True)
        (bold_path, (bw, bh)) = vectorize(bold_mask, scale, offset)
        (bold_italic_path, (biw, bih)) = vectorize(bold_mask, scale, italic_offset, italic=True)
        add_width = m_height / height
        fonts['Regular'][char] = {'width': (w + add_width) * scale, 'height': h * scale, 'path': path}
        fonts['Italic'][char] = {'width': (iw + add_width) * scale, 'height': ih * scale, 'path': italic_path}
        fonts['Bold'][char] = {'width': (bw + add_width) * scale, 'height': bh * scale, 'path': bold_path}
        fonts['Bold Italic'][char] = {'width': (biw + add_width) * scale, 'height': bih * scale, 'path': bold_italic_path}
    mw, mh = (5, 8)
    missing = pygame.mask.Mask((mw, mh), fill=False)
    for y in range(mh):
        for x in range(mw):
            if x == 0 or y == 0 or x == mw - 1 or y == mh - 1:
                missing.set_at((x, y), 1)
    add_bitmap_glyph('.notdef', missing, 8, 8)
    for provider in providers:
        if provider['type'] == 'space':
            for char,width in provider['advances'].items():
                if char in seen_chars:
                    continue
                seen_chars.add(char)
                fonts['Regular'][char] = {'width': width * pixel_scale, 'height': 0, 'path': None}
                fonts['Italic'][char] = {'width': width * pixel_scale, 'height': 0, 'path': None}
                fonts['Bold'][char] = {'width': (width + 1) * pixel_scale, 'height': 0, 'path': None}
                fonts['Bold Italic'][char] = {'width': (width + 1) * pixel_scale, 'height': 0, 'path': None}
        elif provider['type'] == 'bitmap':
            (img, date) = read_image(jar, provider['file'])
            if date > modified_date:
                modified_date = date
            height = provider.get('height', 8)
            ascent = provider['ascent']
            glyph_width = img.width // len(provider['chars'][0])
            glyph_height = img.height // len(provider['chars'])
            for y,row in enumerate(provider['chars']):
                for x,char in enumerate(row):
                    if char == '\u0000':
                        continue
                    if char in seen_chars:
                        continue
                    glyph = img.crop((x * glyph_width, y * glyph_height, (x + 1) * glyph_width, (y + 1) * glyph_height)).convert('RGBA')
                    surface = pygame.image.fromstring(glyph.tobytes(), glyph.size, 'RGBA')
                    mask = pygame.mask.from_surface(surface)
                    add_bitmap_glyph(char, mask, height, ascent)
    for style, data in fonts.items():
        full_name = 'Minecraft ' + name
        ttf_name = full_name.replace(' ', '') + '-' + style.replace(' ', '')
        font = make_font(full_name, style, font_em, (created_date, modified_date), data, aglfn)
        os.makedirs('out', exist_ok=True)
        font.save(f'out/{ttf_name}.ttf')

def make_font(name: str, style: str, font_em: int, dates: tuple[datetime.datetime, datetime.datetime], char_data: dict, aglfn: dict[str, str]) -> fontTools.fontBuilder.FontBuilder:
    nameStrings = dict(
        copyright = 'Copyright (c) 2009 Mojang AB',
        familyName = name,
        styleName = style,
        uniqueFontIdentifier = name.replace(' ', '') + '.' + style.replace(' ', ''),
        fullName = name + ' ' + style,
        version = 'Version 1.000',
        psName = name.replace(' ','') + style.replace(' ', ''),
        sampleText = 'and the universe said I love you'
    )
    empty_glyph = fontTools.ttLib.tables._g_l_y_f.Glyph()
    defined_glyphs = ['.notdef', '.null']
    codepoints = {}
    char_widths = {'.notdef': 0, '.null': 0}
    char_paths = {'.notdef': empty_glyph, '.null': empty_glyph}
    for char, data in char_data.items():
        if char not in ('.notdef', '.null'):
            char_name = aglfn.get(char, 'uni' + format(ord(char), '04x'))
            defined_glyphs.append(char_name)
            codepoints[ord(char)] = char_name
        else:
            char_name = char
        char_widths[char_name] = data['width']
        if data['path'] is not None:
            char_paths[char_name] = data['path']
        else:
            char_paths[char_name] = empty_glyph
    widest = max(map(lambda x: x['width'], char_data.values()))
    tallest = max(map(lambda x: x['height'], char_data.values()))
    font = fontTools.fontBuilder.FontBuilder(unitsPerEm=font_em, isTTF=True)
    font.setupGlyphOrder(defined_glyphs)
    font.setupCharacterMap(codepoints)
    font.setupGlyf(char_paths)
    metrics = {}
    glyphTable = font.font["glyf"]
    for gn, advanceWidth in char_widths.items():
        metrics[gn] = (advanceWidth, glyphTable[gn].xMin)
    font.setupHorizontalMetrics(metrics)
    ascent = font_em*9//12
    descent = font_em*2//12
    font.setupHorizontalHeader(ascent=ascent, descent=-descent)
    font.setupNameTable(nameStrings)
    fs_selection = 0
    mac_style = 0
    weight = 400
    if 'Bold' in style:
        mac_style += 1
        fs_selection += 32
        weight = 700
    if 'Italic' in style:
        mac_style += 2
        fs_selection += 1
    if 'Bold' not in style and 'Italic' not in style:
        fs_selection += 64
    font.setupOS2(sTypoAscender=ascent, sTypoDescender=-descent, usWinAscent=ascent, usWinDescent=descent, sCapHeight=font_em*7//12, sxHeight=font_em*5//12, yStrikeoutPosition=font_em*4//12, yStrikeoutSize=font_em*1//12, sTypoLineGap=0, fsSelection=fs_selection, achVendID="", usWeightClass=weight)
    italic_angle = 14.05598 if 'Italic' in style else 0
    font.setupPost(underlinePosition=-font_em*1//12, underlineThickness=font_em*1//12, italicAngle=-italic_angle)
    epoch = datetime.datetime.fromisoformat('1904-01-01T00:00:00Z')
    created, modified = dates
    font.updateHead(xMin=0, xMax=int(widest), yMin=-descent, yMax=int(tallest), created=int((created - epoch).total_seconds()), modified=int((modified - epoch).total_seconds()), macStyle=mac_style)
    return font

def start_point(mask: pygame.mask.Mask) -> tuple[int, int]:
    w, h = mask.get_size()
    for y in range(h):
        for x in range(w):
            if mask.get_at((x, y)) == 1:
                return (x, y)
    raise ValueError(mask)

def is_set(mask: pygame.mask.Mask, point: tuple[int, int]) -> bool:
    x, y = point
    if x < 0 or y < 0:
        return False
    w, h = mask.get_size()
    if x >= w or y >= h:
        return False
    return mask.get_at(point) == 1

def outline(mask: pygame.mask.Mask) -> list[tuple[int, int]]:
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

def neighbor_connected(mask: pygame.mask.Mask) -> list[pygame.mask.Mask]:
    w, h = mask.get_size()
    pixels_checked = set()
    result = []
    for y in range(h):
        for x in range(w):
            pos = (x, y)
            if pos not in pixels_checked:
                if mask.get_at(pos) == 1:
                    region = pygame.mask.Mask((w, h))
                    pixel_queue = [pos]
                    while len(pixel_queue) > 0:
                        pixel = pixel_queue.pop()
                        px, py = pixel
                        if px < 0 or px >= w or py < 0 or py >= h or pixel in pixels_checked or mask.get_at(pixel) != 1:
                            pixels_checked.add(pixel)
                            continue
                        pixels_checked.add(pixel)
                        region.set_at(pixel, 1)
                        pixel_queue.append((px - 1, py))
                        pixel_queue.append((px + 1, py))
                        pixel_queue.append((px, py - 1))
                        pixel_queue.append((px, py + 1))
                    result.append(region)
                pixels_checked.add(pos)
    return result 

def separate_regions(mask: pygame.mask.Mask) -> tuple[list[pygame.mask.Mask], list[pygame.mask.Mask]]:
    filled = mask.connected_components()
    w, h = mask.get_size()
    inverted = pygame.mask.Mask((w + 2, h + 2))
    inverted.draw(mask, (1, 1))
    inverted.invert()
    big_unfilled = neighbor_connected(inverted)
    unfilled = []
    for big in big_unfilled[1:]:
        fixed = pygame.mask.Mask((w, h))
        fixed.draw(big, (-1, -1))
        unfilled.append(fixed)
    return (filled, unfilled)

def collinear(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]) -> bool:
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return abs(x1 * y2 - x2 * y1) < 1e-12

def vectorize(mask: pygame.mask.Mask, scale: float, offset: tuple[float, float], italic: bool=False) -> tuple[fontTools.ttLib.tables._g_l_y_f.Glyph | None, tuple[int, int]]:
    ox, oy = offset
    pen = fontTools.pens.ttGlyphPen.TTGlyphPen(None)
    pen_pos: dict[str, tuple[int, int] | None] = {'current': None, 'next': None}
    width, height = mask.get_size()
    def draw_last():
        if pen_pos['next'] is not None:
            x, y = pen_pos['next']
            x += ox
            y += oy
            if italic:
                x += (height - y) / 4
            pen.lineTo((x * scale, (height - y) * scale))
            pen_pos['current'] = pen_pos['next']
            pen_pos['next'] = None
    def move_pen(point: tuple[int, int]):
        draw_last()
        pen_pos['current'] = point
        pen_pos['next'] = None
        x, y = point
        x += ox
        y += oy
        if italic:
            x += (height - y) / 4
        pen.moveTo((x * scale, (height - y) * scale))
    def line_pen(point: tuple[int, int]):
        if pen_pos['next'] is not None and not collinear(pen_pos['current'], pen_pos['next'], point):
            draw_last()
        pen_pos['next'] = point
    filled, empty = separate_regions(mask)
    if len(filled) == 0:
        return (None, (0, 0))
    else:
        rects = mask.get_bounding_rects()
        size = (max(map(lambda x: x.right, rects)), max(map(lambda x: x.top, rects)))
        for region in filled:
            outline_points = outline(region)
            move_pen(outline_points[0])
            for point in outline_points[1:]:
                line_pen(point)
            pen_pos['next'] = None
            pen.closePath()
        for region in empty:
            outline_points = list(reversed(outline(region)))
            move_pen(outline_points[0])
            for point in outline_points[1:]:
                line_pen(point)
            pen_pos['next'] = None
            pen.closePath()
    return (pen.glyph(), size)

if __name__ == '__main__':
    main()
