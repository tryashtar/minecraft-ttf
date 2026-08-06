import datetime
import io
import json
import pathlib
import zipfile

import PIL.Image
import requests

from bitmap import Bitmap, bitmap_from_image
from font import CharInfo, FontInfo, make_font, vectorize


def main():
    latest = get_latest()
    name = latest['id']
    meta_url = latest['url']
    cached_path = pathlib.Path('cache') / f'minecraft-{name}.jar'
    if not cached_path.exists():
        print(f'Downloading minecraft jar {name}...')
        response = requests.get(meta_url)
        data = response.json()
        client_jar = data['downloads']['client']['url']
        response = requests.get(client_jar)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'wb') as f:
            f.writelines(response.iter_content(chunk_size=16 * 1024))
    aglfn = get_aglfn()
    print('Converting fonts...')
    with zipfile.ZipFile(cached_path, 'r') as jar:
        # TTF metadata includes a creation date
        # this information isn't in the jar, so we have to provide it ourselves
        convert_font('Default', 'assets/minecraft/font/default.json', jar, datetime.datetime.fromisoformat('2009-05-16T16:52:00Z'), aglfn)
        convert_font('Enchanting', 'assets/minecraft/font/alt.json', jar, datetime.datetime.fromisoformat('2011-10-06T00:00:00Z'), aglfn)
        convert_font('Illager Runes', 'assets/minecraft/font/illageralt.json', jar, datetime.datetime.fromisoformat('2021-09-15T16:04:30Z'), aglfn)
    print('Done!')

def get_latest() -> dict:
    cached_path = pathlib.Path('cache') / 'manifest.json'
    try:
        with open(cached_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('Downloading version manifest...')
        manifest_url = 'https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'
        response = requests.get(manifest_url)
        data = response.json()
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    snapshot_id = data['latest']['snapshot']
    for version in data['versions']:
        if version['id'] == snapshot_id:
            return version
    raise ValueError(snapshot_id)

# The Adobe Glyph List For New Fonts tells us what names to use for the glyphs that characters are mapped to
def get_aglfn() -> dict[str, str]:
    cached_path = pathlib.Path('cache') / 'aglfn.txt'
    if not cached_path.exists():
        print('Downloading Adobe AGLFN...')
        response = requests.get('https://raw.githubusercontent.com/adobe-type-tools/agl-aglfn/refs/heads/master/aglfn.txt')
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'wb') as f:
            f.writelines(response.iter_content(chunk_size=16 * 1024))
    aglfn_map = {}
    with open(cached_path, 'r', encoding='utf-8') as aglfn:
        for line in aglfn:
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
    return datetime.datetime(y, m, d, h, mm, s, 0, tzinfo=datetime.UTC)

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
            modified_date = max(modified_date, date)
            del providers[index]
            providers[index:index] = reference['providers']
        index += 1
    seen_chars: set[str] = set()
    fonts: dict[str, dict[str, CharInfo]] = {'Regular': {}, 'Bold': {}, 'Italic': {}, 'Bold Italic': {}}
    chatbox_height = 12
    font_em = 1200
    pixel_scale = font_em / chatbox_height
    # font textures have a color depth of 1 bit, so they are just 2D bitmasks
    # this lets us leverage some efficient operations provided by pygame
    def add_bitmap_glyph(char: str, mask: Bitmap, height: int, ascent: int):
        m_width, m_height = mask.get_size()
        seen_chars.add(char)
        # bold characters are created by overlapping two copies of the texture
        bold_mask = Bitmap((m_width + 1, m_height))
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
        fonts['Regular'][char] = CharInfo(width = (w + add_width) * scale, height = h * scale, path = path)
        fonts['Italic'][char] = CharInfo(width = (iw + add_width) * scale, height = ih * scale, path = italic_path)
        fonts['Bold'][char] = CharInfo(width = (bw + add_width) * scale, height = bh * scale, path = bold_path)
        fonts['Bold Italic'][char] = CharInfo(width = (biw + add_width) * scale, height = bih * scale, path = bold_italic_path)
    mw, mh = (5, 8)
    missing = Bitmap((mw, mh))
    for y in range(mh):
        for x in range(mw):
            if x == 0 or y == 0 or x == mw - 1 or y == mh - 1:
                missing.set_at((x, y), True)
    add_bitmap_glyph('.notdef', missing, 8, 8)
    for provider in providers:
        if provider['type'] == 'space':
            for char, width in provider['advances'].items():
                if char in seen_chars:
                    continue
                seen_chars.add(char)
                fonts['Regular'][char] = CharInfo(width = width * pixel_scale, height = 0, path = None)
                fonts['Italic'][char] = CharInfo(width = width * pixel_scale, height = 0, path = None)
                fonts['Bold'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, path = None)
                fonts['Bold Italic'][char] = CharInfo(width = (width + 1) * pixel_scale, height = 0, path = None)
        elif provider['type'] == 'bitmap':
            (img, date) = read_image(jar, provider['file'])
            modified_date = max(modified_date, date)
            height = provider.get('height', 8)
            ascent = provider['ascent']
            glyph_width = img.width // len(provider['chars'][0])
            glyph_height = img.height // len(provider['chars'])
            for y, row in enumerate(provider['chars']):
                for x, char in enumerate(row):
                    if char == '\u0000':
                        continue
                    if char in seen_chars:
                        continue
                    glyph = img.crop((x * glyph_width, y * glyph_height, (x + 1) * glyph_width, (y + 1) * glyph_height)).convert('RGBA')
                    mask = bitmap_from_image(glyph)
                    add_bitmap_glyph(char, mask, height, ascent)
    for style, data in fonts.items():
        full_name = 'Minecraft ' + name
        ttf_name = full_name.replace(' ', '') + '-' + style.replace(' ', '')
        info = FontInfo(
            name = full_name,
            style = style,
            copyright = 'Copyright (c) 2009 Mojang AB',
            sample = 'and the universe said I love you',
            em = font_em,
            created = created_date,
            modified = modified_date,
            version = 'Version 1.000'
        )
        font = make_font(info, data, aglfn)
        dest = pathlib.Path('out') / f'{ttf_name}.ttf'
        dest.parent.mkdir(parents=True, exist_ok=True)
        font.save(dest)

if __name__ == '__main__':
    main()
