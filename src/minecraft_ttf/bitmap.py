import dataclasses

import bitarray
import PIL.Image


@dataclasses.dataclass
class Bitmap:
    bits: bitarray.bitarray
    width: int
    height: int

    def __init__(self, size: tuple[int, int]):
        self.width, self.height = size
        self.bits = bitarray.bitarray(self.width * self.height)
        
    def __repr__(self) -> str:
        result: list[str] = []
        for y in range(self.height):
            string = ''
            for x in range(self.width):
                if self.get_at((x, y)):
                    string += '█'
                else:
                    string += ' '
            result.append(string)
        return '\n'.join(result)

    def get_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def get_index(self, pos: tuple[int, int]) -> int:
       x, y = pos
       return y * self.width + x

    def set_at(self, pos: tuple[int, int], value: bool):
        self.bits[self.get_index(pos)] = value

    def get_at(self, pos: tuple[int, int]) -> bool:
        return self.bits[self.get_index(pos)] == 1

    def content_box(self)-> tuple[int, int, int, int] | None:
        left = self.width
        top = self.height
        right = 0
        bottom = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.get_at((x, y)):
                    left = min(left, x)
                    top = min(top, y)
                    right = max(right, x + 1)
                    bottom = max(bottom, y + 1)
        if left == self.width:
            return None
        return (left, top, right, bottom)

    def resized(self, box: tuple[int, int, int, int]) -> 'Bitmap':
        left, top, right, bottom = box
        result = Bitmap((right - left, bottom - top))
        for y in range(top, bottom):
            for x in range(left, right):
                dest = self.get_index((x, y))
                if dest >= 0 and dest < len(self.bits) and self.bits[dest] == 1:
                    result.set_at((x - left, y - top), True)
        return result

    def invert(self):
        self.bits.invert()

    def draw(self, map: 'Bitmap', pos: tuple[int, int]):
        x, y = pos
        for yy in range(map.height):
            for xx in range(map.width):
                dest = self.get_index((x + xx, y + yy))
                if dest >= 0 and dest < len(self.bits):
                   val = self.bits[dest] or map.get_at((xx, yy))
                   self.bits[dest] = val

def get_palette_rgba(palette: list[int], transparent: int | None, pixel: int) -> tuple[int, int, int, int]:
    r, g, b = palette[pixel * 3:pixel * 3 + 3]
    a = 0 if transparent is not None and pixel == transparent else 255
    return (r, g, b, a)

def bitmap_from_image(image: PIL.Image.Image) -> Bitmap:
    mask = Bitmap(image.size)
    if image.mode == 'RGB':
        mask.invert()
        return mask
    pixels = image.load()
    assert pixels is not None
    transparent_index = image.info.get('transparency')
    for y in range(image.height):
        for x in range(image.width):
            pixel = pixels[(x, y)]
            if image.mode == 'P':
                assert isinstance(pixel, int)
                if transparent_index is None or pixel != transparent_index:
                    mask.set_at((x, y), True)
            elif image.mode == 'RGBA':
                assert isinstance(pixel, tuple)
                _r, _g, _b, a = pixel
                if a >= 10:
                    mask.set_at((x, y), True)
            else:
                raise ValueError(image.mode)
    return mask

def bitmaps_from_colors(image: PIL.Image.Image) -> list[tuple[Bitmap, tuple[int, int, int, int]]]:
    color_results = image.getcolors(maxcolors = image.width * image.height)
    assert color_results is not None
    colors = [color for _count, color in color_results]
    bitmaps: list[tuple[Bitmap, tuple[int, int, int, int]]] = []
    if image.mode == 'P':
        palette = image.getpalette()
        assert palette is not None
        transparent_index = image.info.get('transparency')
        for color in colors:
            assert isinstance(color, int)
            rgba = get_palette_rgba(palette, transparent_index, color)
            bitmaps.append((Bitmap(image.size), rgba))
    elif image.mode == 'RGBA':
        for color in colors:
            assert isinstance(color, tuple)
            r, g, b, a =color
            bitmaps.append((Bitmap(image.size), (r, g, b, a)))
    elif image.mode == 'RGB':
        for color in colors:
            assert isinstance(color, tuple)
            r, g, b =color
            bitmaps.append((Bitmap(image.size), (r, g, b, 255)))
    else:
        raise ValueError(image.mode)
    pixels = image.load()
    assert pixels is not None
    for y in range(image.height):
        for x in range(image.width):
            pixel = pixels[(x, y)]
            index = colors.index(pixel)
            bitmaps[index][0].set_at((x, y), True)
    return bitmaps
