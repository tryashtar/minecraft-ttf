import dataclasses

import bitarray
import PIL.Image


@dataclasses.dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

@dataclasses.dataclass
class BitmapLabels:
    width: int
    height: int
    image: list[int]
    ufind: list[int]
    num_components: int

    def get_bounding_rects(self) -> list[Rect]:
        if self.num_components == 0:
            return []
        rects: list[Rect | None] = [None] * (self.num_components + 1)
        for i, label in enumerate(self.image):
            component = self.ufind[label]
            if component == 0:
                continue
            y, x = divmod(i, self.width)
            rect = rects[component]
            if rect is None:
                rects[component] = Rect(x, y, 1, 1)
            else:
                right = max(rect.x + rect.w, x + 1)
                bottom = max(rect.y + rect.h, y + 1)
                rect.x = min(rect.x, x)
                rect.y = min(rect.y, y)
                rect.w = right - rect.x
                rect.h = bottom - rect.y
        return [x for x in rects[1:] if x is not None]

    def connected_components(self) -> list['Bitmap']:
        if self.num_components == 0:
            return []
        components = [Bitmap((self.width, self.height)) for _ in range(self.num_components)]
        sizes = [0] * (self.num_components + 1)
        for i, label in enumerate(self.image):
            component = self.ufind[label]
            if component:
                components[component - 1].bits[i] = True
                sizes[component] += 1
        return components

@dataclasses.dataclass
class Bitmap:
    bits: bitarray.bitarray
    width: int
    height: int

    def __init__(self, size: tuple[int, int]):
        self.width, self.height = size
        self.bits = bitarray.bitarray(self.width * self.height)

    def get_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def get_index(self, pos: tuple[int, int]) -> int:
       x, y = pos
       return y * self.width + x

    def set_at(self, pos: tuple[int, int], value: bool):
        self.bits[self.get_index(pos)] = value

    def get_at(self, pos: tuple[int, int]) -> bool:
        return self.bits[self.get_index(pos)] == 1

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

    def label(self) -> BitmapLabels:
        image = [0] * (self.width * self.height)
        max_labels = (self.width // 2 + 1) * (self.height // 2 + 1)
        ufind = [0] * max_labels
        largest = [0] * max_labels
        label = cc_label(self, image, ufind, largest)
        num_components = 0
        for x_uf in range(1, label + 1):
            if ufind[x_uf] < x_uf:
                ufind[x_uf] = ufind[ufind[x_uf]]
            else:
                num_components += 1
                ufind[x_uf] = num_components
        return BitmapLabels(self.width, self.height, image, ufind, num_components)

def bitmap_from_image(image: PIL.Image.Image) -> Bitmap:
    mask = Bitmap(image.size)
    pixels = image.load()
    assert pixels is not None
    for y in range(image.height):
        for x in range(image.width):
            _r, _g, _b, a = pixels[(x, y)]
            if a >= 10:
                mask.set_at((x, y), True)
    return mask

def cc_label(mask: Bitmap, image: list[int], ufind: list[int], largest: list[int]) -> int:
    w = mask.width
    h = mask.height
    bits = mask.bits
    label = 0
    ufind[0] = 0

    def find(x: int) -> int:
        while ufind[x] < x:
            x = ufind[x]
        return x

    def union(a: int, b: int) -> int:
        ra = find(a)
        rb = find(b)
        root = min(ra, rb)
        while ufind[a] > root:
            nxt = ufind[a]
            ufind[a] = root
            a = nxt
        while ufind[b] > root:
            nxt = ufind[b]
            ufind[b] = root
            b = nxt
        return root

    if bits[0]:
        label += 1
        image[0] = label
        ufind[label] = label
        largest[label] = 1
    else:
        image[0] = 0
    buf = 1
    for x in range(1, w):
        if bits[x]:
            if image[buf - 1]:
                image[buf] = image[buf - 1]
            else:
                label += 1
                image[buf] = label
                ufind[label] = label
                largest[label] = 0
            largest[image[buf]] += 1
        else:
            image[buf] = 0
        buf += 1
    for y in range(1, h):
        if bits[y * w]:
            b = image[buf - w]
            c = image[buf - w + 1]
            if b:
                image[buf] = b
            elif c:
                image[buf] = c
            else:
                label += 1
                image[buf] = label
                ufind[label] = label
                largest[label] = 0
            largest[image[buf]] += 1
        else:
            image[buf] = 0
        buf += 1
        for x in range(1, w - 1):
            if bits[y * w + x]:
                a = image[buf - w - 1]
                b = image[buf - w]
                c = image[buf - w + 1]
                d = image[buf - 1]
                if b:
                    image[buf] = b
                elif c:
                    if a:
                        image[buf] = union(c, a)
                    elif d:
                        image[buf] = union(c, d)
                    else:
                        image[buf] = c
                elif a:
                    image[buf] = a
                elif d:
                    image[buf] = d
                else:
                    label += 1
                    image[buf] = label
                    ufind[label] = label
                    largest[label] = 0
                largest[image[buf]] += 1
            else:
                image[buf] = 0
            buf += 1
        if w > 1:
            if bits[y * w + (w - 1)]:
                a = image[buf - w - 1]
                b = image[buf - w]
                d = image[buf - 1]
                if b:
                    image[buf] = b
                elif a:
                    image[buf] = a
                elif d:
                    image[buf] = d
                else:
                    label += 1
                    image[buf] = label
                    ufind[label] = label
                    largest[label] = 0
                largest[image[buf]] += 1
            else:
                image[buf] = 0
            buf += 1
    return label
