import fontTools.pens.ttGlyphPen
import fontTools.ttLib.tables._g_l_y_f
import networkx

from minecraft_ttf.bitmap import Bitmap


def to_graph(mask: Bitmap) -> networkx.DiGraph:
    graph = networkx.DiGraph()
    w, h = mask.get_size()
    for y in range(h):
        for x in range(w):
            if mask.get_at((x, y)):
                top_left = (x, y)
                top_right = (x + 1, y)
                bottom_left = (x, y + 1)
                bottom_right = (x + 1, y + 1)
                if not is_set(mask, (x, y - 1)):
                    graph.add_edge(top_left, top_right)
                if not is_set(mask, (x - 1, y)):
                    graph.add_edge(bottom_left, top_left)
                if not is_set(mask, (x + 1, y)):
                    graph.add_edge(top_right, bottom_right)
                if not is_set(mask, (x, y + 1)):
                    graph.add_edge(bottom_right, bottom_left)
    return graph

def is_set(mask: Bitmap, point: tuple[int, int]) -> bool:
    x, y = point
    if x < 0 or y < 0:
        return False
    w, h = mask.get_size()
    if x >= w or y >= h:
        return False
    return mask.get_at(point)

def collinear(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]) -> bool:
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return abs(x1 * y2 - x2 * y1) == 0

class TrackingPen:
    pen: fontTools.pens.ttGlyphPen.TTGlyphPen
    height: int
    scale: float
    offset: tuple[float, float]
    shear: float | None
    current: tuple[int, int]
    next: tuple[int, int] | None

    def __init__(self, height: int, scale: float, offset: tuple[float, float], shear: float | None):
        self.pen = fontTools.pens.ttGlyphPen.TTGlyphPen(None)
        self.height = height
        self.scale = scale
        self.offset = offset
        self.shear = shear
        self.current = (0, 0)
        self.next = None

    def convert_point(self, point: tuple[int, int]) -> tuple[float, float]:
        x, y = point
        ox, oy = self.offset
        x += ox
        y += oy
        if self.shear is not None:
            x += (self.height - y) / self.shear
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

def trace_bitmap(mask: Bitmap) -> list[list[tuple[int, int]]]:
    graph = to_graph(mask)
    walks: list[list[tuple[int, int]]] = []
    subgraphs = (graph.subgraph(x) for x in networkx.strongly_connected_components(graph))
    for subgraph in subgraphs:
        walk: list[tuple[int, int]] = []
        y, x = min((y, x) for x, y in subgraph.nodes)
        circuit = networkx.eulerian_circuit(subgraph, (x, y))
        start, end = next(circuit)
        walk.append(start)
        walk.append(end)
        for _, end in circuit:
            walk.append(end)
        walks.append(walk)
    return walks

def vectorize(walks: list[list[tuple[int, int]]], pen: TrackingPen) -> fontTools.ttLib.tables._g_l_y_f.Glyph | None:
    if len(walks) == 0:
        return None
    pen.current = (0, 0)
    pen.next = None
    for walk in walks:
        pen.move(walk[0])
        for point in walk[1:]:
            pen.line(point)
        pen.next = None
        pen.pen.closePath()
    return pen.pen.glyph()
