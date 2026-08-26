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
    pen = TrackingPen(mask.height, scale, italic, step_size)
    graph = to_graph(mask)
    subgraphs = (graph.subgraph(x) for x in networkx.strongly_connected_components(graph))
    for subgraph in subgraphs:
        y, x = min((y, x) for x, y in subgraph.nodes)
        walk = list(networkx.eulerian_circuit(subgraph, (x, y)))
        pen.move(walk[0][0])
        for point in walk:
            pen.line(point[1])
        pen.next = None
        pen.pen.closePath()
    return pen.pen.glyph()
