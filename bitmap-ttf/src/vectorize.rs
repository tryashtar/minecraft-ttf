use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::{Directed, Graph, algo::tarjan_scc, graph::NodeIndex};
use read_fonts::tables::glyf::CurvePoint;
use write_fonts::tables::glyf::{Bbox, Contour, SimpleGlyph};

use crate::bitmap::Bitmap;

pub type BitmapGraph = Graph<(usize, usize), (), Directed>;

fn bitmap_to_graph(bitmap: &Bitmap) -> BitmapGraph {
    let mut graph = Graph::new();
    let mut nodes: HashMap<(usize, usize), NodeIndex> = HashMap::new();
    let mut get_node = |graph: &mut Graph<_, _, _>, x: usize, y: usize| -> NodeIndex {
        *nodes
            .entry((x, y))
            .or_insert_with(|| graph.add_node((x, y)))
    };
    for y in 0..bitmap.height() {
        for x in 0..bitmap.width() {
            if bitmap.get(x, y) {
                let top_left = get_node(&mut graph, x, y);
                let top_right = get_node(&mut graph, x + 1, y);
                let bottom_left = get_node(&mut graph, x, y + 1);
                let bottom_right = get_node(&mut graph, x + 1, y + 1);
                if y == 0 || !bitmap.try_get(x, y - 1).unwrap_or(false) {
                    graph.add_edge(top_left, top_right, ());
                }
                if !bitmap.try_get(x + 1, y).unwrap_or(false) {
                    graph.add_edge(top_right, bottom_right, ());
                }
                if !bitmap.try_get(x, y + 1).unwrap_or(false) {
                    graph.add_edge(bottom_right, bottom_left, ());
                }
                if x == 0 || !bitmap.try_get(x - 1, y).unwrap_or(false) {
                    graph.add_edge(bottom_left, top_left, ());
                }
            }
        }
    }
    graph
}

fn swap_pair<A, B>(tuple: (A, B)) -> (B, A) {
    let (a, b) = tuple;
    (b, a)
}

fn trace(graph: &BitmapGraph) -> Vec<Vec<(usize, usize)>> {
    let components = tarjan_scc(graph);
    let mut result = vec![];
    for component in components {
        let mut adjacencies = adjacencies(graph, &component);
        let start_node = component
            .iter()
            .min_by_key(|x| swap_pair(graph[**x]))
            .unwrap();
        let circuit = hierholzer_euler_circuit(&mut adjacencies, *start_node);
        let points = circuit.into_iter().map(|x| graph[x]).collect();
        result.push(points);
    }
    result
}

fn adjacencies<A, B>(
    graph: &Graph<A, B, Directed>,
    component: &[NodeIndex],
) -> HashMap<NodeIndex, HashSet<NodeIndex>> {
    let mut map = HashMap::new();
    for node in component {
        let mut set = HashSet::new();
        for neighbor in graph.neighbors(*node) {
            set.insert(neighbor);
        }
        map.insert(*node, set);
    }
    map
}

fn hierholzer_euler_circuit(
    adjacencies: &mut HashMap<NodeIndex, HashSet<NodeIndex>>,
    start_node: NodeIndex,
) -> VecDeque<NodeIndex> {
    let mut circuit = VecDeque::new();
    let mut stack = vec![start_node];
    let mut current = start_node;
    while !stack.is_empty() {
        if let Some(neighbor) = adjacencies
            .get(&current)
            .and_then(|x| x.iter().next())
            .copied()
        {
            stack.push(current);
            adjacencies.get_mut(&current).unwrap().remove(&neighbor);
            adjacencies.get_mut(&neighbor).unwrap().remove(&current);
            current = neighbor;
        } else {
            circuit.push_front(current);
            current = stack.pop().unwrap();
        }
    }
    circuit
}

pub trait PointTracer {
    type Error;

    fn start(&mut self, point: (usize, usize)) -> Result<(), Self::Error>;
    fn line(&mut self, point: (usize, usize)) -> Result<(), Self::Error>;
    fn done(&mut self) -> Vec<CurvePoint>;
}

fn trace_paths<T>(
    pen: &mut impl PointTracer<Error = T>,
    paths: impl Iterator<Item = impl Iterator<Item = (usize, usize)>>,
) -> Result<(Vec<Contour>, Bbox), T> {
    let mut contours = vec![];
    let mut bbox = Bbox::default();
    for mut path in paths {
        if let Some(first) = path.next() {
            pen.start(first)?;
            for next in path {
                pen.line(next)?;
            }
            let contour = pen.done();
            for point in contour.iter() {
                bbox.x_min = bbox.x_min.min(point.x);
                bbox.x_max = bbox.x_max.max(point.x);
                bbox.y_min = bbox.y_min.min(point.y);
                bbox.y_max = bbox.y_max.max(point.y);
            }
            contours.push(Contour::from(contour));
        }
    }
    Ok((contours, bbox))
}

pub fn full_glyph<T>(
    bitmap: &Bitmap,
    pen: &mut impl PointTracer<Error = T>,
) -> Result<SimpleGlyph, T> {
    let graph = bitmap_to_graph(bitmap);
    let paths = trace(&graph);
    let (contours, bbox) = trace_paths(pen, paths.iter().map(|x| x.iter().copied()))?;
    Ok(SimpleGlyph {
        bbox,
        contours,
        instructions: vec![],
        overlaps: false,
    })
}
