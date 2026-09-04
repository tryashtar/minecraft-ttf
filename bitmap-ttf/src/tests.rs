use crate::{
    bitmap::Bitmap,
    vectorize::{bitmap_to_graph, trace},
};

#[test]
fn efficient_square() {
    let mut bitmap = Bitmap::new(8, 8);
    for y in 1..bitmap.height() - 1 {
        for x in 1..bitmap.width() - 1 {
            bitmap.set(x, y, true);
        }
    }
    let graph = bitmap_to_graph(&bitmap);
    assert_eq!(graph.node_count(), 24);
    let paths = trace(&graph);
    assert_eq!(paths.len(), 1);
    let path = &paths[0];
    assert_eq!(path.len(), 25);
}

#[test]
fn draw() {
    let mut bitmap = Bitmap::new(8, 8);
    bitmap.set(6, 6, true);
    let mut smaller = Bitmap::new(1, 8);
    bitmap.draw_onto(&mut smaller, -6, 0);
    assert!(smaller.get(0, 6));
}
