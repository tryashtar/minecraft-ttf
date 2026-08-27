use bitmap_ttf::{
    bitmap::Bitmap,
    vectorize::{bitmap_to_graph, trace},
};

fn main() {
    let mut test = Bitmap::new(8, 8);
    test.set(4, 4, true);
    test.set(4, 3, true);
    let graph = bitmap_to_graph(&test);
    let walks = trace(&graph);
    for walk in walks {
        println!("{:?}", walk);
    }
}
