use std::collections::HashMap;

use bitmap_ttf::bitmap::Bitmap;

#[derive(Debug)]
pub struct CharBitmap {
    bitmap: Bitmap,
    advance: f32,
    bold_offset: f32,
}

#[derive(Debug)]
pub struct BitmapProvider {
    height: i32,
    ascent: i32,
    chars: indexmap::IndexMap<char, CharBitmap>,
}

pub struct CharImage {
    image: image::SubImage<()>,
    bitmap: Bitmap,
    advance: f32,
    bold_offset: f32,
}

pub struct ImageProvider {
    height: i32,
    ascent: i32,
    has_color: bool,
    chars: indexmap::IndexMap<char, CharImage>,
}

#[derive(Debug)]
pub struct SpaceProvider {
    chars: HashMap<char, f32>,
}
