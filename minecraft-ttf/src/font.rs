use std::{collections::HashMap, fmt::Display};

use bitmap_ttf::{bitmap::Bitmap, font::GlyphInfo, read_fonts::tables::cpal::ColorRecord};

use crate::providers;

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
pub enum Style {
    Regular,
    Bold,
    Italic,
    BoldItalic,
}

impl Display for Style {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular => write!(f, "regular"),
            Self::Bold => write!(f, "bold"),
            Self::Italic => write!(f, "italic"),
            Self::BoldItalic => write!(f, "bold_italic"),
        }
    }
}

#[derive(Debug)]
pub struct FontInfo {
    chars: indexmap::IndexMap<char, GlyphInfo>,
    missing_glyph: GlyphInfo,
    scales: HashMap<(u32, u32), Vec<char>>,
    colored: Vec<char>,
    font_em: u16,
}

#[derive(thiserror::Error, Debug)]
pub enum CreateFontError {}

pub fn create_font<'a>(
    providers: impl Iterator<Item = &'a providers::Provider>,
    missing_glyph: Option<&Bitmap>,
    bold: bool,
    italic: bool,
) -> Result<FontInfo, CreateFontError> {
    let mut scales = HashMap::new();
    let mut chars = indexmap::IndexMap::new();
    let mut colored = vec![];
    let font_em = 1200;
    let pixel_scale = 100.0f32;
    let missing_glyph = missing_glyph
        .map(|x| {
            bitmap_glyph(
                x,
                &[],
                x.width() as f32 + 1.0,
                1.0,
                8,
                7,
                pixel_scale,
                bold,
                italic,
            )
        })
        .unwrap_or(GlyphInfo::empty(0));
    for provider in providers {
        match provider {
            providers::Provider::Space(provider) => {
                for (char, width) in &provider.chars {
                    if !chars.contains_key(char) {
                        let glyph = space_glyph(*width);
                        chars.insert(*char, glyph);
                    }
                }
            }
            providers::Provider::Bitmap(provider) => {
                for (char, data) in &provider.chars {
                    if !chars.contains_key(char) {
                        let glyph = bitmap_glyph(
                            &data.bitmap,
                            &[],
                            data.advance,
                            data.bold_offset,
                            provider.height,
                            provider.ascent,
                            pixel_scale,
                            bold,
                            italic,
                        );
                        chars.insert(*char, glyph);
                        add_scale(&mut scales, *char, data.bitmap.height(), provider.height);
                    }
                }
            }
            providers::Provider::Image(provider) => {
                for (char, data) in &provider.chars {
                    if !chars.contains_key(char) {
                        if provider.height > 0 && provider.ascent > -16384 {
                            let mut colored_layers = vec![];
                            if provider.has_color {
                                colored.push(*char);
                                todo!();
                            }
                            let glyph = bitmap_glyph(
                                &data.bitmap,
                                colored_layers.as_slice(),
                                data.advance,
                                data.bold_offset,
                                provider.height,
                                provider.ascent,
                                pixel_scale,
                                bold,
                                italic,
                            );
                            chars.insert(*char, glyph);
                            add_scale(&mut scales, *char, data.bitmap.height(), provider.height);
                        }
                    } else {
                        chars.insert(*char, GlyphInfo::empty(0));
                    }
                }
            }
        }
    }
    Ok(FontInfo {
        chars,
        missing_glyph,
        scales,
        colored,
        font_em,
    })
}

fn space_glyph(width: f32) -> GlyphInfo {
    GlyphInfo::empty(width.max(0.0) as u16)
}

fn add_scale(
    scales: &mut HashMap<(u32, u32), Vec<char>>,
    char: char,
    bitmap_height: usize,
    height: i32,
) {
    let height_ratio = (bitmap_height as u32, height.max(0) as u32);
    scales.entry(height_ratio).or_default().push(char);
}

fn bitmap_glyph(
    base_layer: &Bitmap,
    colored_layers: &[(Bitmap, ColorRecord)],
    advance: f32,
    bold_offset: f32,
    height: i32,
    ascent: i32,
    pixel_scale: f32,
    bold: bool,
    italic: bool,
) -> GlyphInfo {
    todo!()
}
