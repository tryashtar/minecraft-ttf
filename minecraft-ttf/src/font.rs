use std::{collections::HashMap, fmt::Display};

use bitmap_ttf::{
    bitmap::Bitmap,
    font::{CharMap, ColoredLayer, CoordsError, FontMeta, GlyphInfo, Names},
    read_fonts::tables::glyf::CurvePoint,
    vectorize::{PointTracer, full_glyph},
    write_fonts::tables::cpal::ColorRecord,
};
use num_traits::ToPrimitive;

use crate::providers::{self, colors_to_bitmaps};

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
pub enum Style {
    Regular,
    Bold,
    Italic,
    BoldItalic,
}

impl Style {
    pub fn info(self) -> StyleInfo {
        match self {
            Self::Regular => StyleInfo {
                name: "Regular",
                bold: false,
                italic: false,
            },
            Self::Bold => StyleInfo {
                name: "Bold",
                bold: true,
                italic: false,
            },
            Self::Italic => StyleInfo {
                name: "Italic",
                bold: false,
                italic: true,
            },
            Self::BoldItalic => StyleInfo {
                name: "Bold Italic",
                bold: true,
                italic: true,
            },
        }
    }
}

#[derive(Debug)]
pub struct StyleInfo {
    pub name: &'static str,
    pub bold: bool,
    pub italic: bool,
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
    pub chars: CharMap,
    pub missing_glyph: GlyphInfo,
    pub scales: HashMap<(u16, u16), Vec<char>>,
    pub colored: Vec<char>,
    pub font_em: u16,
}

pub fn create_font<'a>(
    providers: impl Iterator<Item = &'a providers::Provider>,
    missing_glyph: Option<&Bitmap>,
    bold: bool,
    italic: bool,
) -> Result<FontInfo, CoordsError> {
    let mut scales = HashMap::new();
    let mut chars = CharMap::new();
    let mut colored = vec![];
    let font_em = 1200;
    let pixel_scale = 100.0;
    let missing_glyph = match missing_glyph {
        None => None,
        Some(glyph) => Some(bitmap_glyph(
            glyph,
            &[],
            BitmapGlyphSizes {
                advance: glyph.width().to_f32().ok_or(CoordsError::SizeCast)? + 1.0,
                height: 8,
                ascent: 7,
                pixel_scale,
                bold_offset: bold.then_some(1.0),
                italic: italic.then_some((-6.0 / 8.0, 4.0)),
            },
        )?),
    }
    .unwrap_or(GlyphInfo::empty(0));
    for provider in providers {
        match provider {
            providers::Provider::Space(provider) => {
                for (char, width) in &provider.chars {
                    if chars.contains_key(char) {
                        continue;
                    }
                    let mut advance = *width;
                    if bold {
                        advance += 1.0;
                    }
                    advance *= pixel_scale;
                    let int_advance = advance.to_u16().ok_or(CoordsError::FloatCast)?;
                    let glyph = space_glyph(int_advance);
                    chars.insert(*char, glyph);
                }
            }
            providers::Provider::Bitmap(provider) => {
                for (char, data) in &provider.chars {
                    if chars.contains_key(char) {
                        continue;
                    }
                    let glyph = bitmap_glyph(
                        &data.bitmap,
                        &[],
                        BitmapGlyphSizes {
                            advance: data.advance,
                            height: provider.height,
                            ascent: provider.ascent,
                            pixel_scale,
                            bold_offset: bold.then_some(data.bold_offset),
                            italic: italic.then_some((
                                -6.0 / provider.height.to_f32().ok_or(CoordsError::FloatCast)?,
                                4.0,
                            )),
                        },
                    )?;
                    chars.insert(*char, glyph);
                    add_scale(&mut scales, *char, data.bitmap.height(), provider.height)?;
                }
            }
            providers::Provider::Image(provider) => {
                for (char, data) in &provider.chars {
                    if chars.contains_key(char) {
                        continue;
                    }
                    if provider.height < 0 || provider.ascent <= -16384 {
                        chars.insert(*char, GlyphInfo::empty(0));
                        continue;
                    }
                    let colored_layers = if provider.has_color {
                        colored.push(*char);
                        let image_segment = image::GenericImageView::view(
                            &provider.image,
                            data.content_box
                                .left
                                .to_u32()
                                .ok_or(CoordsError::SizeCast)?,
                            data.content_box.top.to_u32().ok_or(CoordsError::SizeCast)?,
                            data.content_box
                                .width
                                .to_u32()
                                .ok_or(CoordsError::SizeCast)?,
                            data.content_box
                                .height
                                .to_u32()
                                .ok_or(CoordsError::SizeCast)?,
                        );
                        let planes = colors_to_bitmaps(image_segment, 10);
                        planes
                            .into_iter()
                            .map(|(color, bitmap)| {
                                let record = ColorRecord {
                                    blue: color.0[2],
                                    green: color.0[1],
                                    red: color.0[0],
                                    alpha: color.0[3],
                                };
                                (bitmap, record)
                            })
                            .collect()
                    } else {
                        vec![]
                    };
                    let glyph = bitmap_glyph(
                        &data.bitmap,
                        colored_layers.as_slice(),
                        BitmapGlyphSizes {
                            advance: data.advance,
                            height: provider.height,
                            ascent: provider.ascent,
                            pixel_scale,
                            bold_offset: bold.then_some(data.bold_offset),
                            italic: italic.then_some((
                                -6.0 / provider.height.to_f32().ok_or(CoordsError::FloatCast)?,
                                4.0,
                            )),
                        },
                    )?;
                    chars.insert(*char, glyph);
                    add_scale(&mut scales, *char, data.bitmap.height(), provider.height)?;
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

fn space_glyph(width: u16) -> GlyphInfo {
    GlyphInfo::empty(width)
}

fn add_scale(
    scales: &mut HashMap<(u16, u16), Vec<char>>,
    char: char,
    bitmap_height: usize,
    height: i32,
) -> Result<(), CoordsError> {
    let bitmap_height = bitmap_height.to_u16().ok_or(CoordsError::SizeCast)?;
    let height = height.try_into().map_err(CoordsError::IntCast)?;
    scales
        .entry((bitmap_height, height))
        .or_default()
        .push(char);
    Ok(())
}

struct BitmapGlyphSizes {
    advance: f32,
    height: i32,
    ascent: i32,
    pixel_scale: f32,
    bold_offset: Option<f32>,
    italic: Option<(f32, f32)>,
}

fn bitmap_glyph(
    base_layer: &Bitmap,
    colored_layers: &[(Bitmap, ColorRecord)],
    sizes: BitmapGlyphSizes,
) -> Result<GlyphInfo, CoordsError> {
    let mask_height = base_layer.height();
    let mask_height_f = mask_height.to_f32().ok_or(CoordsError::FloatCast)?;
    let height_f = sizes.height.to_f32().ok_or(CoordsError::FloatCast)?;
    let ascent_f = sizes.ascent.to_f32().ok_or(CoordsError::FloatCast)?;
    let offset_x = sizes.italic.map(|x| x.0).unwrap_or(0.0);
    let offset_y = if sizes.height == 0 {
        0.0
    } else {
        ((height_f - ascent_f) * mask_height_f) / height_f
    };
    let scale = if mask_height == 0 {
        0.0
    } else {
        (height_f * sizes.pixel_scale) / mask_height_f
    };
    let mut pen = TracePen::new(
        mask_height,
        scale,
        (offset_x, -offset_y),
        sizes.italic.map(|x| x.1),
    );
    let char_width = sizes.advance * sizes.pixel_scale;
    let char_height = height_f * sizes.pixel_scale;
    let mut colored_paths = vec![];
    let mut colored_base = None;
    if colored_layers.len() == 1 {
        colored_base = Some(colored_layers[0].1.clone());
    } else {
        for (bitmap, color) in colored_layers {
            let path = full_glyph(bitmap, &mut pen)?;
            colored_paths.push(ColoredLayer {
                glyph: path,
                color: color.clone(),
            });
        }
    }
    let glyph_info = match sizes.bold_offset {
        None => GlyphInfo {
            width: char_width.to_u16().ok_or(CoordsError::FloatCast)?,
            height: char_height.to_u16().ok_or(CoordsError::FloatCast)?,
            base_layer: Some(full_glyph(base_layer, &mut pen)?),
            base_offsets: vec![(0, 0)],
            base_color: colored_base,
            colored_layers: colored_paths,
            colored_offsets: vec![(0, 0)],
        },
        Some(val) => {
            let color_offsets = vec![
                (0, 0),
                (
                    (val * sizes.pixel_scale)
                        .to_i16()
                        .ok_or(CoordsError::FloatCast)?,
                    0,
                ),
            ];
            let pixel_offset = if sizes.height == 0 {
                0.0
            } else {
                (val * mask_height_f) / height_f
            };
            if pixel_offset.fract() == 0.0 {
                let int_offset = pixel_offset.to_isize().ok_or(CoordsError::FloatCast)?;
                let mut bold_bitmap = Bitmap::new(
                    (base_layer.width() as isize + int_offset) as usize,
                    mask_height,
                );
                base_layer.draw_onto(&mut bold_bitmap, 0, 0);
                base_layer.draw_onto(&mut bold_bitmap, int_offset, 0);
                GlyphInfo {
                    width: (char_width + (val * sizes.pixel_scale))
                        .to_u16()
                        .ok_or(CoordsError::FloatCast)?,
                    height: char_height.to_u16().ok_or(CoordsError::FloatCast)?,
                    base_layer: Some(full_glyph(&bold_bitmap, &mut pen)?),
                    base_offsets: vec![(0, 0)],
                    base_color: colored_base,
                    colored_layers: colored_paths,
                    colored_offsets: color_offsets,
                }
            } else {
                GlyphInfo {
                    width: char_width.to_u16().ok_or(CoordsError::FloatCast)?,
                    height: char_height.to_u16().ok_or(CoordsError::FloatCast)?,
                    base_layer: Some(full_glyph(base_layer, &mut pen)?),
                    base_offsets: color_offsets.clone(),
                    base_color: colored_base,
                    colored_layers: colored_paths,
                    colored_offsets: color_offsets,
                }
            }
        }
    };
    Ok(glyph_info)
}

#[derive(Debug)]
struct TracePen {
    height: usize,
    scale: f32,
    offset: (f32, f32),
    shear: Option<f32>,
    path: Vec<CurvePoint>,
}

impl TracePen {
    fn new(height: usize, scale: f32, offset: (f32, f32), shear: Option<f32>) -> Self {
        Self {
            height,
            scale,
            offset,
            shear,
            path: vec![],
        }
    }

    fn convert_point(&self, point: (usize, usize)) -> Result<CurvePoint, CoordsError> {
        let (x, y) = point;
        let y = self.height - y;
        let mut xf = x.to_f32().ok_or(CoordsError::SizeCast)?;
        let mut yf = y.to_f32().ok_or(CoordsError::SizeCast)?;
        if let Some(shear) = self.shear {
            xf += yf / shear;
        }
        let (ox, oy) = self.offset;
        xf += ox;
        yf += oy;
        xf *= self.scale;
        yf *= self.scale;
        Ok(CurvePoint {
            x: xf.to_i16().ok_or(CoordsError::FloatCast)?,
            y: yf.to_i16().ok_or(CoordsError::FloatCast)?,
            on_curve: true,
        })
    }

    fn push_point(&mut self, point: CurvePoint) {
        if let Some([p1, p2]) = self.path.last_chunk_mut::<2>()
            && collinear(p1, p2, &point)
        {
            *p2 = point;
        } else {
            self.path.push(point);
        }
    }
}

impl PointTracer for TracePen {
    type Error = CoordsError;

    fn start(&mut self, point: (usize, usize)) -> Result<(), CoordsError> {
        let converted = self.convert_point(point)?;
        self.path = vec![converted];
        Ok(())
    }

    fn line(&mut self, point: (usize, usize)) -> Result<(), CoordsError> {
        let converted = self.convert_point(point)?;
        self.push_point(converted);
        Ok(())
    }

    fn done(&mut self) -> Vec<CurvePoint> {
        self.path.pop();
        std::mem::take(&mut self.path)
    }
}

fn collinear(p1: &CurvePoint, p2: &CurvePoint, p3: &CurvePoint) -> bool {
    let (x1, y1): (i32, i32) = ((p2.x - p1.x).into(), (p2.y - p1.y).into());
    let (x2, y2): (i32, i32) = ((p3.x - p1.x).into(), (p3.y - p1.y).into());
    (x1 * y2) == (x2 * y1)
}

pub fn font_meta(
    name: String,
    style: &StyleInfo,
    font_em: u16,
    created: jiff::Timestamp,
    modified: jiff::Timestamp,
) -> FontMeta {
    let names = Names {
        copyright: String::from("Copyright (c) 2009 Mojang AB"),
        unique: format!("{}.{}", name.replace(' ', ""), style.name.replace(' ', "")),
        full: format!("{} {}", name, style.name),
        version: String::from("Version 1.000"),
        postscript: format!("{}{}", name.replace(' ', ""), style.name.replace(' ', "")),
        sample: String::from("and the universe said I love you"),
        family: name,
        style: String::from(style.name),
    };
    FontMeta {
        names,
        bold: style.bold,
        italic: style.italic,
        em: font_em,
        created,
        modified,
    }
}
