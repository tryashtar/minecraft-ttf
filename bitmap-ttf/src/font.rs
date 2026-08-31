use std::collections::BTreeMap;

use read_fonts::{
    ps::agl,
    tables::glyf::{Anchor, CurvePoint, Transform},
};
use write_fonts::{
    BuilderError, FontBuilder, OffsetMarker,
    tables::{
        cmap::Cmap,
        colr::{BaseGlyph, Colr, Layer},
        cpal::{ColorRecord, Cpal},
        glyf::{
            Bbox, Component, ComponentFlags, CompositeGlyph, GlyfLocaBuilder, Glyph, SimpleGlyph,
            SomeGlyph,
        },
        head::{Head, MacStyle},
        hhea::Hhea,
        hmtx::Hmtx,
        loca::LocaFormat,
        name::{Name, NameRecord},
        os2::{Os2, SelectionFlags},
        post::Post,
    },
    types::{FWord, Fixed, GlyphId16, LongDateTime, NameId},
};

#[derive(Debug, PartialEq)]
pub struct GlyphInfo {
    width: u16,
    height: u16,
    base_layer: Option<SimpleGlyph>,
    base_offsets: Vec<(i16, i16)>,
    colored_layers: Vec<ColoredLayer>,
    colored_offsets: Vec<(i16, i16)>,
}

impl GlyphInfo {
    pub fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            base_layer: None,
            base_offsets: vec![],
            colored_layers: vec![],
            colored_offsets: vec![],
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ColoredLayer {
    glyph: SimpleGlyph,
    color: ColorRecord,
}

#[derive(Debug)]
pub struct FontPositions {
    ascent: f32,
    descent: f32,
    s_cap_height: f32,
    sx_height: f32,
    y_strikeout_position: f32,
    y_strikeout_size: f32,
    underline_position: f32,
    underline_thickness: f32,
    italic_angle: f32,
}

#[derive(Debug)]
pub struct FontInfo {
    names: Names,
    bold: bool,
    italic: bool,
    em: u16,
    created: jiff::Timestamp,
    modified: jiff::Timestamp,
}

#[derive(thiserror::Error, Debug)]
pub enum MakeFontError {
    #[error(transparent)]
    Glyph(#[from] write_fonts::error::Error),
    #[error(transparent)]
    Builder(#[from] write_fonts::error::BuilderError),
}

pub fn make_font(
    info: FontInfo,
    positions: &FontPositions,
    chars: &BTreeMap<char, GlyphInfo>,
    notdef: Glyph,
) -> Result<Vec<u8>, MakeFontError> {
    let mut name_buffer = [0; agl::MAX_NAME_LEN];
    for (char, data) in chars {
        let name = match agl::char_to_name(*char, &mut name_buffer) {
            None => {
                let char_int = *char as u32;
                match char_int {
                    0..=0xffff => format!("uni{:04X}", char_int),
                    _ => format!("u{:X}", char_int),
                }
            }
            Some(name) => name.to_owned(),
        };
    }
    let mut builder = FontBuilder::new();
    let (loca_format, widest, tallest) = add_glyphs(&mut builder, chars, &notdef)?;
    add_metrics(&mut builder, info, positions, widest, tallest, loca_format)?;
    Ok(builder.build())
}

fn translate_glyph(glyph: &mut SimpleGlyph, offset_x: i16, offset_y: i16) {
    glyph.contours = glyph
        .contours
        .iter()
        .map(|x| {
            x.iter()
                .map(|CurvePoint { x, y, on_curve }| CurvePoint {
                    x: *x + offset_x,
                    y: *y + offset_y,
                    on_curve: *on_curve,
                })
                .collect::<Vec<_>>()
                .into()
        })
        .collect();
    translate_bbox(&mut glyph.bbox, offset_x, offset_y);
}

fn translate_bbox(bbox: &mut Bbox, offset_x: i16, offset_y: i16) {
    let Bbox {
        x_min,
        y_min,
        x_max,
        y_max,
    } = bbox;
    *x_min += offset_x;
    *y_min += offset_y;
    *x_max += offset_x;
    *y_max += offset_y;
}

fn make_offset_component(
    base_index: u16,
    offset_x: i16,
    offset_y: i16,
    mut bbox: Bbox,
) -> (Component, Bbox) {
    translate_bbox(&mut bbox, offset_x, offset_y);
    (
        Component::new(
            GlyphId16::new(base_index),
            Anchor::Offset {
                x: offset_x,
                y: offset_y,
            },
            Transform::default(),
            ComponentFlags::default(),
        ),
        bbox,
    )
}

#[derive(Default)]
struct GlyphBuilder {
    builder: GlyfLocaBuilder,
    next_index: u16,
}

impl GlyphBuilder {
    fn add_glyph(&mut self, glyph: &impl SomeGlyph) -> Result<u16, write_fonts::error::Error> {
        self.builder.add_glyph(glyph)?;
        let index = self.next_index;
        self.next_index += 1;
        Ok(index)
    }
}

fn add_glyphs(
    builder: &mut FontBuilder,
    chars: &BTreeMap<char, GlyphInfo>,
    notdef: &Glyph,
) -> Result<(LocaFormat, i16, i16), MakeFontError> {
    let mut glyph_builder = GlyphBuilder::default();
    let mut hmtx = Hmtx::default();
    let mut palettes = vec![];
    let mut color_base_records = vec![];
    let mut color_layer_records = vec![];
    let mut cmap = Cmap::default();
    let mut widest = 0i16;
    let mut tallest = 0i16;
    glyph_builder.add_glyph(notdef)?;
    let mut component_pieces = vec![];
    let mut color_pieces = vec![];
    for (char, info) in chars.iter() {
        let glyph_index = match (&info.base_layer, info.base_offsets.as_slice()) {
            (None, _) | (_, []) => glyph_builder.add_glyph(&Glyph::Empty)?,
            (Some(layer), [(0, 0)]) => glyph_builder.add_glyph(layer)?,
            (Some(layer), [(x, y)]) => {
                let mut cloned = layer.clone();
                translate_glyph(&mut cloned, *x, *y);
                glyph_builder.add_glyph(&cloned)?
            }
            (Some(layer), [(x, y), rest @ ..]) => {
                let base_index = chars.len() + component_pieces.len();
                let (component, bbox) =
                    make_offset_component(base_index as u16, *x, *y, layer.bbox);
                let mut composite = CompositeGlyph::new(component, bbox);
                for (x, y) in rest {
                    let (component, bbox) =
                        make_offset_component(base_index as u16, *x, *y, layer.bbox);
                    composite.add_component(component, bbox);
                }
                component_pieces.push(layer);
                glyph_builder.add_glyph(&Glyph::Composite(composite))?
            }
        };
        if !info.colored_layers.is_empty() {
            color_pieces.push((glyph_index, &info.colored_layers, &info.colored_offsets));
        }
    }
    for glyph in component_pieces {
        glyph_builder.add_glyph(glyph)?;
    }
    let mut next_color_index = 0u16;
    for (main_index, layers, offsets) in color_pieces {
        let num_layers = (layers.len() * offsets.len()) as u16;
        color_base_records.push(BaseGlyph {
            glyph_id: GlyphId16::new(main_index),
            first_layer_index: next_color_index,
            num_layers,
        });
        next_color_index += num_layers;
        for layer in layers.iter() {
            let palette_index = match palettes.iter().position(|x| x == &layer.color) {
                None => {
                    palettes.push(layer.color.clone());
                    palettes.len() - 1
                }
                Some(index) => index,
            } as u16;
            for (x, y) in offsets.iter() {
                let mut cloned = layer.glyph.clone();
                translate_glyph(&mut cloned, *x, *y);
                let glyph_index = glyph_builder.add_glyph(&cloned)?;
                color_layer_records.push(Layer {
                    glyph_id: GlyphId16::new(glyph_index),
                    palette_index,
                });
            }
        }
    }
    let (glyf, loca, loca_format) = glyph_builder.builder.build();
    builder
        .add_table(&glyf)?
        .add_table(&loca)?
        .add_table(&hmtx)?;
    if !palettes.is_empty() {
        let cpal = Cpal::new(
            palettes.len() as u16,
            1,
            palettes.len() as u16,
            Some(palettes),
            vec![0],
        );
        let color_layer_len = color_layer_records.len();
        let colr = Colr::new(
            color_base_records.len() as u16,
            Some(color_base_records),
            Some(color_layer_records),
            color_layer_len as u16,
        );
        builder.add_table(&cpal)?.add_table(&colr)?;
    }
    Ok((loca_format, widest, tallest))
}

fn add_metrics(
    builder: &mut FontBuilder,
    info: FontInfo,
    positions: &FontPositions,
    widest: i16,
    tallest: i16,
    loca_format: LocaFormat,
) -> Result<(), BuilderError> {
    let names = info.names.into_table();
    let ascent = (info.em as f32 * positions.ascent) as u16;
    let descent = (info.em as f32 * positions.descent) as u16;
    let hhea = Hhea {
        ascender: FWord::new(ascent as i16),
        descender: FWord::new(-(descent as i16)),
        ..Default::default()
    };
    let (fs_selection, mac_style) = match (info.bold, info.italic) {
        (true, true) => (
            SelectionFlags::BOLD | SelectionFlags::ITALIC,
            MacStyle::BOLD | MacStyle::ITALIC,
        ),
        (true, false) => (SelectionFlags::BOLD, MacStyle::BOLD),
        (false, true) => (
            SelectionFlags::REGULAR | SelectionFlags::ITALIC,
            MacStyle::ITALIC,
        ),
        (false, false) => (SelectionFlags::REGULAR, MacStyle::empty()),
    };
    let weight = if info.bold { 700 } else { 400 };
    let os2 = Os2 {
        s_typo_ascender: ascent as i16,
        s_typo_descender: -(descent as i16),
        us_win_ascent: ascent,
        us_win_descent: descent,
        s_cap_height: Some((info.em as f32 * positions.s_cap_height) as i16),
        sx_height: Some((info.em as f32 * positions.sx_height) as i16),
        y_strikeout_position: (info.em as f32 * positions.y_strikeout_position) as i16,
        y_strikeout_size: (info.em as f32 * positions.y_strikeout_size) as i16,
        s_typo_line_gap: 0,
        fs_selection,
        us_weight_class: weight,
        ..Default::default()
    };
    let post = Post {
        underline_position: FWord::new((info.em as f32 * positions.underline_position) as i16),
        underline_thickness: FWord::new((info.em as f32 * positions.underline_thickness) as i16),
        italic_angle: Fixed::from_f64(if info.italic {
            positions.italic_angle as f64
        } else {
            0.0
        }),
        ..Default::default()
    };
    let epoch = jiff::Timestamp::constant(-2_082_844_800, 0);
    let head = Head {
        units_per_em: info.em,
        x_min: 0,
        x_max: widest,
        y_min: -(descent as i16),
        y_max: tallest,
        created: LongDateTime::new(info.created.as_second() - epoch.as_second()),
        modified: LongDateTime::new(info.modified.as_second() - epoch.as_second()),
        mac_style,
        index_to_loc_format: loca_format as i16,
        ..Default::default()
    };
    builder
        .add_table(&names)?
        .add_table(&hhea)?
        .add_table(&os2)?
        .add_table(&post)?
        .add_table(&head)?;
    Ok(())
}

#[derive(Debug)]
struct Names {
    copyright: String,
    family: String,
    style: String,
    unique: String,
    full: String,
    version: String,
    postscript: String,
    sample: String,
}

impl Names {
    fn into_table(self) -> Name {
        let mut names = vec![];
        Self::add_record(&mut names, NameId::COPYRIGHT_NOTICE, self.copyright);
        Self::add_record(&mut names, NameId::FAMILY_NAME, self.family);
        Self::add_record(&mut names, NameId::SUBFAMILY_NAME, self.style);
        Self::add_record(&mut names, NameId::UNIQUE_ID, self.unique);
        Self::add_record(&mut names, NameId::FULL_NAME, self.full);
        Self::add_record(&mut names, NameId::VERSION_STRING, self.version);
        Self::add_record(&mut names, NameId::POSTSCRIPT_NAME, self.postscript);
        Self::add_record(&mut names, NameId::SAMPLE_TEXT, self.sample);
        Name::new(names)
    }

    fn add_record(names: &mut Vec<NameRecord>, id: NameId, value: String) {
        names.push(NameRecord {
            platform_id: 3,
            encoding_id: 1,
            language_id: 0x0409,
            name_id: id,
            string: OffsetMarker::new(value.clone()),
        });
        names.push(NameRecord {
            platform_id: 1,
            encoding_id: 0,
            language_id: 0,
            name_id: id,
            string: OffsetMarker::new(value),
        });
    }
}
