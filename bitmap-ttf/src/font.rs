use std::collections::BTreeMap;

use num_traits::ToPrimitive;
use read_fonts::tables::{
    cmap::PlatformId,
    glyf::{Anchor, CurvePoint, Transform},
};
use write_fonts::{
    FontBuilder, OffsetMarker,
    tables::{
        cmap::{Cmap, Cmap4, Cmap12, CmapSubtable, EncodingRecord, SequentialMapGroup},
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
    pub fn empty(width: u16) -> Self {
        Self {
            width,
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
    ascent: f64,
    descent: f64,
    s_cap_height: f64,
    sx_height: f64,
    y_strikeout_position: f64,
    y_strikeout_size: f64,
    underline_position: f64,
    underline_thickness: f64,
    italic_angle: f64,
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
    #[error(transparent)]
    IntCast(#[from] std::num::TryFromIntError),
    #[error("Converting number")]
    FloatCast,
    #[error("Adding number")]
    IntAdd,
}

pub fn make_font(
    info: FontInfo,
    positions: &FontPositions,
    chars: &BTreeMap<char, GlyphInfo>,
    notdef: Glyph,
) -> Result<Vec<u8>, MakeFontError> {
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

#[derive(Debug, Default)]
struct CmapBuilder {
    low_pairs: BTreeMap<u16, u16>,
    all_pairs: BTreeMap<char, u16>,
}

impl CmapBuilder {
    fn add(&mut self, char: char, glyph_id: u16) {
        if let Ok(small) = char.try_into() {
            self.low_pairs.insert(small, glyph_id);
        }
        self.all_pairs.insert(char, glyph_id);
    }

    fn build(&self) -> Cmap {
        let mut records = vec![];
        if !self.low_pairs.is_empty() {
            let cmap4 = build_cmap4(self.low_pairs.iter().map(|(a, b)| (*a, *b)));
            records.push(EncodingRecord::new(
                PlatformId::Unicode,
                3,
                CmapSubtable::Format4(cmap4.clone()),
            ));
            records.push(EncodingRecord::new(
                PlatformId::Windows,
                1,
                CmapSubtable::Format4(cmap4),
            ));
        }
        if self.all_pairs.len() > self.low_pairs.len() {
            let cmap12 = build_cmap12(self.all_pairs.iter().map(|(a, b)| (*a, *b)));
            records.push(EncodingRecord::new(
                PlatformId::Unicode,
                3,
                CmapSubtable::Format12(cmap12.clone()),
            ));
            records.push(EncodingRecord::new(
                PlatformId::Windows,
                1,
                CmapSubtable::Format12(cmap12),
            ));
        }
        Cmap::new(records)
    }
}

fn build_cmap4(pairs: impl Iterator<Item = (u16, u16)>) -> Cmap4 {
    let mut start_code = vec![];
    let mut end_code = vec![];
    let mut id_delta = vec![];
    let mut id_range_offsets = vec![];
    let mut current_range = None;
    for (char, glyph) in pairs {
        current_range = match current_range {
            None => Some((char, char, glyph)),
            Some((start, last, start_glyph)) => {
                if last + 1 == char {
                    Some((start, char, start_glyph))
                } else {
                    start_code.push(start);
                    end_code.push(last);
                    id_delta.push(
                        // "as" is guaranteed to wrap here
                        // https://doc.rust-lang.org/reference/expressions/operator-expr.html#r-expr.as.numeric.int-same-size
                        start_glyph.wrapping_sub(start) as i16,
                    );
                    id_range_offsets.push(0);
                    None
                }
            }
        };
    }
    if let Some((start, last, start_glyph)) = current_range {
        start_code.push(start);
        end_code.push(last);
        id_delta.push(
            // "as" is guaranteed to wrap here
            // https://doc.rust-lang.org/reference/expressions/operator-expr.html#r-expr.as.numeric.int-same-size
            start_glyph.wrapping_sub(start) as i16,
        );
        id_range_offsets.push(0);
        if last < 0xffff {
            start_code.push(0xffff);
            end_code.push(0xffff);
            id_delta.push(1);
            id_range_offsets.push(0);
        }
    }
    Cmap4 {
        language: 0,
        end_code,
        start_code,
        id_delta,
        id_range_offsets,
        glyph_id_array: vec![],
    }
}

fn build_cmap12(pairs: impl Iterator<Item = (char, u16)>) -> Cmap12 {
    let mut groups = vec![];
    let mut current_group = None;
    for (char, glyph) in pairs {
        current_group = match current_group {
            None => Some(SequentialMapGroup {
                start_char_code: char.into(),
                end_char_code: char.into(),
                start_glyph_id: glyph.into(),
            }),
            Some(group) => {
                if group.end_char_code + 1 == char.into() {
                    Some(SequentialMapGroup {
                        start_char_code: group.start_char_code,
                        end_char_code: char.into(),
                        start_glyph_id: group.start_glyph_id,
                    })
                } else {
                    groups.push(group);
                    None
                }
            }
        };
    }
    if let Some(group) = current_group {
        groups.push(group);
    }
    Cmap12 {
        language: 0,
        groups,
    }
}

fn add_glyphs(
    builder: &mut FontBuilder,
    chars: &BTreeMap<char, GlyphInfo>,
    notdef: &Glyph,
) -> Result<(LocaFormat, i16, i16), MakeFontError> {
    let mut glyph_builder = GlyphBuilder::default();
    let mut cmap_builder = CmapBuilder::default();
    let mut hmtx = Hmtx::default();
    let mut palettes = vec![];
    let mut color_base_records = vec![];
    let mut color_layer_records = vec![];
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
                let base_index: u16 = (chars.len() + component_pieces.len()).try_into()?;
                let (component, bbox) = make_offset_component(base_index, *x, *y, layer.bbox);
                let mut composite = CompositeGlyph::new(component, bbox);
                for (x, y) in rest {
                    let (component, bbox) = make_offset_component(base_index, *x, *y, layer.bbox);
                    composite.add_component(component, bbox);
                }
                component_pieces.push(layer);
                glyph_builder.add_glyph(&Glyph::Composite(composite))?
            }
        };
        if !info.colored_layers.is_empty() {
            color_pieces.push((glyph_index, &info.colored_layers, &info.colored_offsets));
        }
        cmap_builder.add(*char, glyph_index);
    }
    for glyph in component_pieces {
        glyph_builder.add_glyph(glyph)?;
    }
    let mut next_color_index = 0u16;
    for (main_index, layers, offsets) in color_pieces {
        let num_layers: u16 = (layers.len() * offsets.len()).try_into()?;
        color_base_records.push(BaseGlyph {
            glyph_id: GlyphId16::new(main_index),
            first_layer_index: next_color_index,
            num_layers,
        });
        next_color_index = next_color_index
            .checked_add(num_layers)
            .ok_or(MakeFontError::IntAdd)?;
        for layer in layers.iter() {
            let palette_index: u16 = match palettes.iter().position(|x| x == &layer.color) {
                None => {
                    palettes.push(layer.color.clone());
                    palettes.len() - 1
                }
                Some(index) => index,
            }
            .try_into()?;
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
    let cmap = cmap_builder.build();
    builder
        .add_table(&glyf)?
        .add_table(&loca)?
        .add_table(&hmtx)?
        .add_table(&cmap)?;
    if !palettes.is_empty() {
        let cpal = Cpal::new(
            palettes.len().try_into()?,
            1,
            palettes.len().try_into()?,
            Some(palettes),
            vec![0],
        );
        let color_layer_len = color_layer_records.len().try_into()?;
        let colr = Colr::new(
            color_base_records.len().try_into()?,
            Some(color_base_records),
            Some(color_layer_records),
            color_layer_len,
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
) -> Result<(), MakeFontError> {
    let names = info.names.into_table();
    let em: f64 = info.em.into();
    let ascent: u16 = (em * positions.ascent)
        .round()
        .to_u16()
        .ok_or(MakeFontError::FloatCast)?;
    let descent: u16 = (em * positions.descent)
        .round()
        .to_u16()
        .ok_or(MakeFontError::FloatCast)?;
    let signed_ascent: i16 = ascent.try_into()?;
    let signed_descent: i16 = descent.try_into()?;
    let hhea = Hhea {
        ascender: FWord::new(signed_ascent),
        descender: FWord::new(-signed_descent),
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
        s_typo_ascender: signed_ascent,
        s_typo_descender: -signed_descent,
        us_win_ascent: ascent,
        us_win_descent: descent,
        s_cap_height: Some(
            (em * positions.s_cap_height)
                .round()
                .to_i16()
                .ok_or(MakeFontError::FloatCast)?,
        ),
        sx_height: Some(
            (em * positions.sx_height)
                .round()
                .to_i16()
                .ok_or(MakeFontError::FloatCast)?,
        ),
        y_strikeout_position: (em * positions.y_strikeout_position)
            .round()
            .to_i16()
            .ok_or(MakeFontError::FloatCast)?,
        y_strikeout_size: (em * positions.y_strikeout_size)
            .round()
            .to_i16()
            .ok_or(MakeFontError::FloatCast)?,
        s_typo_line_gap: 0,
        fs_selection,
        us_weight_class: weight,
        ..Default::default()
    };
    let post = Post {
        underline_position: FWord::new(
            (em * positions.underline_position)
                .round()
                .to_i16()
                .ok_or(MakeFontError::FloatCast)?,
        ),
        underline_thickness: FWord::new(
            (em * positions.underline_thickness)
                .round()
                .to_i16()
                .ok_or(MakeFontError::FloatCast)?,
        ),
        italic_angle: Fixed::from_f64(if info.italic {
            positions.italic_angle
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
        y_min: -signed_descent,
        y_max: tallest,
        created: LongDateTime::new(info.created.as_second() - epoch.as_second()),
        modified: LongDateTime::new(info.modified.as_second() - epoch.as_second()),
        mac_style,
        // enum cast
        // https://doc.rust-lang.org/reference/expressions/operator-expr.html#r-expr.as.enum.discriminant
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
            string: OffsetMarker::new(value),
        });
    }
}
