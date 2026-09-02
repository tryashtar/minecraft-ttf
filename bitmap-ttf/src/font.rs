use std::collections::BTreeMap;

use num_traits::ToPrimitive;
use read_fonts::{
    tables::{
        cmap::PlatformId,
        glyf::{Anchor, CurvePoint, Transform},
        head::Flags,
    },
    types::Version16Dot16,
};
use tracing::{Level, debug, span};
use write_fonts::{
    FontBuilder, OffsetMarker,
    tables::{
        cmap::{Cmap, Cmap4, Cmap12, CmapSubtable, EncodingRecord, SequentialMapGroup},
        colr::{BaseGlyph, Colr, Layer},
        cpal::{ColorRecord, Cpal},
        glyf::{
            Bbox, Component, ComponentFlags, CompositeGlyph, GlyfLocaBuilder, Glyph, SimpleGlyph,
        },
        head::{Head, MacStyle},
        hhea::Hhea,
        hmtx::Hmtx,
        loca::LocaFormat,
        maxp::Maxp,
        name::{Name, NameRecord},
        os2::{Os2, SelectionFlags},
        post::Post,
        vmtx::LongMetric,
    },
    types::{FWord, Fixed, GlyphId16, LongDateTime, NameId},
};

#[derive(Debug, PartialEq)]
pub struct GlyphInfo {
    pub width: u16,
    pub height: u16,
    pub base_layer: Option<SimpleGlyph>,
    pub base_offsets: Vec<(i16, i16)>,
    pub colored_layers: Vec<ColoredLayer>,
    pub colored_offsets: Vec<(i16, i16)>,
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
    pub glyph: SimpleGlyph,
    pub color: ColorRecord,
}

#[derive(Debug)]
pub struct FontPositions {
    pub ascent: f64,
    pub descent: f64,
    pub s_cap_height: f64,
    pub sx_height: f64,
    pub y_strikeout_position: f64,
    pub y_strikeout_size: f64,
    pub underline_position: f64,
    pub underline_thickness: f64,
    pub italic_angle: f64,
}

#[derive(Debug)]
pub struct FontMeta {
    pub names: Names,
    pub bold: bool,
    pub italic: bool,
    pub em: u16,
    pub created: jiff::Timestamp,
    pub modified: jiff::Timestamp,
}

#[derive(thiserror::Error, Debug)]
pub enum MakeFontError {
    #[error(transparent)]
    Glyph(#[from] write_fonts::error::Error),
    #[error(transparent)]
    Builder(#[from] write_fonts::error::BuilderError),
    #[error(transparent)]
    Coords(#[from] CoordsError),
}

#[derive(thiserror::Error, Debug)]
pub enum CoordsError {
    #[error(transparent)]
    IntCast(#[from] std::num::TryFromIntError),
    #[error("Converting number")]
    SizeCast,
    #[error("Converting number")]
    FloatCast,
    #[error("Adding number")]
    IntAdd,
}

pub fn make_font(
    meta: FontMeta,
    positions: &FontPositions,
    smallest_legible: u16,
    chars: &BTreeMap<char, GlyphInfo>,
    notdef: &SimpleGlyph,
) -> Result<Vec<u8>, MakeFontError> {
    let mut builder = FontBuilder::new();
    let (loca_format, glyph_count, widest, tallest) = add_glyphs(&mut builder, chars, notdef)?;
    add_metrics(
        &mut builder,
        meta,
        positions,
        glyph_count,
        widest,
        tallest,
        smallest_legible,
        loca_format,
    )?;
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

#[derive(Default)]
struct GlyphBuilder {
    builder: GlyfLocaBuilder,
    hmtx: Hmtx,
    next_index: u16,
    max_points: u16,
    max_contours: u16,
}

impl GlyphBuilder {
    fn next(&mut self, advance: u16) -> u16 {
        self.hmtx.h_metrics.push(LongMetric {
            advance,
            side_bearing: 0,
        });
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    fn add_empty_glyph(&mut self, advance: u16) -> Result<u16, write_fonts::error::Error> {
        self.builder.add_glyph(&Glyph::Empty)?;
        Ok(self.next(advance))
    }

    fn add_simple_glyph(
        &mut self,
        glyph: &SimpleGlyph,
        advance: u16,
    ) -> Result<u16, write_fonts::error::Error> {
        self.builder.add_glyph(glyph)?;
        self.max_contours = self.max_contours.max(glyph.contours.len() as u16);
        for contour in glyph.contours.iter() {
            self.max_points = self.max_points.max(contour.len() as u16);
        }
        Ok(self.next(advance))
    }

    fn build_maxp(&self) -> Maxp {
        Maxp {
            num_glyphs: self.next_index,
            max_points: Some(self.max_points),
            max_contours: Some(self.max_contours),
            max_composite_points: Some(0),
            max_composite_contours: Some(0),
            max_zones: Some(2),
            max_twilight_points: Some(0),
            max_storage: Some(0),
            max_function_defs: Some(0),
            max_instruction_defs: Some(0),
            max_stack_elements: Some(0),
            max_size_of_instructions: Some(0),
            max_component_elements: Some(0),
            max_component_depth: Some(0),
        }
    }
}

#[derive(Debug, Default)]
struct CmapBuilder {
    low_pairs: indexmap::IndexMap<u16, u16>,
    all_pairs: indexmap::IndexMap<char, u16>,
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

#[derive(Debug, Copy, Clone)]
struct Cmap4Range {
    start: u16,
    end: u16,
    start_glyph: u16,
}

#[derive(Debug, Default)]
struct Cmap4RangeBuilder {
    start_code: Vec<u16>,
    end_code: Vec<u16>,
    id_delta: Vec<i16>,
    id_range_offsets: Vec<u16>,
    current_range: Option<Cmap4Range>,
}

impl Cmap4RangeBuilder {
    fn consume(&mut self, char: u16, glyph: u16) {
        if let Some(current) = self.current_range.as_mut()
            && current.end + 1 == char
        {
            current.end = char;
        } else {
            let new_row = Cmap4Range {
                start: char,
                end: char,
                start_glyph: glyph,
            };
            if let Some(old) = self.current_range.replace(new_row) {
                self.push(old);
            }
        }
    }

    fn push(
        &mut self,
        Cmap4Range {
            start,
            end,
            start_glyph,
        }: Cmap4Range,
    ) {
        debug!(
            "range from {} ({:04X}) to {} ({:04X}) ({} chars)",
            char::from_u32(start.into()).unwrap_or('\0'),
            start,
            char::from_u32(end.into()).unwrap_or('\0'),
            end,
            end - start + 1
        );
        self.start_code.push(start);
        self.end_code.push(end);
        self.id_delta.push(
            // "as" is guaranteed to wrap here
            // https://doc.rust-lang.org/reference/expressions/operator-expr.html#r-expr.as.numeric.int-same-size
            start_glyph.wrapping_sub(start) as i16,
        );
        self.id_range_offsets.push(0);
    }

    fn done(&mut self) {
        if let Some(last) = self.current_range.take() {
            let sentinel = last.end < 0xffff;
            self.push(last);
            if sentinel {
                self.start_code.push(0xffff);
                self.end_code.push(0xffff);
                self.id_delta.push(1);
                self.id_range_offsets.push(0);
            }
        }
    }
}

fn build_cmap4(pairs: impl Iterator<Item = (u16, u16)>) -> Cmap4 {
    let span = span!(Level::DEBUG, "building cmap4");
    let _guard = span.enter();
    let mut builder = Cmap4RangeBuilder::default();
    for (char, glyph) in pairs {
        builder.consume(char, glyph);
    }
    builder.done();
    debug!("{} ranges", builder.end_code.len());
    Cmap4 {
        language: 0,
        end_code: builder.end_code,
        start_code: builder.start_code,
        id_delta: builder.id_delta,
        id_range_offsets: builder.id_range_offsets,
        glyph_id_array: vec![],
    }
}

#[derive(Debug, Default)]
struct Cmap12RangeBuilder {
    ranges: Vec<SequentialMapGroup>,
    current_range: Option<SequentialMapGroup>,
}

impl Cmap12RangeBuilder {
    fn consume(&mut self, char: char, glyph: u16) {
        let char_u32 = char.into();
        if let Some(current) = self.current_range.as_mut()
            && current.end_char_code + 1 == char_u32
        {
            current.end_char_code = char_u32;
        } else {
            let new_row = SequentialMapGroup {
                start_char_code: char_u32,
                end_char_code: char_u32,
                start_glyph_id: glyph.into(),
            };
            if let Some(old) = self.current_range.replace(new_row) {
                self.push(old);
            }
        }
    }

    fn push(&mut self, range: SequentialMapGroup) {
        debug!(
            "range from {} ({:04X}) to {} ({:04X}) ({} chars)",
            char::from_u32(range.start_char_code).unwrap_or('\0'),
            range.start_char_code,
            char::from_u32(range.end_char_code).unwrap_or('\0'),
            range.end_char_code,
            range.end_char_code - range.start_char_code + 1
        );
        self.ranges.push(range);
    }

    fn done(&mut self) {
        if let Some(last) = self.current_range.take() {
            self.push(last);
        }
    }
}

fn build_cmap12(pairs: impl Iterator<Item = (char, u16)>) -> Cmap12 {
    let span = span!(Level::DEBUG, "building cmap12");
    let _guard = span.enter();
    let mut builder = Cmap12RangeBuilder::default();
    for (char, glyph) in pairs {
        builder.consume(char, glyph);
    }
    builder.done();
    debug!("{} ranges", builder.ranges.len());
    Cmap12 {
        language: 0,
        groups: builder.ranges,
    }
}

#[derive(Debug, Default)]
struct GlyphSize {
    start_box: Bbox,
    start_advance: i16,
    largest_box: Bbox,
    largest_advance: i16,
}

impl GlyphSize {
    fn new(size: Bbox, advance: u16) -> Self {
        Self {
            start_box: size,
            start_advance: advance as i16,
            largest_box: size,
            largest_advance: advance as i16,
        }
    }

    fn update(&mut self, offset: (i16, i16)) -> Result<(), CoordsError> {
        let (ox, oy) = offset;
        self.largest_box.x_max = self.largest_box.x_max.max(
            self.start_box
                .x_max
                .checked_add(ox)
                .ok_or(CoordsError::IntAdd)?,
        );
        self.largest_box.x_min = self.largest_box.x_min.min(
            self.start_box
                .x_min
                .checked_add(ox)
                .ok_or(CoordsError::IntAdd)?,
        );
        self.largest_box.y_max = self.largest_box.y_max.max(
            self.start_box
                .y_max
                .checked_add(oy)
                .ok_or(CoordsError::IntAdd)?,
        );
        self.largest_box.y_min = self.largest_box.y_min.min(
            self.start_box
                .y_min
                .checked_add(oy)
                .ok_or(CoordsError::IntAdd)?,
        );
        self.largest_advance = self.largest_advance.max(
            self.start_advance
                .checked_add(ox)
                .ok_or(CoordsError::IntAdd)?,
        );
        Ok(())
    }
}

fn add_glyphs(
    builder: &mut FontBuilder,
    chars: &BTreeMap<char, GlyphInfo>,
    notdef: &SimpleGlyph,
) -> Result<(LocaFormat, u16, i16, i16), MakeFontError> {
    let mut glyph_builder = GlyphBuilder::default();
    let mut cmap_builder = CmapBuilder::default();
    let mut palettes = vec![];
    let mut color_base_records = vec![];
    let mut color_layer_records = vec![];
    let mut widest = 0i16;
    let mut tallest = 0i16;
    glyph_builder.add_simple_glyph(notdef, 0)?;
    let mut color_pieces = vec![];
    for (char, info) in chars.iter() {
        let bbox = info.base_layer.as_ref().map(|x| x.bbox).unwrap_or_default();
        let mut size = GlyphSize::new(bbox, info.width);
        let glyph_index = match (&info.base_layer, info.base_offsets.as_slice()) {
            (None, _) | (_, []) => glyph_builder.add_empty_glyph(
                size.largest_advance
                    .try_into()
                    .map_err(CoordsError::IntCast)?,
            )?,
            (Some(layer), [(0, 0)]) => glyph_builder.add_simple_glyph(
                layer,
                size.largest_advance
                    .try_into()
                    .map_err(CoordsError::IntCast)?,
            )?,
            (Some(layer), offsets) => {
                let mut contours = vec![];
                for (x, y) in offsets {
                    size.update((*x, *y))?;
                    let mut cloned = layer.clone();
                    translate_glyph(&mut cloned, *x, *y);
                    contours.extend(cloned.contours);
                }
                let final_glyph = SimpleGlyph {
                    bbox: size.largest_box,
                    contours,
                    instructions: vec![],
                    overlaps: true,
                };
                glyph_builder.add_simple_glyph(
                    &final_glyph,
                    size.largest_advance
                        .try_into()
                        .map_err(CoordsError::IntCast)?,
                )?
            }
        };
        let advance = size
            .largest_advance
            .try_into()
            .map_err(CoordsError::IntCast)?;
        if !info.colored_layers.is_empty() {
            color_pieces.push((
                glyph_index,
                advance,
                &info.colored_layers,
                &info.colored_offsets,
            ));
        }
        cmap_builder.add(*char, glyph_index);
        widest = widest.max(size.largest_box.x_max);
        tallest = tallest.max(size.largest_box.y_max);
    }
    let mut next_color_index = 0u16;
    for (main_index, advance, layers, offsets) in color_pieces {
        let num_layers = (layers.len() * offsets.len())
            .to_u16()
            .ok_or(CoordsError::SizeCast)?;
        color_base_records.push(BaseGlyph {
            glyph_id: GlyphId16::new(main_index),
            first_layer_index: next_color_index,
            num_layers,
        });
        next_color_index = next_color_index
            .checked_add(num_layers)
            .ok_or(CoordsError::IntAdd)?;
        for layer in layers.iter() {
            let palette_index = match palettes.iter().position(|x| x == &layer.color) {
                None => {
                    palettes.push(layer.color.clone());
                    palettes.len() - 1
                }
                Some(index) => index,
            }
            .to_u16()
            .ok_or(CoordsError::SizeCast)?;
            for (x, y) in offsets.iter() {
                let mut cloned = layer.glyph.clone();
                translate_glyph(&mut cloned, *x, *y);
                let glyph_index = glyph_builder.add_simple_glyph(&cloned, advance)?;
                color_layer_records.push(Layer {
                    glyph_id: GlyphId16::new(glyph_index),
                    palette_index,
                });
            }
        }
    }
    let maxp = glyph_builder.build_maxp();
    let (glyf, loca, loca_format) = glyph_builder.builder.build();
    let cmap = cmap_builder.build();
    builder
        .add_table(&glyf)?
        .add_table(&loca)?
        .add_table(&glyph_builder.hmtx)?
        .add_table(&maxp)?
        .add_table(&cmap)?;
    if !palettes.is_empty() {
        let cpal = Cpal::new(
            palettes.len().to_u16().ok_or(CoordsError::SizeCast)?,
            1,
            palettes.len().to_u16().ok_or(CoordsError::SizeCast)?,
            Some(palettes),
            vec![0],
        );
        let color_layer_len = color_layer_records
            .len()
            .to_u16()
            .ok_or(CoordsError::SizeCast)?;
        let colr = Colr::new(
            color_base_records
                .len()
                .to_u16()
                .ok_or(CoordsError::SizeCast)?,
            Some(color_base_records),
            Some(color_layer_records),
            color_layer_len,
        );
        builder.add_table(&cpal)?.add_table(&colr)?;
    }
    Ok((loca_format, maxp.num_glyphs, widest, tallest))
}

fn add_metrics(
    builder: &mut FontBuilder,
    meta: FontMeta,
    positions: &FontPositions,
    glyph_count: u16,
    widest: i16,
    tallest: i16,
    smallest_legible: u16,
    loca_format: LocaFormat,
) -> Result<(), MakeFontError> {
    let names = meta.names.into_table();
    let em: f64 = meta.em.into();
    let ascent = (em * positions.ascent)
        .round()
        .to_u16()
        .ok_or(CoordsError::FloatCast)?;
    let descent = (em * positions.descent)
        .round()
        .to_u16()
        .ok_or(CoordsError::FloatCast)?;
    let signed_ascent: i16 = ascent.try_into().map_err(CoordsError::IntCast)?;
    let signed_descent: i16 = descent.try_into().map_err(CoordsError::IntCast)?;
    let hhea = Hhea {
        ascender: FWord::new(signed_ascent),
        descender: FWord::new(-signed_descent),
        number_of_h_metrics: glyph_count,
        ..Default::default()
    };
    let (fs_selection, mac_style) = match (meta.bold, meta.italic) {
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
    let weight = if meta.bold { 700 } else { 400 };
    let os2 = Os2 {
        s_typo_ascender: signed_ascent,
        s_typo_descender: -signed_descent,
        us_win_ascent: ascent,
        us_win_descent: descent,
        s_cap_height: Some(
            (em * positions.s_cap_height)
                .round()
                .to_i16()
                .ok_or(CoordsError::FloatCast)?,
        ),
        sx_height: Some(
            (em * positions.sx_height)
                .round()
                .to_i16()
                .ok_or(CoordsError::FloatCast)?,
        ),
        y_strikeout_position: (em * positions.y_strikeout_position)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        y_strikeout_size: (em * positions.y_strikeout_size)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        s_typo_line_gap: 0,
        fs_selection,
        us_weight_class: weight,
        ul_code_page_range_1: Some(0),
        ul_code_page_range_2: Some(0),
        us_default_char: Some(0),
        us_break_char: Some(0),
        us_max_context: Some(0),
        ..Default::default()
    };
    let post = Post {
        version: Version16Dot16::new(3, 0),
        underline_position: FWord::new(
            (em * positions.underline_position)
                .round()
                .to_i16()
                .ok_or(CoordsError::FloatCast)?,
        ),
        underline_thickness: FWord::new(
            (em * positions.underline_thickness)
                .round()
                .to_i16()
                .ok_or(CoordsError::FloatCast)?,
        ),
        italic_angle: Fixed::from_f64(if meta.italic {
            positions.italic_angle
        } else {
            0.0
        }),
        ..Default::default()
    };
    let epoch = jiff::Timestamp::constant(-2_082_844_800, 0);
    let head = Head {
        units_per_em: meta.em,
        x_min: 0,
        x_max: widest,
        y_min: -signed_descent,
        y_max: tallest,
        created: LongDateTime::new(meta.created.as_second() - epoch.as_second()),
        modified: LongDateTime::new(meta.modified.as_second() - epoch.as_second()),
        mac_style,
        // enum cast
        // https://doc.rust-lang.org/reference/expressions/operator-expr.html#r-expr.as.enum.discriminant
        index_to_loc_format: loca_format as i16,
        flags: Flags::BASELINE_AT_Y_0 | Flags::LSB_AT_X_0,
        lowest_rec_ppem: smallest_legible,
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
pub struct Names {
    pub copyright: String,
    pub family: String,
    pub style: String,
    pub unique: String,
    pub full: String,
    pub version: String,
    pub postscript: String,
    pub sample: String,
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
