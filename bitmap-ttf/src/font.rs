use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};

use num_traits::ToPrimitive;
use read_fonts::{
    tables::{cmap::PlatformId, glyf::CurvePoint, head::Flags, os2::OS2_UNICODE_RANGES},
    types::{UfWord, Version16Dot16},
};
use tracing::{Level, debug, span};
use write_fonts::{
    FontBuilder, OffsetMarker,
    tables::{
        cmap::{Cmap, Cmap4, Cmap12, CmapSubtable, EncodingRecord, SequentialMapGroup},
        colr::{BaseGlyph, Colr, Layer},
        cpal::{ColorRecord, Cpal},
        glyf::{Bbox, GlyfLocaBuilder, Glyph, SimpleGlyph},
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

#[derive(Debug, PartialEq, Eq, Default)]
pub struct GlyphInfo {
    pub width: u16,
    pub height: u16,
    pub base_layer: Option<SimpleGlyph>,
    pub base_offsets: Vec<(i16, i16)>,
    pub base_color: Option<ColorRecord>,
    pub colored_layers: Vec<ColoredLayer>,
    pub colored_offsets: Vec<(i16, i16)>,
}

impl GlyphInfo {
    pub fn empty(width: u16) -> Self {
        Self {
            width,
            ..Default::default()
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ColoredLayer {
    pub glyph: SimpleGlyph,
    pub color: ColorRecord,
}

#[derive(Debug)]
pub struct ScriptPositions {
    pub x_size: f64,
    pub y_size: f64,
    pub x_offset: f64,
    pub y_offset: f64,
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
    pub line_gap: f64,
    pub subscript: ScriptPositions,
    pub superscript: ScriptPositions,
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
#[error("Making font")]
pub enum MakeFontError {
    Glyph(#[from] write_fonts::error::Error),
    Builder(#[from] write_fonts::error::BuilderError),
    Coords(#[from] CoordsError),
    Timestamp,
}

#[derive(thiserror::Error, Debug)]
#[error("Converting coordinates")]
pub enum CoordsError {
    IntCast(#[from] std::num::TryFromIntError),
    SizeCast,
    FloatCast,
    IntAdd,
}

pub type CharMap = indexmap::IndexMap<char, GlyphInfo>;

pub fn make_font<'a>(
    meta: FontMeta,
    positions: &FontPositions,
    smallest_legible: u16,
    chars: &CharMap,
    notdef: &GlyphInfo,
) -> Result<FontBuilder<'a>, MakeFontError> {
    let mut builder = FontBuilder::new();
    let glyph_results = add_glyphs(&mut builder, chars, notdef)?;
    add_metrics(
        &mut builder,
        meta,
        positions,
        smallest_legible,
        &glyph_results,
    )?;
    Ok(builder)
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
    fn next(&mut self, metric: LongMetric) -> u16 {
        self.hmtx.h_metrics.push(metric);
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    fn add_empty_glyph(&mut self, advance: u16) -> Result<u16, write_fonts::error::Error> {
        self.builder.add_glyph(&Glyph::Empty)?;
        Ok(self.next(LongMetric {
            advance,
            side_bearing: 0,
        }))
    }

    fn add_simple_glyph(
        &mut self,
        glyph: &SimpleGlyph,
        advance: u16,
    ) -> Result<u16, MakeFontError> {
        self.builder.add_glyph(glyph)?;
        self.max_contours = self
            .max_contours
            .max(glyph.contours.len().to_u16().ok_or(CoordsError::SizeCast)?);
        for contour in glyph.contours.iter() {
            self.max_points = self
                .max_points
                .max(contour.len().to_u16().ok_or(CoordsError::SizeCast)?);
        }
        Ok(self.next(LongMetric {
            advance,
            side_bearing: glyph.bbox.x_min,
        }))
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
    low_pairs: BTreeMap<u16, u16>,
    all_pairs: BTreeMap<char, u16>,
    unicode_ranges: u128,
}

fn get_unicode_range_bit(codepoint: u32) -> Option<u8> {
    OS2_UNICODE_RANGES
        .binary_search_by(|&(a, b, _)| unicode_range_cmp(codepoint, a, b))
        .ok()
        .map(|i| OS2_UNICODE_RANGES[i].2)
}

fn unicode_range_cmp(codepoint: u32, first: u32, last: u32) -> Ordering {
    if codepoint < first {
        Ordering::Greater
    } else if codepoint <= last {
        Ordering::Equal
    } else {
        Ordering::Less
    }
}

impl CmapBuilder {
    fn add(&mut self, char: char, glyph_id: u16) {
        if let Ok(small) = char.try_into() {
            self.low_pairs.insert(small, glyph_id);
        }
        self.all_pairs.insert(char, glyph_id);
        if let Some(bit) = get_unicode_range_bit(char.into())
            && bit < 128
        {
            self.unicode_ranges |= 1 << bit;
        }
    }

    fn build(&self) -> Cmap {
        let mut records = vec![];
        if !self.low_pairs.is_empty() {
            let cmap4 = build_cmap4(&self.low_pairs);
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
            let cmap12 = build_cmap12(&self.all_pairs);
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
    end_glyph: u16,
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
            && current.end_glyph + 1 == glyph
        {
            current.end = char;
            current.end_glyph = glyph;
        } else {
            let new_row = Cmap4Range {
                start: char,
                end: char,
                start_glyph: glyph,
                end_glyph: glyph,
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
            end_glyph: _,
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

fn build_cmap4(pairs: &BTreeMap<u16, u16>) -> Cmap4 {
    let span = span!(Level::DEBUG, "building cmap4");
    let _guard = span.enter();
    let mut builder = Cmap4RangeBuilder::default();
    for (char, glyph) in pairs {
        builder.consume(*char, *glyph);
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

#[derive(Debug, Copy, Clone)]
struct Cmap12Range {
    start: u32,
    end: u32,
    start_glyph: u16,
    end_glyph: u16,
}

impl From<Cmap12Range> for SequentialMapGroup {
    fn from(val: Cmap12Range) -> Self {
        SequentialMapGroup {
            start_char_code: val.start,
            end_char_code: val.end,
            start_glyph_id: val.start_glyph.into(),
        }
    }
}

#[derive(Debug, Default)]
struct Cmap12RangeBuilder {
    ranges: Vec<SequentialMapGroup>,
    current_range: Option<Cmap12Range>,
}

impl Cmap12RangeBuilder {
    fn consume(&mut self, char: char, glyph: u16) {
        let char_u32 = char.into();
        if let Some(current) = self.current_range.as_mut()
            && current.end + 1 == char_u32
            && current.end_glyph + 1 == glyph
        {
            current.end = char_u32;
            current.end_glyph = glyph;
        } else {
            let new_row = Cmap12Range {
                start: char_u32,
                end: char_u32,
                start_glyph: glyph,
                end_glyph: glyph,
            };
            if let Some(old) = self.current_range.replace(new_row) {
                self.push(old);
            }
        }
    }

    fn push(&mut self, range: Cmap12Range) {
        debug!(
            "range from {} ({:04X}) to {} ({:04X}) ({} chars)",
            char::from_u32(range.start).unwrap_or('\0'),
            range.start,
            char::from_u32(range.end).unwrap_or('\0'),
            range.end,
            range.end - range.start + 1
        );
        self.ranges.push(range.into());
    }

    fn done(&mut self) {
        if let Some(last) = self.current_range.take() {
            self.push(last);
        }
    }
}

fn build_cmap12(pairs: &BTreeMap<char, u16>) -> Cmap12 {
    let span = span!(Level::DEBUG, "building cmap12");
    let _guard = span.enter();
    let mut builder = Cmap12RangeBuilder::default();
    for (char, glyph) in pairs {
        builder.consume(*char, *glyph);
    }
    builder.done();
    debug!("{} ranges", builder.ranges.len());
    Cmap12 {
        language: 0,
        groups: builder.ranges,
    }
}

#[derive(Debug, Default, Clone)]
struct GlyphSize {
    bounds: Bbox,
    advance: u16,
}

#[derive(Debug, Default)]
struct TrackedGlyphSize {
    start: GlyphSize,
    largest: GlyphSize,
}

impl TrackedGlyphSize {
    fn new(start: GlyphSize) -> Self {
        Self {
            start: start.clone(),
            largest: start,
        }
    }

    fn update(&mut self, offset: (i16, i16)) -> Result<(), CoordsError> {
        let (ox, oy) = offset;
        let mut modified_box = self.start.bounds;
        translate_bbox(&mut modified_box, ox, oy);
        self.largest.advance = self.largest.advance.max(
            TryInto::<i16>::try_into(self.start.advance)
                .map_err(CoordsError::IntCast)?
                .checked_add(ox)
                .ok_or(CoordsError::IntAdd)?
                .try_into()
                .map_err(CoordsError::IntCast)?,
        );
        Ok(())
    }
}

#[derive(Default)]
struct GlyphAndColorBuilder {
    glyph_builder: GlyphBuilder,
    largest: GlyphSize,
    color_pieces: Vec<ColorPiece>,
    accumulated_advance: i64,
}

#[derive(Debug)]
struct ColorPiece {
    for_glyph_id: u16,
    advance: u16,
    layers: Vec<ColoredLayer>,
    offsets: Vec<(i16, i16)>,
    base: Option<ColorRecord>,
}

impl GlyphAndColorBuilder {
    fn import(&mut self, info: &GlyphInfo) -> Result<u16, MakeFontError> {
        let (glyph_index, size) = add_glyph(
            &mut self.glyph_builder,
            info.base_layer.as_ref(),
            &info.base_offsets,
            info.width,
        )?;
        if !info.colored_layers.is_empty() || info.base_color.is_some() {
            self.color_pieces.push(ColorPiece {
                for_glyph_id: glyph_index,
                advance: size.advance,
                layers: info.colored_layers.clone(),
                offsets: info.colored_offsets.clone(),
                base: info.base_color.clone(),
            });
        }
        self.largest.bounds = self.largest.bounds.union(size.bounds);
        self.largest.advance = self.largest.advance.max(size.advance);
        self.accumulated_advance += Into::<i64>::into(size.advance);
        Ok(glyph_index)
    }

    fn build_color(&mut self) -> Result<Option<(Cpal, Colr)>, MakeFontError> {
        let mut palettes = vec![];
        let mut color_base_records = vec![];
        let mut color_layer_records = vec![];
        let mut next_color_index = 0u16;
        let mut get_palette_index = |color: &ColorRecord| -> Result<u16, CoordsError> {
            match palettes.iter().position(|x| x == color) {
                None => {
                    palettes.push(color.clone());
                    palettes.len() - 1
                }
                Some(index) => index,
            }
            .to_u16()
            .ok_or(CoordsError::SizeCast)
        };
        for piece in self.color_pieces.iter() {
            let mut num_layers = (piece.layers.len() * piece.offsets.len())
                .to_u16()
                .ok_or(CoordsError::SizeCast)?;
            if piece.base.is_some() {
                num_layers += 1;
            }
            color_base_records.push(BaseGlyph {
                glyph_id: GlyphId16::new(piece.for_glyph_id),
                first_layer_index: next_color_index,
                num_layers,
            });
            next_color_index = next_color_index
                .checked_add(num_layers)
                .ok_or(CoordsError::IntAdd)?;
            if let Some(base) = piece.base.as_ref() {
                let palette_index = get_palette_index(base)?;
                color_layer_records.push(Layer {
                    glyph_id: GlyphId16::new(piece.for_glyph_id),
                    palette_index,
                });
            }
            for layer in piece.layers.iter() {
                let palette_index = get_palette_index(&layer.color)?;
                for (x, y) in piece.offsets.iter() {
                    let mut cloned = layer.glyph.clone();
                    translate_glyph(&mut cloned, *x, *y);
                    let glyph_index = self
                        .glyph_builder
                        .add_simple_glyph(&cloned, piece.advance)?;
                    color_layer_records.push(Layer {
                        glyph_id: GlyphId16::new(glyph_index),
                        palette_index,
                    });
                }
            }
        }
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
            Ok(Some((cpal, colr)))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug)]
struct AddGlyphResults {
    loca_format: LocaFormat,
    glyph_count: u16,
    largest: GlyphSize,
    average_advance: i16,
    unicode_ranges: u128,
    low_chars: BTreeSet<u16>,
    chars: BTreeSet<char>,
}

fn add_glyphs(
    builder: &mut FontBuilder,
    chars: &CharMap,
    notdef: &GlyphInfo,
) -> Result<AddGlyphResults, MakeFontError> {
    let mut glyph_builder = GlyphAndColorBuilder::default();
    let mut cmap_builder = CmapBuilder::default();
    glyph_builder.import(notdef)?;
    for (char, info) in chars.iter() {
        let glyph_index = glyph_builder.import(info)?;
        cmap_builder.add(*char, glyph_index);
    }
    let colors = glyph_builder.build_color()?;
    let maxp = glyph_builder.glyph_builder.build_maxp();
    let average_advance = (glyph_builder.accumulated_advance / Into::<i64>::into(maxp.num_glyphs))
        .try_into()
        .map_err(CoordsError::IntCast)?;
    let (glyf, loca, loca_format) = glyph_builder.glyph_builder.builder.build();
    let cmap = cmap_builder.build();
    builder
        .add_table(&glyf)?
        .add_table(&loca)?
        .add_table(&glyph_builder.glyph_builder.hmtx)?
        .add_table(&maxp)?
        .add_table(&cmap)?;
    if let Some((cpal, cmap)) = colors {
        builder.add_table(&cpal)?.add_table(&cmap)?;
    }
    Ok(AddGlyphResults {
        loca_format,
        glyph_count: maxp.num_glyphs,
        largest: glyph_builder.largest,
        average_advance,
        unicode_ranges: cmap_builder.unicode_ranges,
        low_chars: cmap_builder.low_pairs.keys().copied().collect(),
        chars: cmap_builder.all_pairs.keys().copied().collect(),
    })
}

fn add_glyph(
    builder: &mut GlyphBuilder,
    glyph: Option<&SimpleGlyph>,
    offsets: &[(i16, i16)],
    advance: u16,
) -> Result<(u16, GlyphSize), MakeFontError> {
    let bounds = glyph.as_ref().map(|x| x.bbox).unwrap_or_default();
    let mut size = TrackedGlyphSize::new(GlyphSize { bounds, advance });
    let glyph_index = match (glyph, offsets) {
        (None, _) | (_, []) => builder.add_empty_glyph(size.largest.advance)?,
        (Some(layer), [(0, 0)]) => builder.add_simple_glyph(layer, size.largest.advance)?,
        (Some(layer), offsets) => {
            let mut contours = vec![];
            for (x, y) in offsets {
                size.update((*x, *y))?;
                let mut cloned = layer.clone();
                translate_glyph(&mut cloned, *x, *y);
                contours.extend(cloned.contours);
            }
            let final_glyph = SimpleGlyph {
                bbox: size.largest.bounds,
                contours,
                instructions: vec![],
                overlaps: true,
            };
            builder.add_simple_glyph(&final_glyph, size.largest.advance)?
        }
    };
    Ok((glyph_index, size.largest))
}

fn split_unicode_range(range: u128) -> [u32; 4] {
    [
        range as u32,
        (range >> 32) as u32,
        (range >> 64) as u32,
        (range >> 96) as u32,
    ]
}

fn add_metrics(
    builder: &mut FontBuilder,
    meta: FontMeta,
    positions: &FontPositions,
    smallest_legible: u16,
    glyph_results: &AddGlyphResults,
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
        x_max_extent: FWord::new(glyph_results.largest.bounds.x_max),
        advance_width_max: UfWord::new(glyph_results.largest.advance),
        number_of_h_metrics: glyph_results.glyph_count,
        min_left_side_bearing: FWord::new(glyph_results.largest.bounds.x_min),
        min_right_side_bearing: FWord::new(0),
        line_gap: FWord::new(
            (em * positions.line_gap)
                .round()
                .to_i16()
                .ok_or(CoordsError::FloatCast)?,
        ),
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
    let [uni1, uni2, uni3, uni4] = split_unicode_range(glyph_results.unicode_ranges);
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
        s_typo_line_gap: (em * positions.line_gap)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        fs_selection: fs_selection | SelectionFlags::USE_TYPO_METRICS,
        us_weight_class: weight,
        ul_unicode_range_1: uni1,
        ul_unicode_range_2: uni2,
        ul_unicode_range_3: uni3,
        ul_unicode_range_4: uni4,
        ul_code_page_range_1: Some(0),
        ul_code_page_range_2: Some(0),
        us_default_char: Some(0),
        us_break_char: Some(0x20),
        us_max_context: Some(0),
        us_first_char_index: glyph_results.low_chars.first().copied().unwrap_or(0),
        us_last_char_index: glyph_results.low_chars.last().copied().unwrap_or(0),
        x_avg_char_width: glyph_results.average_advance,
        y_subscript_x_size: (em * positions.subscript.x_size)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        y_subscript_y_size: (em * positions.subscript.y_size)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        y_subscript_x_offset: (em * positions.subscript.x_offset)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        y_subscript_y_offset: (em * positions.subscript.y_offset)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        y_superscript_x_size: (em * positions.superscript.x_size)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        y_superscript_y_size: (em * positions.superscript.y_size)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        y_superscript_x_offset: (em * positions.superscript.x_offset)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        y_superscript_y_offset: (em * positions.superscript.y_offset)
            .round()
            .to_i16()
            .ok_or(CoordsError::FloatCast)?,
        ..Default::default()
    };
    let post = Post {
        version: Version16Dot16::new(3, 0),
        num_glyphs: Some(glyph_results.glyph_count),
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
        x_min: glyph_results.largest.bounds.x_min,
        x_max: glyph_results.largest.bounds.x_max,
        y_min: glyph_results.largest.bounds.y_min,
        y_max: glyph_results.largest.bounds.y_max,
        created: LongDateTime::new(meta.created.as_second() - epoch.as_second()),
        modified: LongDateTime::new(meta.modified.as_second() - epoch.as_second()),
        mac_style,
        // enum cast
        // https://doc.rust-lang.org/reference/expressions/operator-expr.html#r-expr.as.enum.discriminant
        index_to_loc_format: glyph_results.loca_format as i16,
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
