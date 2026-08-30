use std::collections::HashMap;

use write_fonts::{
    BuilderError, FontBuilder, OffsetMarker,
    tables::{
        cpal::ColorRecord,
        glyf::Glyph,
        head::{Head, MacStyle},
        name::{Name, NameRecord},
        os2::{Os2, SelectionFlags},
        post::Post,
    },
    types::{FWord, Fixed, LongDateTime, NameId},
};

#[derive(Debug, PartialEq)]
pub struct GlyphInfo {
    width: f32,
    height: f32,
    base_layer: Glyph,
    base_offsets: Vec<(f32, f32)>,
    colored_layers: Vec<ColoredLayer>,
    colored_offsets: Vec<(f32, f32)>,
}

impl GlyphInfo {
    pub fn empty() -> Self {
        Self {
            width: 0.0,
            height: 0.0,
            base_layer: Glyph::Empty,
            base_offsets: vec![],
            colored_layers: vec![],
            colored_offsets: vec![],
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ColoredLayer {
    layer: Glyph,
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

pub fn make_font(
    info: FontInfo,
    positions: &FontPositions,
    chars: impl Iterator<Item = (char, GlyphInfo)>,
    other_glyphs: impl Iterator<Item = (String, GlyphInfo)>,
    aglfn: &HashMap<String, String>,
) -> Result<Vec<u8>, BuilderError> {
    let mut widest = 0.0f32;
    let mut tallest = 0.0f32;
    let names = info.names.into_table();
    let ascent = (info.em as f32 * positions.ascent) as u16;
    let descent = (info.em as f32 * positions.descent) as u16;
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
        x_max: widest as i16,
        y_min: -(descent as i16),
        y_max: tallest as i16,
        created: LongDateTime::new((info.created - epoch).get_seconds()),
        modified: LongDateTime::new((info.modified - epoch).get_seconds()),
        mac_style,
        ..Default::default()
    };
    let mut builder = FontBuilder::new();
    builder
        .add_table(&names)?
        .add_table(&os2)?
        .add_table(&post)?
        .add_table(&head)?;
    Ok(builder.build())
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
            encoding_id: 1,
            language_id: 0x0409,
            name_id: id,
            string: OffsetMarker::new(value),
        });
    }
}
