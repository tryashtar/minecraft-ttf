use std::{
    cmp::{max, min},
    collections::{BTreeMap, HashMap, HashSet},
    ffi::OsStr,
    fmt::Display,
    io::{BufRead, BufReader, Cursor},
    path::{Path, PathBuf},
    str::FromStr,
};

use bitmap_ttf::bitmap::{Bitmap, Rectangle};
use image::GenericImageView;
use tracing::{Level, debug, span};

use crate::{cache, providers::GridCell::SkippedChar, storage};

pub fn serialize_rectangle<S>(rectangle: &Rectangle, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut state = serializer.serialize_struct("Rectangle", 4)?;
    serde::ser::SerializeStruct::serialize_field(&mut state, "left", &rectangle.left)?;
    serde::ser::SerializeStruct::serialize_field(&mut state, "top", &rectangle.top)?;
    serde::ser::SerializeStruct::serialize_field(&mut state, "width", &rectangle.width)?;
    serde::ser::SerializeStruct::serialize_field(&mut state, "height", &rectangle.height)?;
    serde::ser::SerializeStruct::end(state)
}

pub fn serialize_bitmap<S>(bitmap: &Bitmap, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut state = serializer.serialize_struct("Bitmap", 3)?;
    serde::ser::SerializeStruct::serialize_field(&mut state, "width", &bitmap.width())?;
    serde::ser::SerializeStruct::serialize_field(&mut state, "height", &bitmap.height())?;
    let rows: Vec<String> = (0..bitmap.height())
        .map(|y| {
            (0..bitmap.width())
                .map(|x| if bitmap.get(x, y) { '1' } else { '0' })
                .collect()
        })
        .collect();
    serde::ser::SerializeStruct::serialize_field(&mut state, "bits", &rows)?;
    serde::ser::SerializeStruct::end(state)
}

#[derive(Debug, serde::Serialize, PartialEq)]
pub struct CharBitmap {
    #[serde(serialize_with = "serialize_bitmap")]
    pub bitmap: Bitmap,
    pub advance: f32,
    pub bold_offset: f32,
}

#[derive(Debug, serde::Serialize, PartialEq)]
pub struct BitmapProvider {
    pub height: i32,
    pub ascent: i32,
    pub chars: indexmap::IndexMap<char, CharBitmap>,
}

#[derive(Debug, serde::Serialize, PartialEq)]
pub struct CharImage {
    #[serde(serialize_with = "serialize_rectangle")]
    pub content_box: Rectangle,
    #[serde(serialize_with = "serialize_bitmap")]
    pub bitmap: Bitmap,
    pub advance: f32,
    pub bold_offset: f32,
}

#[derive(Debug, serde::Serialize, PartialEq)]
pub struct ImageProvider {
    #[serde(skip)]
    pub image: image::DynamicImage,
    pub image_path: PathBuf,
    pub has_color: bool,
    pub height: i32,
    pub ascent: i32,
    pub chars: indexmap::IndexMap<char, CharImage>,
}

#[derive(Debug, serde::Serialize, PartialEq)]
pub struct SpaceProvider {
    pub chars: BTreeMap<char, f32>,
}

#[derive(Debug)]
pub enum GridCell {
    FullSizeChar(char),
    SizedChar(char, (u32, u32)),
    SkippedChar(char),
    Empty,
}

impl GridCell {
    pub fn full(char: Option<char>) -> Self {
        match char {
            None => Self::Empty,
            Some(char) => Self::FullSizeChar(char),
        }
    }

    pub fn sized(char: Option<char>, size: Option<(u32, u32)>) -> Self {
        match char {
            None => Self::Empty,
            Some(char) => match size {
                None => Self::SkippedChar(char),
                Some(size) => Self::SizedChar(char, size),
            },
        }
    }
}

#[derive(Debug)]
pub struct SheetGrid {
    pub image_path: PathBuf,
    pub chars: ndarray::Array2<GridCell>,
    pub height: i32,
    pub ascent: i32,
    pub bold_offset: f32,
    pub advance: AdvanceStrategy,
    pub is_unifont: bool,
}

impl SheetGrid {
    fn resolve(
        self,
        store: &mut impl storage::Storage,
        options: &impl ProviderOptions,
    ) -> Result<(ImageProvider, ModifiedTimes), ProvidersError> {
        let mut times = ModifiedTimes::default();
        let image = storage::read_image(store, &self.image_path)?;
        times.update(image.modified_time);
        let has_color = options.has_color(&image.data);
        let char_images = image_grid(&image.data, &self.chars, |x| {
            options.include_char(x) && (!self.is_unifont || options.include_unifont_char(x))
        });
        let chars = char_images
            .into_iter()
            .map(|(char, (content_box, bitmap))| {
                let advance = self.advance.get(char, &bitmap, self.height);
                let glyph = CharImage {
                    content_box,
                    bitmap,
                    advance,
                    bold_offset: self.bold_offset,
                };
                (char, glyph)
            })
            .collect();
        let provider = ImageProvider {
            image: image.data,
            image_path: self.image_path,
            has_color,
            height: self.height,
            ascent: self.ascent,
            chars,
        };
        Ok((provider, times))
    }
}

#[derive(Debug)]
pub enum PartialProvider {
    Bitmap(BitmapProvider),
    Image(SheetGrid),
    Space(SpaceProvider),
}

impl PartialProvider {
    fn resolve(
        self,
        store: &mut impl storage::Storage,
        options: &impl ProviderOptions,
    ) -> Result<(Provider, ModifiedTimes), ProvidersError> {
        match self {
            PartialProvider::Bitmap(bitmap) => {
                Ok((Provider::Bitmap(bitmap), ModifiedTimes::default()))
            }
            PartialProvider::Space(space) => Ok((Provider::Space(space), ModifiedTimes::default())),
            PartialProvider::Image(grid) => {
                let (provider, times) = grid.resolve(store, options)?;
                Ok((Provider::Image(provider), times))
            }
        }
    }
}

#[derive(Debug, serde::Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Provider {
    Bitmap(BitmapProvider),
    Image(ImageProvider),
    Space(SpaceProvider),
}

#[derive(
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
    Debug,
    Clone,
    Hash,
    Eq,
    PartialEq,
    PartialOrd,
    Ord,
)]
pub struct Identifier {
    pub namespace: String,
    pub body: PathBuf,
}

impl Identifier {
    pub fn new(namespace: String, body: PathBuf) -> Self {
        Self { namespace, body }
    }

    pub fn from_string_unchecked(string: &str) -> Self {
        let colon_index = string.find(':');
        let (namespace, body) = match colon_index {
            None => ("minecraft", string),
            Some(index) => (&string[..index], &string[index + 1..]),
        };
        Self {
            namespace: String::from(namespace),
            body: PathBuf::from(body),
        }
    }

    pub fn to_entry(&self, kind: Option<&str>, suffix: Option<&str>) -> PathBuf {
        let mut result = PathBuf::from("assets");
        result.push(&self.namespace);
        if let Some(kind) = kind {
            result.push(kind);
        }
        result.push(&self.body);
        if let Some(suffix) = suffix {
            result = cache::push_path_str(result, OsStr::new("."));
            result = cache::push_path_str(result, OsStr::new(suffix));
        }
        result
    }
}

#[derive(thiserror::Error, Debug)]
pub enum IdentifierError {
    #[error("Invalid character {0}")]
    InvalidChar(char),
    #[error("Empty namespace")]
    EmptyNamespace,
    #[error("Empty body")]
    EmptyBody,
}

impl FromStr for Identifier {
    type Err = IdentifierError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut has_namespace = false;
        let mut namespace = String::new();
        let mut body = String::new();
        for char in s.chars() {
            if char.is_ascii_lowercase()
                || char.is_ascii_digit()
                || char == '_'
                || char == '-'
                || char == '.'
            {
                if has_namespace {
                    body.push(char)
                } else {
                    namespace.push(char);
                }
            } else if char == ':' {
                if has_namespace {
                    return Err(IdentifierError::InvalidChar(char));
                }
                has_namespace = true;
            } else if char == '/' {
                if !has_namespace {
                    return Err(IdentifierError::InvalidChar(char));
                }
                body.push(char);
            } else {
                return Err(IdentifierError::InvalidChar(char));
            }
        }
        if has_namespace {
            if namespace.is_empty() {
                Err(IdentifierError::EmptyNamespace)
            } else if body.is_empty() {
                Err(IdentifierError::EmptyBody)
            } else {
                Ok(Self {
                    namespace,
                    body: PathBuf::from(body),
                })
            }
        } else {
            if namespace.is_empty() {
                Err(IdentifierError::EmptyBody)
            } else {
                Ok(Self {
                    namespace: String::from("minecraft"),
                    body: PathBuf::from(namespace),
                })
            }
        }
    }
}

impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.namespace, self.body.display())
    }
}

#[derive(Debug, Default, serde::Serialize)]
pub struct ModifiedTimes {
    pub oldest: Option<jiff::Timestamp>,
    pub newest: Option<jiff::Timestamp>,
}

impl ModifiedTimes {
    pub fn update(&mut self, time: Option<jiff::Timestamp>) {
        if let Some(time) = time {
            self.oldest = Some(match self.oldest {
                None => time,
                Some(before) => min(before, time),
            });
            self.newest = Some(match self.newest {
                None => time,
                Some(before) => max(before, time),
            });
        }
    }

    pub fn merge(&mut self, other: ModifiedTimes) {
        self.update(other.oldest);
        self.update(other.newest);
    }
}

#[derive(Debug, serde::Serialize)]
pub struct Providers {
    pub providers: Vec<Provider>,
    pub times: ModifiedTimes,
}

#[derive(Debug)]
pub struct PartialProviders {
    pub providers: Vec<PartialProvider>,
    pub times: ModifiedTimes,
}

impl PartialProviders {
    pub fn resolve(
        self,
        store: &mut impl storage::Storage,
        options: &impl ProviderOptions,
    ) -> Result<Providers, ProvidersError> {
        let mut providers = vec![];
        let mut times = ModifiedTimes::default();
        for provider in self.providers {
            let (mapped, extra_times) = provider.resolve(store, options)?;
            times.merge(extra_times);
            providers.push(mapped);
        }
        Ok(Providers {
            providers,
            times: self.times,
        })
    }
}

#[derive(thiserror::Error, Debug)]
#[error("Strings were not all the same length")]
pub struct JaggedArrayError;

pub fn split_grid<'a>(
    rows: impl Iterator<Item = &'a String>,
    filter_nul: bool,
) -> Result<ndarray::Array2<Option<char>>, JaggedArrayError> {
    let map = |x: char| match (x, filter_nul) {
        ('\u{0000}', true) => None,
        (char, _) => Some(char),
    };
    let chars: Vec<Vec<Option<char>>> = rows.map(|s| s.chars().map(map).collect()).collect();
    if chars.is_empty() {
        return Ok(ndarray::Array2::default((0, 0)));
    }
    let rows = chars.len();
    let columns = chars[0].len();
    if chars.iter().any(|x| x.len() != columns) {
        return Err(JaggedArrayError);
    }
    let array = ndarray::Array2::from_shape_fn((rows, columns), |(i, j)| chars[i][j]);
    Ok(array)
}

#[derive(thiserror::Error, Debug)]
#[error("Resolving providers")]
pub enum ProvidersError {
    Storage(#[from] storage::StorageError),
    Jagged(#[from] JaggedArrayError),
    #[error("Recursive provider reference")]
    RecursiveReference,
    Unihex(#[from] UnihexError),
}

pub fn cell_bitmap(
    image: &image::DynamicImage,
    rows: u32,
    columns: u32,
    row: usize,
    column: usize,
    size: Option<(u32, u32)>,
) -> (Rectangle, Bitmap) {
    let glyph_width = image.width() / columns;
    let glyph_height = image.height() / rows;
    let left = column * (glyph_width as usize);
    let top = row * (glyph_height as usize);
    let rectangle = Rectangle {
        left,
        top,
        width: glyph_width as usize,
        height: glyph_height as usize,
    };
    let portion = image_portion(image, &rectangle, size);
    (rectangle, portion)
}

pub fn image_grid(
    image: &image::DynamicImage,
    cells: &ndarray::Array2<GridCell>,
    include: impl Fn(char) -> bool,
) -> indexmap::IndexMap<char, (Rectangle, Bitmap)> {
    let mut map = indexmap::IndexMap::new();
    let rows = cells.nrows() as u32;
    let columns = cells.ncols() as u32;
    for ((row, column), cell) in cells.indexed_iter() {
        let (char, size) = match cell {
            GridCell::FullSizeChar(char) => (char, None),
            GridCell::SizedChar(char, (l, r)) => (char, Some((*l, *r))),
            SkippedChar(_) | GridCell::Empty => {
                continue;
            }
        };
        if map.contains_key(char) || !include(*char) {
            continue;
        }
        let data = cell_bitmap(image, rows, columns, row, column, size);
        map.insert(*char, data);
    }
    map
}

fn image_portion(
    image: &image::DynamicImage,
    rectangle: &Rectangle,
    size: Option<(u32, u32)>,
) -> Bitmap {
    let glyph = image::GenericImageView::view(
        image,
        rectangle.left as u32,
        rectangle.top as u32,
        rectangle.width as u32,
        rectangle.height as u32,
    );
    let bitmap = image_to_bitmap(glyph, 10);
    match size {
        None => bitmap,
        Some((left, right)) => {
            let rect = Rectangle {
                left: left as usize,
                top: 0,
                width: (right - left) as usize,
                height: bitmap.height(),
            };
            bitmap.resized(&rect)
        }
    }
}

pub fn image_to_bitmap(image: image::SubImage<&image::DynamicImage>, threshold: u8) -> Bitmap {
    let mut bitmap = Bitmap::new(image.width() as usize, image.height() as usize);
    for (x, y, pixel) in image.pixels() {
        if pixel.0[3] >= threshold {
            bitmap.set(x as usize, y as usize, true);
        }
    }
    bitmap
}

pub fn colors_to_bitmaps(
    image: image::SubImage<&image::DynamicImage>,
    threshold: u8,
) -> indexmap::IndexMap<image::Rgba<u8>, Bitmap> {
    let mut result = indexmap::IndexMap::new();
    for (x, y, pixel) in image.pixels() {
        if pixel.0[3] >= threshold {
            result
                .entry(pixel)
                .or_insert_with(|| Bitmap::new(image.width() as usize, image.height() as usize))
                .set(x as usize, y as usize, true);
        }
    }
    result
}

pub trait ProviderOptions {
    fn has_color(&self, image: &image::DynamicImage) -> bool;
    fn include_char(&self, char: char) -> bool;
    fn include_unifont_char(&self, char: char) -> bool;
    fn option_uniform(&self) -> bool;
    fn option_jp(&self) -> bool;
}

fn passes_filter(options: &impl ProviderOptions, filter: &JsonProviderFilter) -> bool {
    let JsonProviderFilter { uniform, jp } = filter;
    if let Some(uniform) = uniform
        && *uniform != options.option_uniform()
    {
        return false;
    }
    if let Some(jp) = jp
        && *jp != options.option_jp()
    {
        return false;
    }
    true
}

#[derive(Debug)]
pub struct AdvanceStrategy {
    pub specific: BTreeMap<char, f32>,
    pub auto: AutoAdvance,
}

impl AdvanceStrategy {
    pub fn get(&self, char: char, bitmap: &Bitmap, height: i32) -> f32 {
        if let Some(advance) = self.specific.get(&char) {
            return *advance;
        }
        self.auto.get(bitmap, height)
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum AutoAdvance {
    LeftmostEmpty,
    RightmostFilled,
    EvenUniform,
    UnevenUniform,
}

impl AutoAdvance {
    pub fn get(&self, bitmap: &Bitmap, height: i32) -> f32 {
        match self {
            AutoAdvance::LeftmostEmpty => {
                let column = leftmost_empty(bitmap).unwrap_or(bitmap.width() - 1);
                normal_advance(bitmap, height, column)
            }
            AutoAdvance::RightmostFilled => {
                let column = rightmost_filled(bitmap).unwrap_or(0) + 1;
                normal_advance(bitmap, height, column)
            }
            AutoAdvance::EvenUniform => even_uniform_advance(bitmap),
            AutoAdvance::UnevenUniform => uneven_uniform_advance(bitmap),
        }
    }
}

fn rightmost_filled(bitmap: &Bitmap) -> Option<usize> {
    for x in (0..bitmap.width()).rev() {
        for y in (0..bitmap.height()).rev() {
            if bitmap.get(x, y) {
                return Some(x);
            }
        }
    }
    None
}

fn leftmost_empty(bitmap: &Bitmap) -> Option<usize> {
    for x in 0..bitmap.width() {
        let mut empty = true;
        for y in 0..bitmap.height() {
            if bitmap.get(x, y) {
                empty = false;
                break;
            }
        }
        if empty {
            return Some(x);
        }
    }
    None
}

fn normal_advance(bitmap: &Bitmap, height: i32, column: usize) -> f32 {
    let scale = height as f32 / bitmap.height() as f32;
    (0.5 + column as f32 * scale).floor() + 1.0
}

pub fn uneven_uniform_advance(bitmap: &Bitmap) -> f32 {
    let advance = (bitmap.width() / 2) + 1;
    advance as f32
}

pub fn even_uniform_advance(bitmap: &Bitmap) -> f32 {
    (bitmap.width() as f32 / 2.0) + 1.0
}

#[derive(serde::Deserialize, Debug)]
struct RootJsonProvider {
    providers: Vec<JsonProvider>,
}

#[derive(serde::Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum JsonProvider {
    Bitmap(JsonBitmapProvider),
    Space(JsonSpaceProvider),
    Reference(JsonReferenceProvider),
    LegacyUnicode(JsonLegacyUnicodeProvider),
    Unihex(JsonUnihexProvider),
}

impl JsonProvider {
    fn get_filter(&self) -> Option<&JsonProviderFilter> {
        match self {
            JsonProvider::Bitmap(bitmap) => bitmap.filter.as_ref(),
            JsonProvider::Space(space) => space.filter.as_ref(),
            JsonProvider::Reference(reference) => reference.filter.as_ref(),
            JsonProvider::LegacyUnicode(unicode) => unicode.filter.as_ref(),
            JsonProvider::Unihex(unihex) => unihex.filter.as_ref(),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
struct JsonProviderFilter {
    uniform: Option<bool>,
    jp: Option<bool>,
}

#[derive(serde::Deserialize, Debug)]
struct JsonBitmapProvider {
    filter: Option<JsonProviderFilter>,
    file: Identifier,
    height: Option<i32>,
    ascent: i32,
    chars: Vec<String>,
}

#[derive(serde::Deserialize, Debug)]
struct JsonSpaceProvider {
    filter: Option<JsonProviderFilter>,
    advances: BTreeMap<char, f32>,
}

#[derive(serde::Deserialize, Debug)]
struct JsonReferenceProvider {
    filter: Option<JsonProviderFilter>,
    id: Identifier,
}

#[derive(serde::Deserialize, Debug)]
struct JsonLegacyUnicodeProvider {
    filter: Option<JsonProviderFilter>,
    sizes: Identifier,
    template: String,
}

#[derive(serde::Deserialize, Debug)]
struct JsonUnihexProvider {
    filter: Option<JsonProviderFilter>,
    hex_file: Identifier,
    size_overrides: Option<Vec<UnihexSizeOverride>>,
}

#[derive(serde::Deserialize, Debug)]
struct UnihexSizeOverride {
    from: char,
    to: char,
    left: u32,
    right: u32,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct ProviderBehavior {
    pub uniform: ForceUniformBehavior,
    pub uneven_unifont: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum ForceUniformBehavior {
    Filter,
    SkipBitmaps,
    SwitchIdentifier,
}

fn font_entry(
    identifier: &Identifier,
    options: &impl ProviderOptions,
    behavior: &ProviderBehavior,
) -> PathBuf {
    if options.option_uniform()
        && let ForceUniformBehavior::SwitchIdentifier = behavior.uniform
        && identifier.namespace == "minecraft"
        && identifier.body == Path::new("default")
    {
        let uniform = Identifier {
            namespace: String::from("minecraft"),
            body: PathBuf::from("uniform"),
        };
        uniform.to_entry(Some("font"), Some("json"))
    } else {
        identifier.to_entry(Some("font"), Some("json"))
    }
}

pub fn load_partial_providers(
    identifier: Identifier,
    store: &mut storage::StackStorage,
    advance: AutoAdvance,
    options: &impl ProviderOptions,
    behavior: &ProviderBehavior,
    seen: &mut HashSet<Identifier>,
) -> Result<Option<PartialProviders>, ProvidersError> {
    let entry = font_entry(&identifier, options, behavior);
    seen.insert(identifier);
    match read_stacked_providers(store, &entry) {
        Err(storage::StorageError::FileNotFound(_)) => Ok(None),
        Err(err) => Err(ProvidersError::Storage(err)),
        Ok((providers, mut times)) => {
            let mut converted = vec![];
            for single in providers {
                if single
                    .get_filter()
                    .is_none_or(|x| passes_filter(options, x))
                {
                    let subs = convert_provider(store, options, advance, behavior, single, seen)?;
                    converted.extend(subs.providers);
                    times.merge(subs.times);
                }
            }
            Ok(Some(PartialProviders {
                providers: converted,
                times,
            }))
        }
    }
}

fn convert_provider(
    store: &mut storage::StackStorage,
    options: &impl ProviderOptions,
    advance: AutoAdvance,
    behavior: &ProviderBehavior,
    provider: JsonProvider,
    seen: &mut HashSet<Identifier>,
) -> Result<PartialProviders, ProvidersError> {
    match provider {
        JsonProvider::Bitmap(bitmap) => {
            let image_path = bitmap.file.to_entry(Some("textures"), None);
            let height = bitmap.height.unwrap_or(8);
            let chars = split_grid(bitmap.chars.iter(), true)?;
            let sheet = SheetGrid {
                image_path,
                chars: chars.mapv(GridCell::full),
                height,
                ascent: bitmap.ascent,
                bold_offset: 1.0,
                is_unifont: false,
                advance: AdvanceStrategy {
                    specific: BTreeMap::new(),
                    auto: advance,
                },
            };
            if options.option_uniform()
                && let ForceUniformBehavior::SkipBitmaps = behavior.uniform
            {
                return Ok(PartialProviders {
                    providers: vec![],
                    times: ModifiedTimes::default(),
                });
            }
            Ok(PartialProviders {
                providers: vec![PartialProvider::Image(sheet)],
                times: ModifiedTimes::default(),
            })
        }
        JsonProvider::Space(space) => {
            let full = SpaceProvider {
                chars: space.advances,
            };
            Ok(PartialProviders {
                providers: vec![PartialProvider::Space(full)],
                times: ModifiedTimes::default(),
            })
        }
        JsonProvider::Reference(reference) => {
            if seen.contains(&reference.id) {
                return Err(ProvidersError::RecursiveReference);
            }
            let err_entry = reference.id.to_entry(Some("font"), Some("json"));
            let resolved =
                load_partial_providers(reference.id, store, advance, options, behavior, seen)?
                    .ok_or(storage::StorageError::FileNotFound(err_entry))?;
            Ok(resolved)
        }
        JsonProvider::LegacyUnicode(unicode) => {
            let sheets = unicode_sheets(&unicode.template, |x| {
                Identifier::from_string_unchecked(&x).to_entry(Some("textures"), None)
            });
            let (many, times) = legacy_unicode(
                store,
                behavior.uneven_unifont,
                sheets.into_iter(),
                &unicode.sizes.to_entry(None, None),
            )?;
            Ok(PartialProviders {
                providers: many.into_iter().map(PartialProvider::Image).collect(),
                times,
            })
        }
        JsonProvider::Unihex(unihex) => {
            let (result, times) =
                convert_unihex_provider(store, options, behavior.uneven_unifont, &unihex)?;
            Ok(PartialProviders {
                providers: vec![PartialProvider::Bitmap(result)],
                times,
            })
        }
    }
}

fn convert_unihex_provider(
    store: &mut impl storage::Storage,
    options: &impl ProviderOptions,
    uneven: bool,
    unihex: &JsonUnihexProvider,
) -> Result<(BitmapProvider, ModifiedTimes), ProvidersError> {
    let span = span!(
        Level::DEBUG,
        "converting unihex provider",
        hex = %unihex.hex_file
    );
    let _guard = span.enter();
    let mut times = ModifiedTimes::default();
    let zip_entry = unihex.hex_file.to_entry(None, None);
    let zip_time = store.modified_time(&zip_entry)?;
    times.update(zip_time);
    let zip_bytes = store.read(&zip_entry)?;
    let reader = Cursor::new(zip_bytes);
    let mut zip = zip::ZipArchive::new(reader).map_err(storage::StorageError::Zip)?;
    let lines = unihex_lines(&mut zip)?;
    times.update(lines.modified_time);
    let mut chars = indexmap::IndexMap::new();
    for line in &lines.data {
        let colon_index = line.find(':').ok_or(UnihexError::MissingColon)?;
        let code_str = &line[..colon_index];
        let art = &line[colon_index + 1..];
        let code = u32::from_str_radix(code_str, 16).map_err(UnihexError::Parse)?;
        let char = char::from_u32(code).ok_or(UnihexError::InvalidChar(code))?;
        if !options.include_char(char) || !options.include_unifont_char(char) {
            continue;
        }
        let bitmap = unihex_bitmap(art)?;
        let (left, right) = unifont_char_size(
            char,
            &bitmap,
            unihex.size_overrides.as_ref().unwrap_or(&vec![]).iter(),
        );
        let width = (right as usize)
            .checked_sub(left as usize)
            .ok_or(ProvidersError::Unihex(UnihexError::InvalidSize(
                left, right,
            )))?;
        let cropped = bitmap.resized(&Rectangle {
            left: left as usize,
            top: 0,
            width,
            height: bitmap.height(),
        });
        let advance = if uneven {
            uneven_uniform_advance(&cropped)
        } else {
            even_uniform_advance(&cropped)
        };
        debug!(
            "{:?} ({:04X}):\n{}\nleft: {}, right: {}, advance: {}",
            char,
            Into::<u32>::into(char),
            bitmap,
            left,
            right,
            advance
        );
        chars.insert(
            char,
            CharBitmap {
                bitmap: cropped,
                advance,
                bold_offset: 0.5,
            },
        );
    }
    let full = BitmapProvider {
        height: 8,
        ascent: 7,
        chars,
    };
    Ok((full, times))
}

fn unifont_char_size<'a>(
    char: char,
    bitmap: &Bitmap,
    overrides: impl Iterator<Item = &'a UnihexSizeOverride>,
) -> (u32, u32) {
    for entry in overrides {
        if char >= entry.from && char <= entry.to {
            return (entry.left, entry.right + 1);
        }
    }
    match bitmap.content_box() {
        None => (0, 0),
        Some(content) => (
            content.left as u32,
            content.left as u32 + content.width as u32,
        ),
    }
}

fn unihex_lines(
    zip: &mut zip::ZipArchive<impl std::io::Read + std::io::Seek>,
) -> Result<storage::ReadEntry<Vec<String>>, storage::StorageError> {
    for i in 0..zip.len() {
        let file = zip.by_index(i)?;
        if file.name().ends_with(".hex") {
            let time = storage::zip_time(&file).map_err(storage::StorageError::Time)?;
            let lines = BufReader::new(file)
                .lines()
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(storage::ReadEntry {
                data: lines,
                modified_time: time,
            });
        };
    }
    Err(storage::StorageError::FileNotFound(PathBuf::from(".hex")))
}

#[derive(thiserror::Error, Debug)]
#[error("Parsing unihex")]
pub enum UnihexError {
    #[error("Missing colon in unihex entry")]
    MissingColon,
    Parse(#[from] std::num::ParseIntError),
    #[error("Invalid character in unihex entry: {0}")]
    InvalidChar(u32),
    Hex(#[from] hex::FromHexError),
    #[error("Invalid dimensions for unihex entry: {0}")]
    InvalidDimensions(usize),
    #[error("Invalid size for unihex entry: {0}, {1}")]
    InvalidSize(u32, u32),
}

fn unihex_bitmap(art: &str) -> Result<Bitmap, UnihexError> {
    let bytes = hex::decode(art)?;
    let bits = bitmap_ttf::bitvec::vec::BitVec::from_vec(bytes);
    let bitmap = match bits.len() {
        128 => Bitmap::from_array(8, 16, bits).expect("8 * 16 = 128"),
        256 => Bitmap::from_array(16, 16, bits).expect("16 * 16 = 256"),
        other => {
            return Err(UnihexError::InvalidDimensions(other));
        }
    };
    Ok(bitmap)
}

pub fn unicode_sheets(
    template: &str,
    convert: impl Fn(String) -> PathBuf,
) -> Vec<(ndarray::Array2<Option<char>>, PathBuf)> {
    let mut result = vec![];
    for sheet_id in 0u8..=0xffu8 {
        let start: u32 = Into::<u32>::into(sheet_id) * 256;
        let chars = ndarray::Array2::from_shape_fn((16, 16), |(y, x)| {
            char::from_u32(start + (16 * y as u32) + x as u32)
        });
        let path = template
            .replace("%s", &format!("{:02x}", sheet_id))
            .replace("%S", &format!("{:02X}", sheet_id));
        result.push((chars, convert(path)));
    }
    result
}

fn glyph_sizes(size_bytes: impl Iterator<Item = u8>) -> HashMap<char, (u32, u32)> {
    size_bytes
        .zip(0u32..)
        .filter_map(|(byte, char)| {
            if byte > 0 {
                char::from_u32(char).map(|x| (byte, x))
            } else {
                None
            }
        })
        .map(|(byte, char)| {
            (
                char,
                (((byte >> 4) & 0xf).into(), ((byte & 0xf) + 1).into()),
            )
        })
        .collect()
}

pub fn legacy_unicode(
    store: &mut impl storage::Storage,
    uneven: bool,
    sheets: impl Iterator<Item = (ndarray::Array2<Option<char>>, PathBuf)>,
    sizes: &Path,
) -> Result<(Vec<SheetGrid>, ModifiedTimes), ProvidersError> {
    let mut result = vec![];
    let mut times = ModifiedTimes::default();
    let size_time = store.modified_time(sizes)?;
    times.update(size_time);
    let size_bytes = store.read(sizes)?;
    let size_dict = glyph_sizes(size_bytes.into_iter());
    for (chars, image_path) in sheets {
        let mapped_chars =
            chars.mapv(|x| GridCell::sized(x, x.and_then(|y| size_dict.get(&y).copied())));
        let full = SheetGrid {
            chars: mapped_chars,
            image_path,
            height: 8,
            ascent: 7,
            bold_offset: 0.5,
            is_unifont: true,
            advance: AdvanceStrategy {
                specific: BTreeMap::new(),
                auto: if uneven {
                    AutoAdvance::UnevenUniform
                } else {
                    AutoAdvance::EvenUniform
                },
            },
        };
        result.push(full);
    }
    Ok((result, times))
}

fn read_stacked_providers(
    store: &mut storage::StackStorage,
    entry: &Path,
) -> Result<(Vec<JsonProvider>, ModifiedTimes), storage::StorageError> {
    let results = storage::stack_all(
        store,
        |x| storage::read_json::<RootJsonProvider, _>(x, entry),
        entry,
    )?;
    let mut times = ModifiedTimes::default();
    let mapped = results
        .into_iter()
        .rev()
        .flat_map(|x| {
            times.update(x.modified_time);
            x.data.providers
        })
        .collect();
    Ok((mapped, times))
}
