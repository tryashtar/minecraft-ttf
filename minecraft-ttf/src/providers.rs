use std::{
    cmp::{max, min},
    collections::HashMap,
    ffi::OsStr,
    fmt::Display,
    path::{Path, PathBuf},
    str::FromStr,
};

use bitmap_ttf::bitmap::{Bitmap, Rectangle};
use image::GenericImageView;

use crate::{cache, storage};

#[derive(Debug)]
pub struct CharBitmap {
    pub bitmap: Bitmap,
    pub advance: f32,
    pub bold_offset: f32,
}

#[derive(Debug)]
pub struct BitmapProvider {
    pub height: i32,
    pub ascent: i32,
    pub chars: indexmap::IndexMap<char, CharBitmap>,
}

pub struct CharImage {
    pub content_box: Rectangle,
    pub bitmap: Bitmap,
    pub advance: f32,
    pub bold_offset: f32,
}

pub struct ImageProvider {
    pub image: image::DynamicImage,
    pub has_color: bool,
    pub height: i32,
    pub ascent: i32,
    pub chars: indexmap::IndexMap<char, CharImage>,
}

#[derive(Debug)]
pub struct SpaceProvider {
    pub chars: HashMap<char, f32>,
}

pub enum Provider {
    Bitmap(BitmapProvider),
    Image(ImageProvider),
    Space(SpaceProvider),
}

#[derive(serde_with::DeserializeFromStr, Debug, Clone)]
pub struct Identifier {
    pub namespace: String,
    pub body: PathBuf,
}

impl Identifier {
    pub fn new(namespace: String, body: PathBuf) -> Self {
        Self { namespace, body }
    }

    pub fn vanilla(body: PathBuf) -> Self {
        Self::new(String::from("minecraft"), body)
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

#[derive(Debug, Default)]
pub struct ModifiedTimes {
    oldest: Option<jiff::Timestamp>,
    newest: Option<jiff::Timestamp>,
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

pub struct Providers {
    pub providers: Vec<Provider>,
    pub times: ModifiedTimes,
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
pub enum ProvidersError {
    #[error(transparent)]
    Storage(#[from] storage::StorageError),
    #[error(transparent)]
    Jagged(#[from] JaggedArrayError),
}

pub fn image_grid(
    image: &image::DynamicImage,
    chars: &ndarray::Array2<Option<char>>,
    sizes: Option<&HashMap<char, (u32, u32)>>,
    include: impl Fn(char) -> bool,
) -> indexmap::IndexMap<char, (Rectangle, Bitmap)> {
    let mut map = indexmap::IndexMap::new();
    let glyph_width = image.width() / (chars.ncols() as u32);
    let glyph_height = image.height() / (chars.nrows() as u32);
    for ((row, col), char) in chars.indexed_iter() {
        let Some(char) = char else {
            continue;
        };
        if map.contains_key(char) || !include(*char) {
            continue;
        }
        let left = col * (glyph_width as usize);
        let top = row * (glyph_height as usize);
        let rectangle = Rectangle {
            left,
            top,
            width: glyph_width as usize,
            height: glyph_height as usize,
        };
        let size = match sizes {
            None => None,
            Some(sizes) => sizes.get(char).copied(),
        };
        let portion = image_portion(image, &rectangle, size);
        map.insert(*char, (rectangle, portion));
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
    let resize_box = match size {
        None => {
            let width = bitmap.content_box().map(|x| x.width).unwrap_or(0);
            Rectangle {
                left: 0,
                top: 0,
                width,
                height: bitmap.height(),
            }
        }
        Some((left, right)) => Rectangle {
            left: left as usize,
            top: 0,
            width: (right - left) as usize,
            height: bitmap.height(),
        },
    };
    bitmap.resized(&resize_box)
}

fn image_to_bitmap(image: image::SubImage<&image::DynamicImage>, threshold: u8) -> Bitmap {
    let mut bitmap = Bitmap::new(image.width() as usize, image.height() as usize);
    for (x, y, pixel) in image.pixels() {
        if pixel.0[3] >= threshold {
            bitmap.set(x as usize, y as usize, true);
        }
    }
    bitmap
}

pub trait ProviderOptions {
    fn has_color(&self, image: &image::DynamicImage) -> bool;
    fn include_char(&self, char: char) -> bool;
    fn include_unifont_char(&self, char: char) -> bool;
    fn option_uniform(&self) -> bool;
    fn option_jp(&self) -> bool;
}

pub fn normal_advance(bitmap: &Bitmap, height: i32) -> f32 {
    let scale = height as f32 / bitmap.height() as f32;
    (0.5 + bitmap.width() as f32 * scale).floor() + 1.0
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
#[serde(tag = "type", rename_all = "lowercase")]
enum JsonProvider {
    Bitmap(JsonBitmapProvider),
    Space(JsonSpaceProvider),
    Reference(JsonReferenceProvider),
    LegacyUnicode(JsonLegacyUnicodeProvider),
    Unihex(JsonUnihexProvider),
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
    advances: HashMap<char, f32>,
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

#[derive(Debug)]
pub struct ProviderBehavior {
    pub uniform: ForceUniformBehavior,
    pub uneven_unifont: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum ForceUniformBehavior {
    Filter,
    SkipBitmaps,
    SwitchIdentifier,
}

pub fn load_providers(
    identifier: &Identifier,
    store: &mut storage::StackStorage,
    options: &impl ProviderOptions,
    behavior: &ProviderBehavior,
) -> Result<Option<Providers>, ProvidersError> {
    let entry = identifier.to_entry(Some("font"), Some("json"));
    match read_stacked_providers(store, &entry) {
        Err(storage::StorageError::FileNotFound) => Ok(None),
        Err(err) => Err(ProvidersError::Storage(err)),
        Ok((providers, mut times)) => {
            let mut converted = vec![];
            for single in providers {
                let subs = convert_provider(store, options, behavior, single)?;
                converted.extend(subs.providers);
                times.merge(subs.times);
            }
            Ok(Some(Providers {
                providers: converted,
                times,
            }))
        }
    }
}

fn convert_provider(
    store: &mut storage::StackStorage,
    options: &impl ProviderOptions,
    behavior: &ProviderBehavior,
    provider: JsonProvider,
) -> Result<Providers, ProvidersError> {
    let mut times = ModifiedTimes::default();
    match provider {
        JsonProvider::Bitmap(bitmap) => {
            let img_entry = bitmap.file.to_entry(Some("textures"), None);
            let img_data = storage::read_image(store, &img_entry)?;
            times.update(img_data.modified_time);
            let height = bitmap.height.unwrap_or(8);
            let has_color = options.has_color(&img_data.data);
            let grid = split_grid(bitmap.chars.iter(), true)?;
            let char_images = image_grid(&img_data.data, &grid, None, |x| options.include_char(x));
            let chars = char_images
                .into_iter()
                .map(|(char, (content_box, bitmap))| {
                    let advance = normal_advance(&bitmap, height);
                    (
                        char,
                        CharImage {
                            content_box,
                            bitmap,
                            advance,
                            bold_offset: 1.0,
                        },
                    )
                })
                .collect();
            let full = ImageProvider {
                image: img_data.data,
                has_color,
                height,
                ascent: bitmap.ascent,
                chars,
            };
            Ok(Providers {
                providers: vec![Provider::Image(full)],
                times,
            })
        }
        JsonProvider::Space(space) => {
            let full = SpaceProvider {
                chars: space.advances.clone(),
            };
            Ok(Providers {
                providers: vec![Provider::Space(full)],
                times,
            })
        }
        JsonProvider::Reference(reference) => {
            let resolved = load_providers(&reference.id, store, options, behavior)?
                .ok_or(storage::StorageError::FileNotFound)?;
            Ok(resolved)
        }
        JsonProvider::LegacyUnicode(unicode) => todo!(),
        JsonProvider::Unihex(unihex) => todo!(),
    }
}

fn read_stacked_providers(
    store: &mut storage::StackStorage,
    entry: &Path,
) -> Result<(Vec<JsonProvider>, ModifiedTimes), storage::StorageError> {
    let results = storage::stack_all(store, |x| {
        storage::read_json::<RootJsonProvider, _>(x, entry)
    })?;
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
