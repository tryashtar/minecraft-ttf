use std::{
    cmp::{max, min},
    collections::HashMap,
    fmt::Display,
    path::PathBuf,
    str::FromStr,
};

use bitmap_ttf::bitmap::{Bitmap, Rectangle};

use crate::storage;

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

#[derive(Debug, Clone)]
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
    image: image::DynamicImage,
    chars: ndarray::Array2<Option<char>>,
    sizes: Option<HashMap<char, (u32, u32)>>,
    include: impl Fn(char) -> bool,
) -> indexmap::IndexMap<char, CharBitmap> {
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
        let glyph_box = Rectangle {
            left,
            top,
            width: glyph_width as usize,
            height: glyph_height as usize,
        };
        let glyph = image::GenericImageView::view(
            &image,
            left as u32,
            top as u32,
            glyph_width,
            glyph_height,
        );
    }
    map
}
