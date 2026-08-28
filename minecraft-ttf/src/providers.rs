use std::{collections::HashMap, fmt::Display, path::PathBuf, str::FromStr};

use bitmap_ttf::bitmap::Bitmap;

#[derive(Debug)]
pub struct CharBitmap {
    bitmap: Bitmap,
    advance: f32,
    bold_offset: f32,
}

#[derive(Debug)]
pub struct BitmapProvider {
    height: i32,
    ascent: i32,
    chars: indexmap::IndexMap<char, CharBitmap>,
}

pub struct CharImage {
    image: image::SubImage<()>,
    bitmap: Bitmap,
    advance: f32,
    bold_offset: f32,
}

pub struct ImageProvider {
    height: i32,
    ascent: i32,
    has_color: bool,
    chars: indexmap::IndexMap<char, CharImage>,
}

#[derive(Debug)]
pub struct SpaceProvider {
    chars: HashMap<char, f32>,
}

pub enum Provider {
    Bitmap(BitmapProvider),
    Image(ImageProvider),
    Space(SpaceProvider),
}

#[derive(Debug, Clone)]
pub struct Identifier {
    namespace: String,
    body: PathBuf,
}

impl Identifier {
    pub fn new(namespace: String, body: PathBuf) -> Self {
        Self { namespace, body }
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
