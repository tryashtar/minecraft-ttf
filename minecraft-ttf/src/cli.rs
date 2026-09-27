use std::{fmt::Display, path::PathBuf, range::RangeInclusive, str::FromStr};

use image::GenericImageView;

use crate::{font, providers};

#[derive(clap::Parser, Debug)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Generate TTF fonts from the vanilla game jar
    #[clap(subcommand)]
    Vanilla(VanillaCommand),
    /// List or generate fonts from a resource pack
    #[clap(subcommand)]
    Pack(PackCommand),
}

#[derive(clap::Subcommand, Debug)]
pub enum VanillaCommand {
    /// Generate TTF fonts from a specific version
    Generate(VanillaGenerateArgs),
    /// Generate all unique TTF fonts across Minecraft's history
    History(VanillaHistoryArgs),
    /// Generate data about how a Minecraft version handles fonts
    Version(VanillaVersionArgs),
}

#[derive(clap::Subcommand, Debug)]
pub enum PackCommand {
    /// Generate one TTF font from a resource pack
    Generate(PackGenerateArgs),
    /// List available font identifiers
    List(PackListArgs),
}

#[derive(clap::Args, Debug)]
pub struct VanillaGenerateArgs {
    /// Name of the Minecraft version to download, or "latest"
    pub version: String,
    #[command(flatten)]
    pub font_args: FontArgs,
    #[command(flatten)]
    pub generate_args: GenerateArgs,
    #[command(flatten)]
    pub generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
pub struct VanillaHistoryArgs {
    /// First version to scan
    #[arg(long)]
    pub from: Option<String>,
    /// Last version to scan
    #[arg(long)]
    pub to: Option<String>,
    #[command(flatten)]
    pub font_args: FontArgs,
    #[command(flatten)]
    pub generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
pub struct VanillaVersionArgs {
    /// Name of the Minecraft version to download
    pub version: String,
    #[command(flatten)]
    pub generic_args: GenericArgs,
    /// File to save the information to
    #[arg(long, default_value = "version.json")]
    pub output: PathBuf,
}

#[derive(clap::Args, Debug)]
pub struct PackGenerateArgs {
    #[command(flatten)]
    pub pack_args: PackArgs,
    #[command(flatten)]
    pub font_args: FontArgs,
    #[command(flatten)]
    pub generate_args: GenerateArgs,
    #[command(flatten)]
    pub generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
pub struct PackListArgs {
    #[command(flatten)]
    pub pack_args: PackArgs,
    #[command(flatten)]
    pub generic_args: GenericArgs,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
pub enum ColorMode {
    Always,
    Never,
    Auto,
}

impl Display for ColorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Always => write!(f, "always"),
            Self::Never => write!(f, "never"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CharRange {
    ranges: Vec<RangeInclusive<char>>,
}

impl CharRange {
    fn none() -> Self {
        Self { ranges: vec![] }
    }

    fn all() -> Self {
        Self {
            ranges: vec![RangeInclusive {
                start: char::MIN,
                last: char::MAX,
            }],
        }
    }

    fn matches(&self, ch: char) -> bool {
        self.ranges.iter().any(|x| x.contains(&ch))
    }
}

impl Display for CharRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, range) in self.ranges.iter().enumerate() {
            write!(
                f,
                "{0:x}-{1:x}",
                Into::<u32>::into(range.start),
                Into::<u32>::into(range.last)
            )?;
            if i < self.ranges.len() - 1 {
                write!(f, ",")?;
            }
        }
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum CharRangeError {
    #[error("Empty range")]
    EmptyRange,
    #[error("Unexpected character {0}")]
    UnexpectedCharacter(char),
    #[error("{0:x} is not a valid Unicode codepoint")]
    InvalidCodepoint(u32),
    #[error("Error parsing hex: {0}")]
    ParseHex(std::num::ParseIntError),
}

#[derive(Debug, Default)]
struct CharRangeBuilder {
    current_codepoint: String,
    first_codepoint: Option<char>,
    ranges: Vec<RangeInclusive<char>>,
}

impl CharRangeBuilder {
    fn consume(&mut self, char: char) -> Result<(), CharRangeError> {
        match char {
            ',' => self.next_range(),
            '-' => self.next_char(),
            c if c.is_ascii_hexdigit() => {
                self.current_codepoint.push(c);
                Ok(())
            }
            other => Err(CharRangeError::UnexpectedCharacter(other)),
        }
    }

    fn parse_char(&mut self) -> Result<char, CharRangeError> {
        if self.current_codepoint.is_empty() {
            Err(CharRangeError::EmptyRange)
        } else {
            let hex = u32::from_str_radix(&self.current_codepoint, 16)
                .map_err(CharRangeError::ParseHex)?;
            let Some(char) = char::from_u32(hex) else {
                return Err(CharRangeError::InvalidCodepoint(hex));
            };
            self.current_codepoint.clear();
            Ok(char)
        }
    }

    fn next_char(&mut self) -> Result<(), CharRangeError> {
        let char = self.parse_char()?;
        self.first_codepoint = Some(char);
        Ok(())
    }

    fn next_range(&mut self) -> Result<(), CharRangeError> {
        let char = self.parse_char()?;
        let range = match self.first_codepoint {
            None => RangeInclusive {
                start: char,
                last: char,
            },
            Some(first) => RangeInclusive {
                start: first,
                last: char,
            },
        };
        self.ranges.push(range);
        Ok(())
    }

    fn build(mut self) -> Result<CharRange, CharRangeError> {
        self.next_range()?;
        Ok(CharRange {
            ranges: self.ranges,
        })
    }
}

impl FromStr for CharRange {
    type Err = CharRangeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Ok(CharRange::none());
        }
        if s == "*" {
            return Ok(CharRange::all());
        }
        let mut builder = CharRangeBuilder::default();
        for char in s.chars() {
            builder.consume(char)?;
        }
        builder.build()
    }
}

#[derive(clap::Args, Debug)]
pub struct GenericArgs {
    /// Folder for cache files
    #[arg(long, default_value = "cache")]
    pub cache: PathBuf,
}

#[derive(clap::Args, Debug)]
pub struct PackArgs {
    /// Name of the Minecraft version the resource pack is targeting, or "latest"
    pub version: String,
    /// One or more paths to resource pack folders or zip files
    #[arg(num_args = 1..)]
    pub packs: Vec<PathBuf>,
}

#[derive(clap::Args, Debug)]
pub struct FontArgs {
    #[command(flatten)]
    pub font_info: FontInfoArgs,
    /// When to include color for characters that come from images (auto = only if any part of the image is not solid white)
    #[arg(long, default_value_t = ColorMode::Auto)]
    pub color: ColorMode,
    /// Ranges of characters to include. Example: "0020-007e,0370-03ff"
    #[arg(long, default_value_t = CharRange::all())]
    pub chars: CharRange,
    /// Ranges of characters from GNU unifont providers to include. Example: "0000-ffff"
    #[arg(long, default_value_t = CharRange::none())]
    pub unifont_chars: CharRange,
    /// Act as though the "Force Unicode Font" option was enabled
    #[arg(long, default_value_t = false)]
    pub option_uniform: bool,
    /// Act as though the "Japanese Glyph Variants" option was enabled
    #[arg(long, default_value_t = false)]
    pub option_jp: bool,
}

#[derive(clap::Args, Debug)]
pub struct FontInfoArgs {
    /// Identifier of the font definition to generate fonts from
    #[arg(long)]
    pub identifier: providers::Identifier,
    /// Display name for the generated TTF font's metadata
    pub font_name: Option<String>,
    /// Created time for the generated TTF font's metadata
    pub font_created: Option<jiff::Timestamp>,
    /// Modified time for the generated TTF font's metadata
    pub font_modified: Option<jiff::Timestamp>,
}

#[derive(clap::Args, Debug)]
pub struct GenerateArgs {
    /// Styles to generate
    #[arg(long, value_delimiter=',', default_values_t = [font::Style::Regular])]
    pub styles: Vec<font::Style>,
    /// Folder to save the generated fonts in
    #[arg(long, default_value = "out")]
    pub output: PathBuf,
}

fn image_has_color(image: &image::DynamicImage) -> bool {
    !image.pixels().all(|(_, _, pixel)| {
        let [r, g, b, a] = pixel.0;
        a == 0 || (r == 255 && g == 255 && b == 255)
    })
}

impl providers::ProviderOptions for FontArgs {
    fn has_color(&self, image: &image::DynamicImage) -> bool {
        match self.color {
            ColorMode::Always => true,
            ColorMode::Never => false,
            ColorMode::Auto => image_has_color(image),
        }
    }

    fn include_char(&self, char: char) -> bool {
        self.chars.matches(char)
    }

    fn include_unifont_char(&self, char: char) -> bool {
        self.unifont_chars.matches(char)
    }

    fn option_uniform(&self) -> bool {
        self.option_uniform
    }

    fn option_jp(&self) -> bool {
        self.option_jp
    }
}
