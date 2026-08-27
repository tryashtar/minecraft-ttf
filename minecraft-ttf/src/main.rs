use std::{fmt::Display, ops::RangeInclusive, path::PathBuf, str::FromStr};

use crate::font::Style;

mod cache;
mod font;

#[derive(clap::Parser, Debug)]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Generate TTF fonts from the vanilla game jar
    #[clap(subcommand)]
    Vanilla(VanillaCommand),
    /// List or generate fonts from a resource pack
    #[clap(subcommand)]
    Pack(PackCommand),
}

#[derive(clap::Subcommand, Debug)]
enum VanillaCommand {
    /// Generate TTF fonts from a specific version
    Generate(VanillaGenerateArgs),
    /// Generate all unique TTF fonts across Minecraft's history
    History(VanillaHistoryArgs),
}

#[derive(clap::Subcommand, Debug)]
enum PackCommand {
    /// Generate one TTF font from a resource pack
    Generate(PackGenerateArgs),
    /// List available font identifiers
    List(PackListArgs),
}

#[derive(clap::Args, Debug)]
struct VanillaGenerateArgs {
    /// Name of the Minecraft version to download, or "latest"
    version: String,
    #[command(flatten)]
    generate_args: GenerateArgs,
    #[command(flatten)]
    generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
struct VanillaHistoryArgs {
    #[command(flatten)]
    generate_args: GenerateArgs,
    #[command(flatten)]
    generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
struct PackGenerateArgs {
    #[command(flatten)]
    pack_args: PackArgs,
    #[command(flatten)]
    generate_args: GenerateArgs,
    #[command(flatten)]
    generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
struct PackListArgs {
    #[command(flatten)]
    pack_args: PackArgs,
    #[command(flatten)]
    generic_args: GenericArgs,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum ColorMode {
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
struct CharRange {
    ranges: Vec<RangeInclusive<char>>,
}

impl CharRange {
    fn none() -> Self {
        Self { ranges: vec![] }
    }

    fn all() -> Self {
        Self {
            ranges: vec![char::MIN..=char::MAX],
        }
    }

    fn matches(&self, ch: char) -> bool {
        self.ranges.iter().any(|x| x.contains(&ch))
    }
}

impl Display for CharRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, range) in self.ranges.iter().enumerate() {
            write!(f, "{0:x}-{1:x}", *range.start() as u32, *range.end() as u32)?;
            if i < self.ranges.len() - 1 {
                write!(f, ",")?;
            }
        }
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
enum CharRangeError {
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
            None => char..=char,
            Some(first) => first..=char,
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
struct GenericArgs {
    /// Folder for cache files
    #[arg(long, default_value = "cache")]
    cache: PathBuf,
}

#[derive(clap::Args, Debug)]
struct PackArgs {
    /// Name of the Minecraft version the resource pack is targeting, or "latest"
    version: String,
    /// Path to the resource pack folder or zip file
    location: PathBuf,
}

#[derive(clap::Args, Debug)]
struct GenerateArgs {
    /// Styles to generate
    #[arg(long, default_values_t = [Style::Regular])]
    styles: Vec<Style>,
    /// When to include color for characters that come from images (auto = only if any part of the image is not solid white)
    #[arg(long, default_value_t = ColorMode::Auto)]
    color: ColorMode,
    /// Ranges of characters to include. Example: "0020-007e,0370-03ff"
    #[arg(long, default_value_t = CharRange::all())]
    chars: CharRange,
    /// Ranges of characters from GNU unifont providers to include. Example: "0000-ffff"
    #[arg(long, default_value_t = CharRange::none())]
    unifont_chars: CharRange,
    /// Act as though the "Force Unicode Font" option was enabled
    #[arg(long, default_value_t = false)]
    option_uniform: bool,
    /// Act as though the "Japanese Glyph Variants" option was enabled
    #[arg(long, default_value_t = false)]
    option_jp: bool,
    /// Folder to save the generated fonts in
    #[arg(long, default_value = "out")]
    output: PathBuf,
}

fn main() {
    let cli = <Cli as clap::Parser>::parse();
}
