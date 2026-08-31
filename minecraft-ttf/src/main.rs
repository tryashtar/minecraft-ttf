use std::{
    fmt::{Display, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    range::RangeInclusive,
    str::FromStr,
};

use image::GenericImageView;
use tracing::error;
use tracing_subscriber::layer::SubscriberExt;

use crate::{cache::AssetStorage, font::Style, versions::VanillaFontId};

mod cache;
mod font;
mod providers;
mod storage;
mod versions;

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
    #[arg(long, value_delimiter=',', default_values_t = [VanillaFontId::Default])]
    identifiers: Vec<VanillaFontId>,
    #[command(flatten)]
    generate_args: GenerateArgs,
    #[command(flatten)]
    generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
struct VanillaHistoryArgs {
    /// First version to scan
    #[arg(long)]
    from: Option<String>,
    /// Last version to scan
    #[arg(long)]
    to: Option<String>,
    #[command(flatten)]
    generate_args: GenerateArgs,
    #[command(flatten)]
    generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
struct PackGenerateArgs {
    #[command(flatten)]
    pack_args: PackArgs,
    identifier: providers::Identifier,
    name: String,
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
            write!(f, "{0:x}-{1:x}", range.start as u32, range.last as u32)?;
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
    #[arg(long, value_delimiter=',', default_values_t = [Style::Regular])]
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

fn main() -> ExitCode {
    setup_logging();
    let cli = <Cli as clap::Parser>::parse();
    let result = run(cli.command);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            error!("Fatal: {}", report(&err));
            ExitCode::FAILURE
        }
    }
}

fn setup_logging() {
    tracing_subscriber::util::SubscriberInitExt::init(
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::filter::EnvFilter::from_default_env()
                    .add_directive(concat!(env!("CARGO_PKG_NAME"), "=debug").parse().unwrap()),
            )
            .with(tracing_tree::HierarchicalLayer::new(2)),
    );
}

fn report(mut err: &dyn std::error::Error) -> String {
    let mut s = format!("{err}");
    while let Some(src) = err.source() {
        _ = write!(s, "\nCaused by: {src}");
        err = src;
    }
    s
}

#[derive(thiserror::Error, Debug)]
enum CommandError {
    #[error(transparent)]
    Cache(#[from] cache::CacheError),
    #[error(transparent)]
    Version(#[from] versions::VersionError),
    #[error(transparent)]
    Storage(#[from] storage::StorageError),
    #[error(transparent)]
    Providers(#[from] providers::ProvidersError),
    #[error("No version with that name found")]
    UnknownVersion,
}

fn run(command: Command) -> Result<(), CommandError> {
    match command {
        Command::Vanilla(VanillaCommand::Generate(args)) => vanilla_generate(&args),
        Command::Vanilla(VanillaCommand::History(args)) => vanilla_history(&args),
        Command::Pack(PackCommand::Generate(args)) => pack_generate(&args),
        Command::Pack(PackCommand::List(args)) => pack_list(&args),
    }
}

#[derive(Debug)]
struct JarInfo<T> {
    launcher: cache::LauncherData,
    jar_store: storage::ZipStorage<T>,
    asset_store: cache::AssetStorage,
    version: versions::MinecraftVersion,
}

fn load_jar(
    version: &str,
    cache: &Path,
) -> Result<JarInfo<impl std::io::Read + std::io::Seek + use<>>, CommandError> {
    let manifest = cache::get_manifest(cache)?;
    let version_id = match version {
        "latest" => &manifest.latest.snapshot,
        _ => version,
    };
    let version = manifest
        .find_version(version_id)
        .ok_or(CommandError::UnknownVersion)?;
    let launcher_data = cache::get_launcher(version, cache)?;
    let assets = cache::get_asset_source(&launcher_data.asset_index, cache)?;
    let mut jar = cache::get_jar(&launcher_data, cache)?;
    let version = versions::detect_version(&mut jar)?;
    Ok(JarInfo {
        launcher: launcher_data,
        jar_store: storage::ZipStorage::new(jar),
        asset_store: AssetStorage::new(
            assets,
            version.asset_mount.clone(),
            cache.join("assets/objects"),
        ),
        version,
    })
}

fn image_has_color(image: &image::DynamicImage) -> bool {
    !image.pixels().all(|(_, _, pixel)| {
        let [r, g, b, a] = pixel.0;
        a == 0 || (r == 255 && g == 255 && b == 255)
    })
}

impl providers::ProviderOptions for GenerateArgs {
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

fn vanilla_generate(args: &VanillaGenerateArgs) -> Result<(), CommandError> {
    let info = load_jar(&args.version, &args.generic_args.cache)?;
    println!("Generating fonts from Minecraft {}", info.launcher.id);
    println!("Font support level {}", info.version.name);
    let mut stack =
        storage::StackStorage(vec![Box::new(info.jar_store), Box::new(info.asset_store)]);
    for identifier in &args.identifiers {
        let providers = versions::get_providers(
            &info.version,
            &mut stack,
            (*identifier).into(),
            &args.generate_args,
        )?;
        match providers {
            None => {
                println!("No providers found for {}", identifier);
            }
            Some(providers) => {}
        }
    }
    Ok(())
}

fn vanilla_history(args: &VanillaHistoryArgs) -> Result<(), CommandError> {
    Ok(())
}

fn pack_generate(args: &PackGenerateArgs) -> Result<(), CommandError> {
    Ok(())
}

fn pack_list(args: &PackListArgs) -> Result<(), CommandError> {
    Ok(())
}
