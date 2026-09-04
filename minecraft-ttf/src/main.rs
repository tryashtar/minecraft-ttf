use std::{
    collections::{BTreeMap, HashMap, btree_map::Entry},
    fmt::{Display, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    range::RangeInclusive,
    str::FromStr,
};

use bitmap_ttf::{
    bitmap::Bitmap,
    font::{FontPositions, MakeFontError, ScriptPositions, make_font},
};
use clap::Parser;
use image::GenericImageView;
use tracing::error;
use tracing_subscriber::layer::SubscriberExt;

use crate::{
    cache::AssetStorage,
    font::{FontInfo, Style, StyleInfo, create_font, font_meta},
    versions::VanillaFontId,
};

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
    /// Identifiers of the font definitions to generate fonts from
    #[arg(long, value_delimiter=',', default_values_t = [VanillaFontId::Default])]
    identifiers: Vec<VanillaFontId>,
    #[command(flatten)]
    font_args: FontArgs,
    #[command(flatten)]
    generate_args: GenerateArgs,
    #[command(flatten)]
    generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
struct VanillaHistoryArgs {
    /// Identifiers to check
    #[arg(long, value_delimiter=',', default_values_t = [VanillaFontId::Default])]
    identifiers: Vec<VanillaFontId>,
    /// First version to scan
    #[arg(long)]
    from: Option<String>,
    /// Last version to scan
    #[arg(long)]
    to: Option<String>,
    /// Folder to save the changed fonts in
    #[arg(long)]
    output: Option<PathBuf>,
    #[command(flatten)]
    font_args: FontArgs,
    #[command(flatten)]
    generic_args: GenericArgs,
}

#[derive(clap::Args, Debug)]
struct PackGenerateArgs {
    #[command(flatten)]
    pack_args: PackArgs,
    /// Identifier of the font definition to generate fonts from
    identifier: providers::Identifier,
    /// Display name for the generated TTF font's metadata
    name: String,
    /// Created time for the generated TTF font's metadata
    #[arg(long)]
    created_time: Option<jiff::Timestamp>,
    #[command(flatten)]
    font_args: FontArgs,
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
struct FontArgs {
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
}

#[derive(clap::Args, Debug)]
struct GenerateArgs {
    /// Styles to generate
    #[arg(long, value_delimiter=',', default_values_t = [Style::Regular])]
    styles: Vec<Style>,
    /// Folder to save the generated fonts in
    #[arg(long, default_value = "out")]
    output: PathBuf,
}

fn main() -> ExitCode {
    setup_logging();
    let cli = Cli::parse();
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
                tracing_subscriber::filter::EnvFilter::from_default_env().add_directive(
                    concat!(env!("CARGO_PKG_NAME"), "=debug")
                        .parse()
                        .expect("Hardcoded env"),
                ),
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
#[error("Running command")]
enum CommandError {
    Cache(#[from] cache::CacheError),
    Version(#[from] versions::VersionError),
    Storage(#[from] storage::StorageError),
    Providers(#[from] providers::ProvidersError),
    #[error("No version with that name found")]
    UnknownVersion,
    Font(#[from] MakeFontError),
    Io(#[from] std::io::Error),
    Yaml(#[from] serde_saphyr::Error),
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

fn load_jar_name(
    version: &str,
    updates: &[versions::VersionUpdate],
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
    load_jar_version(version, updates, cache)
}

fn load_jar_version(
    version: &cache::ManifestVersion,
    updates: &[versions::VersionUpdate],
    cache: &Path,
) -> Result<JarInfo<impl std::io::Read + std::io::Seek + use<>>, CommandError> {
    let launcher_data = cache::get_launcher(version, cache)?;
    let assets = cache::get_asset_source(&launcher_data.asset_index, cache)?;
    let mut jar = cache::get_jar(&launcher_data, cache)?;
    let version = versions::detect_version(updates, &mut jar)?;
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

fn version_checker() -> Result<Vec<versions::VersionUpdate>, serde_saphyr::Error> {
    let version_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/versions.yaml"));
    let versions: Vec<versions::VersionUpdate> = serde_saphyr::from_str(version_yaml)?;
    Ok(versions)
}

fn vanilla_generate(args: &VanillaGenerateArgs) -> Result<(), CommandError> {
    let checker = version_checker()?;
    let positions = positions();
    let info = load_jar_name(&args.version, &checker, &args.generic_args.cache)?;
    println!("Generating fonts from Minecraft {}", info.launcher.id);
    println!("Font support level {}", info.version.name);
    let mut stack =
        storage::StackStorage(vec![Box::new(info.jar_store), Box::new(info.asset_store)]);
    let missing_glyph = info.version.supports_missing_glyph().then(missing_glyph);
    for identifier in &args.identifiers {
        let id_info = identifier.info();
        let providers = versions::get_providers(
            &info.version,
            &mut stack,
            (*identifier).into(),
            &args.font_args,
        )?;
        match providers {
            None => {
                println!("No providers found for {}", identifier);
            }
            Some(providers) => {
                println!("Generating {}", identifier);
                for style in &args.generate_args.styles {
                    let style_info = style.info();
                    let ttf_name = format!(
                        "{}-{}.ttf",
                        id_info.name.replace(' ', ""),
                        style_info.name.replace(' ', "")
                    );
                    let out_file = args.generate_args.output.join(ttf_name);
                    generate_font(
                        &style_info,
                        &providers,
                        &positions,
                        missing_glyph.as_ref(),
                        id_info.name.clone(),
                        Some(id_info.created),
                        &out_file,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn missing_glyph() -> Bitmap {
    let (mw, mh) = (5, 8);
    let mut bitmap = Bitmap::new(mw, mh);
    for y in 0..mh {
        for x in 0..mw {
            if x == 0 || y == 0 || x == mw - 1 || y == mh - 1 {
                bitmap.set(x, y, true);
            }
        }
    }
    bitmap
}

fn generate_font(
    style_info: &StyleInfo,
    providers: &providers::Providers,
    positions: &FontPositions,
    missing_glyph: Option<&Bitmap>,
    name: String,
    created: Option<jiff::Timestamp>,
    path: &Path,
) -> Result<(), CommandError> {
    let info = create_font(
        providers.providers.iter(),
        missing_glyph,
        style_info.bold,
        style_info.italic,
    )
    .map_err(MakeFontError::Coords)?;
    let (sizes, smallest_legible) = get_pixel_info(&info);
    print_pixel_info(&sizes);
    let created = match created {
        Some(created) => created,
        None => providers.times.oldest.ok_or(MakeFontError::Timestamp)?,
    };
    let modified = providers.times.newest.ok_or(MakeFontError::Timestamp)?;
    let meta = font_meta(name, style_info, info.font_em, created, modified);
    let mut data = make_font(
        meta,
        positions,
        smallest_legible,
        &info.chars,
        &info.missing_glyph,
    )?;
    let bytes = data.build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &bytes)?;
    Ok(())
}

fn get_pixel_info(info: &FontInfo) -> (BTreeMap<u16, Vec<char>>, u16) {
    if !info.colored.is_empty() {
        println!("{} have color", info.colored.len())
    }
    let mut smallest_point = u16::MAX;
    let mut point_sizes: BTreeMap<u16, Vec<char>> = BTreeMap::new();
    for ((num, denom), chars) in &info.scales {
        let top = num * 12;
        let gcd = num::integer::gcd(top, *denom);
        let point_size = top / gcd;
        smallest_point = smallest_point.min(point_size);
        point_sizes.entry(point_size).or_default().extend(chars);
    }
    (point_sizes, smallest_point)
}

fn report_characters(chars: &[char]) -> String {
    if chars.len() > 60 {
        return format!("{} characters", chars.len());
    }
    format!(
        "{} characters ({:?})",
        chars.len(),
        chars.iter().collect::<String>()
    )
}

fn print_pixel_info(point_sizes: &BTreeMap<u16, Vec<char>>) {
    for (point, chars) in point_sizes {
        println!(
            "\t{} will look pixel-perfect at font size multiples of {}px",
            report_characters(chars),
            point
        );
    }
    if let Some(lcm) = point_sizes.keys().copied().reduce(num::integer::lcm) {
        println!(
            "\tAll characters in this font will look pixel-perfect at font size multiples of {}px",
            lcm
        );
    }
}

fn positions() -> FontPositions {
    FontPositions {
        ascent: 9.0 / 12.0,
        descent: 2.0 / 12.0,
        s_cap_height: 7.0 / 12.0,
        sx_height: 5.0 / 12.0,
        y_strikeout_position: 4.0 / 12.0,
        y_strikeout_size: 1.0 / 12.0,
        underline_position: -1.0 / 12.0,
        underline_thickness: 1.0 / 12.0,
        line_gap: -1.0 / 12.0,
        italic_angle: 4.0f64.atan2(1.0).to_degrees() - 90.0,
        subscript: ScriptPositions {
            x_size: 4.0 / 12.0,
            y_size: 5.0 / 12.0,
            x_offset: 0.0 / 12.0,
            y_offset: 1.0 / 12.0,
        },
        superscript: ScriptPositions {
            x_size: 4.0 / 12.0,
            y_size: 5.0 / 12.0,
            x_offset: 0.0 / 12.0,
            y_offset: 2.0 / 12.0,
        },
    }
}

#[derive(Debug, Default, Clone)]
struct ProviderGlyphSummary(BTreeMap<char, GlyphSummaryEntry>);

impl ProviderGlyphSummary {
    fn new(providers: Vec<providers::Provider>) -> Self {
        let mut map = BTreeMap::new();
        for provider in providers {
            match provider {
                providers::Provider::Bitmap(bitmap) => {
                    for (char, glyph) in bitmap.chars {
                        if let Entry::Vacant(e) = map.entry(char) {
                            e.insert(GlyphSummaryEntry::Bitmap {
                                bitmap: glyph.bitmap,
                                height: bitmap.height,
                                ascent: bitmap.ascent,
                                advance: glyph.advance,
                                bold_offset: glyph.bold_offset,
                                has_color: false,
                            });
                        }
                    }
                }
                providers::Provider::Image(image) => {
                    for (char, glyph) in image.chars {
                        if let Entry::Vacant(e) = map.entry(char) {
                            e.insert(GlyphSummaryEntry::Bitmap {
                                bitmap: glyph.bitmap,
                                height: image.height,
                                ascent: image.ascent,
                                advance: glyph.advance,
                                bold_offset: glyph.bold_offset,
                                has_color: image.has_color,
                            });
                        }
                    }
                }
                providers::Provider::Space(space) => {
                    for (char, glyph) in space.chars {
                        if let Entry::Vacant(e) = map.entry(char) {
                            e.insert(GlyphSummaryEntry::Space(glyph));
                        }
                    }
                }
            }
        }
        Self(map)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum GlyphSummaryEntry {
    Bitmap {
        bitmap: Bitmap,
        height: i32,
        ascent: i32,
        advance: f32,
        bold_offset: f32,
        has_color: bool,
    },
    Space(f32),
}

#[derive(Debug, Default)]
struct SummaryChangeReport {
    added: BTreeMap<char, GlyphSummaryEntry>,
    removed: BTreeMap<char, GlyphSummaryEntry>,
    changed: BTreeMap<char, (GlyphSummaryEntry, GlyphSummaryEntry)>,
}

impl SummaryChangeReport {
    fn new(mut old: ProviderGlyphSummary, new: ProviderGlyphSummary) -> Self {
        let mut result = Self::default();
        for (char, entry) in new.0 {
            if let Some(existing) = old.0.remove(&char) {
                if existing != entry {
                    result.changed.insert(char, (existing, entry));
                }
            } else {
                result.added.insert(char, entry);
            }
        }
        result.removed = old.0;
        result
    }

    fn any(&self) -> bool {
        !self.added.is_empty() || !self.removed.is_empty() || !self.changed.is_empty()
    }
}

fn vanilla_history(args: &VanillaHistoryArgs) -> Result<(), CommandError> {
    let checker = version_checker()?;
    let manifest = cache::get_manifest(&args.generic_args.cache)?;
    let mut history: HashMap<VanillaFontId, Vec<ProviderGlyphSummary>> = HashMap::new();
    for version in manifest.versions.iter().rev() {
        match load_jar_version(version, &checker, &args.generic_args.cache) {
            Err(CommandError::Version(versions::VersionError::UnknownVersion)) => {
                continue;
            }
            Err(e) => {
                return Err(e);
            }
            Ok(info) => {
                let mut stack = storage::StackStorage(vec![
                    Box::new(info.jar_store),
                    Box::new(info.asset_store),
                ]);
                for identifier in &args.identifiers {
                    let providers = versions::get_providers(
                        &info.version,
                        &mut stack,
                        (*identifier).into(),
                        &args.font_args,
                    )?;
                    if let Some(providers) = providers {
                        let summary = ProviderGlyphSummary::new(providers.providers);
                        let history_entry = history.entry(*identifier).or_default();
                        let last_version = history_entry.last().cloned().unwrap_or_default();
                        let changes = SummaryChangeReport::new(last_version, summary.clone());
                        if changes.any() {
                            history_entry.push(summary);
                            println!(
                                "{} ({}): {} changed",
                                version.id, info.version.name, identifier
                            );
                            if !changes.added.is_empty() {
                                println!(
                                    "\tadded {}",
                                    report_characters(
                                        &changes.added.keys().copied().collect::<Vec<_>>()
                                    ),
                                );
                            }
                            if !changes.removed.is_empty() {
                                println!(
                                    "\tremoved {}",
                                    report_characters(
                                        &changes.removed.keys().copied().collect::<Vec<_>>()
                                    ),
                                );
                            }
                            if !changes.changed.is_empty() {
                                println!(
                                    "\tchanged {}",
                                    report_characters(
                                        &changes.changed.keys().copied().collect::<Vec<_>>()
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn pack_generate(args: &PackGenerateArgs) -> Result<(), CommandError> {
    Ok(())
}

fn pack_list(args: &PackListArgs) -> Result<(), CommandError> {
    Ok(())
}
