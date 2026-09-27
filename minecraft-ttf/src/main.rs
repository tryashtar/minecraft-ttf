use std::{
    collections::{BTreeMap, HashMap, btree_map::Entry},
    fmt::Write,
    path::Path,
    process::ExitCode,
};

use bitmap_ttf::{
    bitmap::Bitmap,
    font::{FontPositions, MakeFontError, ScriptPositions, make_font},
};
use clap::Parser;
use tracing::error;
use tracing_subscriber::layer::SubscriberExt;

use crate::{
    cache::AssetStorage,
    font::{StyleInfo, create_font, font_meta},
    storage::get_storage,
    versions::VanillaFontId,
};

mod cache;
mod cli;
mod font;
mod providers;
mod storage;
mod versions;

fn main() -> ExitCode {
    setup_logging();
    let cli = cli::Cli::parse();
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
    Serde(#[from] serde_json::Error),
}

fn run(command: cli::Command) -> Result<(), CommandError> {
    match command {
        cli::Command::Vanilla(cli::VanillaCommand::Generate(args)) => vanilla_generate(&args),
        cli::Command::Vanilla(cli::VanillaCommand::History(args)) => vanilla_history(&args),
        cli::Command::Vanilla(cli::VanillaCommand::Version(args)) => vanilla_version(&args),
        cli::Command::Pack(cli::PackCommand::Generate(args)) => pack_generate(&args),
        cli::Command::Pack(cli::PackCommand::List(args)) => pack_list(&args),
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
        asset_store: AssetStorage::new(assets, version.asset_mount.clone(), cache.to_owned()),
        version,
    })
}

fn version_checker() -> Vec<versions::VersionUpdate> {
    let version_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/updates.yaml"));
    let versions: Vec<versions::VersionUpdate> =
        serde_saphyr::from_str(version_yaml).expect("versions should be configured correctly");
    versions
}

fn vanilla_generate(args: &cli::VanillaGenerateArgs) -> Result<(), CommandError> {
    let checker = version_checker();
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
    let (sizes, smallest_legible) = get_pixel_info(&info.scales);
    print_pixel_info(&info.colored, &sizes);
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

fn get_pixel_info(scales: &HashMap<(u16, u16), Vec<char>>) -> (BTreeMap<u16, Vec<char>>, u16) {
    let mut smallest_point = u16::MAX;
    let mut point_sizes: BTreeMap<u16, Vec<char>> = BTreeMap::new();
    for ((num, denom), chars) in scales {
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

fn print_pixel_info(colored: &[char], point_sizes: &BTreeMap<u16, Vec<char>>) {
    if !colored.is_empty() {
        println!("\t{} have color", report_characters(colored))
    }
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

#[derive(Debug, Default, Clone, PartialEq)]
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

type HistoryTrack = HashMap<VanillaFontId, Vec<(String, ProviderGlyphSummary)>>;

fn vanilla_history(args: &cli::VanillaHistoryArgs) -> Result<(), CommandError> {
    let checker = version_checker();
    let manifest = cache::get_manifest(&args.generic_args.cache)?;
    let mut history = HistoryTrack::new();
    let versions = manifest
        .versions
        .iter()
        .rev()
        .skip_while(|x| args.from.as_ref().is_some_and(|y| &x.id != y))
        .take_while(|x| args.to.as_ref().is_none_or(|y| &x.id != y));
    for version in versions {
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
                        vanilla_history_entry(
                            providers.providers,
                            &mut history,
                            identifier,
                            &version.id,
                            &info.version.name,
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

fn vanilla_version(args: &cli::VanillaVersionArgs) -> Result<(), CommandError> {
    let checker = version_checker();
    let manifest = cache::get_manifest(&args.generic_args.cache)?;
    let mversion = manifest
        .find_version(&args.version)
        .ok_or(CommandError::UnknownVersion)?;
    let cache = &args.generic_args.cache;
    let launcher_data = cache::get_launcher(mversion, cache)?;
    let mut jar = cache::get_jar(&launcher_data, cache)?;
    let version = versions::detect_version(&checker, &mut jar)?;
    let text = serde_json::to_string(&version)?;
    std::fs::write(&args.output, text)?;
    Ok(())
}

fn vanilla_history_entry(
    providers: Vec<providers::Provider>,
    history: &mut HistoryTrack,
    identifier: &VanillaFontId,
    version: &str,
    version_name: &str,
) {
    let summary = ProviderGlyphSummary::new(providers);
    let history_entry = history.entry(*identifier).or_default();
    let last_version = history_entry
        .last()
        .cloned()
        .map(|x| x.1)
        .unwrap_or_default();
    if summary == last_version {
        return;
    }
    let existing_match = history_entry.iter().find(|x| x.1 == summary);
    if let Some((name, _)) = existing_match {
        println!(
            "{} ({}): {} reverted to {}",
            version, version_name, identifier, name
        );
        history_entry.push((String::from(version), summary));
        return;
    }
    let changes = SummaryChangeReport::new(last_version, summary.clone());
    if changes.any() {
        println!("{} ({}): {} changed", version, version_name, identifier);
        history_entry.push((String::from(version), summary));
        print_changes(&changes);
    }
}

fn print_changes(changes: &SummaryChangeReport) {
    if !changes.added.is_empty() {
        println!(
            "\tadded {}",
            report_characters(&changes.added.keys().copied().collect::<Vec<_>>()),
        );
    }
    if !changes.removed.is_empty() {
        println!(
            "\tremoved {}",
            report_characters(&changes.removed.keys().copied().collect::<Vec<_>>()),
        );
    }
    if !changes.changed.is_empty() {
        println!(
            "\tchanged {}",
            report_characters(&changes.changed.keys().copied().collect::<Vec<_>>()),
        );
    }
}

fn pack_generate(args: &cli::PackGenerateArgs) -> Result<(), CommandError> {
    let pack_storage = get_storage(&args.pack_args.location)?;
    let checker = version_checker();
    let positions = positions();
    let info = load_jar_name(&args.pack_args.version, &checker, &args.generic_args.cache)?;
    println!(
        "Generating font {} from resource pack {} on Minecraft {}",
        args.identifier,
        args.pack_args
            .location
            .file_name()
            .unwrap_or_default()
            .display(),
        info.launcher.id
    );
    println!("Font support level {}", info.version.name);
    let mut stack = storage::StackStorage(vec![
        Box::new(info.jar_store),
        Box::new(info.asset_store),
        pack_storage,
    ]);
    let missing_glyph = info.version.supports_missing_glyph().then(missing_glyph);
    let providers = versions::get_providers(
        &info.version,
        &mut stack,
        args.identifier.clone(),
        &args.font_args,
    )?;
    match providers {
        None => {
            println!("No providers found for {}", args.identifier);
        }
        Some(providers) => {
            for style in &args.generate_args.styles {
                let style_info = style.info();
                let ttf_name = format!(
                    "{}-{}.ttf",
                    args.name.replace(' ', ""),
                    style_info.name.replace(' ', "")
                );
                let out_file = args.generate_args.output.join(ttf_name);
                generate_font(
                    &style_info,
                    &providers,
                    &positions,
                    missing_glyph.as_ref(),
                    args.name.clone(),
                    args.created_time,
                    &out_file,
                )?;
            }
        }
    }
    Ok(())
}

fn pack_list(args: &cli::PackListArgs) -> Result<(), CommandError> {
    let pack_storage = get_storage(&args.pack_args.location)?;
    let checker = version_checker();
    let info = load_jar_name(&args.pack_args.version, &checker, &args.generic_args.cache)?;
    println!(
        "Available font identifiers in resource pack {} on Minecraft {}:",
        args.pack_args
            .location
            .file_name()
            .unwrap_or_default()
            .display(),
        info.launcher.id
    );
    let mut stack = storage::StackStorage(vec![
        Box::new(info.jar_store),
        Box::new(info.asset_store),
        pack_storage,
    ]);
    let identifiers = info.version.available_identifiers(&mut stack)?;
    for identifier in identifiers {
        println!("{}", identifier);
    }
    Ok(())
}
