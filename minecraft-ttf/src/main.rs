use std::{
    collections::{BTreeMap, HashMap, btree_map::Entry},
    fmt::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

use bitmap_ttf::{
    bitmap::Bitmap,
    font::{FontPositions, MakeFontError, make_font},
};
use clap::Parser;
use tracing::error;
use tracing_subscriber::layer::SubscriberExt;

use crate::{
    cache::AssetStorage,
    font::{StyleInfo, create_font, font_meta},
    storage::get_storage,
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
                    concat!(env!("CARGO_CRATE_NAME"), "=info")
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
    #[error("No providers for that identifier")]
    NoProviders,
    #[error("No version with that name found")]
    UnknownVersion,
    Font(#[from] MakeFontError),
    Io(#[from] std::io::Error),
    Serde(#[from] serde_json::Error),
}

fn run(command: cli::Command) -> Result<(), CommandError> {
    match command {
        cli::Command::Generate(args) => run_generate(&args),
        cli::Command::History(args) => run_history(&args),
        cli::Command::List(args) => run_list(&args),
        cli::Command::Export(cli::ExportCommand::Version(args)) => run_export_version(&args),
        cli::Command::Export(cli::ExportCommand::Providers(args)) => run_export_providers(&args),
        cli::Command::Export(cli::ExportCommand::Images(args)) => run_images_providers(&args),
    }
}

#[derive(Debug)]
struct JarInfo<T> {
    jar_store: storage::ZipStorage<T>,
    asset_store: cache::AssetStorage,
    version: versions::MinecraftVersion,
}

fn load_jar_source(
    source: &cli::JarSource,
    cache: &Path,
) -> Result<(zip::ZipArchive<std::fs::File>, cache::LauncherData), CommandError> {
    match source {
        cli::JarSource::Latest => {
            let manifest = cache::get_manifest(cache)?;
            let version = manifest
                .find_version(&manifest.latest.snapshot)
                .ok_or(CommandError::UnknownVersion)?;
            let launcher_data = cache::get_launcher(version, cache)?;
            let jar = cache::get_jar(&launcher_data, cache)?;
            Ok((jar, launcher_data))
        }
        cli::JarSource::Vanilla(version_id) => {
            let manifest = cache::get_manifest(cache)?;
            let version = manifest
                .find_version(version_id)
                .ok_or(CommandError::UnknownVersion)?;
            let launcher_data = cache::get_launcher(version, cache)?;
            let jar = cache::get_jar(&launcher_data, cache)?;
            Ok((jar, launcher_data))
        }
        cli::JarSource::Launcher { json } => {
            let file = std::fs::File::open(json)?;
            let launcher_data: cache::LauncherData = serde_json::from_reader(file)?;
            let jar = cache::get_jar(&launcher_data, cache)?;
            Ok((jar, launcher_data))
        }
        cli::JarSource::Full { jar, json } => {
            let file = std::fs::File::open(json)?;
            let launcher_data: cache::LauncherData = serde_json::from_reader(file)?;
            let file = std::fs::File::open(jar)?;
            let jar = zip::ZipArchive::new(file)
                .map_err(|x| CommandError::Storage(storage::StorageError::Zip(x)))?;
            Ok((jar, launcher_data))
        }
    }
}

fn load_jar_version(
    local: Option<&Path>,
    jar: &mut zip::ZipArchive<impl std::io::Read + std::io::Seek>,
    updates: &[versions::VersionUpdate],
) -> Result<versions::MinecraftVersion, CommandError> {
    if let Some(local) = local {
        let file = std::fs::File::open(local)?;
        let version: versions::MinecraftVersion = serde_json::from_reader(file)?;
        return Ok(version);
    }
    Ok(versions::detect_version(updates, jar)?)
}

fn build_jar_info<T>(
    assets: &cache::LauncherAssets,
    jar: zip::ZipArchive<T>,
    version: versions::MinecraftVersion,
    cache: &Path,
) -> Result<JarInfo<T>, CommandError> {
    let asset_source = cache::get_asset_source(assets, cache)?;
    Ok(JarInfo {
        jar_store: storage::ZipStorage::new(jar),
        asset_store: AssetStorage::new(asset_source, version.asset_mount.clone(), cache.to_owned()),
        version,
    })
}

fn version_checker() -> Vec<versions::VersionUpdate> {
    let version_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/updates.yaml"));
    let versions: Vec<versions::VersionUpdate> =
        serde_saphyr::from_str(version_yaml).expect("versions should be configured correctly");
    versions
}

fn get_name(args: &cli::FontMetaArgs, identifier: &providers::Identifier) -> String {
    if let Some(name) = args.font_name.as_ref() {
        return name.clone();
    }
    let providers::Identifier { namespace, body } = identifier;
    match (namespace.as_str(), body) {
        ("minecraft", path) if path == Path::new("default") => String::from("Minecraft Default"),
        ("minecraft", path) if path == Path::new("alt") => String::from("Minecraft Enchanting"),
        ("minecraft", path) if path == Path::new("illageralt") => {
            String::from("Minecraft Illager Runes")
        }
        ("minecraft", path) if path == Path::new("uniform") => String::from("Minecraft Unicode"),
        (namespace, body) => {
            let mut result = String::new();
            result.push_str(namespace);
            result.push(' ');
            for part in body.components() {
                result.push_str(&part.as_os_str().to_string_lossy());
                result.push(' ');
            }
            convert_case::ccase!(title, result)
        }
    }
}

fn get_created(
    args: &cli::FontMetaArgs,
    identifier: &providers::Identifier,
) -> Option<jiff::Timestamp> {
    if let Some(date) = args.font_created.as_ref() {
        return Some(*date);
    }
    let providers::Identifier { namespace, body } = identifier;
    match (namespace.as_str(), body) {
        ("minecraft", path) if path == Path::new("default") => {
            Some(jiff::Timestamp::constant(1_242_492_720, 0))
        }
        ("minecraft", path) if path == Path::new("alt") => {
            Some(jiff::Timestamp::constant(1_317_913_020, 0))
        }
        ("minecraft", path) if path == Path::new("illageralt") => {
            Some(jiff::Timestamp::constant(1_631_721_870, 0))
        }
        ("minecraft", path) if path == Path::new("uniform") => {
            Some(jiff::Timestamp::constant(1_323_360_326, 0))
        }
        (_, _) => None,
    }
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
    modified: Option<jiff::Timestamp>,
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
    let modified = match modified {
        Some(modified) => modified,
        None => providers.times.newest.ok_or(MakeFontError::Timestamp)?,
    };
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

type HistoryTrack = HashMap<providers::Identifier, Vec<(String, ProviderGlyphSummary)>>;

fn run_history(args: &cli::HistoryArgs) -> Result<(), CommandError> {
    let checker = version_checker();
    let manifest = cache::get_manifest(&args.generic_args.cache)?;
    let mut history = HistoryTrack::new();
    let versions = manifest
        .versions
        .iter()
        .rev()
        .skip_while(|x| args.from.as_ref().is_some_and(|y| &x.id != y))
        .take_while(|x| args.to.as_ref().is_none_or(|y| &x.id != y));
    for mversion in versions {
        match history_check_version(mversion, &checker, &mut history, args) {
            Ok(()) => {}
            Err(err) => {
                error!("Version {}: {}", mversion.id, err);
            }
        }
    }
    Ok(())
}

fn history_check_version(
    mversion: &cache::ManifestVersion,
    checker: &[versions::VersionUpdate],
    history: &mut HistoryTrack,
    args: &cli::HistoryArgs,
) -> Result<(), CommandError> {
    let launcher_data = cache::get_launcher(mversion, &args.generic_args.cache)?;
    let mut jar = cache::get_jar(&launcher_data, &args.generic_args.cache)?;
    let version = versions::detect_version(checker, &mut jar)?;
    let info = build_jar_info(
        &launcher_data.asset_index,
        jar,
        version,
        &args.generic_args.cache,
    )?;
    let mut stack =
        storage::StackStorage(vec![Box::new(info.jar_store), Box::new(info.asset_store)]);
    let providers = versions::get_partial_providers(
        &info.version,
        &mut stack,
        args.identifier.clone(),
        &args.font_args,
    )?;
    if let Some(providers) = providers {
        let providers = providers.resolve(&mut stack, &args.font_args)?;
        vanilla_history_entry(
            providers.providers,
            history,
            &args.identifier,
            &mversion.id,
            &info.version.name,
        );
    }
    Ok(())
}

fn run_export_version(args: &cli::ExportVersionArgs) -> Result<(), CommandError> {
    let checker = version_checker();
    let (mut jar, _) = load_jar_source(&args.jar_args.source(), &args.generic_args.cache)?;
    let version = versions::detect_version(&checker, &mut jar)?;
    let text = serde_json::to_string(&version)?;
    std::fs::write(&args.output, text)?;
    Ok(())
}

fn run_export_providers(args: &cli::ExportProvidersArgs) -> Result<(), CommandError> {
    let checker = version_checker();
    let (mut jar, launcher) = load_jar_source(&args.jar_args.source(), &args.generic_args.cache)?;
    let version = load_jar_version(args.jar_args.version_data.as_deref(), &mut jar, &checker)?;
    let info = build_jar_info(
        &launcher.asset_index,
        jar,
        version,
        &args.generic_args.cache,
    )?;
    let mut storage_vec: Vec<Box<dyn storage::Storage>> =
        vec![Box::new(info.jar_store), Box::new(info.asset_store)];
    for pack in &args.jar_args.packs {
        let pack_storage = get_storage(pack)?;
        storage_vec.push(pack_storage);
    }
    let mut stack = storage::StackStorage(storage_vec);
    let Some(providers) = versions::get_partial_providers(
        &info.version,
        &mut stack,
        args.identifier.clone(),
        &args.font_args,
    )?
    else {
        return Err(CommandError::NoProviders);
    };
    let providers = providers.resolve(&mut stack, &args.font_args)?;
    let text = serde_json::to_string(&providers)?;
    std::fs::write(&args.output, text)?;
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct ExportGridCell {
    char: Option<char>,
    #[serde(serialize_with = "providers::serialize_bitmap")]
    bitmap: Bitmap,
}

#[derive(Debug, serde::Serialize)]
struct ExportSheetGrid {
    image_path: PathBuf,
    image_width: u32,
    image_height: u32,
    cells: ndarray::Array2<ExportGridCell>,
}

fn run_images_providers(args: &cli::ExportImagesArgs) -> Result<(), CommandError> {
    let checker = version_checker();
    let (mut jar, launcher) = load_jar_source(&args.jar_args.source(), &args.generic_args.cache)?;
    let version = load_jar_version(args.jar_args.version_data.as_deref(), &mut jar, &checker)?;
    let info = build_jar_info(
        &launcher.asset_index,
        jar,
        version,
        &args.generic_args.cache,
    )?;
    let mut storage_vec: Vec<Box<dyn storage::Storage>> =
        vec![Box::new(info.jar_store), Box::new(info.asset_store)];
    for pack in &args.jar_args.packs {
        let pack_storage = get_storage(pack)?;
        storage_vec.push(pack_storage);
    }
    let mut stack = storage::StackStorage(storage_vec);
    let Some(providers) = versions::get_partial_providers(
        &info.version,
        &mut stack,
        args.identifier.clone(),
        &args.font_args,
    )?
    else {
        return Err(CommandError::NoProviders);
    };
    let mut result = vec![];
    for provider in providers.providers {
        if let providers::PartialProvider::Image(grid) = provider {
            if grid.is_unifont && !args.include_unifont {
                continue;
            }
            let image = storage::read_image(&mut stack, &grid.image_path)?;
            let rows = grid.chars.nrows();
            let columns = grid.chars.ncols();
            let mut vec = vec![];
            for ((row, column), cell) in grid.chars.indexed_iter() {
                let char = match cell {
                    providers::GridCell::FullSizeChar(char) => Some(char),
                    providers::GridCell::SizedChar(char, _) => Some(char),
                    providers::GridCell::SkippedChar(char) => Some(char),
                    providers::GridCell::Empty => None,
                }
                .copied();
                let (_, bitmap) = providers::cell_bitmap(
                    &image.data,
                    rows as u32,
                    columns as u32,
                    row,
                    column,
                    None,
                );
                let cell = ExportGridCell { char, bitmap };
                vec.push(cell);
            }
            let cells =
                ndarray::Array2::from_shape_vec((rows, columns), vec).expect("Matching shape");
            let entry = ExportSheetGrid {
                image_path: grid.image_path,
                image_width: image.data.width(),
                image_height: image.data.height(),
                cells,
            };
            result.push(entry);
        }
    }
    let text = serde_json::to_string(&result)?;
    std::fs::write(&args.output, text)?;
    Ok(())
}

fn vanilla_history_entry(
    providers: Vec<providers::Provider>,
    history: &mut HistoryTrack,
    identifier: &providers::Identifier,
    version: &str,
    version_name: &str,
) {
    let summary = ProviderGlyphSummary::new(providers);
    let history_entry = history.entry(identifier.clone()).or_default();
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

fn run_generate(args: &cli::GenerateArgs) -> Result<(), CommandError> {
    let checker = version_checker();
    let (mut jar, launcher) = load_jar_source(&args.jar_args.source(), &args.generic_args.cache)?;
    println!(
        "Generating font {} from {} resource packs on Minecraft {}",
        args.identifier,
        args.jar_args.packs.len(),
        launcher.id
    );
    let version = load_jar_version(args.jar_args.version_data.as_deref(), &mut jar, &checker)?;
    let positions = version.positions();
    println!("Font support level {}", version.name);
    let info = build_jar_info(
        &launcher.asset_index,
        jar,
        version,
        &args.generic_args.cache,
    )?;
    let mut storage_vec: Vec<Box<dyn storage::Storage>> =
        vec![Box::new(info.jar_store), Box::new(info.asset_store)];
    for pack in &args.jar_args.packs {
        let pack_storage = get_storage(pack)?;
        storage_vec.push(pack_storage);
    }
    let mut stack = storage::StackStorage(storage_vec);
    let missing_glyph = info.version.supports_missing_glyph().then(missing_glyph);
    let Some(providers) = versions::get_partial_providers(
        &info.version,
        &mut stack,
        args.identifier.clone(),
        &args.font_args,
    )?
    else {
        return Err(CommandError::NoProviders);
    };
    let providers = providers.resolve(&mut stack, &args.font_args)?;
    let name = get_name(&args.meta_args, &args.identifier);
    let created = get_created(&args.meta_args, &args.identifier);
    for style in &args.styles {
        let style_info = style.info();
        let ttf_name = format!(
            "{}-{}.ttf",
            name.replace(' ', ""),
            style_info.name.replace(' ', "")
        );
        let out_file = args.output.join(ttf_name);
        generate_font(
            &style_info,
            &providers,
            &positions,
            missing_glyph.as_ref(),
            name.clone(),
            created,
            args.meta_args.font_modified,
            &out_file,
        )?;
    }
    Ok(())
}

fn run_list(args: &cli::ListArgs) -> Result<(), CommandError> {
    let checker = version_checker();
    let (mut jar, launcher) = load_jar_source(&args.jar_args.source(), &args.generic_args.cache)?;
    println!(
        "Available font identifiers across {} resource packs on Minecraft {}:",
        args.jar_args.packs.len(),
        launcher.id
    );
    let version = load_jar_version(args.jar_args.version_data.as_deref(), &mut jar, &checker)?;
    let info = build_jar_info(
        &launcher.asset_index,
        jar,
        version,
        &args.generic_args.cache,
    )?;
    let mut storage_vec: Vec<Box<dyn storage::Storage>> =
        vec![Box::new(info.jar_store), Box::new(info.asset_store)];
    for pack in &args.jar_args.packs {
        let pack_storage = get_storage(pack)?;
        storage_vec.push(pack_storage);
    }
    let mut stack = storage::StackStorage(storage_vec);
    let identifiers = info.version.available_identifiers(&mut stack)?;
    for identifier in identifiers {
        println!("{}", identifier);
    }
    Ok(())
}
