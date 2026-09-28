use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ffi::OsStr,
    io::Read,
    path::{Path, PathBuf},
};

use bitmap_ttf::{
    bitmap::Bitmap,
    font::{FontPositions, ScriptPositions},
};
use tracing::debug;

use crate::{
    providers::{self, split_grid},
    storage,
};

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct MinecraftVersion {
    pub name: String,
    providers: ProviderSupport,
    hardcoded_spaces: BTreeMap<char, f32>,
    auto_advance: providers::AutoAdvance,
    aligned_strikethrough: bool,
    pub asset_mount: PathBuf,
}

impl Default for MinecraftVersion {
    fn default() -> Self {
        Self {
            name: Default::default(),
            providers: ProviderSupport::Hardcoded(Box::new(HardcodedFont {
                chars: CharSource::Hardcoded(ndarray::Array2::default((0, 0))),
                textures: Default::default(),
                hardcoded_advances: Default::default(),
                unifont: None,
            })),
            hardcoded_spaces: Default::default(),
            auto_advance: providers::AutoAdvance::LeftmostEmpty,
            asset_mount: Default::default(),
            aligned_strikethrough: true,
        }
    }
}

impl MinecraftVersion {
    pub fn supports_missing_glyph(&self) -> bool {
        matches!(self.providers, ProviderSupport::Supported(_))
    }

    pub fn available_identifiers(
        &self,
        storage: &mut impl storage::Storage,
    ) -> Result<Vec<providers::Identifier>, storage::StorageError> {
        match &self.providers {
            ProviderSupport::Hardcoded(hardcoded) => {
                let mut results = hardcoded.textures.keys().cloned().collect::<Vec<_>>();
                results.sort();
                Ok(results)
            }
            ProviderSupport::Supported(_) => {
                let mut results = vec![];
                let assets = storage.get_entries(Path::new("assets"))?;
                for asset in assets {
                    if let Some(name) = asset.file_stem()
                        && let Some(name) = name.to_str()
                        && asset.extension().is_some_and(|x| x == OsStr::new("json"))
                    {
                        let parts = asset.iter().filter_map(|x| x.to_str()).collect::<Vec<_>>();
                        if let [_, namespace, "font", body @ .., _end] = parts.as_slice() {
                            let mut body = PathBuf::from_iter(body);
                            body.push(name);
                            let identifier =
                                providers::Identifier::new(String::from(*namespace), body);
                            results.push(identifier);
                        }
                    }
                }
                results.sort();
                Ok(results)
            }
        }
    }

    pub fn positions(&self) -> FontPositions {
        FontPositions {
            ascent: 9.0 / 12.0,
            descent: 2.0 / 12.0,
            s_cap_height: 7.0 / 12.0,
            sx_height: 5.0 / 12.0,
            y_strikeout_position: if self.aligned_strikethrough {
                4.0 / 12.0
            } else {
                3.5 / 12.0
            },
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
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct LegacyUnifont {
    template: String,
    sizes: PathBuf,
    uneven_spacing: bool,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CharSource {
    FromFile(PathBuf),
    Hardcoded(ndarray::Array2<Option<char>>),
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ProviderSupport {
    Supported(providers::ProviderBehavior),
    Hardcoded(Box<HardcodedFont>),
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct HardcodedFont {
    chars: CharSource,
    textures: HashMap<providers::Identifier, Option<PathBuf>>,
    hardcoded_advances: BTreeMap<char, f32>,
    unifont: Option<LegacyUnifont>,
}

#[derive(thiserror::Error, Debug)]
#[error("Determining version")]
pub enum VersionError {
    Zip(#[from] zip::result::ZipError),
    Serde(#[from] json5::Error),
    Io(#[from] std::io::Error),
    #[error("Unknown version")]
    UnknownVersion,
}

#[derive(serde::Deserialize, Debug)]
#[serde(untagged)]
pub enum PackVersion {
    Int(u32),
    Split { resource: u32 },
    Modern { resource_major: u32 },
}

impl PackVersion {
    fn as_int(&self) -> u32 {
        match self {
            Self::Int(val) => *val,
            Self::Split { resource } => *resource,
            Self::Modern { resource_major } => *resource_major,
        }
    }
}

#[derive(serde::Deserialize, Debug, Default)]
pub struct VersionJson {
    pack_version: Option<PackVersion>,
    world_version: Option<u32>,
}

#[derive(serde::Deserialize, Debug)]
pub struct VersionUpdate {
    check: VersionCheck,
    name: String,
    changes: Vec<VersionChange>,
}

impl VersionUpdate {
    fn apply_to(&self, version: &mut MinecraftVersion) {
        version.name = self.name.clone();
        for change in self.changes.iter() {
            change.apply_to(version);
        }
    }
}

#[derive(serde::Deserialize, Debug)]
struct VersionCheck {
    file: Option<String>,
    no_file: Option<String>,
    contains: Option<HashMap<String, String>>,
    pack_version: Option<u32>,
    world_version: Option<u32>,
}

impl VersionCheck {
    fn passes(
        &self,
        jar: &mut zip::ZipArchive<impl std::io::Read + std::io::Seek>,
        version_json: &VersionJson,
    ) -> Result<bool, zip::result::ZipError> {
        if let Some(file) = self.file.as_ref()
            && let Err(zip::result::ZipError::FileNotFound) = jar.by_name(file)
        {
            return Ok(false);
        }
        if let Some(file) = self.no_file.as_ref() {
            let Err(zip::result::ZipError::FileNotFound) = jar.by_name(file) else {
                return Ok(false);
            };
        }
        if let Some(pack_version) = self.pack_version
            && !version_json
                .pack_version
                .as_ref()
                .is_some_and(|x| x.as_int() >= pack_version)
        {
            return Ok(false);
        }
        if let Some(world_version) = self.world_version
            && !version_json
                .world_version
                .is_some_and(|x| x >= world_version)
        {
            return Ok(false);
        }
        if let Some(contains) = self.contains.as_ref() {
            for (file, data) in contains {
                match jar.by_name(file) {
                    Ok(mut content) => {
                        let mut string = String::new();
                        content.read_to_string(&mut string)?;
                        if !string.contains(data) {
                            return Ok(false);
                        }
                    }
                    Err(zip::result::ZipError::FileNotFound) => {
                        return Ok(false);
                    }
                    Err(err) => {
                        return Err(err);
                    }
                }
            }
        }
        Ok(true)
    }
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
enum VersionChange {
    AddTextures(HashMap<providers::Identifier, Option<PathBuf>>),
    HardcodedSpaces(BTreeMap<char, f32>),
    SetChars(SetCharChange),
    SpecialAdvances(Vec<SpecialAdvance>),
    AutoAdvance(providers::AutoAdvance),
    Unifont(LegacyUnifont),
    AssetMount(PathBuf),
    Providers(providers::ProviderBehavior),
    AlignedStrikethrough(bool),
}

#[derive(serde::Deserialize, Debug)]
struct SpecialAdvance {
    advance: f32,
    chars: String,
}

impl VersionChange {
    fn get_legacy(version: &mut MinecraftVersion) -> &mut HardcodedFont {
        match &mut version.providers {
            ProviderSupport::Supported(_) => {
                panic!("Versions configured incorrectly")
            }
            ProviderSupport::Hardcoded(hardcoded) => hardcoded.as_mut(),
        }
    }

    fn apply_to(&self, version: &mut MinecraftVersion) {
        match self {
            Self::AddTextures(adding) => {
                let textures = &mut Self::get_legacy(version).textures;
                for (id, path) in adding {
                    textures.insert(id.clone(), path.clone());
                }
            }
            Self::HardcodedSpaces(spaces) => {
                version.hardcoded_spaces = spaces.clone();
            }
            Self::SetChars(char_change) => {
                let legacy = &mut Self::get_legacy(version);
                match char_change {
                    SetCharChange::File(path) => {
                        legacy.chars = CharSource::FromFile(path.clone());
                    }
                    SetCharChange::Hardcoded(items) => {
                        let mut builder = HardcodedCharsBuilder::new(16, 16);
                        for item in items {
                            match item {
                                CharBuilderEntry::String(str) => {
                                    builder = builder.add_chars(str);
                                }
                                CharBuilderEntry::Blanks(count) => {
                                    builder = builder.add_blanks(*count);
                                }
                            }
                        }
                        legacy.chars = CharSource::Hardcoded(builder.build());
                    }
                }
            }
            Self::SpecialAdvances(data) => {
                let legacy = &mut Self::get_legacy(version);
                let mut map = BTreeMap::new();
                for entry in data {
                    for char in entry.chars.chars() {
                        map.insert(char, entry.advance);
                    }
                }
                legacy.hardcoded_advances = map;
            }
            Self::AutoAdvance(advance) => {
                version.auto_advance = *advance;
            }
            Self::Unifont(unifont) => {
                let legacy = &mut Self::get_legacy(version);
                legacy.unifont = Some(unifont.clone());
            }
            Self::AssetMount(path) => {
                version.asset_mount = path.clone();
            }
            Self::Providers(behavior) => {
                version.providers = ProviderSupport::Supported(behavior.clone());
            }
            Self::AlignedStrikethrough(aligned) => {
                version.aligned_strikethrough = *aligned;
            }
        }
    }
}

#[derive(serde::Deserialize, Debug)]
#[serde(untagged)]
enum SetCharChange {
    File(PathBuf),
    Hardcoded(Vec<CharBuilderEntry>),
}

#[derive(serde::Deserialize, Debug)]
#[serde(untagged)]
enum CharBuilderEntry {
    String(String),
    Blanks(usize),
}

pub fn detect_version(
    updates: &[VersionUpdate],
    jar: &mut zip::ZipArchive<impl std::io::Read + std::io::Seek>,
) -> Result<MinecraftVersion, VersionError> {
    let version_json: VersionJson = match jar.by_name("version.json") {
        Ok(mut file) => {
            let mut text = String::new();
            file.read_to_string(&mut text)?;
            json5::from_str(&text)?
        }
        Err(zip::result::ZipError::FileNotFound) => VersionJson::default(),
        Err(e) => return Err(VersionError::Zip(e)),
    };
    let mut index = None;
    for i in (0..updates.len()).rev() {
        if updates[i].check.passes(jar, &version_json)? {
            index = Some(i);
            break;
        }
    }
    let Some(index) = index else {
        return Err(VersionError::UnknownVersion);
    };
    let mut version = MinecraftVersion::default();
    for update in updates.iter().take(index + 1) {
        update.apply_to(&mut version);
    }
    Ok(version)
}

#[derive(Debug, Default)]
struct HardcodedCharsBuilder {
    width: usize,
    height: usize,
    chars: Vec<Option<char>>,
}

impl HardcodedCharsBuilder {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            chars: vec![],
        }
    }

    fn add_chars(mut self, string: &str) -> Self {
        self.chars.extend(string.chars().map(Some));
        self
    }

    fn add_blanks(mut self, num: usize) -> Self {
        for _ in 0..num {
            self.chars.push(None)
        }
        self
    }

    fn build(self) -> ndarray::Array2<Option<char>> {
        ndarray::Array2::from_shape_vec((self.width, self.height), self.chars)
            .expect("Versions should be configured correctly")
    }
}

pub fn get_providers(
    version: &MinecraftVersion,
    store: &mut storage::StackStorage,
    identifier: providers::Identifier,
    options: &impl providers::ProviderOptions,
) -> Result<Option<providers::Providers>, providers::ProvidersError> {
    let mut providers = vec![];
    let mut times = providers::ModifiedTimes::default();
    if !version.hardcoded_spaces.is_empty() {
        let provider = providers::SpaceProvider {
            chars: version.hardcoded_spaces.clone(),
        };
        providers.push(providers::Provider::Space(provider));
    }
    match &version.providers {
        ProviderSupport::Supported(behavior) => {
            let Some(real) = providers::load_providers(
                identifier,
                store,
                version.auto_advance,
                options,
                behavior,
                &mut HashSet::new(),
            )?
            else {
                return Ok(None);
            };
            providers.extend(real.providers);
            times.merge(real.times);
        }
        ProviderSupport::Hardcoded(data) => {
            let HardcodedFont {
                chars,
                textures,
                hardcoded_advances,
                unifont,
            } = Box::as_ref(data);
            let Some(texture) = textures.get(&identifier) else {
                return Ok(None);
            };
            if let Some(texture) = texture {
                let (provider, subtimes) = legacy_bitmap(
                    store,
                    texture,
                    chars,
                    hardcoded_advances,
                    version.auto_advance,
                    options,
                )?;
                times.merge(subtimes);
                providers.push(providers::Provider::Image(provider));
            }
            if let Some(LegacyUnifont {
                template,
                sizes,
                uneven_spacing,
            }) = unifont
            {
                let sheets = providers::unicode_sheets(template, PathBuf::from);
                let (many, subtimes) = providers::legacy_unicode(
                    store,
                    options,
                    *uneven_spacing,
                    sheets.iter().map(|(a, b)| (a, b)),
                    sizes,
                )?;
                times.merge(subtimes);
                providers.extend(many.into_iter().map(providers::Provider::Image));
            }
        }
    }
    Ok(Some(providers::Providers { providers, times }))
}

fn legacy_bitmap(
    store: &mut impl storage::Storage,
    texture: &Path,
    chars: &CharSource,
    advances: &BTreeMap<char, f32>,
    auto_advances: providers::AutoAdvance,
    options: &impl providers::ProviderOptions,
) -> Result<(providers::ImageProvider, providers::ModifiedTimes), providers::ProvidersError> {
    let mut times = providers::ModifiedTimes::default();
    let image = storage::read_image(store, texture)?;
    times.update(image.modified_time);
    let char_images = match chars {
        CharSource::FromFile(path) => {
            let lines = storage::read_font_txt(store, path)?;
            times.update(lines.modified_time);
            let content = split_grid(lines.data.iter(), false)?;
            let mut array = ndarray::Array2::from_elem((2, content.ncols()), None);
            array
                .append(ndarray::Axis(0), content.view())
                .expect("Shape matches");
            let blank_rows = ndarray::Array2::from_elem((5, array.ncols()), None);
            array
                .append(ndarray::Axis(0), blank_rows.view())
                .expect("Shape matches");
            providers::image_grid(&image.data, &array, None, |x| options.include_char(x))
        }
        CharSource::Hardcoded(array) => {
            providers::image_grid(&image.data, array, None, |x| options.include_char(x))
        }
    };
    let height = 8i32;
    let char_advance = |char: char, bitmap: &Bitmap| -> f32 {
        match advances.get(&char) {
            None => auto_advances.get(bitmap, height),
            Some(advance) => *advance,
        }
    };
    let has_color = options.has_color(&image.data);
    let chars = char_images
        .into_iter()
        .map(|(char, (content_box, bitmap))| {
            let advance = char_advance(char, &bitmap);
            debug!(
                "{:?} ({:04X}):\n{}\nadvance: {}",
                char,
                Into::<u32>::into(char),
                bitmap,
                advance
            );
            let glyph = providers::CharImage {
                content_box,
                bitmap,
                advance,
                bold_offset: 1.0,
            };
            (char, glyph)
        })
        .collect();
    let provider = providers::ImageProvider {
        image: image.data,
        image_path: texture.to_owned(),
        has_color,
        height,
        ascent: 7,
        chars,
    };
    Ok((provider, times))
}
