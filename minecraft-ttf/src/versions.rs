use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Display,
    io::Read,
    path::{Path, PathBuf},
};

use bitmap_ttf::bitmap::Bitmap;

use crate::{
    providers::{self, normal_advance, split_grid},
    storage,
};

#[derive(clap::ValueEnum, serde::Deserialize, Debug, Clone, Copy, Hash, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VanillaFontId {
    Default,
    Alt,
    Illageralt,
    Uniform,
}

impl VanillaFontId {
    pub fn info(self) -> FontIdInfo {
        match self {
            Self::Default => FontIdInfo {
                name: String::from("Minecraft Default"),
                created: jiff::Timestamp::constant(1_242_492_720, 0),
            },
            Self::Alt => FontIdInfo {
                name: String::from("Minecraft Enchanting"),
                created: jiff::Timestamp::constant(1_317_913_020, 0),
            },
            Self::Illageralt => FontIdInfo {
                name: String::from("Minecraft Illager Runes"),
                created: jiff::Timestamp::constant(1_631_721_870, 0),
            },
            Self::Uniform => FontIdInfo {
                name: String::from("Minecraft Unicode"),
                created: jiff::Timestamp::constant(1_323_360_326, 0),
            },
        }
    }
}

#[derive(Debug)]
pub struct FontIdInfo {
    pub name: String,
    pub created: jiff::Timestamp,
}

impl TryFrom<&providers::Identifier> for VanillaFontId {
    type Error = ();

    fn try_from(value: &providers::Identifier) -> Result<Self, Self::Error> {
        let providers::Identifier { namespace, body } = value;
        match (namespace.as_str(), body) {
            ("minecraft", path) if path == Path::new("default") => Ok(Self::Default),
            ("minecraft", path) if path == Path::new("alt") => Ok(Self::Alt),
            ("minecraft", path) if path == Path::new("illageralt") => Ok(Self::Illageralt),
            ("minecraft", path) if path == Path::new("uniform") => Ok(Self::Uniform),
            _ => Err(()),
        }
    }
}

impl From<VanillaFontId> for providers::Identifier {
    fn from(value: VanillaFontId) -> Self {
        match value {
            VanillaFontId::Default => Self::vanilla(PathBuf::from("default")),
            VanillaFontId::Alt => Self::vanilla(PathBuf::from("alt")),
            VanillaFontId::Illageralt => Self::vanilla(PathBuf::from("illageralt")),
            VanillaFontId::Uniform => Self::vanilla(PathBuf::from("uniform")),
        }
    }
}

impl Display for VanillaFontId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "default"),
            Self::Alt => write!(f, "alt"),
            Self::Illageralt => write!(f, "illageralt"),
            Self::Uniform => write!(f, "uniform"),
        }
    }
}

#[derive(Debug)]
pub struct MinecraftVersion {
    pub name: String,
    providers: ProviderSupport,
    hardcoded_spaces: Option<BTreeMap<char, f32>>,
    pub asset_mount: PathBuf,
}

impl Default for MinecraftVersion {
    fn default() -> Self {
        Self {
            name: Default::default(),
            providers: ProviderSupport::Hardcoded(Box::new(HardcodedFont {
                chars: CharSource::Hardcoded(ndarray::Array2::default((0, 0))),
                textures: Default::default(),
                hardcoded_advances: None,
                unifont: None,
            })),
            hardcoded_spaces: Default::default(),
            asset_mount: Default::default(),
        }
    }
}

impl MinecraftVersion {
    pub fn supports_missing_glyph(&self) -> bool {
        matches!(self.providers, ProviderSupport::Supported(_))
    }
}

#[derive(serde::Deserialize, Debug, Clone)]
struct LegacyUnifont {
    template: String,
    sizes: PathBuf,
    uneven_spacing: bool,
}

#[derive(Debug)]
pub enum CharSource {
    FromFile(PathBuf),
    Hardcoded(ndarray::Array2<Option<char>>),
}

#[derive(Debug)]
enum ProviderSupport {
    Supported(providers::ProviderBehavior),
    Hardcoded(Box<HardcodedFont>),
}

#[derive(Debug)]
struct HardcodedFont {
    chars: CharSource,
    textures: HashMap<VanillaFontId, Option<PathBuf>>,
    hardcoded_advances: Option<HashMap<char, f32>>,
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
    pack_version: Option<u32>,
    world_version: Option<u32>,
}

impl VersionCheck {
    fn passes(&self, names: &HashSet<&str>, version_json: &VersionJson) -> bool {
        if let Some(file) = self.file.as_ref()
            && !names.contains(file.as_str())
        {
            return false;
        }
        if let Some(file) = self.no_file.as_ref()
            && names.contains(file.as_str())
        {
            return false;
        }
        if let Some(pack_version) = self.pack_version
            && !version_json
                .pack_version
                .as_ref()
                .is_some_and(|x| x.as_int() == pack_version)
        {
            return false;
        }
        if let Some(world_version) = self.world_version
            && !version_json
                .world_version
                .is_some_and(|x| x == world_version)
        {
            return false;
        }
        true
    }
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
enum VersionChange {
    AddTextures(HashMap<VanillaFontId, Option<PathBuf>>),
    HardcodedSpaces(Option<BTreeMap<char, f32>>),
    SetChars(SetCharChange),
    ZeroAdvances(Option<String>),
    Unifont(LegacyUnifont),
    AssetMount(PathBuf),
    Providers(providers::ProviderBehavior),
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
                    textures.insert(*id, path.clone());
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
            Self::ZeroAdvances(data) => {
                let legacy = &mut Self::get_legacy(version);
                legacy.hardcoded_advances = data
                    .as_ref()
                    .map(|string| string.chars().map(|x| (x, 0.0)).collect());
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
    let names = jar.file_names().collect::<HashSet<_>>();
    let mut index = None;
    for i in (0..updates.len()).rev() {
        if updates[i].check.passes(&names, &version_json) {
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
    if let Some(spaces) = &version.hardcoded_spaces {
        let provider = providers::SpaceProvider {
            chars: spaces.clone(),
        };
        providers.push(providers::Provider::Space(provider));
    }
    match &version.providers {
        ProviderSupport::Supported(behavior) => {
            let Some(real) = providers::load_providers(
                identifier,
                store,
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
            let Ok(vanilla) = TryInto::<VanillaFontId>::try_into(&identifier) else {
                return Ok(None);
            };
            let Some(texture) = textures.get(&vanilla) else {
                return Ok(None);
            };
            if let Some(texture) = texture {
                let (provider, subtimes) =
                    legacy_bitmap(store, texture, chars, hardcoded_advances.as_ref(), options)?;
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
    advances: Option<&HashMap<char, f32>>,
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
        match advances {
            None => normal_advance(bitmap, height),
            Some(sizes) => match sizes.get(&char) {
                Some(size) => *size,
                None => {
                    let mut normal = normal_advance(bitmap, height);
                    if normal == 8.0 {
                        normal += 1.0;
                    }
                    normal
                }
            },
        }
    };
    let has_color = options.has_color(&image.data);
    let chars = char_images
        .into_iter()
        .map(|(char, (content_box, bitmap))| {
            let advance = char_advance(char, &bitmap);
            (
                char,
                providers::CharImage {
                    content_box,
                    bitmap,
                    advance,
                    bold_offset: 1.0,
                },
            )
        })
        .collect();
    let provider = providers::ImageProvider {
        image: image.data,
        has_color,
        height,
        ascent: 7,
        chars,
    };
    Ok((provider, times))
}
