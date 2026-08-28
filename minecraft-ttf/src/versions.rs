use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    path::PathBuf,
};

use crate::{providers, storage};

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
pub enum VanillaFontId {
    Default,
    Alt,
    Illageralt,
    Uniform,
}

impl VanillaFontId {
    pub fn identifier(self) -> providers::Identifier {
        match self {
            VanillaFontId::Default => {
                providers::Identifier::new(String::from("minecraft"), PathBuf::from("default"))
            }
            VanillaFontId::Alt => {
                providers::Identifier::new(String::from("minecraft"), PathBuf::from("alt"))
            }
            VanillaFontId::Illageralt => {
                providers::Identifier::new(String::from("minecraft"), PathBuf::from("illageralt"))
            }
            VanillaFontId::Uniform => {
                providers::Identifier::new(String::from("minecraft"), PathBuf::from("uniform"))
            }
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
    pub providers: ProviderSupport,
    pub hardcoded_spaces: Option<HashMap<char, f32>>,
    pub uneven_unifont: bool,
    pub asset_mount: PathBuf,
}

#[derive(Debug)]
pub enum ForceUniformBehavior {
    Filter,
    SkipBitmaps,
    SwitchIdentifier,
}

#[derive(Debug)]
pub struct LegacyUnifont {
    template: String,
    sizes: PathBuf,
}

#[derive(Debug)]
pub enum ProviderSupport {
    Supported {
        uniform: ForceUniformBehavior,
    },
    HardcodedChars {
        chars: ndarray::Array2<Option<char>>,
        textures: HashMap<VanillaFontId, Option<PathBuf>>,
        hardcoded_sizes: Option<HashMap<char, f32>>,
        unifont: Option<LegacyUnifont>,
    },
    FileChars {
        path: PathBuf,
        textures: HashMap<VanillaFontId, Option<PathBuf>>,
        unifont: Option<LegacyUnifont>,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum VersionError {
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
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

pub fn detect_version(
    jar: &mut zip::ZipArchive<impl std::io::Read + std::io::Seek>,
) -> Result<MinecraftVersion, VersionError> {
    let version_json: VersionJson = match jar.by_name("version.json") {
        Ok(file) => serde_json::from_reader(file)?,
        Err(zip::result::ZipError::FileNotFound) => VersionJson::default(),
        Err(e) => return Err(VersionError::Zip(e)),
    };
    if version_json
        .pack_version
        .as_ref()
        .is_some_and(|x| x.as_int() >= 89)
    {
        return Ok(MinecraftVersion {
            name: String::from("26.3-snapshot-1+"),
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported {
                uniform: ForceUniformBehavior::Filter,
            },
            uneven_unifont: true,
            hardcoded_spaces: None,
        });
    }
    if version_json
        .pack_version
        .as_ref()
        .is_some_and(|x| x.as_int() >= 88)
        && version_json.world_version.is_some_and(|x| x >= 4896)
    {
        return Ok(MinecraftVersion {
            name: String::from("26.2-pre-3+"),
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported {
                uniform: ForceUniformBehavior::Filter,
            },
            uneven_unifont: false,
            hardcoded_spaces: None,
        });
    }
    if version_json
        .pack_version
        .as_ref()
        .is_some_and(|x| x.as_int() >= 26)
    {
        return Ok(MinecraftVersion {
            name: String::from("24w06a+"),
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported {
                uniform: ForceUniformBehavior::Filter,
            },
            uneven_unifont: true,
            hardcoded_spaces: None,
        });
    }
    if version_json
        .pack_version
        .as_ref()
        .is_some_and(|x| x.as_int() >= 9)
    {
        return Ok(MinecraftVersion {
            name: String::from("22w11a+"),
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported {
                uniform: ForceUniformBehavior::SwitchIdentifier,
            },
            uneven_unifont: true,
            hardcoded_spaces: None,
        });
    }
    if version_json
        .pack_version
        .as_ref()
        .is_some_and(|x| x.as_int() >= 8)
        && version_json.world_version.is_some_and(|x| x >= 2966)
    {
        return Ok(MinecraftVersion {
            name: String::from("22w03a+"),
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported {
                uniform: ForceUniformBehavior::SwitchIdentifier,
            },
            uneven_unifont: true,
            hardcoded_spaces: Some(HashMap::from([(' ', 4.0), ('\u{200c}', 0.0)])),
        });
    }
    if version_json
        .pack_version
        .as_ref()
        .is_some_and(|x| x.as_int() >= 5)
        && version_json.world_version.is_some_and(|x| x >= 2529)
    {
        return Ok(MinecraftVersion {
            name: String::from("20w17a+"),
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported {
                uniform: ForceUniformBehavior::SwitchIdentifier,
            },
            uneven_unifont: true,
            hardcoded_spaces: Some(HashMap::from([(' ', 4.0)])),
        });
    }
    let names = jar.file_names().collect::<HashSet<_>>();
    if names.contains("assets/minecraft/font/default.json") {
        return Ok(MinecraftVersion {
            name: String::from("1.13-pre7+"),
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported {
                uniform: ForceUniformBehavior::SkipBitmaps,
            },
            uneven_unifont: true,
            hardcoded_spaces: Some(HashMap::from([(' ', 4.0)])),
        });
    }
    Err(VersionError::UnknownVersion)
}

pub fn get_providers(
    version: &MinecraftVersion,
    store: &mut impl storage::Storage,
    identifier: &providers::Identifier,
) -> Vec<providers::Provider> {
    vec![]
}
