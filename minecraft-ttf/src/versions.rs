use std::{collections::HashMap, fmt::Display, path::PathBuf};

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
pub enum VanillaFontId {
    Default,
    Alt,
    Illageralt,
    Uniform,
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

pub struct MinecraftVersion {
    name: String,
    providers: ProviderSupport,
    hardcoded_spaces: Option<HashMap<char, f32>>,
    uneven_unifont: bool,
    asset_mount: PathBuf,
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
