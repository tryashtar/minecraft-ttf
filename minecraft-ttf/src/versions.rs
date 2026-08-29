use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    path::{Path, PathBuf},
};

use bitmap_ttf::bitmap::Bitmap;

use crate::{
    providers::{self, normal_advance, split_grid},
    storage,
};

#[derive(clap::ValueEnum, Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum VanillaFontId {
    Default,
    Alt,
    Illageralt,
    Uniform,
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
    pub name: &'static str,
    pub providers: ProviderSupport,
    pub hardcoded_spaces: Option<HashMap<char, f32>>,
    pub asset_mount: PathBuf,
}

#[derive(Debug)]
pub struct LegacyUnifont {
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
pub enum ProviderSupport {
    Supported(providers::ProviderBehavior),
    Hardcoded {
        chars: CharSource,
        textures: HashMap<VanillaFontId, Option<PathBuf>>,
        hardcoded_advances: Option<HashMap<char, f32>>,
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
            name: "26.3-snapshot-1+",
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported(providers::ProviderBehavior {
                uniform: providers::ForceUniformBehavior::Filter,
                uneven_unifont: true,
            }),
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
            name: "26.2-pre-3+",
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported(providers::ProviderBehavior {
                uniform: providers::ForceUniformBehavior::Filter,
                uneven_unifont: false,
            }),
            hardcoded_spaces: None,
        });
    }
    if version_json
        .pack_version
        .as_ref()
        .is_some_and(|x| x.as_int() >= 26)
    {
        return Ok(MinecraftVersion {
            name: "24w06a+",
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported(providers::ProviderBehavior {
                uniform: providers::ForceUniformBehavior::Filter,
                uneven_unifont: true,
            }),
            hardcoded_spaces: None,
        });
    }
    if version_json
        .pack_version
        .as_ref()
        .is_some_and(|x| x.as_int() >= 9)
    {
        return Ok(MinecraftVersion {
            name: "22w11a+",
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported(providers::ProviderBehavior {
                uniform: providers::ForceUniformBehavior::SwitchIdentifier,
                uneven_unifont: true,
            }),
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
            name: "22w03a+",
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported(providers::ProviderBehavior {
                uniform: providers::ForceUniformBehavior::SwitchIdentifier,
                uneven_unifont: true,
            }),
            hardcoded_spaces: Some(HashMap::from([(' ', 4.0), ('\u{200c}', 0.0)])),
        });
    }
    let simple_spaces = Some(HashMap::from([(' ', 4.0)]));
    if version_json
        .pack_version
        .as_ref()
        .is_some_and(|x| x.as_int() >= 5)
        && version_json.world_version.is_some_and(|x| x >= 2529)
    {
        return Ok(MinecraftVersion {
            name: "20w17a+",
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported(providers::ProviderBehavior {
                uniform: providers::ForceUniformBehavior::SwitchIdentifier,
                uneven_unifont: true,
            }),
            hardcoded_spaces: simple_spaces,
        });
    }
    let names = jar.file_names().collect::<HashSet<_>>();
    if names.contains("assets/minecraft/font/default.json") {
        return Ok(MinecraftVersion {
            name: "1.13-pre7+",
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Supported(providers::ProviderBehavior {
                uniform: providers::ForceUniformBehavior::SkipBitmaps,
                uneven_unifont: true,
            }),
            hardcoded_spaces: simple_spaces,
        });
    }
    let asset_map = HashMap::from([
        (
            VanillaFontId::Default,
            Some(PathBuf::from("assets/minecraft/textures/font/ascii.png")),
        ),
        (
            VanillaFontId::Alt,
            Some(PathBuf::from(
                "assets/minecraft/textures/font/ascii_sga.png",
            )),
        ),
        (VanillaFontId::Uniform, None),
    ]);
    let new_unifont = Some(LegacyUnifont {
        template: String::from("assets/minecraft/textures/font/unicode_page_%x.png"),
        sizes: PathBuf::from("assets/minecraft/font/glyph_sizes.bin"),
        uneven_spacing: true,
    });
    if !names.contains("font.txt") && names.contains("assets/minecraft/textures/font/ascii.png") {
        let chars = HardcodedCharsBuilder::new(16, 16)
            .add_chars("ÀÁÂÈÊËÍÓÔÕÚßãõğİıŒœŞşŴŵžȇ")
            .add_blanks(7)
            .add_chars(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
            .add_blanks(1)
            .add_chars("ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜø£Ø×ƒáíóúñÑªº¿®¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀αβΓπΣσμτΦΘΩδ∞∅∈∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■")
            .add_blanks(1)
            .build();
        return Ok(MinecraftVersion {
            name: "13w42b+",
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::Hardcoded(chars),
                textures: asset_map,
                hardcoded_advances: None,
                unifont: new_unifont,
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    if names.contains("assets/minecraft/textures/font/ascii.png") {
        return Ok(MinecraftVersion {
            name: "13w24a+",
            asset_mount: PathBuf::from("assets"),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::FromFile(PathBuf::from("font.txt")),
                textures: asset_map,
                hardcoded_advances: None,
                unifont: new_unifont,
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    if names.contains("font/glyph_sizes.bin") {
        return Ok(MinecraftVersion {
            name: "11w49a+",
            asset_mount: PathBuf::default(),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::FromFile(PathBuf::from("font.txt")),
                textures: HashMap::from([
                    (
                        VanillaFontId::Default,
                        Some(PathBuf::from("font/default.png")),
                    ),
                    (
                        VanillaFontId::Alt,
                        Some(PathBuf::from("font/alternate.png")),
                    ),
                    (VanillaFontId::Uniform, None),
                ]),
                hardcoded_advances: None,
                unifont: Some(LegacyUnifont {
                    template: String::from("font/glyph_%X.png"),
                    sizes: PathBuf::from("font/glyph_sizes.bin"),
                    uneven_spacing: true,
                }),
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    if names.contains("font/alternate.png") {
        return Ok(MinecraftVersion {
            name: "b1.9-pre3+",
            asset_mount: PathBuf::default(),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::FromFile(PathBuf::from("font.txt")),
                textures: HashMap::from([
                    (
                        VanillaFontId::Default,
                        Some(PathBuf::from("font/default.png")),
                    ),
                    (
                        VanillaFontId::Alt,
                        Some(PathBuf::from("font/alternate.png")),
                    ),
                ]),
                hardcoded_advances: None,
                unifont: None,
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    let simple_map = HashMap::from([(
        VanillaFontId::Default,
        Some(PathBuf::from("font/default.png")),
    )]);
    if names.contains("font.txt") {
        return Ok(MinecraftVersion {
            name: "b1.1+",
            asset_mount: PathBuf::default(),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::FromFile(PathBuf::from("font.txt")),
                textures: simple_map,
                hardcoded_advances: None,
                unifont: None,
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    if names.contains("lang/en_US.lang") {
        let chars = HardcodedCharsBuilder::new(16, 16)
            .add_blanks(32)
            .add_chars(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_'abcdefghijklmnopqrstuvwxyz{|}~â")
            .add_chars("Œ‚Ã‡Ã¼Ã©Ã¢Ã¤Ã\u{00a0}Ã¥Ã§ÃªÃ«Ã¨Ã¯Ã®Ã¬Ã„Ã…Ã‰Ã¦Ã†Ã´Ã¶Ã²Ã»Ã¹Ã¿Ã–ÃœÃ¸Â£Ã˜Ã—Æ’Ã¡Ã\u{00ad}Ã³ÃºÃ±Ã‘ÂªÂºÂ¿Â®Â¬Â½Â¼Â¡Â«")
            .add_blanks(30)
            .build();
        return Ok(MinecraftVersion {
            name: "b1.0+",
            asset_mount: PathBuf::default(),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::Hardcoded(chars),
                textures: simple_map,
                hardcoded_advances: None,
                unifont: None,
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    let alpha_chars = HardcodedCharsBuilder::new(16, 16)
        .add_blanks(32)
        .add_chars(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_'abcdefghijklmnopqrstuvwxyz{|}~â")
        .add_chars("ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜø£Ø×ƒáíóúñÑªº¿®¬½¼¡«»")
        .add_blanks(80)
        .build();
    if names.contains("font/default.png") {
        return Ok(MinecraftVersion {
            name: "a1.2.2+",
            asset_mount: PathBuf::default(),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::Hardcoded(alpha_chars),
                textures: simple_map,
                hardcoded_advances: None,
                unifont: None,
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    if names.contains("mob/cow.png") {
        return Ok(MinecraftVersion {
            name: "a1.0.9+",
            asset_mount: PathBuf::default(),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::Hardcoded(alpha_chars),
                textures: HashMap::from([(
                    VanillaFontId::Default,
                    Some(PathBuf::from("default.png")),
                )]),
                hardcoded_advances: None,
                unifont: None,
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    let full_chars = HardcodedCharsBuilder::new(16, 16)
        .add_range('\u{0000}'..='\u{00ff}')
        .build();
    let zero_advances = Some(
        ('\u{0080}'..='\u{00ff}')
            .map(|x| (x, 0.0))
            .collect::<HashMap<_, _>>(),
    );
    if names.contains("default.png") {
        return Ok(MinecraftVersion {
            name: "c0.0.17a+",
            asset_mount: PathBuf::default(),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::Hardcoded(full_chars),
                textures: HashMap::from([(
                    VanillaFontId::Default,
                    Some(PathBuf::from("default.png")),
                )]),
                hardcoded_advances: zero_advances,
                unifont: None,
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    if names.contains("default.gif") {
        return Ok(MinecraftVersion {
            name: "c0.0.2a+",
            asset_mount: PathBuf::default(),
            providers: ProviderSupport::Hardcoded {
                chars: CharSource::Hardcoded(full_chars),
                textures: HashMap::from([(
                    VanillaFontId::Default,
                    Some(PathBuf::from("default.gif")),
                )]),
                hardcoded_advances: zero_advances,
                unifont: None,
            },
            hardcoded_spaces: simple_spaces,
        });
    }
    Err(VersionError::UnknownVersion)
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

    fn add_range(mut self, chars: impl Iterator<Item = char>) -> Self {
        for char in chars {
            self.chars.push(Some(char));
        }
        self
    }

    fn build(self) -> ndarray::Array2<Option<char>> {
        ndarray::Array2::from_shape_vec((self.width, self.height), self.chars).unwrap()
    }
}

pub fn get_providers(
    version: &MinecraftVersion,
    store: &mut storage::StackStorage,
    identifier: &providers::Identifier,
    options: &impl providers::ProviderOptions,
) -> Result<Option<providers::Providers>, providers::ProvidersError> {
    let mut providers = vec![];
    let mut times = providers::ModifiedTimes::default();
    match &version.providers {
        ProviderSupport::Supported(behavior) => {
            let providers = providers::load_providers(identifier, store, options, behavior)?;
        }
        ProviderSupport::Hardcoded {
            chars,
            textures,
            hardcoded_advances,
            unifont,
        } => {
            let Ok(vanilla) = TryInto::<VanillaFontId>::try_into(identifier) else {
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
            if let Some(unifont) = unifont {}
        }
    }
    if let Some(spaces) = &version.hardcoded_spaces {
        let provider = providers::SpaceProvider {
            chars: spaces.clone(),
        };
        providers.push(providers::Provider::Space(provider));
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
            array.append(ndarray::Axis(0), content.view()).unwrap();
            let blank_rows = ndarray::Array2::from_elem((5, array.ncols()), None);
            array.append(ndarray::Axis(0), blank_rows.view()).unwrap();
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
