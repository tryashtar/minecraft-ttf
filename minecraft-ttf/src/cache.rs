use std::{
    collections::{HashMap, HashSet},
    ffi::OsStr,
    fmt::Display,
    io::Read,
    path::{Path, PathBuf},
};

use tracing::info;

use crate::storage;

#[derive(Debug, serde::Deserialize)]
pub struct ManifestLatest {
    pub snapshot: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct ManifestVersion {
    pub id: String,
    pub url: url::Url,
}

#[derive(Debug, serde::Deserialize)]
pub struct Manifest {
    pub latest: ManifestLatest,
    pub versions: Vec<ManifestVersion>,
}

impl Manifest {
    pub fn find_version(&self, version: &str) -> Option<&ManifestVersion> {
        self.versions.iter().find(|x| x.id == version)
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct LauncherDownload {
    pub url: url::Url,
}

#[derive(Debug, serde::Deserialize)]
pub struct LauncherDownloads {
    pub client: LauncherDownload,
}

#[derive(Debug, serde::Deserialize)]
pub struct LauncherAssets {
    pub id: String,
    pub url: url::Url,
}

#[derive(Debug, serde::Deserialize)]
pub struct LauncherData {
    pub id: String,
    pub downloads: LauncherDownloads,
    #[serde(rename = "assetIndex")]
    pub asset_index: LauncherAssets,
}

#[derive(Debug, serde::Deserialize)]
pub struct AssetEntry {
    pub hash: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct AssetSource {
    pub objects: HashMap<String, AssetEntry>,
}

#[derive(thiserror::Error, Debug)]
pub enum CacheError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error(transparent)]
    Request(#[from] reqwest::Error),
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),
}

pub fn get_manifest(cache: &Path) -> Result<Manifest, CacheError> {
    let cached_path = cache.join(Path::new("versions/version_manifest_v2.json"));
    let url = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json";
    read_or_download_json(&cached_path, url)
}

pub fn get_launcher(version: &ManifestVersion, cache: &Path) -> Result<LauncherData, CacheError> {
    let mut cached_path = cache.join(Path::new("versions"));
    cached_path.push(&version.id);
    cached_path.push(&version.id);
    let cached_path = push_path_str(cached_path, OsStr::new(".json"));
    read_or_download_json(&cached_path, version.url.clone())
}

pub fn get_asset_source(index: &LauncherAssets, cache: &Path) -> Result<AssetSource, CacheError> {
    let mut cached_path = cache.join(Path::new("assets/indexes"));
    cached_path.push(&index.id);
    let cached_path = push_path_str(cached_path, OsStr::new(".json"));
    read_or_download_json(&cached_path, index.url.clone())
}

pub fn get_jar(
    launcher: &LauncherData,
    cache: &Path,
) -> Result<zip::ZipArchive<impl std::io::Read + std::io::Seek + use<>>, CacheError> {
    let mut cached_path = cache.join(Path::new("versions"));
    cached_path.push(&launcher.id);
    cached_path.push(&launcher.id);
    let cached_path = push_path_str(cached_path, OsStr::new(".jar"));
    let bytes = read_or_download_bytes(&cached_path, launcher.downloads.client.url.clone())?;
    let zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
    Ok(zip)
}

fn push_path_str(path: PathBuf, str: &OsStr) -> PathBuf {
    let mut string = path.into_os_string();
    string.push(str);
    PathBuf::from(string)
}

fn read_or_download_bytes(
    path: &Path,
    url: impl reqwest::IntoUrl + Display,
) -> Result<Vec<u8>, CacheError> {
    let file = std::fs::File::open(path);
    match file {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            info!("Downloading {} to {:?}", url, path.file_name());
            let response = reqwest::blocking::get(url)?.error_for_status()?;
            let bytes = response.bytes()?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, &bytes)?;
            Ok(bytes.to_vec())
        }
        Err(e) => Err(CacheError::Io(e)),
        Ok(mut file) => {
            let mut bytes = vec![];
            file.read_to_end(&mut bytes)?;
            Ok(bytes)
        }
    }
}

fn read_or_download_json<T: serde::de::DeserializeOwned>(
    path: &Path,
    url: impl reqwest::IntoUrl + Display,
) -> Result<T, CacheError> {
    let bytes = read_or_download_bytes(path, url)?;
    let result = serde_json::from_slice(&bytes)?;
    Ok(result)
}

#[derive(Debug)]
pub struct AssetStorage {
    assets: AssetSource,
    prefix: PathBuf,
    cache: PathBuf,
}

impl AssetStorage {
    pub fn new(assets: AssetSource, prefix: PathBuf, cache: PathBuf) -> Self {
        Self {
            assets,
            prefix,
            cache,
        }
    }
}

impl storage::Storage for AssetStorage {
    fn get_entries(&self, prefix: &Path) -> Result<HashSet<PathBuf>, storage::StorageError> {
        let modified = prefix.strip_prefix(&self.prefix)?;
        let mut result = HashSet::new();
        for entry in self.assets.objects.keys() {
            let path = Path::new(entry);
            if path.starts_with(modified) {
                result.insert(path.to_owned());
            }
        }
        Ok(result)
    }

    fn read(&mut self, entry: &Path) -> Result<Vec<u8>, storage::StorageError> {
        let modified = entry.strip_prefix(&self.prefix)?;
        let Some(name) = modified.as_os_str().to_str() else {
            return Err(storage::StorageError::PathConversion);
        };
        let Some(data) = self.assets.objects.get(name) else {
            return Err(storage::StorageError::FileNotFound);
        };
        let hash = &data.hash;
        let mut cached = self.cache.join("assets/objects");
        cached.push(&hash[..2]);
        cached.push(hash);
        let mut url = String::from("https://resources.download.minecraft.net/");
        url.push_str(&hash[..2]);
        url.push('/');
        url.push_str(hash);
        let result = read_or_download_bytes(&cached, &url)?;
        Ok(result)
    }

    fn modified_time(
        &mut self,
        _entry: &Path,
    ) -> Result<Option<jiff::Timestamp>, storage::StorageError> {
        Ok(None)
    }
}
