use std::path::Path;

use tracing::info;

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
    downloads: LauncherDownloads,
    #[serde(rename = "assetIndex")]
    asset_index: LauncherAssets,
}

#[derive(thiserror::Error, Debug)]
pub enum CacheError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error(transparent)]
    Request(#[from] reqwest::Error),
}

pub fn get_manifest(cache: &Path) -> Result<Manifest, CacheError> {
    let cached_path = cache.join(Path::new("versions/version_manifest_v2.json"));
    let url = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json";
    read_or_download_json(&cached_path, url)
}

fn read_or_download_json<T: serde::de::DeserializeOwned>(
    path: &Path,
    url: &str,
) -> Result<T, CacheError> {
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
            let result = serde_json::from_slice(&bytes)?;
            Ok(result)
        }
        Err(e) => Err(CacheError::Io(e)),
        Ok(file) => Ok(serde_json::from_reader(file)?),
    }
}
