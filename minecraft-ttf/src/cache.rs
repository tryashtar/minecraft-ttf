#[derive(Debug, serde::Deserialize)]
struct ManifestLatest {
    release: String,
    snapshot: String,
}

#[derive(Debug, serde::Deserialize)]
struct ManifestVersion {
    id: String,
    url: reqwest::Url,
}

#[derive(Debug, serde::Deserialize)]
struct Manifest {
    latest: ManifestLatest,
    versions: Vec<ManifestVersion>,
}

#[derive(Debug, serde::Deserialize)]
struct LauncherDownload {
    url: url::Url,
}

#[derive(Debug, serde::Deserialize)]
struct LauncherDownloads {
    client: LauncherDownload,
}

#[derive(Debug, serde::Deserialize)]
struct LauncherAssets {
    id: String,
    url: url::Url,
}

#[derive(Debug, serde::Deserialize)]
struct LauncherData {
    downloads: LauncherDownloads,
    assetIndex: LauncherAssets,
}
