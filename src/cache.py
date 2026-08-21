import dataclasses
import json
import pathlib
import typing
import zipfile

import requests

from minecraft_ttf.minecraft.storage import AssetSource, AssetStorage, ZipStorage
from minecraft_ttf.minecraft.versions import MinecraftVersion, detect_version

ManifestLatest = typing.TypedDict('ManifestLatest', {
    'release': str,
    'snapshot': str
})

ManifestVersion = typing.TypedDict('ManifestVersion', {
    'id': str,
    'url': str
})

Manifest = typing.TypedDict('Manifest', {
    'latest': ManifestLatest,
    'versions': list[ManifestVersion]
})

LauncherDownload = typing.TypedDict('LauncherDownload', {
    'url': str
})

LauncherDownloads = typing.TypedDict('LauncherDownloads', {
    'client': LauncherDownload
})

LauncherAssets = typing.TypedDict('LauncherAssets', {
    'id': str,
    'url': str,
})

LauncherData = typing.TypedDict('LauncherData', {
    'downloads': LauncherDownloads,
    'assetIndex': LauncherAssets,
})

@dataclasses.dataclass
class JarInformation:
    jar_storage: ZipStorage
    asset_storage: AssetStorage
    manifest: ManifestVersion
    launcher: LauncherData
    assets: AssetSource
    version: MinecraftVersion

def jar_info(version_id: str, cache: pathlib.Path) -> JarInformation | None:
    manifest = get_manifest(cache)
    if version_id == 'latest':
        version_data = get_version(manifest, manifest['latest']['snapshot'])
    else:
        version_data = get_version(manifest, version_id)
    if version_data is None:
        print(f'Version {version_id} not found in manifest')
        return None
    data = version_from_data(version_data, cache)
    if data is None:
        print(f'Unable to determine capabilities of jar {version_data['id']}!')
        return None
    return data

def version_from_data(version_data: ManifestVersion, cache: pathlib.Path) -> JarInformation | None:
    launcher_data = get_launcher_json(version_data['id'], version_data['url'], cache)
    assets = get_asset_source(launcher_data, cache)
    jar_path = get_jar(version_data['id'], launcher_data, cache)
    zip = zipfile.ZipFile(jar_path, 'r')
    version = detect_version(zip)
    if version is None:
        return None
    jar_storage = ZipStorage(zip)
    asset_storage = AssetStorage('https://resources.download.minecraft.net', version.asset_mount, assets, cache / 'assets/objects')
    return JarInformation(jar_storage, asset_storage, version_data, launcher_data, assets, version)

def get_launcher_json(version_id: str, meta_url: str, cache: pathlib.Path) -> LauncherData:
    cached_path = cache / 'versions' / version_id / f'{version_id}.json'
    try:
        with open(cached_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'Downloading Minecraft launcher data {version_id}')
        response = requests.get(meta_url)
        response.raise_for_status()
        data = response.json()
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    return data

def get_asset_source(launcher_data: LauncherData, cache: pathlib.Path) -> AssetSource:
    cached_path = cache / 'assets' / 'indexes' /  f'{launcher_data['assetIndex']['id']}.json'
    try:
        with open(cached_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'Downloading Minecraft assets {launcher_data['assetIndex']['id']}')
        response = requests.get(launcher_data['assetIndex']['url'])
        response.raise_for_status()
        data = response.json()
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    return data

def get_jar(version_id: str, launcher_data: LauncherData, cache: pathlib.Path) -> pathlib.Path:
    cached_path = cache / 'versions' / version_id / f'{version_id}.jar'
    if not cached_path.exists():
        print(f'Downloading Minecraft jar {version_id}')
        client_jar = launcher_data['downloads']['client']['url']
        response = requests.get(client_jar)
        response.raise_for_status()
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'wb') as f:
            f.writelines(response.iter_content(chunk_size=16 * 1024))
    return cached_path

def get_version(manifest: Manifest, version_id: str) -> ManifestVersion | None:
    for version in manifest['versions']:
        if version['id'] == version_id:
            return version
    return None

def get_manifest(cache: pathlib.Path) -> Manifest:
    cached_path = cache / 'versions/version_manifest_v2.json'
    try:
        with open(cached_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('Downloading version manifest')
        manifest_url = 'https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'
        response = requests.get(manifest_url)
        response.raise_for_status()
        data = response.json()
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    return data

# The Adobe Glyph List For New Fonts tells us what names to use for the glyphs that characters are mapped to
def get_aglfn(cache: pathlib.Path) -> dict[str, str]:
    cached_path = cache / 'aglfn.txt'
    if not cached_path.exists():
        print('Downloading Adobe AGLFN')
        response = requests.get('https://raw.githubusercontent.com/adobe-type-tools/agl-aglfn/refs/heads/master/aglfn.txt')
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_path, 'wb') as f:
            f.writelines(response.iter_content(chunk_size=16 * 1024))
    aglfn_map = {}
    with open(cached_path, 'r', encoding='utf-8') as aglfn:
        for line in aglfn:
            if line.startswith('#') or line.isspace() or len(line) == 0:
                continue
            unihex, name, _uniname = line.split(';')
            uninum = int(unihex, 16)
            codepoint = chr(uninum)
            aglfn_map[codepoint] = name
    return aglfn_map
