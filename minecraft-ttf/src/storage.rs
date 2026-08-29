use std::{
    collections::{HashSet, VecDeque},
    io::Read,
    path::{Path, PathBuf},
};

use crate::cache;

pub trait Storage {
    fn get_entries(&self, prefix: &Path) -> Result<HashSet<PathBuf>, StorageError>;
    fn read(&mut self, entry: &Path) -> Result<Vec<u8>, StorageError>;
    fn modified_time(&mut self, entry: &Path) -> Result<Option<jiff::Timestamp>, StorageError>;
}

#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("file not found")]
    FileNotFound,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Walk(#[from] walkdir::Error),
    #[error(transparent)]
    StripPrefix(#[from] std::path::StripPrefixError),
    #[error("couldn't convert path to string")]
    PathConversion,
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),
    #[error(transparent)]
    Time(#[from] jiff::Error),
    #[error(transparent)]
    Cache(#[from] cache::CacheError),
    #[error(transparent)]
    Parse(#[from] std::string::FromUtf8Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug)]
pub struct FilesystemStorage {
    root: PathBuf,
}

impl Storage for FilesystemStorage {
    fn get_entries(&self, prefix: &Path) -> Result<HashSet<PathBuf>, StorageError> {
        let path = self.root.join(prefix);
        let mut result = HashSet::new();
        for entry in walkdir::WalkDir::new(path) {
            let entry = entry?;
            if entry.file_type().is_file() {
                let path = entry.into_path();
                let partial = path.strip_prefix(&path)?;
                result.insert(prefix.join(partial));
            }
        }
        Ok(result)
    }

    fn read(&mut self, entry: &Path) -> Result<Vec<u8>, StorageError> {
        let path = self.root.join(entry);
        let result = std::fs::read(&path);
        match result {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(StorageError::FileNotFound),
            other => Ok(other?),
        }
    }

    fn modified_time(&mut self, entry: &Path) -> Result<Option<jiff::Timestamp>, StorageError> {
        let path = self.root.join(entry);
        let metadata = std::fs::metadata(path);
        match metadata {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(StorageError::FileNotFound),
            Err(e) => Err(StorageError::Io(e)),
            Ok(metadata) => {
                let Ok(modified) = metadata.modified() else {
                    return Ok(None);
                };
                let timestamp = jiff::Timestamp::try_from(modified)?;
                Ok(Some(timestamp))
            }
        }
    }
}

#[derive(Debug)]
pub struct ZipStorage<T> {
    zip: zip::ZipArchive<T>,
}

impl<T> ZipStorage<T> {
    pub fn new(zip: zip::ZipArchive<T>) -> Self {
        Self { zip }
    }
}

impl<T: std::io::Read + std::io::Seek> Storage for ZipStorage<T> {
    fn get_entries(&self, prefix: &Path) -> Result<HashSet<PathBuf>, StorageError> {
        let mut result = HashSet::new();
        for name in self.zip.file_names() {
            let path = Path::new(name);
            if path.starts_with(prefix) {
                result.insert(path.to_owned());
            }
        }
        Ok(result)
    }

    fn read(&mut self, entry: &Path) -> Result<Vec<u8>, StorageError> {
        let Some(name) = entry.as_os_str().to_str() else {
            return Err(StorageError::PathConversion);
        };
        let file = self.zip.by_name(name);
        match file {
            Err(zip::result::ZipError::FileNotFound) => Err(StorageError::FileNotFound),
            Err(e) => Err(StorageError::Zip(e)),
            Ok(mut file) => {
                let mut bytes = vec![];
                file.read_to_end(&mut bytes)?;
                Ok(bytes)
            }
        }
    }

    fn modified_time(&mut self, entry: &Path) -> Result<Option<jiff::Timestamp>, StorageError> {
        let Some(name) = entry.as_os_str().to_str() else {
            return Err(StorageError::PathConversion);
        };
        let file = self.zip.by_name(name);
        match file {
            Err(zip::result::ZipError::FileNotFound) => Err(StorageError::FileNotFound),
            Err(e) => Err(StorageError::Zip(e)),
            Ok(file) => {
                let Some(modified) = file.last_modified() else {
                    return Ok(None);
                };
                let civil = jiff::civil::DateTime::try_from(modified)?;
                let zoned = civil.to_zoned(jiff::tz::TimeZone::unknown())?;
                Ok(Some(zoned.timestamp()))
            }
        }
    }
}

pub struct StackStorage(pub Vec<Box<dyn Storage>>);

fn stack_first<T>(
    storage: &mut StackStorage,
    mut callback: impl FnMut(&mut dyn Storage) -> Result<T, StorageError>,
) -> Result<T, StorageError> {
    for sub in storage.0.iter_mut() {
        let result = callback(Box::as_mut(sub));
        match result {
            Err(StorageError::FileNotFound) => {
                continue;
            }
            _ => {
                return result;
            }
        }
    }
    Err(StorageError::FileNotFound)
}

impl Storage for StackStorage {
    fn get_entries(&self, prefix: &Path) -> Result<HashSet<PathBuf>, StorageError> {
        let mut result = HashSet::new();
        for sub in &self.0 {
            let entries = sub.get_entries(prefix)?;
            result.extend(entries);
        }
        Ok(result)
    }

    fn read(&mut self, entry: &Path) -> Result<Vec<u8>, StorageError> {
        stack_first(self, |x| x.read(entry))
    }

    fn modified_time(&mut self, entry: &Path) -> Result<Option<jiff::Timestamp>, StorageError> {
        stack_first(self, |x| x.modified_time(entry))
    }
}

#[derive(Debug)]
pub struct ReadEntry<T> {
    pub data: T,
    pub modified_time: Option<jiff::Timestamp>,
}

pub fn read_image(
    store: &mut impl Storage,
    entry: &Path,
) -> Result<ReadEntry<image::DynamicImage>, StorageError> {
    let modified_time = store.modified_time(entry)?;
    let data = store.read(entry)?;
    let image = image::load_from_memory(&data)?;
    Ok(ReadEntry {
        data: image,
        modified_time,
    })
}

pub fn read_font_txt(
    store: &mut impl Storage,
    entry: &Path,
) -> Result<ReadEntry<VecDeque<String>>, StorageError> {
    let modified_time = store.modified_time(entry)?;
    let data = store.read(entry)?;
    let text = String::from_utf8(data)?;
    let lines = text
        .lines()
        .filter(|x| !x.starts_with('#'))
        .map(String::from)
        .collect();
    Ok(ReadEntry {
        data: lines,
        modified_time,
    })
}

pub fn read_json<T: serde::de::DeserializeOwned, J: Storage + ?Sized>(
    store: &mut J,
    entry: &Path,
) -> Result<ReadEntry<T>, StorageError> {
    let modified_time = store.modified_time(entry)?;
    let data = store.read(entry)?;
    let result = serde_json::from_slice(&data)?;
    Ok(ReadEntry {
        data: result,
        modified_time,
    })
}
