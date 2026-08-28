use std::{
    collections::HashSet,
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
}

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
        Ok(std::fs::read(&path)?)
    }

    fn modified_time(&mut self, entry: &Path) -> Result<Option<jiff::Timestamp>, StorageError> {
        let path = self.root.join(entry);
        let metadata = std::fs::metadata(path)?;
        let Ok(modified) = metadata.modified() else {
            return Ok(None);
        };
        let timestamp = jiff::Timestamp::try_from(modified)?;
        Ok(Some(timestamp))
    }
}

pub struct ZipStorage {
    zip: zip::ZipArchive<std::fs::File>,
}

impl Storage for ZipStorage {
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
        let mut file = self.zip.by_name(name)?;
        let mut bytes = vec![];
        file.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    fn modified_time(&mut self, entry: &Path) -> Result<Option<jiff::Timestamp>, StorageError> {
        let Some(name) = entry.as_os_str().to_str() else {
            return Err(StorageError::PathConversion);
        };
        let file = self.zip.by_name(name)?;
        let Some(modified) = file.last_modified() else {
            return Ok(None);
        };
        let civil = jiff::civil::DateTime::try_from(modified)?;
        let zoned = civil.to_zoned(jiff::tz::TimeZone::system())?;
        Ok(Some(zoned.timestamp()))
    }
}

pub struct StackStorage(Vec<Box<dyn Storage>>);

fn stack_first<T>(
    storage: &mut StackStorage,
    mut callback: impl FnMut(&mut dyn Storage) -> Result<T, StorageError>,
) -> Result<T, StorageError> {
    for sub in storage.0.iter_mut() {
        let result = callback(Box::as_mut(sub));
        match result {
            Err(StorageError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                continue;
            }
            _ => {
                return result;
            }
        }
    }
    Err(StorageError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "",
    )))
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
