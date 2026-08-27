use std::{
    io::Read,
    path::{Path, PathBuf},
};

pub trait Storage {
    fn get_entries(&self, prefix: &Path) -> Result<Vec<PathBuf>, StorageError>;
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
}

pub struct FilesystemStorage {
    root: PathBuf,
}

impl Storage for FilesystemStorage {
    fn get_entries(&self, prefix: &Path) -> Result<Vec<PathBuf>, StorageError> {
        let path = self.root.join(prefix);
        let mut result: Vec<PathBuf> = vec![];
        for entry in walkdir::WalkDir::new(path) {
            let entry = entry?;
            if entry.file_type().is_file() {
                let path = entry.into_path();
                let partial = path.strip_prefix(&path)?;
                result.push(prefix.join(partial));
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
    fn get_entries(&self, prefix: &Path) -> Result<Vec<PathBuf>, StorageError> {
        let mut result = vec![];
        for name in self.zip.file_names() {
            let path = Path::new(name);
            if path.starts_with(prefix) {
                result.push(path.to_owned());
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
