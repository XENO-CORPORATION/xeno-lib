//! EXIF metadata helpers built atop `nom-exif`.

use crate::error::TransformError;
use nom_exif::{Exif, ExifIter, MediaParser, MediaSource};
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

/// Read EXIF metadata from any Read + Seek source containing supported container data (JPEG, HEIF, TIFF).
pub fn read_exif_from_reader<R>(reader: R) -> Result<Exif, TransformError>
where
    R: Read + Seek,
{
    let media = MediaSource::seekable(reader).map_err(map_exif_err)?;
    let mut parser = MediaParser::new();
    let iter: ExifIter = parser.parse(media).map_err(map_exif_err)?;
    Ok(iter.into())
}

/// Convenience helper to read EXIF metadata from a file path.
pub fn read_exif_from_path<P>(path: P) -> Result<Exif, TransformError>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    read_exif_from_reader(reader)
}

fn map_exif_err<E: std::fmt::Display>(err: E) -> TransformError {
    TransformError::ExifRead {
        message: err.to_string(),
    }
}
