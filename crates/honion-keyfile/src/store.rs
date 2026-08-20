//! Writing a hidden-service directory in the layout Tor expects.
//!
//! # File formats
//!
//! Tor reads three files from a `HiddenServiceDir`:
//!
//! | file | contents |
//! |---|---|
//! | `hs_ed25519_secret_key` | 32-byte tag `"== ed25519v1-secret: type0 =="` NUL-padded, then the 64-byte expanded key |
//! | `hs_ed25519_public_key` | 32-byte tag `"== ed25519v1-public: type0 =="` NUL-padded, then the 32-byte public key |
//! | `hostname` | the address followed by `".onion"` and a newline |
//!
//! The tags are fixed-width fields, not strings: Tor compares all 32 bytes, so
//! the padding is part of the format rather than cosmetic.
//!
//! # Writing
//!
//! Files are created with mode `0600` inside a `0700` directory, and written
//! through a temporary file that is renamed into place. The rename is atomic on
//! any POSIX filesystem, so an interrupted run leaves either no file or a
//! complete one — never a half-written secret key that Tor would load and fail
//! on, or worse, that would look valid while being truncated.

use std::fs;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use crate::key::VerifiedKey;

/// Tag prefixing the secret key file.
const SECRET_TAG: &[u8] = b"== ed25519v1-secret: type0 ==";
/// Tag prefixing the public key file.
const PUBLIC_TAG: &[u8] = b"== ed25519v1-public: type0 ==";
/// Width of the tag field in both files.
const TAG_LEN: usize = 32;

/// Build a tag field: the label, NUL-padded to [`TAG_LEN`].
fn tag_field(label: &[u8]) -> [u8; TAG_LEN] {
    let mut out = [0u8; TAG_LEN];
    // Every tag this module uses is shorter than the field; the assert makes
    // that a checked property rather than an assumption about string lengths.
    assert!(label.len() <= TAG_LEN, "tag does not fit its field");
    out[..label.len()].copy_from_slice(label);
    out
}

/// The exact bytes of `hs_ed25519_secret_key`.
#[must_use]
pub fn secret_key_file(key: &VerifiedKey) -> Vec<u8> {
    let mut out = Vec::with_capacity(TAG_LEN + 64);
    out.extend_from_slice(&tag_field(SECRET_TAG));
    out.extend_from_slice(&key.key().to_bytes());
    out
}

/// The exact bytes of `hs_ed25519_public_key`.
#[must_use]
pub fn public_key_file(key: &VerifiedKey) -> Vec<u8> {
    let mut out = Vec::with_capacity(TAG_LEN + 32);
    out.extend_from_slice(&tag_field(PUBLIC_TAG));
    out.extend_from_slice(key.public());
    out
}

/// The exact bytes of `hostname`.
#[must_use]
pub fn hostname_file(key: &VerifiedKey) -> Vec<u8> {
    let mut out = key.address().to_hostname().into_bytes();
    out.push(b'\n');
    out
}

/// Write a complete hidden-service directory for `key` under `parent`.
///
/// The directory is named for the full onion hostname — `<address>.onion` —
/// so many results accumulate in one output folder without collision, and the
/// name is directly recognisable as an address rather than as a bare 56-character
/// token. This is also the layout `mkp224o` produces, so results from either
/// tool are interchangeable.
///
/// The *file* names inside are fixed by Tor and cannot be changed: it looks for
/// `hs_ed25519_secret_key` and `hs_ed25519_public_key` by those exact names.
/// Returns the directory created.
///
/// # Errors
///
/// [`StoreError::AlreadyExists`] if the directory is already present — an
/// existing result is never overwritten, because doing so would destroy a key
/// that cannot be recovered. Otherwise, any underlying I/O error.
pub fn write_service_dir(parent: &Path, key: &VerifiedKey) -> Result<PathBuf, StoreError> {
    let dir = parent.join(key.address().to_hostname());
    if dir.exists() {
        return Err(StoreError::AlreadyExists { path: dir });
    }
    fs::create_dir_all(&dir).map_err(|e| StoreError::Io {
        path: dir.clone(),
        reason: e.to_string(),
    })?;
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).map_err(|e| StoreError::Io {
        path: dir.clone(),
        reason: e.to_string(),
    })?;

    write_private(&dir, "hs_ed25519_secret_key", &secret_key_file(key))?;
    write_private(&dir, "hs_ed25519_public_key", &public_key_file(key))?;
    write_private(&dir, "hostname", &hostname_file(key))?;

    Ok(dir)
}

/// Write one file atomically with mode 0600.
fn write_private(dir: &Path, name: &str, contents: &[u8]) -> Result<(), StoreError> {
    let final_path = dir.join(name);
    let tmp_path = dir.join(format!(".{name}.tmp"));

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp_path)
        .map_err(|e| StoreError::Io {
            path: tmp_path.clone(),
            reason: e.to_string(),
        })?;
    file.write_all(contents).map_err(|e| StoreError::Io {
        path: tmp_path.clone(),
        reason: e.to_string(),
    })?;
    // Flush to the device before the rename, so a crash cannot leave a renamed
    // but empty file.
    file.sync_all().map_err(|e| StoreError::Io {
        path: tmp_path.clone(),
        reason: e.to_string(),
    })?;
    drop(file);

    fs::rename(&tmp_path, &final_path).map_err(|e| StoreError::Io {
        path: final_path,
        reason: e.to_string(),
    })
}

/// Why a key could not be stored.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StoreError {
    /// A directory for this address already exists.
    #[error("{path} already exists; refusing to overwrite an existing key")]
    AlreadyExists {
        /// The conflicting path.
        path: PathBuf,
    },
    /// An I/O operation failed.
    #[error("writing {path}: {reason}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// The underlying error, rendered. Not named `source`, which
        /// `thiserror` reserves for a nested `Error`.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::SecretScalar;
    use honion_core::address::OnionAddress;
    use honion_core::pattern::{Pattern, PatternSet};

    fn a_verified_key() -> VerifiedKey {
        let scalar = SecretScalar::generate().expect("randomness");
        let address = OnionAddress::from_pubkey(&scalar.public_key());
        let prefix: String = address.body().chars().take(4).collect();
        let set = PatternSet::compile(&[Pattern::parse(&prefix).expect("valid")]).expect("non-empty");
        VerifiedKey::verify(&scalar, 0, &set).expect("verifies")
    }

    #[test]
    fn secret_file_has_the_layout_tor_reads() {
        let key = a_verified_key();
        let bytes = secret_key_file(&key);
        assert_eq!(bytes.len(), 96, "32-byte tag plus 64-byte expanded key");
        assert_eq!(&bytes[..SECRET_TAG.len()], SECRET_TAG);
        assert!(
            bytes[SECRET_TAG.len()..TAG_LEN].iter().all(|b| *b == 0),
            "tag field is NUL-padded, not space-padded"
        );
        assert_eq!(&bytes[TAG_LEN..], &key.key().to_bytes());
    }

    #[test]
    fn public_file_has_the_layout_tor_reads() {
        let key = a_verified_key();
        let bytes = public_key_file(&key);
        assert_eq!(bytes.len(), 64, "32-byte tag plus 32-byte key");
        assert_eq!(&bytes[..PUBLIC_TAG.len()], PUBLIC_TAG);
        assert!(bytes[PUBLIC_TAG.len()..TAG_LEN].iter().all(|b| *b == 0));
        assert_eq!(&bytes[TAG_LEN..], key.public());
    }

    #[test]
    fn hostname_file_is_the_address_and_a_newline() {
        let key = a_verified_key();
        let bytes = hostname_file(&key);
        let text = String::from_utf8(bytes).expect("ascii");
        assert!(text.ends_with(".onion\n"));
        assert_eq!(text.trim_end(), key.address().to_hostname());
        assert_eq!(text.len(), 56 + ".onion\n".len());
    }

    #[test]
    fn the_public_file_matches_the_key_in_the_secret_file() {
        // The check that catches a mismatched pair, which would produce a
        // service that starts and then fails to be reachable.
        let key = a_verified_key();
        let secret = secret_key_file(&key);
        let public = public_key_file(&key);
        let scalar_bytes: [u8; 32] = secret[TAG_LEN..TAG_LEN + 32].try_into().expect("32 bytes");
        let scalar = SecretScalar::from_clamped(scalar_bytes).expect("written scalars are clamped");
        assert_eq!(&scalar.public_key()[..], &public[TAG_LEN..]);
    }

    #[test]
    fn writes_the_expected_files_with_tight_permissions() {
        let base = std::env::temp_dir().join(format!("honion-store-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).expect("temp dir");

        let key = a_verified_key();
        let dir = write_service_dir(&base, &key).expect("writes");

        assert_eq!(
            dir.file_name().and_then(|s| s.to_str()),
            Some(key.address().to_hostname().as_str()),
            "the directory is named for the full hostname, including .onion"
        );
        assert!(
            dir.file_name().and_then(|s| s.to_str()).is_some_and(|n| n.ends_with(".onion")),
        );
        assert_eq!(
            fs::metadata(&dir).expect("stat").permissions().mode() & 0o777,
            0o700
        );
        for name in ["hs_ed25519_secret_key", "hs_ed25519_public_key", "hostname"] {
            let p = dir.join(name);
            assert!(p.exists(), "{name} was not written");
            assert_eq!(
                fs::metadata(&p).expect("stat").permissions().mode() & 0o777,
                0o600,
                "{name} must not be readable by anyone else"
            );
        }
        // No temporary files left behind.
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .expect("readdir")
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with('.'))
            .collect();
        assert!(leftovers.is_empty(), "temporary files were left behind");

        // Writing the same address twice must not clobber the first result.
        assert!(matches!(
            write_service_dir(&base, &key),
            Err(StoreError::AlreadyExists { .. })
        ));

        fs::remove_dir_all(&base).ok();
    }
}
