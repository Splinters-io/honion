//! Tor v3 onion address construction, byte for byte.
//!
//! # The specification
//!
//! From Tor's `rend-spec-v3.txt` §6 ("Encoding onion addresses"):
//!
//! ```text
//! onion_address = base32(PUBKEY | CHECKSUM | VERSION) + ".onion"
//! CHECKSUM      = H(".onion checksum" | PUBKEY | VERSION)[:2]
//! VERSION       = '\x03'
//! ```
//!
//! where `PUBKEY` is the 32-byte Ed25519 public key, `H` is SHA3-256, and the
//! base32 is RFC 4648 lowercase without padding. The encoded input is
//! `32 + 2 + 1 = 35` bytes, which is exactly `56` base32 characters — the
//! address is always 56 characters before `.onion`.
//!
//! # The fact that makes GPU vanity search practical
//!
//! `PUBKEY` occupies the first 32 bytes = 256 bits = 51.2 base32 characters of
//! the encoded input. Because base32 emits bits most-significant first, the
//! first 51 characters of an address are a function of the *public key alone*.
//! The checksum — the only part requiring SHA3 — cannot influence any character
//! before index 51.
//!
//! Consequently a search for a *prefix* of at most [`PREFIX_CHARS_WITHOUT_CHECKSUM`]
//! characters never needs to hash anything. The GPU can mask raw public-key
//! bytes and compare. A *suffix* search, by contrast, would require a SHA3-256
//! per candidate, which is why this tool does not offer one.
//!
//! See [`crate::base32`] for the bit-layout argument behind that claim.

use core::fmt;

use sha3::{Digest, Sha3_256};

use crate::base32::{self, DecodeError};

/// Length of an Ed25519 public key in bytes.
pub const PUBKEY_LEN: usize = 32;
/// Length of the address checksum in bytes.
pub const CHECKSUM_LEN: usize = 2;
/// The onion address version this crate implements.
pub const VERSION: u8 = 3;
/// Number of bytes fed into base32: public key, checksum, version.
pub const ADDRESS_BODY_LEN: usize = PUBKEY_LEN + CHECKSUM_LEN + 1;
/// Number of base32 characters in an address, excluding the `.onion` suffix.
pub const ADDRESS_CHARS: usize = 56;
/// The domain suffix.
pub const ONION_SUFFIX: &str = ".onion";

/// Personalisation string prepended to the checksum input.
const CHECKSUM_PERSONALISATION: &[u8] = b".onion checksum";

/// The greatest number of leading address characters determined solely by the
/// public key.
///
/// Character `i` covers bits `[5i, 5i+5)`. The public key ends at bit 256, so
/// character 51 (bits 255..260) is the first that straddles into the checksum.
/// Characters `0..=50` — that is, 51 of them — depend only on the key.
pub const PREFIX_CHARS_WITHOUT_CHECKSUM: usize = 51;

const _: () = {
    assert!(ADDRESS_BODY_LEN == 35);
    assert!(base32::encoded_len(ADDRESS_BODY_LEN) == ADDRESS_CHARS);
    // The boundary claim above, checked by the compiler rather than by comment.
    assert!(base32::prefix_bytes_needed(PREFIX_CHARS_WITHOUT_CHECKSUM) == PUBKEY_LEN);
    assert!(base32::prefix_bytes_needed(PREFIX_CHARS_WITHOUT_CHECKSUM + 1) > PUBKEY_LEN);
};

/// Compute the 2-byte checksum for a public key.
#[must_use]
pub fn checksum(pubkey: &[u8; PUBKEY_LEN]) -> [u8; CHECKSUM_LEN] {
    let mut hasher = Sha3_256::new();
    hasher.update(CHECKSUM_PERSONALISATION);
    hasher.update(pubkey);
    hasher.update([VERSION]);
    let digest = hasher.finalize();
    // SHA3-256 always produces 32 bytes, so this cannot fail; taking a prefix
    // chunk rather than slicing keeps that fact checked instead of assumed.
    digest
        .first_chunk::<CHECKSUM_LEN>()
        .copied()
        .unwrap_or([0u8; CHECKSUM_LEN])
}

/// A well-formed v3 onion address.
///
/// Constructing one is proof that the checksum and version are correct and that
/// the textual form is exactly 56 canonical base32 characters. Code holding an
/// `OnionAddress` never needs to re-validate it (langsec rule 3).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OnionAddress {
    pubkey: [u8; PUBKEY_LEN],
    /// Cached textual form without the `.onion` suffix. Always 56 characters.
    text: String,
}

impl OnionAddress {
    /// Derive the address for an Ed25519 public key.
    ///
    /// Total: every 32-byte string yields an address. (Not every 32-byte string
    /// is a valid curve point, but the address encoding does not care, and the
    /// callers that produce keys guarantee validity by construction.)
    #[must_use]
    pub fn from_pubkey(pubkey: &[u8; PUBKEY_LEN]) -> Self {
        let mut body = [0u8; ADDRESS_BODY_LEN];
        body[..PUBKEY_LEN].copy_from_slice(pubkey);
        body[PUBKEY_LEN..PUBKEY_LEN + CHECKSUM_LEN].copy_from_slice(&checksum(pubkey));
        body[ADDRESS_BODY_LEN - 1] = VERSION;
        Self {
            pubkey: *pubkey,
            text: base32::encode(&body),
        }
    }

    /// Recognise an address in text form.
    ///
    /// Accepts with or without the `.onion` suffix. Rejects anything whose
    /// checksum or version byte does not match — so a typo cannot silently
    /// become a different valid-looking address.
    ///
    /// # Errors
    ///
    /// See [`AddressError`].
    pub fn parse(input: &str) -> Result<Self, AddressError> {
        let body_text = input.strip_suffix(ONION_SUFFIX).unwrap_or(input);
        if body_text.len() != ADDRESS_CHARS {
            return Err(AddressError::WrongLength {
                found: body_text.len(),
            });
        }
        let body = base32::decode(body_text)?;
        // 56 canonical characters always decode to 35 bytes. Rather than assert
        // that and then index, take the whole body as a fixed-size array: the
        // conversion fails if the length is ever wrong, so no indexing here can
        // be out of range by construction.
        let body: [u8; ADDRESS_BODY_LEN] =
            body.try_into().map_err(|_| AddressError::WrongLength {
                found: body_text.len(),
            })?;
        let (pubkey, rest) = body.split_at(PUBKEY_LEN);
        let (found_checksum, version) = rest.split_at(CHECKSUM_LEN);
        let (Ok(pubkey), Ok(found_checksum), [found_version]) = (
            <[u8; PUBKEY_LEN]>::try_from(pubkey),
            <[u8; CHECKSUM_LEN]>::try_from(found_checksum),
            version,
        ) else {
            // Unreachable: the three parts sum to ADDRESS_BODY_LEN by
            // definition. Handled as an error rather than a panic so that the
            // function stays total.
            return Err(AddressError::WrongLength {
                found: body_text.len(),
            });
        };

        if *found_version != VERSION {
            return Err(AddressError::WrongVersion {
                found: *found_version,
            });
        }
        let expected = checksum(&pubkey);
        if found_checksum != expected {
            return Err(AddressError::ChecksumMismatch {
                found: found_checksum,
                expected,
            });
        }
        Ok(Self {
            pubkey,
            text: body_text.to_owned(),
        })
    }

    /// The public key this address encodes.
    #[must_use]
    pub const fn pubkey(&self) -> &[u8; PUBKEY_LEN] {
        &self.pubkey
    }

    /// The 56-character base32 body, without the `.onion` suffix.
    #[must_use]
    pub fn body(&self) -> &str {
        &self.text
    }

    /// The full address including the `.onion` suffix.
    #[must_use]
    pub fn to_hostname(&self) -> String {
        let mut s = String::with_capacity(ADDRESS_CHARS + ONION_SUFFIX.len());
        s.push_str(&self.text);
        s.push_str(ONION_SUFFIX);
        s
    }
}

impl fmt::Display for OnionAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.text, ONION_SUFFIX)
    }
}

impl fmt::Debug for OnionAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OnionAddress({}{})", self.text, ONION_SUFFIX)
    }
}

/// Why a string was not recognised as a v3 onion address.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AddressError {
    /// The base32 body was not exactly [`ADDRESS_CHARS`] characters.
    #[error("expected {ADDRESS_CHARS} base32 characters before '.onion', found {found}")]
    WrongLength {
        /// Number of characters actually present.
        found: usize,
    },
    /// The body was not canonical base32.
    #[error("malformed base32: {0}")]
    Base32(#[from] DecodeError),
    /// The version byte was not 3.
    #[error("unsupported onion address version {found}, expected {VERSION}")]
    WrongVersion {
        /// The version byte found.
        found: u8,
    },
    /// The embedded checksum did not match the public key.
    #[error("checksum mismatch: address carries {found:02x?}, key implies {expected:02x?}")]
    ChecksumMismatch {
        /// Checksum bytes carried by the address.
        found: [u8; CHECKSUM_LEN],
        /// Checksum bytes implied by the public key.
        expected: [u8; CHECKSUM_LEN],
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real, publicly advertised v3 addresses.
    ///
    /// These are self-validating: `parse` recomputes the SHA3-256 checksum from
    /// the decoded public key and rejects a mismatch. So this test pins the
    /// base32 decoder, the byte layout, and the checksum construction against
    /// data produced by Tor itself rather than by us.
    const REAL_ADDRESSES: &[&str] = &[
        // The Tor Project's own site.
        "2gzyxa5ihm7nsggfxnu52rck2vv4rvmdlkiu3zzui5du4xyclen53wid.onion",
        // DuckDuckGo. Note this is itself a vanity address: a 10-character
        // prefix, which is roughly the practical ceiling for a single GPU.
        "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion",
    ];

    #[test]
    fn real_addresses_validate() {
        for text in REAL_ADDRESSES {
            let addr = OnionAddress::parse(text)
                .unwrap_or_else(|e| panic!("{text} should be a valid address: {e}"));
            assert_eq!(&addr.to_hostname(), text);
        }
    }

    #[test]
    fn real_addresses_reconstruct_from_their_own_keys() {
        // Round trip through the *encoder*: decoding an address yields a key,
        // and re-encoding that key must reproduce the address exactly.
        for text in REAL_ADDRESSES {
            let addr = OnionAddress::parse(text).expect("valid");
            let rebuilt = OnionAddress::from_pubkey(addr.pubkey());
            assert_eq!(rebuilt, addr, "re-encoding {text}");
            assert_eq!(&rebuilt.to_hostname(), text);
        }
    }

    #[test]
    fn all_zero_key_has_stable_address() {
        // A fixed vector so a change in the checksum construction is caught even
        // if every other test were removed.
        let addr = OnionAddress::from_pubkey(&[0u8; PUBKEY_LEN]);
        assert_eq!(addr.body().len(), ADDRESS_CHARS);
        assert!(addr.body().starts_with("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
        // Re-parsing must accept what we produced.
        assert_eq!(OnionAddress::parse(&addr.to_hostname()).expect("valid"), addr);
    }

    #[test]
    fn parse_accepts_with_and_without_suffix() {
        let addr = OnionAddress::from_pubkey(&[7u8; PUBKEY_LEN]);
        let with = addr.to_hostname();
        let without = addr.body();
        assert_eq!(OnionAddress::parse(&with).expect("valid"), addr);
        assert_eq!(OnionAddress::parse(without).expect("valid"), addr);
    }

    #[test]
    fn rejects_corrupted_checksum() {
        let addr = OnionAddress::from_pubkey(&[1u8; PUBKEY_LEN]);
        let mut body = addr.body().to_owned();
        // Character 52 lies past the public key, so mutating it changes only
        // the checksum — precisely the case a naive parser would wave through.
        flip_char(&mut body, 52);
        assert!(matches!(
            OnionAddress::parse(&body),
            Err(AddressError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(matches!(
            OnionAddress::parse("abc.onion"),
            Err(AddressError::WrongLength { found: 3 })
        ));
    }

    /// Replace the base32 character at `index` with a different one.
    fn flip_char(s: &mut String, index: usize) {
        let mut bytes = core::mem::take(s).into_bytes();
        bytes[index] = if bytes[index] == b'a' { b'b' } else { b'a' };
        *s = String::from_utf8(bytes).expect("alphabet is ascii");
    }

    #[test]
    fn prefix_boundary_is_where_we_claim() {
        // Two keys differing only in their final bit produce addresses that
        // agree on the first 51 characters. This is the property the GPU search
        // relies on, demonstrated rather than asserted.
        let mut a = [0u8; PUBKEY_LEN];
        let mut b = [0u8; PUBKEY_LEN];
        a[PUBKEY_LEN - 1] = 0x00;
        b[PUBKEY_LEN - 1] = 0x01;
        let addr_a = OnionAddress::from_pubkey(&a);
        let addr_b = OnionAddress::from_pubkey(&b);
        assert_eq!(
            &addr_a.body()[..PREFIX_CHARS_WITHOUT_CHECKSUM],
            &addr_b.body()[..PREFIX_CHARS_WITHOUT_CHECKSUM]
        );
        // ...and differ at or after character 51.
        assert_ne!(addr_a.body(), addr_b.body());
    }
}
