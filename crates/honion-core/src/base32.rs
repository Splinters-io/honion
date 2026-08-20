//! RFC 4648 base32, lowercase, unpadded — the *only* base32 implementation in
//! this workspace.
//!
//! # Why this module exists in isolation
//!
//! Langsec rule 6: one encoder, one decoder, proven inverse. Tor's v3 address is
//! defined as `base32(pubkey ‖ checksum ‖ version)`, and a vanity search is
//! precisely a search over the *output* of this function. If the encoder here
//! disagreed by one bit with the bit-masking performed on the GPU, the search
//! would find keys whose addresses do not match the requested pattern — and the
//! failure would be silent. So there is exactly one definition of the mapping
//! from bits to characters, and everything (address formatting, pattern
//! compilation, the mask/target computation uploaded to the device) derives
//! from it.
//!
//! # Bit layout
//!
//! Characters are emitted most-significant-bit first. Character `i` of the
//! output covers input bits `[5i, 5i+5)`, where bit 0 is the *most* significant
//! bit of byte 0. This "big-endian bitstream" convention is what makes prefix
//! matching cheap: character `i` depends only on input bytes
//! `[floor(5i/8), floor((5i+4)/8)]`, so a prefix of the address is a function of
//! a prefix of the bytes. See [`prefix_bytes_needed`].

use core::fmt;

/// The RFC 4648 base32 alphabet in lowercase, indexed by 5-bit value.
///
/// Tor addresses are always lowercase; we neither emit nor accept uppercase, so
/// that the encoding is a bijection on the character set rather than a
/// many-to-one map. Accepting both cases would make `decode` non-injective and
/// give two spellings of the same address — a classic parser-differential
/// hazard.
const ALPHABET: [u8; 32] = *b"abcdefghijklmnopqrstuvwxyz234567";

/// Number of bits each base32 character encodes.
pub const BITS_PER_CHAR: usize = 5;

/// A single valid base32 character.
///
/// Holding one of these is proof that the byte is in [`ALPHABET`]. The inner
/// field is private, so the only way to obtain one is [`Base32Char::from_ascii`]
/// or [`Base32Char::from_value`] — there is no path that constructs an invalid
/// instance (langsec rule 3: parse, don't validate).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Base32Char(u8);

impl Base32Char {
    /// Recognise an ASCII byte as a base32 character.
    ///
    /// Returns `None` for anything outside the lowercase alphabet, including
    /// uppercase forms and the RFC 4648 padding byte `=`.
    #[must_use]
    pub const fn from_ascii(byte: u8) -> Option<Self> {
        match byte {
            b'a'..=b'z' => Some(Self(byte - b'a')),
            b'2'..=b'7' => Some(Self(byte - b'2' + 26)),
            _ => None,
        }
    }

    /// Build a character from its 5-bit value.
    ///
    /// Returns `None` if `value >= 32`.
    #[must_use]
    pub const fn from_value(value: u8) -> Option<Self> {
        if value < 32 { Some(Self(value)) } else { None }
    }

    /// The 5-bit value this character encodes, in `0..32`.
    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// The ASCII byte for this character.
    #[must_use]
    // `self.0 < 32` is an invariant established by every constructor, and
    // `ALPHABET` has exactly 32 entries, so this index is in range for every
    // value of this type. That is the whole point of the newtype: the check
    // happened once, at construction, and does not have to happen again here.
    #[allow(clippy::indexing_slicing, reason = "index bounded by the type invariant")]
    pub const fn as_ascii(self) -> u8 {
        ALPHABET[self.0 as usize]
    }
}

impl fmt::Debug for Base32Char {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Base32Char({:?})", self.as_ascii() as char)
    }
}

impl fmt::Display for Base32Char {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(core::str::from_utf8(&[self.as_ascii()]).unwrap_or("?"))
    }
}

/// Why a byte string failed to be recognised as unpadded lowercase base32.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    /// A byte was not in the lowercase base32 alphabet.
    #[error("byte {byte:#04x} at offset {offset} is not a lowercase base32 character")]
    BadCharacter {
        /// Zero-based offset of the offending byte.
        offset: usize,
        /// The offending byte.
        byte: u8,
    },
    /// The character count cannot be produced by encoding any byte string.
    ///
    /// Unpadded base32 lengths are `n` where `n % 8` is one of 0, 2, 4, 5, 7.
    /// Counts congruent to 1, 3 or 6 mod 8 encode a fractional byte and are
    /// therefore not in the image of [`encode`].
    #[error("length {length} is not a valid unpadded base32 length ({length} mod 8 == {residue})")]
    BadLength {
        /// The offending length.
        length: usize,
        /// `length % 8`.
        residue: usize,
    },
    /// The final character carried non-zero bits beyond the last whole byte.
    ///
    /// Such an input decodes to the same bytes as a different, canonical
    /// spelling. Rejecting it keeps `decode` injective.
    #[error("trailing bits in final character are non-zero; encoding is non-canonical")]
    NonCanonicalTrailingBits,
}

/// Number of base32 characters produced by encoding `n` bytes.
#[must_use]
pub const fn encoded_len(n: usize) -> usize {
    // 8 characters per 5 bytes, rounding up.
    n.div_ceil(5) * 8 - match n % 5 {
        0 => 0,
        1 => 6,
        2 => 4,
        3 => 3,
        _ => 1, // n % 5 == 4
    }
}

/// Number of leading input bytes that a prefix of `chars` characters depends on.
///
/// Because character `i` covers bits `[5i, 5i+5)`, a prefix of `chars`
/// characters is fully determined by the first `ceil(5 * chars / 8)` bytes.
/// This is the function that lets the GPU test a pattern against a raw public
/// key without ever encoding anything.
#[must_use]
pub const fn prefix_bytes_needed(chars: usize) -> usize {
    (chars * BITS_PER_CHAR).div_ceil(8)
}

/// Encode bytes as unpadded lowercase base32.
///
/// Total: every byte string has an encoding.
#[must_use]
pub fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(encoded_len(bytes.len()));
    // A sliding window of bits: `acc` holds `nbits` unconsumed low-order bits.
    let mut acc: u32 = 0;
    let mut nbits: u32 = 0;
    for &byte in bytes {
        acc = (acc << 8) | u32::from(byte);
        nbits += 8;
        while nbits >= BITS_PER_CHAR as u32 {
            nbits -= BITS_PER_CHAR as u32;
            let value = ((acc >> nbits) & 0x1f) as u8;
            // `value < 32`, so `from_value` cannot fail.
            if let Some(c) = Base32Char::from_value(value) {
                out.push(c.as_ascii() as char);
            }
        }
    }
    if nbits > 0 {
        // Pad the final partial group with zero bits, per RFC 4648.
        let value = ((acc << (BITS_PER_CHAR as u32 - nbits)) & 0x1f) as u8;
        if let Some(c) = Base32Char::from_value(value) {
            out.push(c.as_ascii() as char);
        }
    }
    out
}

/// Decode unpadded lowercase base32 into bytes.
///
/// Rejects non-alphabet bytes, lengths outside the image of [`encode`], and
/// non-canonical trailing bits — so that `decode` is injective and
/// `encode(decode(s)) == s` for every accepted `s`.
///
/// # Errors
///
/// See [`DecodeError`].
pub fn decode(input: &str) -> Result<Vec<u8>, DecodeError> {
    let bytes = input.as_bytes();
    let residue = bytes.len() % 8;
    if matches!(residue, 1 | 3 | 6) {
        return Err(DecodeError::BadLength {
            length: bytes.len(),
            residue,
        });
    }

    let mut out = Vec::with_capacity(bytes.len() * BITS_PER_CHAR / 8);
    let mut acc: u32 = 0;
    let mut nbits: u32 = 0;
    for (offset, &byte) in bytes.iter().enumerate() {
        let c = Base32Char::from_ascii(byte)
            .ok_or(DecodeError::BadCharacter { offset, byte })?;
        acc = (acc << BITS_PER_CHAR) | u32::from(c.value());
        nbits += BITS_PER_CHAR as u32;
        if nbits >= 8 {
            nbits -= 8;
            out.push(((acc >> nbits) & 0xff) as u8);
        }
    }
    // Any bits left over must be the zero padding that `encode` emits.
    if nbits > 0 && (acc & ((1 << nbits) - 1)) != 0 {
        return Err(DecodeError::NonCanonicalTrailingBits);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc4648_vectors_lowercased() {
        // RFC 4648 §10 test vectors. The RFC gives them uppercase and padded;
        // ours are lowercase and unpadded, so `=` padding is stripped.
        let cases: &[(&[u8], &str)] = &[
            (b"", ""),
            (b"f", "my"),
            (b"fo", "mzxq"),
            (b"foo", "mzxw6"),
            (b"foob", "mzxw6yq"),
            (b"fooba", "mzxw6ytb"),
            (b"foobar", "mzxw6ytboi"),
        ];
        for (raw, expected) in cases {
            assert_eq!(&encode(raw), expected, "encoding {raw:?}");
            assert_eq!(&decode(expected).expect("valid"), raw, "decoding {expected:?}");
        }
    }

    #[test]
    fn encoded_len_matches_encode() {
        for n in 0..64usize {
            let input = vec![0xa5u8; n];
            assert_eq!(encoded_len(n), encode(&input).len(), "n = {n}");
        }
    }

    #[test]
    fn thirty_five_bytes_gives_fifty_six_chars() {
        // The v3 address case: 32-byte key + 2-byte checksum + 1 version byte.
        assert_eq!(encoded_len(35), 56);
    }

    #[test]
    fn prefix_bytes_needed_is_ceil_five_eighths() {
        assert_eq!(prefix_bytes_needed(0), 0);
        assert_eq!(prefix_bytes_needed(1), 1); // 5 bits -> 1 byte
        assert_eq!(prefix_bytes_needed(8), 5); // 40 bits -> 5 bytes
        assert_eq!(prefix_bytes_needed(51), 32); // 255 bits -> 32 bytes
        // Character 51 is the first that reaches past the 32-byte public key,
        // which is the boundary that makes checksum-free prefix search possible.
        assert_eq!(prefix_bytes_needed(52), 33);
    }

    #[test]
    fn rejects_uppercase() {
        assert!(matches!(
            decode("MZXW6"),
            Err(DecodeError::BadCharacter { offset: 0, byte: b'M' })
        ));
    }

    #[test]
    fn rejects_padding_character() {
        assert!(matches!(
            decode("mzxw6ytb===="),
            Err(DecodeError::BadCharacter { byte: b'=', .. })
        ));
    }

    #[test]
    fn rejects_impossible_lengths() {
        for len in [1usize, 3, 6, 9, 11, 14] {
            let s = "a".repeat(len);
            assert!(
                matches!(decode(&s), Err(DecodeError::BadLength { .. })),
                "length {len} should be rejected"
            );
        }
    }

    #[test]
    fn rejects_non_canonical_trailing_bits() {
        // "my" decodes 'f'; the next character value up sets a padding bit that
        // `encode` would never emit.
        assert_eq!(decode("my").expect("valid"), b"f");
        assert!(matches!(
            decode("mz"),
            Err(DecodeError::NonCanonicalTrailingBits)
        ));
    }

    #[test]
    fn char_roundtrip_over_whole_alphabet() {
        for value in 0..32u8 {
            let c = Base32Char::from_value(value).expect("value < 32");
            let back = Base32Char::from_ascii(c.as_ascii()).expect("alphabet byte");
            assert_eq!(c.value(), back.value());
        }
        assert_eq!(Base32Char::from_value(32), None);
    }

    #[test]
    fn alphabet_has_no_duplicates() {
        let mut seen = [false; 256];
        for &byte in &ALPHABET {
            assert!(!seen[byte as usize], "duplicate {:?} in alphabet", byte as char);
            seen[byte as usize] = true;
        }
    }
}
