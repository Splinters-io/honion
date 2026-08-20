//! Secret scalars: generation, the `+ 8k` walk, and their invariants.
//!
//! # Why scalars and not seeds
//!
//! An Ed25519 key is normally derived from a 32-byte seed: `SHA-512(seed)`
//! gives 64 bytes, the first 32 are clamped into the secret scalar and the rest
//! become the nonce prefix. A vanity search cannot work that way, because it
//! moves through scalar space directly — and SHA-512 cannot be inverted to find
//! a seed that would have produced the scalar it landed on.
//!
//! So `honion` generates the scalar itself, and writes the *expanded* key
//! format, which is what Tor's `hs_ed25519_secret_key` file stores anyway. No
//! functionality is lost: Tor never needs the seed.
//!
//! Since the seed relationship is broken regardless, there is no reason to run
//! SHA-512 at all. A clamped scalar drawn straight from the system CSPRNG is
//! uniform over exactly the same set of values that a hashed-and-clamped seed
//! would produce, and the nonce half is 32 independent random bytes. This is
//! both simpler and one less primitive to get right. See
//! `docs/05-security-model.md` for the argument in full.
//!
//! # Clamping
//!
//! Clamping clears the three low bits and bit 255, and sets bit 254. Two
//! consequences matter here:
//!
//! * Every valid scalar is a multiple of 8, which is why the search can step by
//!   8 and stay in the valid set.
//! * Every valid scalar lies in `[2^254, 2^255)`, so adding `8k` can in
//!   principle carry into bit 255 and leave the set. It essentially never does
//!   — the starting point would have to fall within `2^35` of the top of a
//!   `2^254`-wide range — but [`SecretScalar::offset`] checks rather than
//!   assumes, because "essentially never" is not a property to rely on silently.

use core::fmt;

use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
use curve25519_dalek::scalar::Scalar;
use zeroize::{Zeroize, ZeroizeOnDrop};

use honion_core::address::PUBKEY_LEN;

/// Length of a scalar in bytes.
pub const SCALAR_LEN: usize = 32;

/// A clamped Ed25519 secret scalar.
///
/// Holding one is proof the value is in clamped form: low three bits clear,
/// bit 254 set, bit 255 clear. The bytes are zeroized on drop, and `Debug`
/// deliberately does not print them.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SecretScalar {
    bytes: [u8; SCALAR_LEN],
}

impl SecretScalar {
    /// Draw a fresh clamped scalar from the operating system CSPRNG.
    ///
    /// # Errors
    ///
    /// If the system random source is unavailable. This is not recoverable and
    /// must never be worked around with a weaker source: the entire security of
    /// the resulting key rests on this call.
    pub fn generate() -> Result<Self, ScalarError> {
        let mut bytes = [0u8; SCALAR_LEN];
        getrandom::fill(&mut bytes).map_err(|e| ScalarError::Random(e.to_string()))?;
        Ok(Self::clamp(bytes))
    }

    /// Force arbitrary bytes into clamped form.
    #[must_use]
    pub fn clamp(mut bytes: [u8; SCALAR_LEN]) -> Self {
        bytes[0] &= 248;  // clear the low three bits: the scalar is a multiple of 8
        bytes[31] &= 127; // clear bit 255
        bytes[31] |= 64;  // set bit 254
        Self { bytes }
    }

    /// Recognise bytes that are already clamped.
    ///
    /// # Errors
    ///
    /// [`ScalarError::NotClamped`] if any clamping bit is wrong. Used when
    /// reading a key back, so that a corrupted file is rejected rather than
    /// silently repaired into a different key.
    pub fn from_clamped(bytes: [u8; SCALAR_LEN]) -> Result<Self, ScalarError> {
        let candidate = Self::clamp(bytes);
        if candidate.bytes != bytes {
            return Err(ScalarError::NotClamped);
        }
        Ok(candidate)
    }

    /// The raw scalar bytes, little-endian.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; SCALAR_LEN] {
        &self.bytes
    }

    /// The public key `scalar * B`, compressed.
    #[must_use]
    pub fn public_key(&self) -> [u8; PUBKEY_LEN] {
        (ED25519_BASEPOINT_POINT * self.as_dalek()).compress().to_bytes()
    }

    /// As a reduced `dalek` scalar.
    ///
    /// Reduction modulo the group order does not change the point the scalar
    /// generates, so this is safe for deriving and signing. The *stored* form
    /// stays unreduced, matching what Tor and other Ed25519 implementations
    /// expect in an expanded key file.
    #[must_use]
    pub fn as_dalek(&self) -> Scalar {
        Scalar::from_bytes_mod_order(self.bytes)
    }

    /// The scalar `self + 8m`, the `m`-th step of the search walk.
    ///
    /// `m` is **signed**. The search covers a symmetric range either side of
    /// each starting scalar — the dual addition law produces `base + off` and
    /// `base - off` in the same breath — so a match may lie below where the
    /// thread began.
    ///
    /// # Errors
    ///
    /// [`ScalarError::WalkOverflow`] if the result leaves clamped form. See the
    /// module documentation for why this is checked despite being astronomically
    /// unlikely.
    pub fn offset(&self, m: i64) -> Result<Self, ScalarError> {
        let delta = m
            .checked_mul(8)
            .ok_or(ScalarError::WalkOverflow { offset: m })?;

        let mut out = self.bytes;
        if delta >= 0 {
            let mut carry = delta.unsigned_abs() as u128;
            for byte in &mut out {
                if carry == 0 {
                    break;
                }
                let sum = u128::from(*byte) + (carry & 0xff);
                *byte = (sum & 0xff) as u8;
                carry = (carry >> 8) + (sum >> 8);
            }
            if carry != 0 {
                return Err(ScalarError::WalkOverflow { offset: m });
            }
        } else {
            let mut borrow = delta.unsigned_abs() as u128;
            for byte in &mut out {
                if borrow == 0 {
                    break;
                }
                let (value, underflow) = byte.overflowing_sub((borrow & 0xff) as u8);
                *byte = value;
                borrow = (borrow >> 8) + u128::from(underflow);
            }
            if borrow != 0 {
                return Err(ScalarError::WalkOverflow { offset: m });
            }
        }
        // The walk must land back inside the clamped set: adding or subtracting
        // a multiple of 8 preserves the low bits, but a carry or borrow could
        // disturb the top ones.
        Self::from_clamped(out).map_err(|_| ScalarError::WalkOverflow { offset: m })
    }
}

impl fmt::Debug for SecretScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never print secret material, not even at debug level: logs outlive
        // the process, and a leaked scalar is a permanently compromised onion
        // service.
        f.write_str("SecretScalar(<redacted>)")
    }
}

/// Why a scalar operation failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScalarError {
    /// The system random source failed.
    #[error("could not read from the system random source: {0}")]
    Random(String),
    /// Bytes were not in clamped form.
    #[error("scalar is not in clamped form (low three bits must be clear, bit 254 set, bit 255 clear)")]
    NotClamped,
    /// `self + 8m` left the clamped range.
    #[error(
        "walking {offset} steps from this scalar leaves clamped form; \
         the starting scalar was within 2^35 of an end of the range"
    )]
    WalkOverflow {
        /// The signed step count that left the range.
        offset: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamping_sets_the_bits_it_claims() {
        let s = SecretScalar::clamp([0xff; SCALAR_LEN]);
        assert_eq!(s.as_bytes()[0] & 7, 0, "low three bits clear");
        assert_eq!(s.as_bytes()[31] & 0x80, 0, "bit 255 clear");
        assert_eq!(s.as_bytes()[31] & 0x40, 0x40, "bit 254 set");
    }

    #[test]
    fn clamping_is_idempotent() {
        let once = SecretScalar::clamp([0x5a; SCALAR_LEN]);
        let twice = SecretScalar::clamp(*once.as_bytes());
        assert_eq!(once.as_bytes(), twice.as_bytes());
        // ...and recognising an already-clamped value accepts it unchanged.
        assert_eq!(
            SecretScalar::from_clamped(*once.as_bytes())
                .expect("clamped")
                .as_bytes(),
            once.as_bytes()
        );
    }

    #[test]
    fn unclamped_bytes_are_rejected_rather_than_repaired() {
        // Silently clamping on read would turn a corrupted file into a valid
        // key for a *different* address than its own hostname file claims.
        let mut bad = *SecretScalar::clamp([1; SCALAR_LEN]).as_bytes();
        bad[0] |= 1;
        assert!(matches!(
            SecretScalar::from_clamped(bad),
            Err(ScalarError::NotClamped)
        ));
    }

    #[test]
    fn every_generated_scalar_is_clamped() {
        for _ in 0..256 {
            let s = SecretScalar::generate().expect("system randomness");
            SecretScalar::from_clamped(*s.as_bytes()).expect("generated scalars are clamped");
        }
    }

    #[test]
    fn generated_scalars_differ() {
        // A trivially broken CSPRNG path would show up here immediately.
        let a = SecretScalar::generate().expect("randomness");
        let b = SecretScalar::generate().expect("randomness");
        assert_ne!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn offset_adds_exactly_eight_k() {
        let base = SecretScalar::clamp([0x11; SCALAR_LEN]);
        for k in [0i64, 1, 2, 1000, 65535, 1 << 20, -1, -2, -1000, -(1 << 20)] {
            let moved = base.offset(k).expect("no overflow");
            let expected = if k >= 0 {
                base.as_dalek() + Scalar::from(8u64) * Scalar::from(k.unsigned_abs())
            } else {
                base.as_dalek() - Scalar::from(8u64) * Scalar::from(k.unsigned_abs())
            };
            assert_eq!(moved.as_dalek(), expected, "k = {k}");
            // The result must still be clamped, or the walk left the valid set.
            SecretScalar::from_clamped(*moved.as_bytes()).expect("still clamped");
        }
    }

    #[test]
    fn offset_matches_repeated_point_addition() {
        // The property the whole search rests on: stepping the scalar by 8
        // corresponds to adding 8B to the point.
        let base = SecretScalar::clamp([0x37; SCALAR_LEN]);
        let eight_b = ED25519_BASEPOINT_POINT * Scalar::from(8u64);
        let mut point = ED25519_BASEPOINT_POINT * base.as_dalek();
        for k in 0..64i64 {
            let from_scalar = base.offset(k).expect("no overflow").public_key();
            assert_eq!(from_scalar, point.compress().to_bytes(), "k = {k}");
            point += eight_b;
        }
    }

    #[test]
    fn offset_detects_leaving_the_clamped_range() {
        // Hand-build a scalar sitting at the very top of the clamped range, so
        // that a small step carries into bit 255. This cannot arise by chance,
        // which is exactly why it needs a test.
        let mut bytes = [0xffu8; SCALAR_LEN];
        bytes[0] &= 248;
        bytes[31] = 0x7f;
        let top = SecretScalar { bytes };
        assert!(matches!(
            top.offset(1),
            Err(ScalarError::WalkOverflow { offset: 1 })
        ));

        // ...and the mirror case: a scalar at the bottom of the clamped range,
        // where a single backward step clears bit 254 and leaves the set. The
        // search walks in both directions, so both ends must be checked.
        let mut low = [0u8; SCALAR_LEN];
        low[31] = 0x40;
        let bottom = SecretScalar { bytes: low };
        assert!(matches!(
            bottom.offset(-1),
            Err(ScalarError::WalkOverflow { offset: -1 })
        ));

        // A scalar in the middle of the range is unaffected in either direction.
        assert!(SecretScalar::clamp([0x40; SCALAR_LEN]).offset(1 << 30).is_ok());
        assert!(SecretScalar::clamp([0x40; SCALAR_LEN]).offset(-(1 << 30)).is_ok());
    }

    #[test]
    fn debug_never_prints_key_material() {
        let s = SecretScalar::clamp([0xab; SCALAR_LEN]);
        let shown = format!("{s:?}");
        assert_eq!(shown, "SecretScalar(<redacted>)");
        assert!(!shown.contains("ab"));
    }
}
