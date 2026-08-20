//! Expanded secret keys, and the verification every hit must survive.
//!
//! # The rule
//!
//! Nothing reaches disk until it has been proven correct *without* using the
//! GPU's answer as evidence. [`VerifiedKey::verify`] rebuilds the key from the
//! host's own secret scalar, re-derives the public key with `curve25519-dalek`,
//! recomputes the address with [`honion_core`], re-checks the pattern with the
//! reference matcher, and produces and verifies a signature. Only then is a key
//! considered found.
//!
//! This costs a fraction of a millisecond and runs once per hit, against
//! billions of candidates per second on the device — so it is free in practice.
//! What it buys is that a miscompiled kernel, a bit flip in 96 GB of VRAM, or a
//! mistake in this project's own field arithmetic cannot produce a key file
//! whose contents disagree with its hostname. That failure would be discovered
//! only when a hidden service refused to start, or worse, would leave someone
//! publishing an address they cannot actually serve.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use zeroize::{Zeroize, ZeroizeOnDrop};

use honion_core::address::{OnionAddress, PUBKEY_LEN};
use honion_core::pattern::PatternSet;

use crate::scalar::{SCALAR_LEN, ScalarError, SecretScalar};

/// Length of the nonce half of an expanded key.
pub const NONCE_LEN: usize = 32;

/// Message signed during verification. Its content is irrelevant; what matters
/// is that signing and verification both succeed with the key about to be
/// written.
const PROBE_MESSAGE: &[u8] = b"honion key self-check";

/// An Ed25519 key in the expanded form Tor stores.
///
/// The scalar is the secret; the nonce half is the deterministic-signature
/// prefix, and is equally secret. Both are zeroized on drop.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ExpandedSecretKey {
    scalar: SecretScalar,
    nonce: [u8; NONCE_LEN],
}

impl ExpandedSecretKey {
    /// Pair a scalar with a fresh random nonce half.
    ///
    /// The nonce is independent of the scalar. In a seed-derived key the two
    /// halves come from one SHA-512 output, but nothing about Ed25519 requires
    /// that: the nonce's only job is to be secret and fixed for the key, so
    /// that signatures are deterministic. Independent randomness satisfies that
    /// at least as well.
    ///
    /// # Errors
    ///
    /// If the system random source is unavailable.
    pub fn new(scalar: SecretScalar) -> Result<Self, ScalarError> {
        let mut nonce = [0u8; NONCE_LEN];
        getrandom::fill(&mut nonce).map_err(|e| ScalarError::Random(e.to_string()))?;
        Ok(Self { scalar, nonce })
    }

    /// Pair a scalar with a specific nonce half. For reading keys back.
    #[must_use]
    pub const fn from_parts(scalar: SecretScalar, nonce: [u8; NONCE_LEN]) -> Self {
        Self { scalar, nonce }
    }

    /// The secret scalar.
    #[must_use]
    pub const fn scalar(&self) -> &SecretScalar {
        &self.scalar
    }

    /// The nonce half.
    #[must_use]
    pub const fn nonce(&self) -> &[u8; NONCE_LEN] {
        &self.nonce
    }

    /// The 64-byte expanded key: scalar followed by nonce.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; SCALAR_LEN + NONCE_LEN] {
        let mut out = [0u8; SCALAR_LEN + NONCE_LEN];
        out[..SCALAR_LEN].copy_from_slice(self.scalar.as_bytes());
        out[SCALAR_LEN..].copy_from_slice(&self.nonce);
        out
    }

    /// Sign a message with this key.
    fn sign(&self, message: &[u8], public: &VerifyingKey) -> Signature {
        let hazmat = ed25519_dalek::hazmat::ExpandedSecretKey {
            scalar: self.scalar.as_dalek(),
            hash_prefix: self.nonce,
        };
        ed25519_dalek::hazmat::raw_sign::<sha2::Sha512>(&hazmat, message, public)
    }
}

/// A key that has passed every check, together with its address.
///
/// This type cannot be constructed except through [`VerifiedKey::verify`]
/// returning `Ok`, so possessing one is proof that the key, the public key and
/// the address all agree and that the key can actually sign.
pub struct VerifiedKey {
    key: ExpandedSecretKey,
    public: [u8; PUBKEY_LEN],
    address: OnionAddress,
}

impl core::fmt::Debug for VerifiedKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The address is public; the key is not.
        write!(f, "VerifiedKey({}, key=<redacted>)", self.address)
    }
}

impl VerifiedKey {
    /// Rebuild and fully check a candidate the device reported.
    ///
    /// `start` is the thread's starting scalar, held by the caller and never
    /// sent to the device; `offset` is the signed step count the device
    /// reported. The candidate is accepted only if all of the following hold:
    ///
    /// 1. `start + 8 * offset` is still a clamped scalar.
    /// 2. Its public key, derived independently with `curve25519-dalek`,
    ///    matches at least one pattern under the reference matcher.
    /// 3. The address derived from that key is well formed.
    /// 4. The key produces a signature that verifies against the public key.
    ///
    /// # Errors
    ///
    /// See [`VerifyError`]. Every variant means something is wrong with the
    /// hardware or with this program, never with the user's input — so callers
    /// should surface these loudly rather than skipping the candidate.
    pub fn verify(
        start: &SecretScalar,
        offset: i64,
        patterns: &PatternSet,
    ) -> Result<Self, VerifyError> {
        // 1. Reconstruct, checking the clamping invariant survives the walk.
        let scalar = start.offset(offset)?;

        // 2. Derive the public key independently of the search kernel.
        let public = scalar.public_key();
        let matched = patterns.matching_patterns(&public);
        if matched.is_empty() {
            return Err(VerifyError::PatternMismatch { offset });
        }

        // 3. Build the address and confirm it round-trips through the parser,
        //    which recomputes the SHA3-256 checksum from scratch.
        let address = OnionAddress::from_pubkey(&public);
        let reparsed = OnionAddress::parse(&address.to_hostname())
            .map_err(|e| VerifyError::AddressMalformed(e.to_string()))?;
        if reparsed != address {
            return Err(VerifyError::AddressMalformed(
                "address did not survive a parse round trip".into(),
            ));
        }

        // 4. Prove the key can sign for that public key. This is the check that
        //    catches a scalar which generates the right point but is somehow
        //    unusable — and it exercises the exact bytes about to be written.
        let key = ExpandedSecretKey::new(scalar)?;
        let verifying = VerifyingKey::from_bytes(&public)
            .map_err(|e| VerifyError::NotAValidPublicKey(e.to_string()))?;
        let signature = key.sign(PROBE_MESSAGE, &verifying);
        verifying
            .verify(PROBE_MESSAGE, &signature)
            .map_err(|e| VerifyError::SignatureFailed(e.to_string()))?;

        Ok(Self {
            key,
            public,
            address,
        })
    }

    /// The verified key.
    #[must_use]
    pub const fn key(&self) -> &ExpandedSecretKey {
        &self.key
    }

    /// The public key.
    #[must_use]
    pub const fn public(&self) -> &[u8; PUBKEY_LEN] {
        &self.public
    }

    /// The onion address.
    #[must_use]
    pub const fn address(&self) -> &OnionAddress {
        &self.address
    }
}

/// Why a reported candidate failed verification.
///
/// None of these are ordinary conditions. Each means the device disagreed with
/// the host, which points at a kernel bug, a hardware fault, or a mistake in
/// this program.
#[derive(Debug, Clone, thiserror::Error)]
pub enum VerifyError {
    /// Reconstructing the scalar failed.
    #[error("could not reconstruct the secret scalar: {0}")]
    Scalar(#[from] ScalarError),
    /// The key the device found does not actually match any pattern.
    #[error(
        "device reported a match at offset {offset}, but the key derived from that \
         position matches no pattern; the kernel and the host reference disagree"
    )]
    PatternMismatch {
        /// The signed offset the device reported.
        offset: i64,
    },
    /// The address could not be built or re-parsed.
    #[error("address construction failed: {0}")]
    AddressMalformed(String),
    /// The derived public key is not a usable Ed25519 verifying key.
    #[error("derived public key was rejected: {0}")]
    NotAValidPublicKey(String),
    /// A signature made with the key did not verify.
    #[error("the key could not produce a valid signature: {0}")]
    SignatureFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use honion_core::pattern::Pattern;

    fn patterns_for(public: &[u8; PUBKEY_LEN], len: usize) -> PatternSet {
        let address = OnionAddress::from_pubkey(public);
        let prefix: String = address.body().chars().take(len).collect();
        let p = Pattern::parse(&prefix).expect("an address prefix is valid base32");
        PatternSet::compile(&[p]).expect("non-empty")
    }

    #[test]
    fn verifies_a_genuine_hit() {
        let start = SecretScalar::generate().expect("randomness");
        let k = 12_345i64;
        let target = start.offset(k).expect("no overflow");
        let set = patterns_for(&target.public_key(), 8);

        let verified = VerifiedKey::verify(&start, k, &set).expect("a real hit must verify");
        assert_eq!(verified.public(), &target.public_key());
        assert_eq!(
            verified.address().body(),
            OnionAddress::from_pubkey(&target.public_key()).body()
        );
        assert!(verified.address().to_hostname().ends_with(".onion"));
    }

    #[test]
    fn rejects_a_wrong_iteration() {
        // Exactly the failure mode verification exists to catch: the device
        // reports a position one step away from the real match.
        let start = SecretScalar::generate().expect("randomness");
        let k = 5_000i64;
        let set = patterns_for(&start.offset(k).expect("no overflow").public_key(), 8);

        match VerifiedKey::verify(&start, k + 1, &set) {
            Err(VerifyError::PatternMismatch { offset }) => assert_eq!(offset, k + 1),
            other => panic!("expected a pattern mismatch, got {other:?}"),
        }
    }

    #[test]
    fn expanded_key_is_scalar_then_nonce() {
        let scalar = SecretScalar::generate().expect("randomness");
        let expected_scalar = *scalar.as_bytes();
        let key = ExpandedSecretKey::new(scalar).expect("randomness");
        let bytes = key.to_bytes();
        assert_eq!(bytes.len(), 64);
        assert_eq!(&bytes[..32], &expected_scalar);
        assert_eq!(&bytes[32..], key.nonce());
    }

    #[test]
    fn signatures_verify_for_generated_keys() {
        for _ in 0..16 {
            let scalar = SecretScalar::generate().expect("randomness");
            let public = scalar.public_key();
            let set = patterns_for(&public, 1);
            // Verifying at k = 0 is the identity case, and still must sign.
            VerifiedKey::verify(&scalar, 0, &set).expect("k = 0 must verify");
        }
    }
}
