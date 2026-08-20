//! Compilation of patterns into the form the search kernel consumes.
//!
//! # What the device is given, and why
//!
//! Langsec rule 4: the device is not a parser. It never sees pattern text,
//! never reads a length out of data, and performs no allocation. It receives
//! fixed-size integers whose meaning was fully determined on the host.
//!
//! Compilation splits a pattern into two parts:
//!
//! 1. A **prefilter**: a `(mask, target)` pair over the leading
//!    [`PREFILTER_BYTES`] bytes of the public key, packed big-endian into a
//!    `u64`. A candidate passes when `key & mask == target`. This is a
//!    *necessary* condition, derived only from the pattern's single-character
//!    positions — literals, and classes that happen to admit one character.
//!    It is what runs on every one of the ~10^9 candidates per second.
//!
//! 2. A **residual**: the per-position character sets that the prefilter could
//!    not express — wildcards constrain nothing and are dropped entirely, while
//!    multi-character classes and any position beyond byte 8 are kept as
//!    explicit 32-bit sets. This runs only on candidates that already passed the
//!    prefilter, which for any realistic pattern is a vanishingly rare event.
//!
//! The prefilter is exact whenever a pattern has no multi-character classes and
//! fits within [`PREFILTER_BYTES`]; [`CompiledPattern::prefilter_is_exact`]
//! reports which. When it is not exact, correctness still holds because the
//! residual check is mandatory before a hit is reported.
//!
//! # Why 8 bytes
//!
//! `8 bytes = 64 bits = 12.8 base32 characters`. A 12-character prefix has
//! difficulty `32^12 ≈ 1.15 × 10^18`; at 10^9 addresses per second that is
//! roughly 36 years. The prefilter therefore covers every pattern that can
//! actually be searched to completion, and a single `u64` compare — the
//! cheapest possible test — suffices in the hot loop.
//!
//! # Grouping
//!
//! Many patterns are searched at once by grouping them by identical mask and
//! sorting each group's targets. A candidate is then tested with one mask and
//! one binary search per group, so searching ten thousand patterns of the same
//! shape costs barely more than searching one. See [`PatternSet`].

use std::collections::BTreeMap;

use crate::address::PUBKEY_LEN;
use crate::base32::BITS_PER_CHAR;
use crate::pattern::parse::Pattern;

/// Number of leading public-key bytes covered by the `u64` prefilter.
pub const PREFILTER_BYTES: usize = 8;

/// Number of base32 characters fully covered by the prefilter.
///
/// `floor(64 / 5) = 12`. Character 12 is only partly inside the first 8 bytes,
/// so it is left to the residual.
pub const PREFILTER_CHARS: usize = (PREFILTER_BYTES * 8) / BITS_PER_CHAR;

/// One position that the prefilter could not express exactly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResidualPosition {
    /// Index of the address character this constrains.
    pub char_index: u32,
    /// Bit set of admitted 5-bit values.
    pub allowed: u32,
}

/// A pattern in the form the search kernel consumes.
///
/// Construction is only via [`CompiledPattern::compile`], and the invariants —
/// `mask` covering exactly the bits `target` constrains, residual positions
/// sorted and in range — hold for every value of this type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CompiledPattern {
    mask: u64,
    target: u64,
    residual: Vec<ResidualPosition>,
    char_len: usize,
    source: String,
}

impl CompiledPattern {
    /// Compile a parsed pattern.
    ///
    /// Total: every [`Pattern`] compiles. The parser has already rejected
    /// everything that could not be represented here, which is why this
    /// function cannot fail (langsec rule 2 — recognition happens once, up
    /// front, and later stages are total).
    #[must_use]
    pub fn compile(pattern: &Pattern) -> Self {
        let mut mask: u64 = 0;
        let mut target: u64 = 0;
        let mut residual = Vec::new();

        for (index, atom) in pattern.atoms().iter().enumerate() {
            let class = atom.class();
            if class.is_any() {
                // A wildcard constrains nothing; it contributes to neither the
                // prefilter nor the residual. It still occupies a position, so
                // subsequent atoms keep their character indices.
                continue;
            }
            let bit_offset = index * BITS_PER_CHAR;
            let fits_in_prefilter = bit_offset + BITS_PER_CHAR <= PREFILTER_BYTES * 8;
            match (class.as_single(), fits_in_prefilter) {
                (Some(c), true) => {
                    // Character `index` occupies bits [5i, 5i+5) counting from
                    // the most significant bit of byte 0, which in a big-endian
                    // u64 is a left shift of (64 - 5 - 5i).
                    let shift = 64 - BITS_PER_CHAR - bit_offset;
                    mask |= 0x1fu64 << shift;
                    target |= u64::from(c.value()) << shift;
                }
                _ => residual.push(ResidualPosition {
                    char_index: u32::try_from(index).unwrap_or(u32::MAX),
                    allowed: class.bits(),
                }),
            }
        }

        Self {
            mask,
            target,
            residual,
            char_len: pattern.char_len(),
            source: pattern.source().to_owned(),
        }
    }

    /// The prefilter mask, applied to the first 8 key bytes read big-endian.
    #[must_use]
    pub const fn mask(&self) -> u64 {
        self.mask
    }

    /// The prefilter target: what `key & mask` must equal.
    #[must_use]
    pub const fn target(&self) -> u64 {
        self.target
    }

    /// Positions the prefilter could not express.
    #[must_use]
    pub fn residual(&self) -> &[ResidualPosition] {
        &self.residual
    }

    /// Whether passing the prefilter is sufficient, not merely necessary.
    #[must_use]
    pub fn prefilter_is_exact(&self) -> bool {
        self.residual.is_empty()
    }

    /// Number of address characters the pattern spans.
    #[must_use]
    pub const fn char_len(&self) -> usize {
        self.char_len
    }

    /// The pattern's source text.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The reference matching predicate.
    ///
    /// This is the *definition* of what it means for a key to match a pattern.
    /// The GPU kernel is an optimised implementation of exactly this function,
    /// and every hit it reports is re-tested here before a key is written to
    /// disk. If the two ever disagree, this one is right.
    #[must_use]
    pub fn matches_pubkey(&self, pubkey: &[u8; PUBKEY_LEN]) -> bool {
        if key_prefix_u64(pubkey) & self.mask != self.target {
            return false;
        }
        self.residual
            .iter()
            .all(|r| (r.allowed >> char_value(pubkey, r.char_index as usize)) & 1 == 1)
    }
}

/// The first [`PREFILTER_BYTES`] bytes of a key, big-endian.
#[must_use]
pub fn key_prefix_u64(pubkey: &[u8; PUBKEY_LEN]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&pubkey[..PREFILTER_BYTES]);
    u64::from_be_bytes(bytes)
}

/// Extract the 5-bit value that base32 character `index` encodes.
///
/// Characters run most-significant-bit first across the key, so character `i`
/// occupies bits `[5i, 5i+5)` where bit 0 is the top bit of byte 0. Reading
/// three bytes covers any 5-bit field regardless of alignment. Bits past the end
/// of the key read as zero, matching the zero padding [`crate::base32::encode`]
/// appends.
#[must_use]
pub fn char_value(pubkey: &[u8; PUBKEY_LEN], index: usize) -> u8 {
    let bit = index * BITS_PER_CHAR;
    let byte = bit / 8;
    let offset = bit % 8;
    let b0 = u32::from(*pubkey.get(byte).unwrap_or(&0));
    let b1 = u32::from(*pubkey.get(byte + 1).unwrap_or(&0));
    let window = (b0 << 8) | b1;
    ((window >> (11 - offset)) & 0x1f) as u8
}

/// A group of compiled patterns sharing one prefilter mask.
///
/// Targets are sorted so the device can binary-search them, and deduplicated so
/// a repeated pattern does not cost extra work.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MaskGroup {
    mask: u64,
    /// Sorted, deduplicated targets. Parallel to [`Self::pattern_indices`].
    targets: Vec<u64>,
    /// For each target, the indices of patterns in the set that produced it.
    pattern_indices: Vec<Vec<u32>>,
}

impl MaskGroup {
    /// The shared mask.
    #[must_use]
    pub const fn mask(&self) -> u64 {
        self.mask
    }

    /// The sorted target list.
    #[must_use]
    pub fn targets(&self) -> &[u64] {
        &self.targets
    }

    /// Patterns that produced the target at `slot`.
    #[must_use]
    pub fn patterns_for_slot(&self, slot: usize) -> &[u32] {
        self.pattern_indices
            .get(slot)
            .map_or(&[][..], Vec::as_slice)
    }
}

/// A complete, compiled search target: every pattern the run is looking for.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PatternSet {
    patterns: Vec<CompiledPattern>,
    groups: Vec<MaskGroup>,
}

impl PatternSet {
    /// Compile a collection of parsed patterns into grouped device tables.
    ///
    /// # Errors
    ///
    /// Returns [`PatternSetError::Empty`] if no patterns were supplied — a
    /// search with no target would never terminate.
    pub fn compile(patterns: &[Pattern]) -> Result<Self, PatternSetError> {
        if patterns.is_empty() {
            return Err(PatternSetError::Empty);
        }
        let compiled: Vec<CompiledPattern> =
            patterns.iter().map(CompiledPattern::compile).collect();

        // Group by mask, then by target within the mask, accumulating which
        // patterns map to each. BTreeMap gives a deterministic ordering, so the
        // bytes uploaded to the device are a pure function of the input — which
        // makes runs reproducible and the PTX cache key meaningful.
        let mut by_mask: BTreeMap<u64, BTreeMap<u64, Vec<u32>>> = BTreeMap::new();
        for (index, c) in compiled.iter().enumerate() {
            let slot = by_mask.entry(c.mask()).or_default().entry(c.target()).or_default();
            slot.push(u32::try_from(index).unwrap_or(u32::MAX));
        }

        let groups = by_mask
            .into_iter()
            .map(|(mask, targets)| {
                let (targets, pattern_indices) = targets.into_iter().unzip();
                MaskGroup {
                    mask,
                    targets,
                    pattern_indices,
                }
            })
            .collect();

        Ok(Self {
            patterns: compiled,
            groups,
        })
    }

    /// The compiled patterns, in input order.
    #[must_use]
    pub fn patterns(&self) -> &[CompiledPattern] {
        &self.patterns
    }

    /// The mask groups, in ascending mask order.
    #[must_use]
    pub fn groups(&self) -> &[MaskGroup] {
        &self.groups
    }

    /// Find every pattern in the set that a key matches.
    ///
    /// Reference implementation, mirroring what the kernel does: prefilter by
    /// group, then confirm residuals. Used to verify device hits.
    #[must_use]
    pub fn matching_patterns(&self, pubkey: &[u8; PUBKEY_LEN]) -> Vec<u32> {
        let key = key_prefix_u64(pubkey);
        let mut out = Vec::new();
        for group in &self.groups {
            let probe = key & group.mask;
            if let Ok(slot) = group.targets.binary_search(&probe) {
                for &index in group.patterns_for_slot(slot) {
                    if self
                        .patterns
                        .get(index as usize)
                        .is_some_and(|p| p.matches_pubkey(pubkey))
                    {
                        out.push(index);
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Base-2 log of the expected trials before *any* pattern in the set hits.
    ///
    /// Treats the patterns as independent events, which slightly understates
    /// difficulty when patterns overlap. Used only for progress estimation.
    #[must_use]
    pub fn difficulty_log2(&self) -> f64 {
        let combined: f64 = self
            .patterns
            .iter()
            .map(|p| {
                let per_pattern = p
                    .residual
                    .iter()
                    .map(|r| f64::from(r.allowed.count_ones()) / 32.0)
                    .product::<f64>();
                let fixed_chars = (p.mask.count_ones() / BITS_PER_CHAR as u32) as f64;
                per_pattern * 2f64.powf(-5.0 * fixed_chars)
            })
            .sum();
        if combined <= 0.0 {
            f64::INFINITY
        } else {
            -combined.log2()
        }
    }
}

/// Why a set of patterns could not be compiled.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PatternSetError {
    /// No patterns were supplied.
    #[error("no patterns supplied; a search needs at least one target")]
    Empty,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::OnionAddress;
    use crate::pattern::parse::Pattern;

    fn compile(src: &str) -> CompiledPattern {
        CompiledPattern::compile(&Pattern::parse(src).unwrap_or_else(|e| panic!("{src}: {e}")))
    }

    #[test]
    fn prefilter_covers_twelve_characters() {
        assert_eq!(PREFILTER_CHARS, 12);
        // Twelve literals fit entirely in the prefilter, so it is exact.
        let c = compile("aaaaaaaaaaaa");
        assert!(c.prefilter_is_exact());
        assert_eq!(c.mask().count_ones(), 60); // 12 characters x 5 bits
        // A thirteenth character straddles the 64-bit boundary and becomes a
        // residual, since a u64 compare cannot express it.
        let c13 = compile("aaaaaaaaaaaaa");
        assert!(!c13.prefilter_is_exact());
        assert_eq!(c13.mask().count_ones(), 60);
        assert_eq!(c13.residual().len(), 1);
        assert_eq!(c13.residual()[0].char_index, 12);
    }

    #[test]
    fn mask_and_target_use_big_endian_character_order() {
        // 'b' has value 1. As the first character it occupies the top 5 bits.
        let c = compile("b");
        assert_eq!(c.mask(), 0x1fu64 << 59);
        assert_eq!(c.target(), 1u64 << 59);
        // As the second character it occupies the next 5 bits down.
        let c2 = compile("ab");
        assert_eq!(c2.mask(), (0x1fu64 << 59) | (0x1fu64 << 54));
        assert_eq!(c2.target(), 1u64 << 54); // 'a' is 0, so only 'b' shows
    }

    #[test]
    fn wildcards_contribute_nothing_but_hold_position() {
        let c = compile("?b");
        // Only the second character is constrained.
        assert_eq!(c.mask(), 0x1fu64 << 54);
        assert_eq!(c.target(), 1u64 << 54);
        assert_eq!(c.char_len(), 2);
        assert!(c.prefilter_is_exact(), "a wildcard needs no residual check");
    }

    #[test]
    fn multi_character_classes_become_residuals() {
        let c = compile("a[bc]d");
        // 'a' and 'd' are fixed; the class is not expressible as a masked compare.
        assert_eq!(c.mask().count_ones(), 10);
        assert_eq!(c.residual().len(), 1);
        assert_eq!(c.residual()[0].char_index, 1);
        assert_eq!(c.residual()[0].allowed.count_ones(), 2);
        assert!(!c.prefilter_is_exact());
    }

    #[test]
    fn single_member_class_folds_into_the_prefilter() {
        // `[b]` denotes exactly what `b` denotes, so it must compile identically.
        assert_eq!(compile("[b]").mask(), compile("b").mask());
        assert_eq!(compile("[b]").target(), compile("b").target());
        assert!(compile("[b]").prefilter_is_exact());
    }

    #[test]
    fn char_value_extracts_the_same_characters_base32_emits() {
        // The device reads 5-bit fields out of key bytes; the address is text.
        // They must be the same characters.
        for seed in 0u8..32 {
            let key = [seed.wrapping_mul(37).wrapping_add(11); PUBKEY_LEN];
            let address = OnionAddress::from_pubkey(&key);
            let body = address.body().as_bytes();
            for (index, &ch) in body
                .iter()
                .enumerate()
                .take(crate::address::PREFIX_CHARS_WITHOUT_CHECKSUM)
            {
                let from_bits = char_value(&key, index);
                let from_text = crate::base32::Base32Char::from_ascii(ch)
                    .expect("address is base32")
                    .value();
                assert_eq!(from_bits, from_text, "character {index} of {address}");
            }
        }
    }

    #[test]
    fn grouping_is_deterministic_and_deduplicates() {
        let patterns: Vec<Pattern> = ["abc", "abd", "abc", "xy?z"]
            .iter()
            .map(|s| Pattern::parse(s).expect("valid"))
            .collect();
        let set = PatternSet::compile(&patterns).expect("non-empty");

        // "abc"/"abd" share a mask (3 literals); "xy?z" has a different one.
        assert_eq!(set.groups().len(), 2);
        // Targets within a group are sorted, so the device can binary-search.
        for g in set.groups() {
            assert!(g.targets().windows(2).all(|w| w[0] < w[1]), "targets sorted");
        }
        // Note both groups happen to have 15 mask bits ("xy?z" also fixes
        // three characters), so the group must be selected by mask *value*.
        let abc_mask = compile("abc").mask();
        let abd_mask = compile("abd").mask();
        assert_eq!(abc_mask, abd_mask, "same shape, so same group");
        let three_char = set
            .groups()
            .iter()
            .find(|g| g.mask() == abc_mask)
            .expect("group present");

        // "abc" and "abd" differ, so the group holds two distinct targets...
        assert_eq!(three_char.targets().len(), 2);
        // ...and the duplicate "abc" collapses into one of them, carrying two
        // pattern ids rather than occupying a second slot.
        let total_ids: usize = (0..three_char.targets().len())
            .map(|s| three_char.patterns_for_slot(s).len())
            .sum();
        assert_eq!(total_ids, 3, "abc, abd, abc");
        let abc_slot = three_char
            .targets()
            .binary_search(&compile("abc").target())
            .expect("abc present");
        assert_eq!(three_char.patterns_for_slot(abc_slot), &[0, 2]);

        // Recompiling the same input yields byte-identical tables.
        let again = PatternSet::compile(&patterns).expect("non-empty");
        assert_eq!(set, again);
    }

    #[test]
    fn empty_set_is_rejected() {
        assert_eq!(PatternSet::compile(&[]), Err(PatternSetError::Empty));
    }

    #[test]
    fn set_difficulty_accounts_for_multiple_patterns() {
        // Two independent 5-character patterns are about twice as likely to hit
        // as one, i.e. one bit easier.
        let one = PatternSet::compile(&[Pattern::parse("abcde").expect("valid")]).expect("ok");
        let two = PatternSet::compile(&[
            Pattern::parse("abcde").expect("valid"),
            Pattern::parse("fghij").expect("valid"),
        ])
        .expect("ok");
        assert!((one.difficulty_log2() - 25.0).abs() < 1e-9);
        assert!((two.difficulty_log2() - 24.0).abs() < 1e-9);
    }
}
