//! The property that makes the whole system trustworthy.
//!
//! `honion` searches for keys by masking raw public-key bytes on a GPU. Users,
//! however, care about the *text* of an onion address. These tests establish
//! that the two views coincide: a key passes [`CompiledPattern::matches_pubkey`]
//! if and only if the address derived from that key, spelled out in base32,
//! literally begins with characters the pattern admits.
//!
//! Everything else in this project is an optimisation of that predicate. If
//! these tests hold, a fast implementation that agrees with `matches_pubkey`
//! is correct by construction; if they fail, no amount of GPU throughput
//! matters.

// Integration tests are separate crates, so the `cfg(test)` relaxation in the
// library does not reach them; the same reasoning applies here.
#![allow(clippy::indexing_slicing, clippy::panic, clippy::unwrap_used, clippy::expect_used)]

use honion_core::address::{OnionAddress, PUBKEY_LEN};
use honion_core::base32::Base32Char;
use honion_core::pattern::{Atom, CompiledPattern, Pattern, PatternSet};
use proptest::prelude::*;

/// Independent reference: does `address` begin with something `pattern` admits?
///
/// Deliberately naive — it walks characters of the finished address string and
/// consults each atom directly. It shares no code with the compiler, so
/// agreement between the two is evidence rather than tautology.
fn text_matches(pattern: &Pattern, address: &OnionAddress) -> bool {
    let body = address.body().as_bytes();
    pattern.atoms().iter().enumerate().all(|(i, atom)| {
        let Some(&byte) = body.get(i) else { return false };
        let Some(c) = Base32Char::from_ascii(byte) else {
            return false;
        };
        match atom {
            Atom::Literal(expected) => c == *expected,
            Atom::Wildcard => true,
            Atom::Class(k) => k.admits_value(c.value()),
        }
    })
}

/// Generate a pattern of `len` atoms, mixing literals, wildcards and classes.
fn pattern_strategy(max_len: usize) -> impl Strategy<Value = Pattern> {
    let atom = prop_oneof![
        // Literals are weighted heavily: they are the common case and the one
        // the prefilter must get exactly right.
        6 => (0u8..32).prop_map(|v| {
            (Base32Char::from_value(v).expect("v < 32").as_ascii() as char).to_string()
        }),
        2 => Just("?".to_owned()),
        2 => prop::collection::btree_set(0u8..32, 1..6).prop_map(|set| {
            let inner: String = set
                .into_iter()
                .map(|v| Base32Char::from_value(v).expect("v < 32").as_ascii() as char)
                .collect();
            format!("[{inner}]")
        }),
    ];
    prop::collection::vec(atom, 1..=max_len)
        .prop_map(|parts| parts.concat())
        .prop_map(|src| Pattern::parse(&src).expect("generated patterns are well formed"))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// The central claim: masking bytes and reading characters agree.
    #[test]
    fn compiled_matcher_agrees_with_address_text(
        key in prop::array::uniform32(any::<u8>()),
        pattern in pattern_strategy(20),
    ) {
        let address = OnionAddress::from_pubkey(&key);
        let compiled = CompiledPattern::compile(&pattern);
        prop_assert_eq!(
            compiled.matches_pubkey(&key),
            text_matches(&pattern, &address),
            "key {:02x?} pattern {} address {}",
            key, pattern, address
        );
    }

    /// Patterns spanning the full searchable width must also agree — this is
    /// where residual handling beyond the 8-byte prefilter is exercised.
    #[test]
    fn agreement_holds_past_the_prefilter_window(
        key in prop::array::uniform32(any::<u8>()),
        pattern in pattern_strategy(51),
    ) {
        let address = OnionAddress::from_pubkey(&key);
        let compiled = CompiledPattern::compile(&pattern);
        prop_assert_eq!(
            compiled.matches_pubkey(&key),
            text_matches(&pattern, &address)
        );
    }

    /// A pattern taken from a real address must match the key it came from.
    /// This is the "does it actually find what we asked for" direction.
    #[test]
    fn prefix_of_an_address_matches_its_own_key(
        key in prop::array::uniform32(any::<u8>()),
        len in 1usize..=51,
    ) {
        let address = OnionAddress::from_pubkey(&key);
        let src: String = address.body().chars().take(len).collect();
        let pattern = Pattern::parse(&src).expect("a real address is valid base32");
        let compiled = CompiledPattern::compile(&pattern);
        prop_assert!(
            compiled.matches_pubkey(&key),
            "address {} should match its own {}-character prefix {}",
            address, len, src
        );
    }

    /// The prefilter must never reject a key the full predicate accepts.
    /// (It may accept keys the full predicate rejects — that is the whole point
    /// of a prefilter — but a false negative would silently lose hits.)
    #[test]
    fn prefilter_never_produces_a_false_negative(
        key in prop::array::uniform32(any::<u8>()),
        pattern in pattern_strategy(30),
    ) {
        let compiled = CompiledPattern::compile(&pattern);
        if compiled.matches_pubkey(&key) {
            let probe = honion_core::pattern::key_prefix_u64(&key) & compiled.mask();
            prop_assert_eq!(probe, compiled.target());
        }
    }

    /// Grouping many patterns must not change any individual verdict.
    #[test]
    fn set_matching_agrees_with_individual_matching(
        key in prop::array::uniform32(any::<u8>()),
        patterns in prop::collection::vec(pattern_strategy(12), 1..8),
    ) {
        let set = PatternSet::compile(&patterns).expect("non-empty");
        let via_set = set.matching_patterns(&key);
        let expected: Vec<u32> = patterns
            .iter()
            .enumerate()
            .filter(|(_, p)| CompiledPattern::compile(p).matches_pubkey(&key))
            .map(|(i, _)| u32::try_from(i).expect("small"))
            .collect();
        prop_assert_eq!(via_set, expected);
    }
}

/// A wildcard-only pattern matches every key: worth pinning explicitly, since
/// it is the degenerate case where mask and target are both zero and a careless
/// binary search would misbehave.
#[test]
fn all_wildcards_matches_everything() {
    let pattern = Pattern::parse("????????").expect("valid");
    let compiled = CompiledPattern::compile(&pattern);
    assert_eq!(compiled.mask(), 0);
    assert_eq!(compiled.target(), 0);
    for seed in 0u8..64 {
        assert!(compiled.matches_pubkey(&[seed; PUBKEY_LEN]));
    }
    let set = PatternSet::compile(&[pattern]).expect("non-empty");
    assert_eq!(set.matching_patterns(&[0u8; PUBKEY_LEN]), vec![0]);
}

/// The DuckDuckGo address is a real 10-character vanity result. Searching for
/// its own prefix must match its own key — a real-world end-to-end check of the
/// predicate against data this project did not produce.
#[test]
fn real_vanity_address_matches_its_prefix() {
    let address = OnionAddress::parse(
        "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion",
    )
    .expect("valid address");
    let pattern = Pattern::parse("duckduckgo").expect("valid pattern");
    let compiled = CompiledPattern::compile(&pattern);
    assert!(compiled.matches_pubkey(address.pubkey()));
    // ...and its difficulty is what the table in the docs claims.
    assert!((pattern.difficulty_log2() - 50.0).abs() < 1e-9);
}
