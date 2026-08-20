//! The vanity pattern language: grammar and recogniser.
//!
//! # Grammar
//!
//! ```ebnf
//! pattern   = atom , { atom } ;              (* 1 to 51 atoms *)
//! atom      = literal | wildcard | class ;
//! literal   = "a".."z" | "2".."7" ;          (* the base32 alphabet *)
//! wildcard  = "?" ;
//! class     = "[" , [ "^" ] , literal , { literal } , "]" ;
//! ```
//!
//! A pattern denotes a set of onion-address prefixes: atom `i` constrains
//! address character `i`. A `literal` admits one character, a `wildcard` admits
//! all 32, and a `class` admits the listed characters (or, when negated with
//! `^`, all characters *except* those listed).
//!
//! The upper bound of 51 atoms is not arbitrary: it is
//! [`crate::address::PREFIX_CHARS_WITHOUT_CHECKSUM`], the last address character
//! determined by the public key alone. A 52-character pattern would depend on
//! the SHA3-256 checksum and could not be tested by masking a public key, so the
//! language simply does not contain such a sentence.
//!
//! # Why hand-written
//!
//! This is the only place in the workspace that turns text into structure
//! (langsec rule 1). It is a single-pass recogniser over bytes with no
//! backtracking, no regular-expression engine, and no `split`. Every rejection
//! carries the byte offset and what was expected there, and recognition is
//! complete before any consumer sees a value (langsec rule 2): [`Pattern`]
//! cannot be constructed except by [`Pattern::parse`] returning `Ok`.

use core::fmt;

use crate::address::PREFIX_CHARS_WITHOUT_CHECKSUM;
use crate::base32::Base32Char;

/// Greatest number of atoms a pattern may contain.
///
/// See the module documentation for why this is exactly the number of address
/// characters that depend only on the public key.
pub const MAX_PATTERN_ATOMS: usize = PREFIX_CHARS_WITHOUT_CHECKSUM;

/// The set of base32 values a single position may take.
///
/// Represented as a 32-bit set: bit `v` is set when the 5-bit value `v` is
/// admitted. The invariant — at least one bit set — is maintained by the
/// constructors, so a `CharClass` can never denote the empty set. An empty
/// class would make the whole pattern unsatisfiable, and a search that can never
/// terminate is a bug we prefer to reject at parse time.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CharClass(u32);

impl CharClass {
    /// Every base32 character. This is what `?` denotes.
    pub const ANY: Self = Self(u32::MAX);

    /// A class admitting exactly one character.
    #[must_use]
    pub const fn single(c: Base32Char) -> Self {
        Self(1u32 << c.value())
    }

    /// Build a class from a bit set, rejecting the empty set.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits == 0 { None } else { Some(Self(bits)) }
    }

    /// The underlying bit set. Bit `v` set means value `v` is admitted.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether this class admits the given 5-bit value.
    #[must_use]
    pub const fn admits_value(self, value: u8) -> bool {
        value < 32 && (self.0 >> value) & 1 == 1
    }

    /// Whether this class admits exactly one character, and which.
    #[must_use]
    pub const fn as_single(self) -> Option<Base32Char> {
        if self.0.count_ones() == 1 {
            Base32Char::from_value(self.0.trailing_zeros() as u8)
        } else {
            None
        }
    }

    /// Whether this class admits every character (i.e. constrains nothing).
    #[must_use]
    pub const fn is_any(self) -> bool {
        self.0 == u32::MAX
    }

    /// How many characters this class admits.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// Whether the class admits no characters. Always `false` by construction;
    /// provided so `clippy::len_without_is_empty` is satisfied honestly.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Debug for CharClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_any() {
            return f.write_str("?");
        }
        if let Some(c) = self.as_single() {
            return write!(f, "{c}");
        }
        f.write_str("[")?;
        for v in 0..32u8 {
            if self.admits_value(v)
                && let Some(c) = Base32Char::from_value(v)
            {
                write!(f, "{c}")?;
            }
        }
        f.write_str("]")
    }
}

/// One position of a pattern.
///
/// Every atom is ultimately a [`CharClass`]; the distinct variants are retained
/// so that diagnostics and `Display` can echo the user's own syntax back to
/// them rather than a normalised form.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Atom {
    /// A single required character, written literally.
    Literal(Base32Char),
    /// `?` — any character.
    Wildcard,
    /// `[...]` or `[^...]` — one of a set.
    Class(CharClass),
}

impl Atom {
    /// The set of characters this atom admits.
    #[must_use]
    pub const fn class(self) -> CharClass {
        match self {
            Self::Literal(c) => CharClass::single(c),
            Self::Wildcard => CharClass::ANY,
            Self::Class(k) => k,
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(c) => write!(f, "{c}"),
            Self::Wildcard => f.write_str("?"),
            Self::Class(k) => write!(f, "{k:?}"),
        }
    }
}

/// A syntactically valid pattern.
///
/// Holding one guarantees: between 1 and [`MAX_PATTERN_ATOMS`] atoms, every
/// atom's class non-empty. There is no public constructor other than
/// [`Pattern::parse`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pattern {
    atoms: Vec<Atom>,
    /// The exact source text, retained for diagnostics and progress display.
    source: String,
}

impl Pattern {
    /// Recognise a pattern.
    ///
    /// # Errors
    ///
    /// See [`PatternError`]. Recognition is complete: the returned value is
    /// either a fully valid pattern or an error naming the first offending byte.
    pub fn parse(source: &str) -> Result<Self, PatternError> {
        let bytes = source.as_bytes();
        let mut atoms: Vec<Atom> = Vec::new();
        let mut i = 0usize;

        // Driven by `get` rather than a length test plus an index, so the loop
        // is total: it ends when input is exhausted, with no way to read past
        // the end even if the body's arithmetic were later changed.
        while let Some(&byte) = bytes.get(i) {
            let atom = match byte {
                b'?' => {
                    i += 1;
                    Atom::Wildcard
                }
                b'[' => {
                    let (class, next) = parse_class(bytes, i)?;
                    i = next;
                    Atom::Class(class)
                }
                b']' => {
                    return Err(PatternError::UnmatchedClassClose { offset: i });
                }
                _ => {
                    let c = Base32Char::from_ascii(byte).ok_or(PatternError::UnexpectedByte {
                        offset: i,
                        byte,
                        context: ByteContext::Pattern,
                    })?;
                    i += 1;
                    Atom::Literal(c)
                }
            };
            if atoms.len() == MAX_PATTERN_ATOMS {
                return Err(PatternError::TooLong {
                    max: MAX_PATTERN_ATOMS,
                });
            }
            atoms.push(atom);
        }

        if atoms.is_empty() {
            return Err(PatternError::Empty);
        }
        Ok(Self {
            atoms,
            source: source.to_owned(),
        })
    }

    /// The atoms, in address-character order.
    #[must_use]
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    /// Number of address characters this pattern constrains.
    #[must_use]
    pub fn char_len(&self) -> usize {
        self.atoms.len()
    }

    /// The text this pattern was parsed from.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Base-2 logarithm of the expected number of trials to find a match.
    ///
    /// Each atom independently admits `class.len()` of 32 values, so the
    /// probability a uniformly random address matches is the product of
    /// `len/32`. The expected trial count is the reciprocal; this returns its
    /// log so that callers can display it without overflowing.
    #[must_use]
    pub fn difficulty_log2(&self) -> f64 {
        self.atoms
            .iter()
            .map(|a| 5.0 - f64::from(a.class().len()).log2())
            .sum()
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.source)
    }
}

/// Parse a bracketed class starting at `open` (which must index a `[`).
///
/// Returns the class and the offset just past the closing `]`.
fn parse_class(bytes: &[u8], open: usize) -> Result<(CharClass, usize), PatternError> {
    let mut i = open + 1;
    let negated = bytes.get(i) == Some(&b'^');
    if negated {
        i += 1;
    }
    let members_start = i;
    let mut bits: u32 = 0;
    loop {
        let Some(&byte) = bytes.get(i) else {
            return Err(PatternError::UnterminatedClass { open_offset: open });
        };
        match byte {
            b']' => break,
            b'[' => {
                return Err(PatternError::UnexpectedByte {
                    offset: i,
                    byte,
                    context: ByteContext::Class,
                });
            }
            _ => {
                let c = Base32Char::from_ascii(byte).ok_or(PatternError::UnexpectedByte {
                    offset: i,
                    byte,
                    context: ByteContext::Class,
                })?;
                bits |= 1u32 << c.value();
                i += 1;
            }
        }
    }
    if i == members_start {
        return Err(PatternError::EmptyClass { open_offset: open });
    }
    let effective = if negated { !bits } else { bits };
    let class = CharClass::from_bits(effective).ok_or(PatternError::UnsatisfiableClass {
        open_offset: open,
    })?;
    Ok((class, i + 1))
}

/// Where in the grammar an unexpected byte appeared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteContext {
    /// At the top level of a pattern.
    Pattern,
    /// Inside a `[...]` class.
    Class,
}

impl fmt::Display for ByteContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pattern => f.write_str("expected a base32 character (a-z, 2-7), '?', or '['"),
            Self::Class => f.write_str("expected a base32 character (a-z, 2-7) or ']'"),
        }
    }
}

/// Why a string was not a valid pattern.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PatternError {
    /// The pattern had no atoms.
    #[error("pattern is empty; it must constrain at least one character")]
    Empty,
    /// The pattern had more atoms than the public key can determine.
    #[error(
        "pattern is longer than {max} characters; \
         beyond that an address depends on its checksum, which cannot be searched by prefix"
    )]
    TooLong {
        /// The limit, [`MAX_PATTERN_ATOMS`].
        max: usize,
    },
    /// A byte was not valid at that point in the grammar.
    #[error("unexpected byte {byte:#04x} at offset {offset}: {context}")]
    UnexpectedByte {
        /// Zero-based byte offset.
        offset: usize,
        /// The offending byte.
        byte: u8,
        /// Where in the grammar it appeared.
        context: ByteContext,
    },
    /// A `[` was never closed.
    #[error("unterminated character class opened at offset {open_offset}; expected ']'")]
    UnterminatedClass {
        /// Offset of the `[`.
        open_offset: usize,
    },
    /// A `]` appeared with no matching `[`.
    #[error("unmatched ']' at offset {offset}")]
    UnmatchedClassClose {
        /// Offset of the `]`.
        offset: usize,
    },
    /// A class listed no members, as in `[]`.
    #[error("empty character class at offset {open_offset}; a class must list at least one character")]
    EmptyClass {
        /// Offset of the `[`.
        open_offset: usize,
    },
    /// A negated class excluded the entire alphabet, as in `[^abc...7]`.
    #[error(
        "character class at offset {open_offset} admits no characters; \
         no address could ever match"
    )]
    UnsatisfiableClass {
        /// Offset of the `[`.
        open_offset: usize,
    },
}

impl PatternError {
    /// The byte offset the error refers to, when it refers to one.
    #[must_use]
    pub const fn offset(&self) -> Option<usize> {
        match self {
            Self::UnexpectedByte { offset, .. } | Self::UnmatchedClassClose { offset } => {
                Some(*offset)
            }
            Self::UnterminatedClass { open_offset }
            | Self::EmptyClass { open_offset }
            | Self::UnsatisfiableClass { open_offset } => Some(*open_offset),
            Self::Empty | Self::TooLong { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse and return the atoms, panicking with the error on failure.
    fn atoms(src: &str) -> Vec<Atom> {
        Pattern::parse(src)
            .unwrap_or_else(|e| panic!("{src:?} should parse: {e}"))
            .atoms()
            .to_vec()
    }

    #[test]
    fn literals_only() {
        let a = atoms("carroll");
        assert_eq!(a.len(), 7);
        assert!(a.iter().all(|x| matches!(x, Atom::Literal(_))));
        assert_eq!(Pattern::parse("carroll").expect("valid").to_string(), "carroll");
    }

    #[test]
    fn digits_two_through_seven_are_literals() {
        // The base32 alphabet's digits are 2-7; 0, 1, 8, 9 are not in it.
        for good in ["2", "3", "4", "5", "6", "7"] {
            assert!(Pattern::parse(good).is_ok(), "{good} should parse");
        }
        for bad in ["0", "1", "8", "9"] {
            assert!(
                matches!(
                    Pattern::parse(bad),
                    Err(PatternError::UnexpectedByte { offset: 0, .. })
                ),
                "{bad} should be rejected"
            );
        }
    }

    #[test]
    fn wildcard_admits_everything() {
        assert_eq!(atoms("?"), vec![Atom::Wildcard]);
        assert!(Atom::Wildcard.class().is_any());
        assert_eq!(Atom::Wildcard.class().len(), 32);
    }

    #[test]
    fn class_admits_listed_members() {
        let a = atoms("[abc]");
        let Atom::Class(k) = a[0] else { panic!("expected a class") };
        assert_eq!(k.len(), 3);
        for c in ['a', 'b', 'c'] {
            let ch = Base32Char::from_ascii(c as u8).expect("alphabet");
            assert!(k.admits_value(ch.value()), "{c} should be admitted");
        }
        let d = Base32Char::from_ascii(b'd').expect("alphabet");
        assert!(!k.admits_value(d.value()));
    }

    #[test]
    fn negated_class_admits_the_complement() {
        let a = atoms("[^a]");
        let Atom::Class(k) = a[0] else { panic!("expected a class") };
        assert_eq!(k.len(), 31);
        let ch = Base32Char::from_ascii(b'a').expect("alphabet");
        assert!(!k.admits_value(ch.value()));
    }

    #[test]
    fn duplicate_class_members_are_idempotent() {
        let Atom::Class(k) = atoms("[aab]")[0] else { panic!("expected a class") };
        assert_eq!(k.len(), 2);
    }

    #[test]
    fn single_member_class_is_equivalent_to_a_literal() {
        let Atom::Class(k) = atoms("[q]")[0] else { panic!("expected a class") };
        let q = Base32Char::from_ascii(b'q').expect("alphabet");
        assert_eq!(k.as_single(), Some(q));
        assert_eq!(k, CharClass::single(q));
    }

    // --- Rejection cases. Each asserts a *specific* error, not merely "an
    // error": a parser that rejects for the wrong reason is still a parser
    // whose language we do not know.

    #[test]
    fn rejects_empty() {
        assert_eq!(Pattern::parse(""), Err(PatternError::Empty));
    }

    #[test]
    fn rejects_over_length() {
        let long = "a".repeat(MAX_PATTERN_ATOMS + 1);
        assert_eq!(
            Pattern::parse(&long),
            Err(PatternError::TooLong { max: MAX_PATTERN_ATOMS })
        );
        // Exactly at the limit is accepted.
        let at_limit = "a".repeat(MAX_PATTERN_ATOMS);
        assert_eq!(
            Pattern::parse(&at_limit).expect("valid").char_len(),
            MAX_PATTERN_ATOMS
        );
    }

    #[test]
    fn rejects_uppercase_with_offset() {
        assert_eq!(
            Pattern::parse("carRoll"),
            Err(PatternError::UnexpectedByte {
                offset: 3,
                byte: b'R',
                context: ByteContext::Pattern,
            })
        );
    }

    #[test]
    fn rejects_unterminated_class() {
        assert_eq!(
            Pattern::parse("ab[cd"),
            Err(PatternError::UnterminatedClass { open_offset: 2 })
        );
    }

    #[test]
    fn rejects_unmatched_close() {
        assert_eq!(
            Pattern::parse("ab]cd"),
            Err(PatternError::UnmatchedClassClose { offset: 2 })
        );
    }

    #[test]
    fn rejects_empty_class() {
        assert_eq!(
            Pattern::parse("ab[]"),
            Err(PatternError::EmptyClass { open_offset: 2 })
        );
        assert_eq!(
            Pattern::parse("ab[^]"),
            Err(PatternError::EmptyClass { open_offset: 2 })
        );
    }

    #[test]
    fn rejects_class_excluding_whole_alphabet() {
        let all = "abcdefghijklmnopqrstuvwxyz234567";
        assert_eq!(
            Pattern::parse(&format!("[^{all}]")),
            Err(PatternError::UnsatisfiableClass { open_offset: 0 })
        );
    }

    #[test]
    fn rejects_nested_class_open() {
        assert_eq!(
            Pattern::parse("[a[b]]"),
            Err(PatternError::UnexpectedByte {
                offset: 2,
                byte: b'[',
                context: ByteContext::Class,
            })
        );
    }

    #[test]
    fn rejects_wildcard_inside_class() {
        // '?' is a pattern-level atom, not a class member. Silently treating it
        // as a literal would give the class a meaning the grammar never defined.
        assert_eq!(
            Pattern::parse("[a?b]"),
            Err(PatternError::UnexpectedByte {
                offset: 2,
                byte: b'?',
                context: ByteContext::Class,
            })
        );
    }

    #[test]
    fn rejects_whitespace_and_punctuation() {
        for (src, offset) in [("ab cd", 2), ("ab.cd", 2), ("ab\ncd", 2), ("ab-cd", 2)] {
            match Pattern::parse(src) {
                Err(PatternError::UnexpectedByte { offset: got, .. }) => {
                    assert_eq!(got, offset, "for {src:?}");
                }
                other => panic!("{src:?} should be rejected, got {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_non_ascii() {
        // Multi-byte UTF-8 is rejected at its first byte; offsets are byte
        // offsets, which is what a caller needs to point at the input.
        match Pattern::parse("caré") {
            Err(PatternError::UnexpectedByte { offset, .. }) => assert_eq!(offset, 3),
            other => panic!("should be rejected, got {other:?}"),
        }
    }

    #[test]
    fn every_error_reports_a_useful_offset() {
        let cases = [
            "carRoll", "ab[cd", "ab]cd", "ab[]", "[a[b]]", "ab cd",
        ];
        for src in cases {
            let err = Pattern::parse(src).expect_err("should fail");
            let offset = err.offset().expect("should carry an offset");
            assert!(offset < src.len(), "offset {offset} out of range for {src:?}");
        }
    }

    #[test]
    fn difficulty_matches_hand_calculation() {
        // Seven fixed characters: 5 bits each.
        assert!((Pattern::parse("carroll").expect("valid").difficulty_log2() - 35.0).abs() < 1e-9);
        // A wildcard constrains nothing.
        assert!((Pattern::parse("?").expect("valid").difficulty_log2() - 0.0).abs() < 1e-9);
        // A 2-member class is 1 bit of constraint less than a literal.
        assert!((Pattern::parse("[ab]").expect("valid").difficulty_log2() - 4.0).abs() < 1e-9);
        // A 31-member negated class barely constrains at all.
        let d = Pattern::parse("[^a]").expect("valid").difficulty_log2();
        assert!(d > 0.0 && d < 0.05, "got {d}");
    }
}
