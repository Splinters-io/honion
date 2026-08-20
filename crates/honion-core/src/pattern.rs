//! The vanity pattern language.
//!
//! Text enters through [`parse::Pattern::parse`] and leaves as a
//! [`compile::PatternSet`] — fixed-size integers ready for the GPU. Nothing
//! downstream of this module ever looks at pattern text again.
//!
//! ```
//! use honion_core::pattern::{Pattern, PatternSet};
//!
//! // "carroll", then any character, then an 'e' or an 'i'.
//! let patterns = vec![Pattern::parse("carroll?[ei]")?];
//! let set = PatternSet::compile(&patterns)?;
//!
//! // Eight fixed characters out of the first twelve are expressible as a
//! // single masked compare; the trailing class becomes a residual check.
//! assert_eq!(set.groups().len(), 1);
//! assert_eq!(set.patterns()[0].residual().len(), 1);
//!
//! // Expected work is 2^difficulty_log2 trials: 5 bits per fixed character
//! // plus 4 for the two-way class.
//! assert!((set.difficulty_log2() - 39.0).abs() < 0.01);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod compile;
pub mod parse;

pub use compile::{
    CompiledPattern, MaskGroup, PREFILTER_BYTES, PREFILTER_CHARS, PatternSet, PatternSetError,
    ResidualPosition, char_value, key_prefix_u64,
};
pub use parse::{Atom, ByteContext, CharClass, MAX_PATTERN_ATOMS, Pattern, PatternError};
