//! Collecting patterns from the command line and from files.
//!
//! Langsec rule 2 in practice: every pattern from every source is recognised
//! completely, and the first violation aborts the run, before any GPU memory is
//! allocated or any key material is generated. There is no mode in which some
//! patterns are used and others are quietly dropped.
//!
//! Pattern files are a list of patterns, one per line. Blank lines and lines
//! beginning with `#` are skipped; everything else must be a valid pattern.
//! That skipping is the *only* leniency in the whole pipeline, it is defined
//! here, and it is defined in terms of whole lines rather than by trying to
//! find a pattern inside arbitrary text.

use std::path::{Path, PathBuf};

use honion_core::pattern::{Pattern, PatternError};

/// A pattern together with where it came from, for diagnostics.
#[derive(Debug, Clone)]
pub struct SourcedPattern {
    /// The parsed pattern.
    pub pattern: Pattern,
    /// Human-readable origin, e.g. `--prefix` or `patterns.txt:12`.
    pub origin: String,
}

/// Parse patterns given directly on the command line.
///
/// # Errors
///
/// [`PatternInputError::Invalid`] naming the offending argument and offset.
pub fn from_args(args: &[String]) -> Result<Vec<SourcedPattern>, PatternInputError> {
    args.iter()
        .map(|src| {
            Pattern::parse(src)
                .map(|pattern| SourcedPattern {
                    pattern,
                    origin: "--prefix".to_owned(),
                })
                .map_err(|error| PatternInputError::Invalid {
                    origin: format!("--prefix {src:?}"),
                    text: src.clone(),
                    error,
                })
        })
        .collect()
}

/// Parse patterns from a file, one per line.
///
/// # Errors
///
/// [`PatternInputError::Io`] if the file cannot be read, or
/// [`PatternInputError::Invalid`] naming the line and offset of the first
/// malformed pattern.
pub fn from_file(path: &Path) -> Result<Vec<SourcedPattern>, PatternInputError> {
    let text = std::fs::read_to_string(path).map_err(|e| PatternInputError::Io {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })?;

    let mut out = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let pattern = Pattern::parse(line).map_err(|error| PatternInputError::Invalid {
            origin: format!("{}:{}", path.display(), index + 1),
            text: line.to_owned(),
            error,
        })?;
        out.push(SourcedPattern {
            pattern,
            origin: format!("{}:{}", path.display(), index + 1),
        });
    }
    Ok(out)
}

/// Render a parse error with a caret under the offending byte.
#[must_use]
pub fn render_error(origin: &str, text: &str, error: &PatternError) -> String {
    let mut out = format!("{origin}: {error}\n  {text}\n");
    if let Some(offset) = error.offset() {
        out.push_str("  ");
        out.push_str(&" ".repeat(offset));
        out.push_str("^\n");
    }
    out
}

/// Why patterns could not be collected.
#[derive(Debug, thiserror::Error)]
pub enum PatternInputError {
    /// A pattern was malformed.
    #[error("{}", render_error(origin, text, error))]
    Invalid {
        /// Where the pattern came from.
        origin: String,
        /// The offending text.
        text: String,
        /// What was wrong with it.
        error: PatternError,
    },
    /// A pattern file could not be read.
    #[error("reading {path}: {reason}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_parse_in_order() {
        let got = from_args(&["abc".to_owned(), "de?f".to_owned()]).expect("valid");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].pattern.source(), "abc");
        assert_eq!(got[1].pattern.source(), "de?f");
    }

    #[test]
    fn a_bad_argument_names_itself_and_the_offset() {
        let err = from_args(&["abC".to_owned()]).expect_err("uppercase is invalid");
        let text = err.to_string();
        assert!(text.contains("--prefix"), "{text}");
        assert!(text.contains("offset 2"), "{text}");
        // The caret must sit under the offending byte.
        assert!(text.contains("\n  abC\n    ^"), "{text}");
    }

    #[test]
    fn files_skip_blanks_and_comments_only() {
        let dir = std::env::temp_dir().join(format!("honion-pat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("patterns.txt");
        std::fs::write(&path, "# a comment\n\nabc\n  def  \n").expect("write");

        let got = from_file(&path).expect("valid");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].pattern.source(), "abc");
        assert_eq!(got[1].pattern.source(), "def", "lines are trimmed");
        assert_eq!(got[0].origin, format!("{}:3", path.display()));

        // One bad line fails the whole file: no partial acceptance.
        std::fs::write(&path, "abc\nde!f\nghi\n").expect("write");
        let err = from_file(&path).expect_err("invalid line");
        assert!(err.to_string().contains(":2"), "{err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_file_is_reported_clearly() {
        let err = from_file(Path::new("/nonexistent/honion/patterns.txt")).expect_err("missing");
        assert!(matches!(err, PatternInputError::Io { .. }));
    }
}
