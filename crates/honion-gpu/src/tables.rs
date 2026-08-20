//! Turning a compiled pattern set into the flat integer tables the kernel reads.
//!
//! This is the host side of langsec rule 4. Everything the device will ever
//! look at is built here, in ordinary testable Rust, from a
//! [`PatternSet`](honion_core::pattern::PatternSet) that has already been fully
//! recognised. The device receives fixed-size integers and nothing else: no
//! strings, no lengths read out of the data it is processing, no structure it
//! has to interpret.
//!
//! # Layout
//!
//! ```text
//! group_mask[g]                    mask for group g
//! group_off[g] ..= group_off[g+1]  that group's slice of `target`
//! target[t]                        sorted ascending within each group
//! target_pat[t]                    the pattern that produced target[t]
//! res_off[p] ..= res_off[p+1]      pattern p's slice of `res`
//! res[i]                           char index << 32 | allowed-value bitmap
//! ```
//!
//! Two patterns can compile to the same `(mask, target)` while differing in
//! their residual constraints — `ab[cd]` and `ab[ef]` do. Rather than a side
//! table mapping each target to a list of patterns, such targets are simply
//! stored more than once, adjacently. Sorted order is preserved because the
//! duplicated keys are equal, and the kernel's binary search finds the start of
//! the run and walks it. One fewer indirection on the device, at the cost of a
//! few bytes on a table that is read almost never.

use honion_core::pattern::PatternSet;

/// The flattened tables, ready to upload.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeviceTables {
    /// One mask per group.
    pub group_mask: Vec<u64>,
    /// Group boundaries in [`Self::target`]; length is `group_mask.len() + 1`.
    pub group_off: Vec<u32>,
    /// Prefilter targets, sorted ascending within each group.
    pub target: Vec<u64>,
    /// Pattern index for each entry of [`Self::target`].
    pub target_pat: Vec<u32>,
    /// Pattern boundaries in [`Self::res`]; length is `num_patterns + 1`.
    pub res_off: Vec<u32>,
    /// Residual constraints: character index in the high 32 bits, allowed-value
    /// bitmap in the low 32.
    pub res: Vec<u64>,
}

impl DeviceTables {
    /// Flatten a compiled pattern set.
    #[must_use]
    pub fn build(set: &PatternSet) -> Self {
        let mut group_mask = Vec::with_capacity(set.groups().len());
        let mut group_off = Vec::with_capacity(set.groups().len() + 1);
        let mut target = Vec::new();
        let mut target_pat = Vec::new();

        for group in set.groups() {
            group_off.push(u32::try_from(target.len()).unwrap_or(u32::MAX));
            group_mask.push(group.mask());
            for (slot, &t) in group.targets().iter().enumerate() {
                for &pid in group.patterns_for_slot(slot) {
                    target.push(t);
                    target_pat.push(pid);
                }
            }
        }
        group_off.push(u32::try_from(target.len()).unwrap_or(u32::MAX));

        let mut res_off = Vec::with_capacity(set.patterns().len() + 1);
        let mut res = Vec::new();
        for pattern in set.patterns() {
            res_off.push(u32::try_from(res.len()).unwrap_or(u32::MAX));
            for r in pattern.residual() {
                res.push((u64::from(r.char_index) << 32) | u64::from(r.allowed));
            }
        }
        res_off.push(u32::try_from(res.len()).unwrap_or(u32::MAX));

        Self {
            group_mask,
            group_off,
            target,
            target_pat,
            res_off,
            res,
        }
    }

    /// Number of mask groups.
    #[must_use]
    pub fn num_groups(&self) -> u32 {
        u32::try_from(self.group_mask.len()).unwrap_or(u32::MAX)
    }

    /// Base-2 log of the reciprocal probability that a random key passes the
    /// prefilter.
    ///
    /// This is *not* the search difficulty — it is how often the device will
    /// hand the host a candidate to verify. A pattern whose constraints all sit
    /// past character 12, or which is mostly character classes, can be hard to
    /// satisfy yet trivial to pass the prefilter, and would deluge the host
    /// with candidates that all fail the full check. [`Self::check_selectivity`]
    /// turns that into an error rather than a mysterious collapse in throughput.
    #[must_use]
    pub fn prefilter_selectivity_log2(&self) -> f64 {
        let mut probability = 0.0f64;
        for (g, mask) in self.group_mask.iter().enumerate() {
            let lo = self.group_off.get(g).copied().unwrap_or(0) as usize;
            let hi = self.group_off.get(g + 1).copied().unwrap_or(0) as usize;
            // Distinct targets, since duplicates denote the same probe value.
            let slice = self.target.get(lo..hi).unwrap_or(&[]);
            let mut distinct = 0u64;
            let mut prev: Option<u64> = None;
            for &t in slice {
                if prev != Some(t) {
                    distinct += 1;
                    prev = Some(t);
                }
            }
            probability += distinct as f64 * 2f64.powi(-(mask.count_ones() as i32));
        }
        if probability <= 0.0 {
            f64::INFINITY
        } else {
            -probability.log2()
        }
    }

    /// Reject pattern sets whose prefilter is too weak to be worth running.
    ///
    /// At roughly 10^9 candidates per second, a prefilter with `n` bits of
    /// selectivity produces `2^(30-n)` candidates per second for the host to
    /// verify. Twenty bits leaves about a thousand per second, which is
    /// comfortable; ten bits would be a million per second, which is not a
    /// search but a denial of service against our own verifier.
    ///
    /// # Errors
    ///
    /// [`TableError::PrefilterTooWeak`] when the prefilter cannot carry its
    /// share of the work.
    pub fn check_selectivity(&self, min_bits: f64) -> Result<(), TableError> {
        let bits = self.prefilter_selectivity_log2();
        if bits < min_bits {
            return Err(TableError::PrefilterTooWeak {
                bits,
                required: min_bits,
            });
        }
        Ok(())
    }
}

/// Why a pattern set cannot be searched as given.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TableError {
    /// The device-side prefilter would pass too many candidates to the host.
    #[error(
        "these patterns give the GPU prefilter only {bits:.1} bits of selectivity, \
         but {required:.0} are needed. The device would forward candidates faster than \
         the host can check them. This happens when a pattern's fixed characters all \
         sit past position 12, or when it is mostly wildcards and character classes; \
         anchoring more literal characters near the start fixes it."
    )]
    PrefilterTooWeak {
        /// Selectivity the given patterns achieve.
        bits: f64,
        /// Selectivity required.
        required: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use honion_core::pattern::Pattern;

    fn set(sources: &[&str]) -> PatternSet {
        let patterns: Vec<Pattern> = sources
            .iter()
            .map(|s| Pattern::parse(s).unwrap_or_else(|e| panic!("{s}: {e}")))
            .collect();
        PatternSet::compile(&patterns).expect("non-empty")
    }

    #[test]
    fn a_single_pattern_flattens_to_one_group_and_one_target() {
        let t = DeviceTables::build(&set(&["carroll"]));
        assert_eq!(t.group_mask.len(), 1);
        assert_eq!(t.group_off, vec![0, 1]);
        assert_eq!(t.target.len(), 1);
        assert_eq!(t.target_pat, vec![0]);
        assert!(t.res.is_empty(), "an all-literal pattern has no residual");
        assert_eq!(t.res_off, vec![0, 0]);
    }

    #[test]
    fn targets_are_sorted_within_each_group() {
        // Sorted order is what makes the device's binary search valid.
        let t = DeviceTables::build(&set(&["zzzz", "aaaa", "mmmm", "bbbb"]));
        assert_eq!(t.group_mask.len(), 1);
        assert!(
            t.target.windows(2).all(|w| w[0] <= w[1]),
            "targets must be ascending: {:?}",
            t.target
        );
    }

    #[test]
    fn patterns_sharing_a_target_are_stored_adjacently() {
        // "ab[cd]" and "ab[ef]" have identical mask and target and differ only
        // in their residuals, so the target appears twice, side by side.
        let t = DeviceTables::build(&set(&["ab[cd]", "ab[ef]"]));
        assert_eq!(t.target.len(), 2);
        assert_eq!(t.target[0], t.target[1], "same probe value");
        let mut pats = t.target_pat.clone();
        pats.sort_unstable();
        assert_eq!(pats, vec![0, 1]);
        // Each carries its own residual.
        assert_eq!(t.res_off, vec![0, 1, 2]);
        assert_eq!(t.res.len(), 2);
        assert_ne!(t.res[0], t.res[1]);
    }

    #[test]
    fn residual_entries_pack_index_and_bitmap() {
        let t = DeviceTables::build(&set(&["a[bc]"]));
        assert_eq!(t.res.len(), 1);
        let entry = t.res[0];
        assert_eq!(entry >> 32, 1, "the class is at character 1");
        assert_eq!((entry as u32).count_ones(), 2, "two admitted characters");
    }

    #[test]
    fn group_offsets_cover_the_whole_target_array() {
        let t = DeviceTables::build(&set(&["abcd", "ef?gh", "z"]));
        assert_eq!(t.group_off.len(), t.group_mask.len() + 1);
        assert_eq!(t.group_off[0], 0);
        assert_eq!(
            *t.group_off.last().expect("non-empty"),
            t.target.len() as u32
        );
        assert!(t.group_off.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(t.target.len(), t.target_pat.len());
    }

    #[test]
    fn selectivity_tracks_the_number_of_fixed_characters() {
        // Each fixed character contributes five bits.
        for (src, expected) in [("abcdef", 30.0), ("abcd", 20.0), ("ab", 10.0)] {
            let t = DeviceTables::build(&set(&[src]));
            let bits = t.prefilter_selectivity_log2();
            assert!((bits - expected).abs() < 1e-9, "{src}: got {bits}");
        }
    }

    #[test]
    fn selectivity_halves_with_each_added_pattern() {
        let one = DeviceTables::build(&set(&["abcdef"])).prefilter_selectivity_log2();
        let two = DeviceTables::build(&set(&["abcdef", "ghijkl"])).prefilter_selectivity_log2();
        assert!((one - two - 1.0).abs() < 1e-9, "{one} vs {two}");
    }

    #[test]
    fn weak_prefilters_are_rejected_with_an_actionable_message() {
        // Every constraint sits past character 12, so the 64-bit prefilter is
        // empty and the device would forward every single candidate.
        let far = format!("{}abcdefgh", "?".repeat(13));
        let t = DeviceTables::build(&set(&[&far]));
        assert_eq!(t.prefilter_selectivity_log2(), 0.0);
        let err = t.check_selectivity(20.0).expect_err("should be rejected");
        let TableError::PrefilterTooWeak { bits, required } = err;
        assert!((bits - 0.0).abs() < 1e-9);
        assert!((required - 20.0).abs() < 1e-9);

        // A normal prefix passes comfortably.
        DeviceTables::build(&set(&["carroll"]))
            .check_selectivity(20.0)
            .expect("a seven-character prefix is plenty");
    }
}
