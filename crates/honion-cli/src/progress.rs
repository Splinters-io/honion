//! Progress reporting.
//!
//! The numbers shown are chosen so a user can decide whether to keep waiting.
//! A rate alone does not answer that; what does is the *expected* remaining
//! time given the pattern's difficulty, and how much of the expected work has
//! already been done. Because the search is memoryless — every key is an
//! independent trial — passing the expected time does not mean a result is
//! overdue, so the display says "expected", never "remaining", and reports the
//! probability of having found something by now.

use std::time::{Duration, Instant};

/// Accumulates and renders search progress.
pub struct Progress {
    started: Instant,
    examined: u64,
    found: usize,
    /// Base-2 log of the expected number of trials per hit.
    difficulty_log2: f64,
    quiet: bool,
}

impl Progress {
    /// Start tracking a search of the given difficulty.
    #[must_use]
    pub fn new(difficulty_log2: f64, quiet: bool) -> Self {
        Self {
            started: Instant::now(),
            examined: 0,
            found: 0,
            difficulty_log2,
            quiet,
        }
    }

    /// Record a completed launch.
    pub fn record(&mut self, examined: u64, found: usize) {
        self.examined += examined;
        self.found += found;
    }

    /// Keys examined so far.
    #[must_use]
    pub const fn examined(&self) -> u64 {
        self.examined
    }

    /// Matches found so far.
    #[must_use]
    pub const fn found(&self) -> usize {
        self.found
    }

    /// Time since the search began.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Keys per second, averaged over the whole run.
    #[must_use]
    pub fn rate(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs <= 0.0 {
            0.0
        } else {
            self.examined as f64 / secs
        }
    }

    /// Probability that a search of this length has produced at least one hit.
    ///
    /// Trials are independent with probability `2^-difficulty` each, so the
    /// chance of no hit after `n` trials is `(1 - p)^n`, computed here as
    /// `exp(-n * p)` since `p` is tiny.
    #[must_use]
    pub fn probability_of_success(&self) -> f64 {
        let p = 2f64.powf(-self.difficulty_log2);
        1.0 - (-(self.examined as f64) * p).exp()
    }

    /// Expected seconds per hit at the current rate.
    #[must_use]
    pub fn expected_seconds(&self) -> f64 {
        let rate = self.rate();
        if rate <= 0.0 {
            return f64::INFINITY;
        }
        2f64.powf(self.difficulty_log2) / rate
    }

    /// Emit a one-line progress update, unless quiet.
    pub fn report(&self) {
        if self.quiet {
            return;
        }
        eprintln!(
            "  {:>9} examined  {:>8}/s  elapsed {:>10}  expected {:>10}  P(found) {:>5.1}%  hits {}",
            si(self.examined as f64),
            si(self.rate()),
            humantime(self.elapsed().as_secs_f64()),
            humantime(self.expected_seconds()),
            self.probability_of_success() * 100.0,
            self.found,
        );
    }
}

/// Format a count with an SI suffix.
#[must_use]
pub fn si(v: f64) -> String {
    const UNITS: [(&str, f64); 5] = [
        ("P", 1e15),
        ("T", 1e12),
        ("G", 1e9),
        ("M", 1e6),
        ("k", 1e3),
    ];
    for (suffix, scale) in UNITS {
        if v >= scale {
            return format!("{:.2}{suffix}", v / scale);
        }
    }
    format!("{v:.0}")
}

/// Format a duration in the largest unit that keeps it readable.
#[must_use]
pub fn humantime(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "unknown".to_owned();
    }
    const YEAR: f64 = 365.25 * 86400.0;
    match seconds {
        s if s < 1.0 => format!("{:.0}ms", s * 1000.0),
        s if s < 90.0 => format!("{s:.1}s"),
        s if s < 5400.0 => format!("{:.1}min", s / 60.0),
        s if s < 172_800.0 => format!("{:.1}h", s / 3600.0),
        s if s < YEAR => format!("{:.1}d", s / 86400.0),
        s => format!("{:.1}y", s / YEAR),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn si_uses_sensible_units() {
        assert_eq!(si(999.0), "999");
        assert_eq!(si(1500.0), "1.50k");
        assert_eq!(si(4.9e9), "4.90G");
        assert_eq!(si(1.15e18), "1150.00P");
    }

    #[test]
    fn humantime_switches_units() {
        assert_eq!(humantime(0.25), "250ms");
        assert_eq!(humantime(45.0), "45.0s");
        assert_eq!(humantime(600.0), "10.0min");
        assert_eq!(humantime(7200.0), "2.0h");
        assert_eq!(humantime(864_000.0), "10.0d");
        assert!(humantime(f64::INFINITY) == "unknown");
    }

    #[test]
    fn probability_reaches_sixty_three_percent_at_the_expected_count() {
        // After exactly 2^d trials the chance of at least one hit is 1 - 1/e.
        let mut p = Progress::new(20.0, true);
        p.record(1 << 20, 0);
        let prob = p.probability_of_success();
        assert!((prob - (1.0 - (-1.0f64).exp())).abs() < 1e-6, "got {prob}");
    }

    #[test]
    fn probability_starts_at_zero_and_is_monotonic() {
        let mut p = Progress::new(30.0, true);
        assert!(p.probability_of_success().abs() < 1e-12);
        let mut last = 0.0;
        for _ in 0..8 {
            p.record(1 << 28, 0);
            let now = p.probability_of_success();
            assert!(now > last, "probability must increase");
            assert!(now < 1.0, "and never reach certainty");
            last = now;
        }
    }
}
