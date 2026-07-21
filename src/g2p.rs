//! Grapheme-to-phoneme via the CMU Pronouncing Dictionary (ARPAbet),
//! with deterministic fallbacks for out-of-vocabulary words.
//!
//! Fallback order per word: dictionary lookup -> digit-word expansion
//! ("42" -> "four" "two") -> single-letter names ("x" -> "EH1 K S").
//! Words that still resolve to nothing are reported as unresolved so the
//! pipeline can skip phoneme units for them (and count them).

use crate::Result;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Default)]
pub struct G2p {
    map: HashMap<String, Vec<String>>,
}

const DIGIT_WORDS: [&str; 10] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];

impl G2p {
    /// Parse a cmudict file (`word PH1 PH2 ...` per line, `;;;` comments,
    /// `(n)` pronunciation-variant suffixes — first variant wins).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(Self::from_str(&text))
    }

    pub fn from_str(text: &str) -> Self {
        let mut map = HashMap::new();
        for line in text.lines() {
            if line.starts_with(";;;") || line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(word) = parts.next() else { continue };
            // skip alternate pronunciations: "word(2)"
            if word.contains('(') {
                continue;
            }
            let phonemes: Vec<String> = parts.map(str::to_string).collect();
            if !phonemes.is_empty() {
                map.insert(word.to_string(), phonemes);
            }
        }
        Self { map }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// ARPAbet phonemes for a normalized (lowercase) word.
    pub fn lookup(&self, word: &str) -> Option<&[String]> {
        self.map.get(word).map(Vec::as_slice)
    }

    /// Dictionary-first lookup with deterministic OOV fallbacks.
    /// Returns None only when nothing resolves at all.
    pub fn phonemes(&self, word: &str) -> Option<Vec<String>> {
        if let Some(p) = self.lookup(word) {
            return Some(p.to_vec());
        }
        let mut out = Vec::new();
        for c in word.chars() {
            let piece = if c.is_ascii_digit() {
                self.lookup(DIGIT_WORDS[c as usize - '0' as usize])
            } else {
                self.lookup(c.to_string().as_str())
            };
            match piece {
                Some(p) => out.extend(p.iter().cloned()),
                None => return None,
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }

    /// Strip the stress digit from a vowel phoneme ("AH0" -> "AH").
    /// Useful for stress-insensitive comparisons and context keys.
    pub fn base(phoneme: &str) -> &str {
        phoneme.trim_end_matches(['0', '1', '2'])
    }
}
