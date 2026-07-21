//! Linguistic unit levels and text normalization.

use std::fmt;
use std::str::FromStr;

/// Granularity of a cached audio primitive. The composer always tries the
/// highest available level first (phrase > word > morpheme > diphone).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UnitLevel {
    Phrase,
    Word,
    Morpheme,
    /// A phoneme-pair transition slice (mid-left-phoneme to mid-right),
    /// the classic concatenative-synthesis unit. Text key is stress-stripped:
    /// "OW+W". Props carry full phonemes and outer phonetic context.
    Diphone,
    Phoneme,
}

impl UnitLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            UnitLevel::Phrase => "phrase",
            UnitLevel::Word => "word",
            UnitLevel::Morpheme => "morpheme",
            UnitLevel::Diphone => "diphone",
            UnitLevel::Phoneme => "phoneme",
        }
    }
}

impl fmt::Display for UnitLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for UnitLevel {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "phrase" => Ok(UnitLevel::Phrase),
            "word" => Ok(UnitLevel::Word),
            "morpheme" => Ok(UnitLevel::Morpheme),
            "diphone" => Ok(UnitLevel::Diphone),
            "phoneme" => Ok(UnitLevel::Phoneme),
            other => Err(format!("unknown unit level: {other}")),
        }
    }
}

/// Lowercase, strip punctuation (keeping intra-word apostrophes), collapse
/// whitespace. All cache keys are normalized text.
pub fn normalize(text: &str) -> String {
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut out = String::with_capacity(chars.len());
    let mut last_was_space = true;
    for (i, &c) in chars.iter().enumerate() {
        if c.is_alphanumeric() {
            out.push(c);
            last_was_space = false;
        } else if c == '\''
            && i > 0
            && chars[i - 1].is_alphanumeric()
            && i + 1 < chars.len()
            && chars[i + 1].is_alphanumeric()
        {
            out.push(c);
        } else if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }
    out.trim().to_string()
}

/// Normalized word tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    normalize(text)
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// Candidates for morpheme fallback: longer cached words that may contain
/// `word` as a stem. Returns `(candidate, start_frac, end_frac)` — the
/// fractional span of the candidate's audio that approximates `word`.
///
/// v1 approximation: stem location inside the candidate is estimated by
/// character proportion, not phonetic alignment.
pub fn morpheme_candidates(word: &str) -> Vec<(String, f64, f64)> {
    let wlen = word.chars().count() as f64;
    let mut cands: Vec<String> = vec![
        format!("{word}s"),
        format!("{word}es"),
        format!("{word}ed"),
        format!("{word}d"),
        format!("{word}ing"),
    ];
    if let Some(last) = word.chars().last() {
        // consonant doubling: run -> running
        if last.is_ascii_alphabetic() && !"aeiou".contains(last) {
            cands.push(format!("{word}{last}ing"));
        }
    }
    if let Some(stem) = word.strip_suffix('e') {
        // make -> making
        if !stem.is_empty() {
            cands.push(format!("{stem}ing"));
        }
    }
    let mut out: Vec<(String, f64, f64)> = cands
        .into_iter()
        .map(|c| {
            let end = wlen / c.chars().count() as f64;
            (c, 0.0, end)
        })
        .collect();
    // un- prefix: stem sits at the tail of the candidate
    let un = format!("un{word}");
    let start = 2.0 / un.chars().count() as f64;
    out.push((un, start, 1.0));
    out
}
