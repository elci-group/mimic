//! Mimic: a phonetic memoization layer for TTS.
//!
//! Generated speech is decomposed into reusable audio primitives (phrases,
//! words, morphemes) that are stored in a padagonia graph/vector database.
//! Future speech is reconstructed by stitching cached units, sending only the
//! missing spans to the TTS provider.

pub mod align;
pub mod audio;
pub mod codec;
pub mod daemon;
pub mod eval;
pub mod features;
pub mod g2p;
pub mod http;
pub mod metrics;
pub mod pipeline;
pub mod plan;
pub mod providers;
pub mod select;
pub mod server;
pub mod ssml;
pub mod store;
pub mod tts;
pub mod units;

#[derive(Debug)]
pub enum MimicError {
    Io(std::io::Error),
    Store(padagonia::StoreError),
    Wav(String),
    NotFound(String),
    SampleRateMismatch(u32, u32),
}

impl std::fmt::Display for MimicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}; check the path and retry"),
            Self::Store(e) => write!(
                f,
                "padagonia store error: {e}; validate the store and retry"
            ),
            Self::Wav(e) => write!(f, "wav error: {e}; provide 16 kHz mono PCM and retry"),
            Self::NotFound(e) => write!(f, "not found: {e}; create a new plan and retry"),
            Self::SampleRateMismatch(a, b) => write!(
                f,
                "sample rate mismatch: {a} Hz vs {b} Hz; resample to 16 kHz and retry"
            ),
        }
    }
}

impl std::error::Error for MimicError {}

impl From<std::io::Error> for MimicError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<padagonia::StoreError> for MimicError {
    fn from(value: padagonia::StoreError) -> Self {
        Self::Store(value)
    }
}

pub type Result<T> = std::result::Result<T, MimicError>;

/// Canonical audio format for the whole pipeline: 16 kHz mono i16 PCM.
pub const SAMPLE_RATE: u32 = 16_000;

/// Crossfade applied at unit joins when splicing (boundary smoothing).
pub const CROSSFADE_MS: u32 = 10;
