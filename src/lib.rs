//! Mimic: a phonetic memoization layer for TTS.
//!
//! Generated speech is decomposed into reusable audio primitives (phrases,
//! words, morphemes) that are stored in a padagonia graph/vector database.
//! Future speech is reconstructed by stitching cached units, sending only the
//! missing spans to the TTS provider.

pub mod align;
pub mod audio;
pub mod codec;
pub mod eval;
pub mod features;
pub mod g2p;
pub mod http;
pub mod metrics;
pub mod pipeline;
pub mod providers;
pub mod select;
pub mod server;
pub mod ssml;
pub mod store;
pub mod tts;
pub mod units;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MimicError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("padagonia store error: {0}")]
    Store(#[from] padagonia::StoreError),
    #[error("wav error: {0}")]
    Wav(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("sample rate mismatch: {0} Hz vs {1} Hz")]
    SampleRateMismatch(u32, u32),
}

pub type Result<T> = std::result::Result<T, MimicError>;

/// Canonical audio format for the whole pipeline: 16 kHz mono i16 PCM.
pub const SAMPLE_RATE: u32 = 16_000;

/// Crossfade applied at unit joins when splicing (boundary smoothing).
pub const CROSSFADE_MS: u32 = 10;
