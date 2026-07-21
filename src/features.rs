//! Deterministic unit embeddings for approximate acoustic/phonetic retrieval.
//!
//! 64 dims: 8 crude audio features + 56 hashed char-trigram text dims,
//! L2-normalized. v1 keeps this dependency-free; the vector is swappable for
//! real MFCC/speaker embeddings later without changing the store schema.

use crate::audio::WavAudio;
use crate::units;

pub const DIM: usize = 64;
const TEXT_DIMS: usize = 56;

/// FNV-1a 32-bit.
pub fn fnv(s: &str) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Goertzel magnitude at `freq` Hz, normalized by length.
pub(crate) fn goertzel(samples: &[i16], sample_rate: u32, freq: f64) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let w = 2.0 * std::f64::consts::PI * freq / sample_rate as f64;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &x in samples {
        let s0 = x as f64 / 32768.0 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).sqrt() / n as f64
}

pub fn embed(text: &str, audio: &WavAudio) -> Vec<f32> {
    let mut v = [0.0f32; DIM];

    // --- audio features (dims 0..8)
    v[0] = (audio.duration_ms().ln_1p() / 8.0) as f32;
    v[1] = audio.rms() as f32;
    v[2] = audio.zero_crossing_rate() as f32;
    let bands: Vec<f64> = [250.0, 750.0, 2000.0, 5000.0]
        .iter()
        .map(|&f| goertzel(&audio.samples, audio.sample_rate, f))
        .collect();
    let bsum: f64 = bands.iter().sum::<f64>().max(1e-12);
    for (i, b) in bands.iter().enumerate() {
        v[3 + i] = (b / bsum) as f32;
    }
    // crude pitch proxy, scaled to ~0..1
    v[7] = (audio.zero_crossing_rate() * audio.sample_rate as f64 / 2.0 / 2000.0) as f32;

    // --- text trigram hashing (dims 8..64)
    let padded = format!(" {} ", units::normalize(text));
    let chars: Vec<char> = padded.chars().collect();
    for w in chars.windows(3) {
        let tri: String = w.iter().collect();
        let h = fnv(&tri);
        let idx = 8 + (h as usize % TEXT_DIMS);
        let sign = if (h >> 13) & 1 == 1 { 1.0 } else { -1.0 };
        v[idx] += sign;
    }

    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v.to_vec()
}

/// Cosine distance (0 = identical direction). Exposed for tests and CLI.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        return 1.0;
    }
    1.0 - dot / denom
}

/// Deterministic "voice signature" (native stand-in for a learned speaker
/// embedding like ECAPA-TDNN): long-term spectral envelope + dynamics.
/// 12 dims, L2-normalized: 8 band energies (250..6000 Hz, normalized),
/// RMS, ZCR, pitch proxy, log-duration.
pub fn voice_signature(audio: &WavAudio) -> Vec<f32> {
    let mut v = [0.0f32; 12];
    let bands: Vec<f64> = [250.0, 500.0, 750.0, 1000.0, 1500.0, 2500.0, 4000.0, 6000.0]
        .iter()
        .map(|&f| goertzel(&audio.samples, audio.sample_rate, f))
        .collect();
    let sum: f64 = bands.iter().sum::<f64>().max(1e-12);
    for (i, b) in bands.iter().enumerate() {
        v[i] = (b / sum) as f32;
    }
    v[8] = audio.rms() as f32;
    v[9] = audio.zero_crossing_rate() as f32;
    v[10] = (audio.zero_crossing_rate() * audio.sample_rate as f64 / 2.0 / 2000.0) as f32;
    v[11] = (audio.duration_ms().ln_1p() / 8.0) as f32;
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v.to_vec()
}
