//! TTS provider abstraction and a deterministic offline mock.

use crate::audio::WavAudio;
use crate::{Result, SAMPLE_RATE};
use std::sync::Mutex;

pub trait TtsProvider {
    fn name(&self) -> &str;
    fn synthesize(&self, text: &str, voice: &str) -> Result<WavAudio>;
    /// Approximate USD per 1M generated characters (0 for local/mock).
    fn cost_per_million_chars(&self) -> f64 {
        0.0
    }
}

/// Deterministic offline stand-in for a real TTS provider (ElevenLabs,
/// Gemini, ...). Each word becomes a buzzy harmonic tone whose pitch is a
/// hash of (voice, word); silence separates words. Records every synthesized
/// text in `calls` so tests and the CLI can show exactly what was generated
/// (i.e. what the cache did *not* save). Mutex-wrapped so it is `Sync` for
/// the HTTP server.
pub struct MockTts {
    pub calls: Mutex<Vec<String>>,
}

impl MockTts {
    pub fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl Default for MockTts {
    fn default() -> Self {
        Self::new()
    }
}

fn envelope(i: usize, n: usize) -> f64 {
    let edge = (SAMPLE_RATE as usize / 100).min(n / 2); // 10 ms
    if edge == 0 {
        return 1.0;
    }
    let mut g = 1.0;
    if i < edge {
        g *= 0.5 * (1.0 - (std::f64::consts::PI * i as f64 / edge as f64).cos());
    }
    if i >= n - edge {
        g *= 0.5
            * (1.0 + (std::f64::consts::PI * (i - (n - edge)) as f64 / edge as f64).cos());
    }
    g
}

/// Per-word synthesized duration (shared by `synthesize` and
/// `mock_word_spans` so the two can't drift).
fn word_duration_ms(word: &str) -> f64 {
    (word.chars().count() as f64 * 90.0).clamp(180.0, 900.0)
}

/// Inter-word silence inserted by `synthesize`.
pub const GAP_MS: f64 = 60.0;

/// Deterministic ground-truth word spans (in milliseconds) for a MockTts
/// rendering of `text`: (word, start_ms, end_ms), gaps excluded.
/// Used by the eval harness as the alignment reference.
pub fn mock_word_spans(text: &str) -> Vec<(String, f64, f64)> {
    let mut spans = Vec::new();
    let mut cursor = 0.0;
    for word in text.split_whitespace() {
        let dur = word_duration_ms(word);
        spans.push((word.to_string(), cursor, cursor + dur));
        cursor += dur + GAP_MS;
    }
    spans
}

impl TtsProvider for MockTts {
    fn name(&self) -> &str {
        "mock-tts"
    }

    fn synthesize(&self, text: &str, voice: &str) -> Result<WavAudio> {
        self.calls.lock().unwrap().push(text.to_string());
        let mut samples = Vec::new();
        for word in text.split_whitespace() {
            let h = crate::features::fnv(&format!("{voice}:{word}"));
            let freq = 180.0 + (h % 340) as f64; // 180..520 Hz
            let dur_ms = word_duration_ms(word);
            let n = (SAMPLE_RATE as f64 * dur_ms / 1000.0) as usize;
            for i in 0..n {
                let t = i as f64 / SAMPLE_RATE as f64;
                let vib = 1.0 + 0.05 * (2.0 * std::f64::consts::PI * 5.0 * t).sin();
                let s = (2.0 * std::f64::consts::PI * freq * t * vib).sin()
                    + 0.5 * (4.0 * std::f64::consts::PI * freq * t).sin()
                    + 0.25 * (6.0 * std::f64::consts::PI * freq * t).sin();
                samples.push((s * envelope(i, n) * 0.35 * i16::MAX as f64) as i16);
            }
            // inter-word silence
            samples.extend(
                std::iter::repeat(0).take(SAMPLE_RATE as usize * GAP_MS as usize / 1000),
            );
        }
        Ok(WavAudio::new(samples, SAMPLE_RATE))
    }
}
