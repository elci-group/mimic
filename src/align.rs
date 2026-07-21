//! Word/phoneme boundary alignment.
//!
//! v1 used character-proportional segmentation (kept as
//! [`AlignMode::Proportional`] — the eval harness's historical baseline).
//! P1 refines each seeded boundary by snapping it to the deepest energy
//! trough in a search window — a classic silence/low-energy segmentation
//! technique that needs no external models. Boundaries are snapped to the
//! *middle* of the trough region (inter-word pauses are plateaus, and the
//! middle is the least-biased estimate of the true word edge).

use crate::audio::WavAudio;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlignMode {
    /// v1 behavior: character-proportional splits, no signal analysis.
    Proportional,
    /// Seed proportional, snap to energy troughs within the window.
    EnergyRefined { search_ms: f64 },
}

impl Default for AlignMode {
    fn default() -> Self {
        AlignMode::EnergyRefined { search_ms: 100.0 }
    }
}

/// Half-open sample span [start, end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// Proportional seed boundaries from weights (v1 math).
fn proportional_bounds(total: usize, weights: &[usize]) -> Vec<usize> {
    let sum: usize = weights.iter().sum();
    let mut bounds = Vec::with_capacity(weights.len() + 1);
    bounds.push(0);
    let mut cum = 0usize;
    for w in weights {
        cum += w;
        bounds.push((total as u128 * cum as u128 / sum as u128) as usize);
    }
    bounds
}

/// Frame RMS energy profile.
fn energy_profile(samples: &[i16], frame: usize, hop: usize) -> Vec<f32> {
    if samples.len() < frame {
        return Vec::new();
    }
    samples
        .chunks(hop)
        .take(samples.len() / hop)
        .map(|w| {
            let n = w.len().min(frame);
            if n == 0 {
                return 0.0;
            }
            let sum: f64 = w[..n]
                .iter()
                .map(|&s| {
                    let x = s as f64 / 32768.0;
                    x * x
                })
                .sum();
            (sum / n as f64).sqrt() as f32
        })
        .collect()
}

/// Snap a seeded boundary (in samples) to the middle of the deepest energy
/// trough within ±search. If the window is energetically flat (no reliable
/// trough, e.g. a steady tone), keep the seed.
fn snap_to_trough(seed: usize, profile: &[f32], hop: usize, search: usize) -> usize {
    if profile.is_empty() || search == 0 {
        return seed;
    }
    let center = seed / hop;
    let radius = search / hop;
    let lo = center.saturating_sub(radius);
    let hi = (center + radius + 1).min(profile.len());
    if lo >= hi {
        return seed;
    }
    let window = &profile[lo..hi];
    let min = window.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = window.iter().cloned().fold(0.0f32, f32::max);
    // Flat window => no trough information; keep the seed.
    if max < 1e-6 || (max - min) / max < 0.05 {
        return seed;
    }
    // Middle of the (near-)minimum plateau — the least-biased trough point.
    let thresh = min + (max - min) * 0.05;
    let idxs: Vec<usize> = window
        .iter()
        .enumerate()
        .filter(|(_, &e)| e <= thresh)
        .map(|(i, _)| lo + i)
        .collect();
    let mid = idxs[idxs.len() / 2];
    mid * hop + hop / 2
}

/// Refine seed boundaries left-to-right, keeping them monotonic with a
/// minimum unit duration.
fn refine(bounds: &mut [usize], audio: &WavAudio, search_ms: f64, min_unit_ms: f64) {
    let sr = audio.sample_rate as f64;
    let hop = (sr * 0.0025) as usize; // 2.5 ms
    let frame = (sr * 0.005) as usize; // 5 ms
    let profile = energy_profile(&audio.samples, frame, hop);
    let search = (sr * search_ms / 1000.0) as usize;
    let min_unit = (sr * min_unit_ms / 1000.0) as usize;
    for i in 1..bounds.len() - 1 {
        let floor = bounds[i - 1] + min_unit;
        let snapped = snap_to_trough(bounds[i], &profile, hop, search);
        bounds[i] = snapped.max(floor).min(audio.samples.len());
    }
}

/// Word spans for `words` over `audio`. Weights include a leading-space
/// share for every word after the first (same convention as v1).
pub fn word_spans(audio: &WavAudio, words: &[String], mode: AlignMode) -> Vec<Span> {
    if words.is_empty() {
        return Vec::new();
    }
    let weights: Vec<usize> = words
        .iter()
        .enumerate()
        .map(|(i, w)| w.chars().count() + usize::from(i > 0))
        .collect();
    let mut bounds = proportional_bounds(audio.samples.len(), &weights);
    if let AlignMode::EnergyRefined { search_ms } = mode {
        refine(&mut bounds, audio, search_ms, 30.0);
    }
    bounds
        .windows(2)
        .map(|w| Span {
            start: w[0],
            end: w[1],
        })
        .collect()
}

/// Phoneme spans inside a word span: proportional by phoneme count, with a
/// tighter refinement window (phoneme coarticulation makes deep troughs
/// rare, so this mainly catches stop gaps and silibant boundaries).
pub fn phoneme_spans(word_audio: &WavAudio, n_phonemes: usize, search_ms: f64) -> Vec<Span> {
    if n_phonemes == 0 {
        return Vec::new();
    }
    let weights = vec![1usize; n_phonemes];
    let mut bounds = proportional_bounds(word_audio.samples.len(), &weights);
    refine(&mut bounds, word_audio, search_ms, 10.0);
    bounds
        .windows(2)
        .map(|w| Span {
            start: w[0],
            end: w[1],
        })
        .collect()
}
