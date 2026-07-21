//! Native objective metrics. STOI is the P4 gate metric; cosine backs the
//! voice-fidelity measurement. Heavy perceptual models (ViSQOL, UTMOS,
//! ECAPA-SECS) remain on the external adapter — see scripts/eval_external.py.

use crate::audio::WavAudio;

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let d = na.sqrt() * nb.sqrt();
    if d < 1e-12 {
        0.0
    } else {
        dot / d
    }
}

/// In-place iterative radix-2 FFT (re/im, length must be a power of two).
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j &= !bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * std::f64::consts::PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let (wr, wi) = (ang.cos(), ang.sin());
        for i in (0..n).step_by(len) {
            let (mut cur_wr, mut cur_wi) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (a, b) = (i + k, i + k + len / 2);
                let (tr, ti) = (re[b] * cur_wr - im[b] * cur_wi, re[b] * cur_wi + im[b] * cur_wr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nwr = cur_wr * wr - cur_wi * wi;
                cur_wi = cur_wr * wi + cur_wi * wr;
                cur_wr = nwr;
            }
        }
        len <<= 1;
    }
}

const NFFT: usize = 256;
const HOP: usize = 128;
const SEG_FRAMES: usize = 30; // ~384 ms analysis segments

/// One-third-octave band edges: 15 bands from 150 Hz upward.
fn band_edges(sample_rate: u32) -> Vec<(usize, usize)> {
    let hz_per_bin = sample_rate as f64 / NFFT as f64;
    let mut edges = Vec::new();
    for m in 0..15 {
        let fc = 150.0 * 2f64.powf(m as f64 / 3.0);
        let lo = (fc / 2f64.powf(1.0 / 6.0) / hz_per_bin).floor() as usize;
        let hi = (fc * 2f64.powf(1.0 / 6.0) / hz_per_bin).ceil() as usize;
        let hi = hi.min(NFFT / 2).max(lo + 1);
        edges.push((lo, hi));
    }
    edges
}

/// Band-energy envelopes: [band][frame].
fn band_envelopes(samples: &[i16], sample_rate: u32) -> Vec<Vec<f64>> {
    let edges = band_edges(sample_rate);
    let mut bands = vec![Vec::new(); edges.len()];
    let hann: Vec<f64> = (0..NFFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (NFFT - 1) as f64).cos()))
        .collect();
    let mut re = vec![0.0f64; NFFT];
    let mut im = vec![0.0f64; NFFT];
    let mut f = 0usize;
    while f * HOP + NFFT <= samples.len() {
        for (i, r) in re.iter_mut().enumerate() {
            *r = samples[f * HOP + i] as f64 / 32768.0 * hann[i];
        }
        im.iter_mut().for_each(|v| *v = 0.0);
        fft(&mut re, &mut im, false);
        for (b, &(lo, hi)) in edges.iter().enumerate() {
            let e: f64 = (lo..hi).map(|k| re[k] * re[k] + im[k] * im[k]).sum();
            bands[b].push((e + 1e-12).ln());
        }
        f += 1;
    }
    bands
}

/// Short-Time Objective Intelligibility (correlation-core variant of
/// Taal et al.): per-band, per-segment normalized correlation of the
/// log band-energy envelopes, averaged. 1.0 = identical; ~0 = unintelligible.
pub fn stoi(clean: &WavAudio, degraded: &WavAudio) -> f64 {
    let n = clean.samples.len().min(degraded.samples.len());
    if n < NFFT * 2 || clean.sample_rate != degraded.sample_rate {
        return 0.0;
    }
    let x = band_envelopes(&clean.samples[..n], clean.sample_rate);
    let y = band_envelopes(&degraded.samples[..n], degraded.sample_rate);
    let mut total = 0.0f64;
    let mut count = 0usize;
    for (xb, yb) in x.iter().zip(y.iter()) {
        let frames = xb.len().min(yb.len());
        if frames < 6 {
            continue;
        }
        let mut s = 0usize;
        while s < frames {
            let e = (s + SEG_FRAMES).min(frames);
            let xs = &xb[s..e];
            let ys = &yb[s..e];
            let mx = xs.iter().sum::<f64>() / xs.len() as f64;
            let my = ys.iter().sum::<f64>() / ys.len() as f64;
            let mut num = 0.0f64;
            let mut dx = 0.0f64;
            let mut dy = 0.0f64;
            for i in 0..xs.len() {
                let a = xs[i] - mx;
                let b = ys[i] - my;
                num += a * b;
                dx += a * a;
                dy += b * b;
            }
            if dx > 1e-9 && dy > 1e-9 {
                total += num / (dx.sqrt() * dy.sqrt());
                count += 1;
            }
            s += SEG_FRAMES;
        }
    }
    if count == 0 {
        return 0.0;
    }
    (total / count as f64).clamp(0.0, 1.0)
}
