//! `mimic-mct-v1`: a small, self-contained transform speech codec.
//!
//! 20 ms frames (320 samples @ 16 kHz), MDCT analysis with a 50%-lapped
//! sine window, then per-frame encoding: silence flag, or a 320-bit
//! significance bitmap + 3-bit quantized coefficients. Decode is IMDCT +
//! overlap-add, so concatenated token streams decode in a single pass with
//! OLA smoothing frame joins — that is what makes token-space stitching
//! seam-friendlier than PCM splicing.
//!
//! This is the native codec behind the `AudioCodec` seam. Neural codecs
//! (DAC, XCodec2, EnCodec) produce far better tokens at far lower bitrates
//! but need a torch stack; they plug in via the sidecar contract
//! (scripts/codec_external.py) without changing the storage format seams.
//!
//! Token stream layout:
//!   header: "MCT1" | orig_samples u32 | sample_rate u32 | frame_count u32
//!   per frame: type u8
//!     0 = silence (1 byte total)
//!     1 = tonal: scale u16 | bitmap 40 B | 3-bit packed values

use crate::audio::WavAudio;
use crate::{MimicError, Result};
use std::sync::OnceLock;

pub const FRAME: usize = 320; // 20 ms @ 16 kHz
pub const WINDOW: usize = 640; // 50% lapped MDCT window
const SILENCE_THRESH: f64 = 1e-4;

pub trait AudioCodec {
    fn name(&self) -> &str;
    fn encode(&self, audio: &WavAudio) -> Vec<u8>;
    fn decode(&self, tokens: &[u8]) -> Result<WavAudio>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MimicMct;

/// Number of analysis frames for a sample count. One extra frame beyond
/// ceil(n/FRAME) so the *tail* is covered by two lapped windows — without
/// it the last ~20 ms of every unit decodes at half window energy.
pub fn frames_for(samples: usize) -> usize {
    samples.div_ceil(FRAME) + 1
}

/// Cosine table for the DCT-IV-based MDCT (k-major, n-minor).
fn tables() -> &'static Vec<Vec<f64>> {
    static T: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
    T.get_or_init(|| {
        let n = WINDOW as f64;
        (0..FRAME)
            .map(|k| {
                (0..WINDOW)
                    .map(|i| {
                        ((2.0 * std::f64::consts::PI / n)
                            * (i as f64 + 0.5 + n / 4.0)
                            * (k as f64 + 0.5))
                            .cos()
                    })
                    .collect()
            })
            .collect()
    })
}

fn sine_window() -> &'static [f64] {
    static W: OnceLock<Vec<f64>> = OnceLock::new();
    W.get_or_init(|| {
        (0..WINDOW)
            .map(|i| (std::f64::consts::PI * (i as f64 + 0.5) / WINDOW as f64).sin())
            .collect()
    })
    .as_slice()
}

/// MDCT of one 640-sample window (already windowed input) -> 320 coeffs.
fn mdct(windowed: &[f64]) -> Vec<f64> {
    let t = tables();
    let mut out = vec![0.0; FRAME];
    for (k, row) in t.iter().enumerate() {
        let mut acc = 0.0;
        for (i, &c) in row.iter().enumerate() {
            acc += windowed[i] * c;
        }
        out[k] = acc;
    }
    out
}

/// IMDCT: 320 coeffs -> 640 windowed samples (caller overlap-adds).
fn imdct(coeffs: &[f64]) -> Vec<f64> {
    let t = tables();
    let w = sine_window();
    let scale = 4.0 / WINDOW as f64;
    let mut out = vec![0.0; WINDOW];
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, row) in t.iter().enumerate() {
            acc += coeffs[k] * row[i];
        }
        *o = scale * acc * w[i];
    }
    out
}

impl AudioCodec for MimicMct {
    fn name(&self) -> &str {
        "mimic-mct-v1"
    }

    fn encode(&self, audio: &WavAudio) -> Vec<u8> {
        let w = sine_window();
        let n_frames = frames_for(audio.samples.len());
        let mut out = Vec::with_capacity(16 + n_frames * 48);
        out.extend_from_slice(b"MCT1");
        out.extend_from_slice(&(audio.samples.len() as u32).to_le_bytes());
        out.extend_from_slice(&audio.sample_rate.to_le_bytes());
        out.extend_from_slice(&(n_frames as u32).to_le_bytes());

        for f in 0..n_frames {
            // 640-sample window centered on this frame (previous half + current)
            let center = f * FRAME;
            let mut windowed = [0.0f64; WINDOW];
            for i in 0..WINDOW {
                let idx = center as i64 + i as i64 - (FRAME as i64);
                let s = if idx >= 0 && (idx as usize) < audio.samples.len() {
                    audio.samples[idx as usize] as f64 / 32768.0
                } else {
                    0.0
                };
                windowed[i] = s * w[i];
            }
            let coeffs = mdct(&windowed);
            let maxc = coeffs.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
            if maxc < SILENCE_THRESH {
                out.push(0u8);
                continue;
            }
            out.push(1u8);
            let scale = ((maxc * 32768.0).round() as u32).min(u16::MAX as u32) as u16;
            out.extend_from_slice(&scale.to_le_bytes());
            let maxc = scale as f64 / 32768.0;
            let mut bitmap = [0u8; 40];
            let mut values = Vec::new();
            for (k, &c) in coeffs.iter().enumerate() {
                let q = (c / maxc * 3.0).round().clamp(-3.0, 3.0) as i8;
                if q != 0 {
                    bitmap[k / 8] |= 1 << (k % 8);
                    values.push((q + 4) as u8); // 1..=7
                }
            }
            out.extend_from_slice(&bitmap);
            // 3-bit packing, LSB-first
            let mut acc: u32 = 0;
            let mut bits = 0u32;
            for v in values {
                acc |= (v as u32 & 0x7) << bits;
                bits += 3;
                while bits >= 8 {
                    out.push((acc & 0xff) as u8);
                    acc >>= 8;
                    bits -= 8;
                }
            }
            if bits > 0 {
                out.push((acc & 0xff) as u8);
            }
        }
        out
    }

    fn decode(&self, tokens: &[u8]) -> Result<WavAudio> {
        if tokens.len() < 16 || &tokens[0..4] != b"MCT1" {
            return Err(MimicError::Wav("not an MCT1 token stream".into()));
        }
        let rd = |o: usize| u32::from_le_bytes(tokens[o..o + 4].try_into().unwrap());
        let orig_samples = rd(4) as usize;
        let sample_rate = rd(8);
        let n_frames = rd(12) as usize;
        let mut pos = 16usize;
        let mut pcm = vec![0.0f64; n_frames * FRAME + FRAME];
        for f in 0..n_frames {
            if pos >= tokens.len() {
                return Err(MimicError::Wav("truncated token stream".into()));
            }
            let ftype = tokens[pos];
            pos += 1;
            let coeffs = if ftype == 0 {
                vec![0.0; FRAME]
            } else if ftype == 1 {
                if pos + 42 > tokens.len() {
                    return Err(MimicError::Wav("truncated tonal frame".into()));
                }
                let scale = u16::from_le_bytes([tokens[pos], tokens[pos + 1]]) as f64 / 32768.0;
                pos += 2;
                let bitmap = &tokens[pos..pos + 40];
                pos += 40;
                let n_vals = bitmap.iter().map(|b| b.count_ones() as usize).sum::<usize>();
                let n_bytes = n_vals * 3 / 8 + usize::from(n_vals * 3 % 8 != 0);
                if pos + n_bytes > tokens.len() {
                    return Err(MimicError::Wav("truncated values".into()));
                }
                let packed = &tokens[pos..pos + n_bytes];
                pos += n_bytes;
                let mut coeffs = vec![0.0; FRAME];
                let mut bitpos = 0usize;
                for (k, c) in coeffs.iter_mut().enumerate() {
                    if bitmap[k / 8] & (1 << (k % 8)) != 0 {
                        let byte = bitpos / 8;
                        let shift = bitpos % 8;
                        let raw = if shift <= 5 {
                            (packed[byte] >> shift) & 0x7
                        } else {
                            let lo = packed[byte] >> shift;
                            let hi = packed.get(byte + 1).copied().unwrap_or(0) << (8 - shift);
                            (lo | hi) & 0x7
                        };
                        *c = (raw as i8 - 4) as f64 / 3.0 * scale;
                        bitpos += 3;
                    }
                }
                coeffs
            } else {
                return Err(MimicError::Wav(format!("bad frame type {ftype}")));
            };
            let y = imdct(&coeffs);
            // synthesis lands where the analysis window stood:
            // [center - FRAME, center + FRAME)
            let center = f * FRAME;
            let base = center as i64 - FRAME as i64;
            for (i, v) in y.iter().enumerate() {
                let idx = base + i as i64;
                if idx >= 0 && (idx as usize) < pcm.len() {
                    pcm[idx as usize] += v;
                }
            }
        }
        let samples: Vec<i16> = pcm
            .iter()
            .take(orig_samples)
            .map(|v| (v * 32767.0).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
            .collect();
        Ok(WavAudio::new(samples, sample_rate))
    }
}

#[cfg(test)]
mod mdct_sanity {
    use super::*;

    #[test]
    fn mdct_ola_is_lossless_unquantized() {
        // window + MDCT/IMDCT + overlap-add without quantization
        let w = sine_window();
        let sr = 16000usize;
        let n = sr / 4; // 250 ms
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 220.0 * i as f64 / sr as f64).sin() * 0.5)
            .collect();
        let n_frames = frames_for(n);
        let mut out = vec![0.0f64; n + WINDOW];
        for f in 0..n_frames {
            let center = f * FRAME;
            let mut windowed = vec![0.0f64; WINDOW];
            for (i, wv) in windowed.iter_mut().enumerate() {
                let idx = center as i64 + i as i64 - FRAME as i64;
                if idx >= 0 && (idx as usize) < x.len() {
                    *wv = x[idx as usize] * w[i];
                }
            }
            let c = mdct(&windowed);
            let y = imdct(&c);
            let base = center as i64 - FRAME as i64;
            for (i, v) in y.iter().enumerate() {
                let idx = base + i as i64;
                if idx >= 0 {
                    out[idx as usize] += v;
                }
            }
        }
        let mut max_err = 0.0f64;
        for i in 0..n {
            max_err = max_err.max((out[i] - x[i]).abs());
        }
        assert!(max_err < 1e-6, "TDAC reconstruction error {max_err}");
    }
}

// ---- token-stream surgery (frame-granular) ----

struct ParsedStream<'a> {
    sample_rate: u32,
    orig_samples: u32,
    frames: Vec<&'a [u8]>,
}

/// Parse a token stream into header fields and raw per-frame byte slices.
fn parse_stream(tokens: &[u8]) -> Result<ParsedStream<'_>> {
    if tokens.len() < 16 || &tokens[0..4] != b"MCT1" {
        return Err(MimicError::Wav("not an MCT1 token stream".into()));
    }
    let rd = |o: usize| u32::from_le_bytes(tokens[o..o + 4].try_into().unwrap());
    let orig_samples = rd(4);
    let sample_rate = rd(8);
    let n_frames = rd(12) as usize;
    let mut frames = Vec::with_capacity(n_frames);
    let mut pos = 16usize;
    for _ in 0..n_frames {
        if pos >= tokens.len() {
            return Err(MimicError::Wav("truncated token stream".into()));
        }
        let start = pos;
        let ftype = tokens[pos];
        pos += 1;
        if ftype == 1 {
            if pos + 42 > tokens.len() {
                return Err(MimicError::Wav("truncated tonal frame".into()));
            }
            let bitmap = &tokens[pos + 2..pos + 42];
            let n_vals: usize = bitmap.iter().map(|b| b.count_ones() as usize).sum();
            pos += 42 + n_vals * 3 / 8 + usize::from(n_vals * 3 % 8 != 0);
            if pos > tokens.len() {
                return Err(MimicError::Wav("truncated values".into()));
            }
        } else if ftype != 0 {
            return Err(MimicError::Wav(format!("bad frame type {ftype}")));
        }
        frames.push(&tokens[start..pos]);
    }
    Ok(ParsedStream {
        sample_rate,
        orig_samples,
        frames,
    })
}

fn emit_stream(sample_rate: u32, orig_samples: u32, frames: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + frames.iter().map(|f| f.len()).sum::<usize>());
    out.extend_from_slice(b"MCT1");
    out.extend_from_slice(&orig_samples.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    for f in frames {
        out.extend_from_slice(f);
    }
    out
}

/// Slice a token stream by fractional offsets (frame-aligned). The unit's
/// morpheme/diphone sub-segments are cut without decoding.
pub fn slice_tokens(tokens: &[u8], f0: f64, f1: f64) -> Result<Vec<u8>> {
    let p = parse_stream(tokens)?;
    let nf = p.frames.len();
    let s = (f0.clamp(0.0, 1.0) * nf as f64).round() as usize;
    let e = (f1.clamp(0.0, 1.0) * nf as f64).round() as usize;
    let s = s.min(nf);
    let e = e.min(nf).max(s);
    let samples = ((e - s) * FRAME) as u32;
    Ok(emit_stream(p.sample_rate, samples, &p.frames[s..e]))
}

/// Number of frames in a token stream (header read, no decode).
pub fn token_frames(tokens: &[u8]) -> Result<u32> {
    if tokens.len() < 16 || &tokens[0..4] != b"MCT1" {
        return Err(MimicError::Wav("not an MCT1 token stream".into()));
    }
    Ok(u32::from_le_bytes(tokens[12..16].try_into().unwrap()))
}

/// Concatenate token streams frame-wise (no resampling, no re-encode).
pub fn concat_tokens(streams: &[&[u8]]) -> Result<Vec<u8>> {
    let mut sample_rate = 0u32;
    let mut samples = 0u32;
    let mut frames: Vec<&[u8]> = Vec::new();
    for s in streams {
        let p = parse_stream(s)?;
        if sample_rate == 0 {
            sample_rate = p.sample_rate;
        }
        if p.sample_rate != sample_rate {
            return Err(MimicError::SampleRateMismatch(sample_rate, p.sample_rate));
        }
        samples += p.orig_samples;
        frames.extend(p.frames);
    }
    Ok(emit_stream(sample_rate, samples, &frames))
}
