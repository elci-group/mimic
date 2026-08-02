//! Minimal WAV codec and splicing. No external crates.

use crate::{MimicError, Result};
use std::path::Path;

/// Mono 16-bit PCM audio.
#[derive(Debug, Clone, PartialEq)]
pub struct WavAudio {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
}

impl WavAudio {
    pub fn new(samples: Vec<i16>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn duration_ms(&self) -> f64 {
        self.samples.len() as f64 * 1000.0 / self.sample_rate as f64
    }

    /// Max absolute amplitude (as i32 so i16::MIN doesn't overflow).
    pub fn peak(&self) -> i32 {
        self.samples
            .iter()
            .map(|s| s.unsigned_abs() as i32)
            .max()
            .unwrap_or(0)
    }

    /// Root-mean-square amplitude, normalized to 0..1.
    pub fn rms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .samples
            .iter()
            .map(|&s| {
                let x = s as f64 / 32768.0;
                x * x
            })
            .sum();
        (sum / self.samples.len() as f64).sqrt()
    }

    /// Fraction of samples that cross zero (crude voicing/pitch indicator).
    pub fn zero_crossing_rate(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let crossings = self
            .samples
            .windows(2)
            .filter(|w| (w[0] < 0) != (w[1] < 0))
            .count();
        crossings as f64 / (self.samples.len() - 1) as f64
    }

    /// Sub-slice by fractional offsets, clamped to [0, 1].
    pub fn slice_frac(&self, start: f64, end: f64) -> WavAudio {
        let n = self.samples.len();
        let s = (start.clamp(0.0, 1.0) * n as f64).round() as usize;
        let e = (end.clamp(0.0, 1.0) * n as f64).round() as usize;
        let s = s.min(n);
        let e = e.min(n).max(s);
        WavAudio::new(self.samples[s..e].to_vec(), self.sample_rate)
    }
}

fn wav_err(msg: &str) -> MimicError {
    MimicError::Wav(msg.to_string())
}

pub fn write_wav<P: AsRef<Path>>(audio: &WavAudio, path: P) -> Result<()> {
    std::fs::write(path, to_wav_bytes(audio))?;
    Ok(())
}

/// Serialize as a standard 44-byte-header RIFF/WAVE PCM file.
pub fn to_wav_bytes(audio: &WavAudio) -> Vec<u8> {
    let data_len = (audio.samples.len() * 2) as u32;
    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&audio.sample_rate.to_le_bytes());
    buf.extend_from_slice(&(audio.sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for s in &audio.samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}

pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<WavAudio> {
    let data = std::fs::read(path)?;
    parse_wav(&data)
}

/// Parse RIFF/WAVE PCM. Chunk walk is defensive: fmt chunk may be 16/18/40
/// bytes, unknown chunks are skipped, chunks are padded to even sizes.
pub fn parse_wav(data: &[u8]) -> Result<WavAudio> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(wav_err("not a RIFF/WAVE file"));
    }
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut raw: Option<Vec<i16>> = None;
    let mut pos = 12usize;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + size > data.len() {
            return Err(wav_err("truncated chunk"));
        }
        match id {
            b"fmt " => {
                if size < 16 {
                    return Err(wav_err("fmt chunk too small"));
                }
                let c = &data[pos..pos + 16];
                fmt = Some((
                    u16::from_le_bytes(c[0..2].try_into().unwrap()),
                    u16::from_le_bytes(c[2..4].try_into().unwrap()),
                    u32::from_le_bytes(c[4..8].try_into().unwrap()),
                    u16::from_le_bytes(c[14..16].try_into().unwrap()),
                ));
            }
            b"data" => {
                let mut s = Vec::with_capacity(size / 2);
                for pair in data[pos..pos + size].chunks_exact(2) {
                    s.push(i16::from_le_bytes([pair[0], pair[1]]));
                }
                raw = Some(s);
            }
            _ => {}
        }
        pos += size + (size & 1);
    }
    let (format, channels, rate, bits) =
        fmt.ok_or_else(|| wav_err("missing fmt chunk; provide a valid WAV file and try again"))?;
    if format != 1 {
        return Err(wav_err(
            "only PCM (format 1) supported; convert the source to PCM",
        ));
    }
    if bits != 16 {
        return Err(wav_err(
            "only 16-bit samples supported; convert the source to 16-bit",
        ));
    }
    let raw =
        raw.ok_or_else(|| wav_err("missing data chunk; provide a valid WAV file and try again"))?;
    let samples = match channels {
        1 => raw,
        2 => raw
            .chunks_exact(2)
            .map(|lr| ((lr[0] as i32 + lr[1] as i32) / 2) as i16)
            .collect(),
        other => return Err(wav_err(&format!("unsupported channel count: {other}"))),
    };
    Ok(WavAudio::new(samples, rate))
}

/// Discontinuity proxy across a join between two parts: combines the RMS
/// step and zero-crossing-rate step across the boundary (10 ms windows).
/// Lower is smoother. Used as the native seam metric until neural
/// perceptual metrics (UTMOS et al.) are wired in.
pub fn seam_discontinuity(a: &WavAudio, b: &WavAudio) -> f64 {
    let w = (a.sample_rate as usize / 100).max(1);
    let tail = &a.samples[a.samples.len().saturating_sub(w)..];
    let head = &b.samples[..b.samples.len().min(w)];
    let rms = |s: &[i16]| {
        if s.is_empty() {
            return 0.0;
        }
        (s.iter()
            .map(|&x| {
                let v = x as f64 / 32768.0;
                v * v
            })
            .sum::<f64>()
            / s.len() as f64)
            .sqrt()
    };
    let zcr = |s: &[i16]| {
        if s.len() < 2 {
            return 0.0;
        }
        s.windows(2).filter(|p| (p[0] < 0) != (p[1] < 0)).count() as f64 / (s.len() - 1) as f64
    };
    (rms(tail) - rms(head)).abs() + (zcr(tail) - zcr(head)).abs()
}

/// Concatenate parts with a raised-cosine crossfade at each join. The fade
/// gains sum to 1.0, so splicing cannot clip louder than the louder part.
pub fn splice(parts: &[WavAudio], crossfade_ms: u32) -> Result<WavAudio> {
    let Some(first) = parts.first() else {
        return Ok(WavAudio::new(Vec::new(), crate::SAMPLE_RATE));
    };
    let sr = first.sample_rate;
    for p in parts {
        if p.sample_rate != sr {
            return Err(MimicError::SampleRateMismatch(sr, p.sample_rate));
        }
    }
    let fade_len = (sr as u64 * crossfade_ms as u64 / 1000) as usize;
    let mut out: Vec<i16> = Vec::new();
    for part in parts {
        if out.is_empty() {
            out.extend_from_slice(&part.samples);
            continue;
        }
        let f = fade_len.min(out.len() / 2).min(part.samples.len() / 2);
        if f == 0 {
            out.extend_from_slice(&part.samples);
            continue;
        }
        let tail = out.split_off(out.len() - f);
        for (i, tail_sample) in tail.iter().enumerate().take(f) {
            let t = (i as f64 + 0.5) / f as f64;
            let g_out = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
            let g_in = 1.0 - g_out;
            let mixed = *tail_sample as f64 * g_out + part.samples[i] as f64 * g_in;
            out.push(mixed.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
        }
        out.extend_from_slice(&part.samples[f..]);
    }
    Ok(WavAudio::new(out, sr))
}
