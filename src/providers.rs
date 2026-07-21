//! Real TTS provider implementations behind the `TtsProvider` seam.
//!
//! Wire formats are implemented per each provider's documented API and are
//! covered by local mock-server tests. They have NOT been validated against
//! live endpoints: this machine has no API keys and the HTTP client is
//! `http://` only (no TLS stack yet) — fronting a provider with an HTTP
//! gateway or adding rustls is the P2.5 follow-up. Cost figures are
//! approximate public list prices; verify before relying on them.

use crate::audio::{self, WavAudio};
use crate::http;
use crate::tts::TtsProvider;
use crate::{MimicError, Result, SAMPLE_RATE};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(30);

/// Approximate USD per 1M generated characters (public list prices, 2025;
/// configurable — check current pricing before trusting cost reports).
pub fn cost_per_million_chars(provider: &str) -> f64 {
    match provider {
        "openai-tts" => 15.0,
        "openai-tts-hd" => 30.0,
        "elevenlabs" => 300.0,
        "cartesia" => 100.0,
        "gemini-tts" => 30.0,
        "mock-tts" => 0.0,
        _ => 0.0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProviderKind {
    OpenAi,
    ElevenLabs,
    Cartesia,
    Gemini,
}

impl ProviderKind {
    pub fn name(&self) -> &'static str {
        match self {
            ProviderKind::OpenAi => "openai-tts",
            ProviderKind::ElevenLabs => "elevenlabs",
            ProviderKind::Cartesia => "cartesia",
            ProviderKind::Gemini => "gemini-tts",
        }
    }
}

/// An HTTP TTS provider. `base_url` includes scheme+host (and any gateway
/// path prefix); endpoints are appended per provider kind.
pub struct HttpProvider {
    pub kind: ProviderKind,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub default_voice: String,
}

impl HttpProvider {
    pub fn new(
        kind: ProviderKind,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        default_voice: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
            default_voice: default_voice.into(),
        }
    }

    fn voice_or<'a>(&'a self, voice: &'a str) -> &'a str {
        if voice == "default" || voice.is_empty() {
            &self.default_voice
        } else {
            voice
        }
    }

    fn request(&self, text: &str, voice: &str) -> (String, Vec<(String, String)>, Vec<u8>) {
        let v = self.voice_or(voice);
        match self.kind {
            ProviderKind::OpenAi => (
                format!("{}/v1/audio/speech", self.base_url),
                vec![
                    ("Authorization".into(), format!("Bearer {}", self.api_key)),
                    ("Content-Type".into(), "application/json".into()),
                ],
                serde_json::json!({
                    "model": self.model,
                    "voice": v,
                    "input": text,
                    "response_format": "wav"
                })
                .to_string()
                .into_bytes(),
            ),
            ProviderKind::ElevenLabs => (
                format!(
                    "{}/v1/text-to-speech/{}?output_format=pcm_16000",
                    self.base_url, v
                ),
                vec![
                    ("xi-api-key".into(), self.api_key.clone()),
                    ("Content-Type".into(), "application/json".into()),
                ],
                serde_json::json!({ "text": text, "model_id": self.model })
                    .to_string()
                    .into_bytes(),
            ),
            ProviderKind::Cartesia => (
                format!("{}/tts/bytes", self.base_url),
                vec![
                    ("X-API-Key".into(), self.api_key.clone()),
                    ("Cartesia-Version".into(), "2024-06-10".into()),
                    ("Content-Type".into(), "application/json".into()),
                ],
                serde_json::json!({
                    "model_id": self.model,
                    "transcript": text,
                    "voice": {"mode": "id", "id": v},
                    "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 16000}
                })
                .to_string()
                .into_bytes(),
            ),
            ProviderKind::Gemini => (
                format!(
                    "{}/v1beta/models/{}:generateContent?key={}",
                    self.base_url, self.model, self.api_key
                ),
                vec![("Content-Type".into(), "application/json".into())],
                serde_json::json!({
                    "contents": [{"parts": [{"text": text}]}],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": v}}}
                    }
                })
                .to_string()
                .into_bytes(),
            ),
        }
    }

    fn parse(&self, body: &[u8]) -> Result<WavAudio> {
        match self.kind {
            ProviderKind::OpenAi | ProviderKind::Cartesia => audio::parse_wav(body),
            // pcm_16000: raw s16le mono at 16 kHz
            ProviderKind::ElevenLabs => Ok(WavAudio::new(
                body.chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect(),
                SAMPLE_RATE,
            )),
            // inline_data: base64 L16 PCM at 24 kHz -> crude 3:2 decimation
            ProviderKind::Gemini => {
                let v: serde_json::Value = serde_json::from_slice(body)
                    .map_err(|e| MimicError::Wav(format!("gemini: bad json: {e}")))?;
                let b64 = v["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                    .as_str()
                    .ok_or_else(|| MimicError::Wav("gemini: no inlineData.data".into()))?;
                let pcm = base64_decode(b64)?;
                let s24: Vec<i16> = pcm
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                // 24000 -> 16000: keep 2 of every 3 samples (documented crude)
                let s16: Vec<i16> = s24
                    .chunks_exact(3)
                    .flat_map(|c| [c[0], c[1]])
                    .collect();
                Ok(WavAudio::new(s16, SAMPLE_RATE))
            }
        }
    }
}

impl TtsProvider for HttpProvider {
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn cost_per_million_chars(&self) -> f64 {
        cost_per_million_chars(self.kind.name())
    }

    fn synthesize(&self, text: &str, voice: &str) -> Result<WavAudio> {
        let (url, headers, body) = self.request(text, voice);
        let h: Vec<(&str, &str)> = headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let resp = http::post(&url, &h, &body, TIMEOUT)?;
        if resp.status != 200 {
            return Err(MimicError::Wav(format!(
                "{}: HTTP {}: {}",
                self.kind.name(),
                resp.status,
                String::from_utf8_lossy(&resp.body[..resp.body.len().min(200)])
            )));
        }
        self.parse(&resp.body)
    }
}

/// Local open-model provider via Python sidecar (Kokoro/F5-TTS/Chatterbox).
/// Contract: `python3 <script> <text> <out.wav>` writes 16 kHz mono wav.
/// A ready-made Kokoro script is planned; any script honoring the contract
/// works. Unavailable models report a clear error, like the eval adapter.
pub struct SidecarTts {
    pub script: std::path::PathBuf,
    pub provider_name: String,
}

impl TtsProvider for SidecarTts {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn synthesize(&self, text: &str, _voice: &str) -> Result<WavAudio> {
        if !self.script.exists() {
            return Err(MimicError::NotFound(format!(
                "sidecar script {} not found — see README P2 section",
                self.script.display()
            )));
        }
        let out = std::env::temp_dir().join(format!("mimic-sidecar-{}.wav", std::process::id()));
        let status = std::process::Command::new("python3")
            .arg(&self.script)
            .arg(text)
            .arg(&out)
            .status()
            .map_err(|e| MimicError::Wav(format!("sidecar run: {e}")))?;
        if !status.success() {
            return Err(MimicError::Wav(format!("sidecar exited {status}")));
        }
        let audio = audio::read_wav(&out)?;
        let _ = std::fs::remove_file(&out);
        Ok(audio)
    }
}

/// Minimal RFC 4648 base64 decoder (standard alphabet, padding optional).
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let mut table = [255u8; 256];
    for (i, c) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        .iter()
        .enumerate()
    {
        table[*c as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits = 0u32;
    for &b in s.as_bytes() {
        if b == b'=' || b.is_ascii_whitespace() {
            continue;
        }
        let v = table[b as usize];
        if v == 255 {
            return Err(MimicError::Wav(format!("base64: bad byte {b:#x}")));
        }
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Ok(out)
}
