//! Provider-free cache planning and deterministic artifact manifests.

use crate::audio::{self, WavAudio};
use crate::g2p::G2p;
use crate::pipeline;
use crate::store::MimicStore;
use crate::units::{self, UnitLevel};
use crate::{MimicError, Result, SAMPLE_RATE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const PLAN_TTL: Duration = Duration::from_secs(60);
const RAM_HEADROOM: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize)]
pub struct PlanRequest {
    pub text: String,
    pub voice_id: String,
    #[serde(default = "default_model")]
    pub model_id: String,
    #[serde(default)]
    pub settings_key: String,
}

fn default_model() -> String {
    "eleven_multilingual_v2".into()
}

#[derive(Debug, Clone, Serialize)]
pub struct ManifestSpan {
    pub span_id: String,
    pub kind: String,
    pub text: String,
    pub node_id: Option<u64>,
    pub link: Option<String>,
    pub sha256: Option<String>,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanResponse {
    pub plan_id: String,
    pub expires_at: u64,
    pub manifest_dir: String,
    pub total_chars: usize,
    pub cached_chars: usize,
    pub missing_chars: usize,
    pub estimated_ram_bytes: u64,
    pub estimated_storage_bytes: u64,
    pub spans: Vec<ManifestSpan>,
}

struct PendingPlan {
    request: PlanRequest,
    response: PlanResponse,
    injected: HashMap<String, WavAudio>,
}

pub struct PlanManager {
    runtime_root: PathBuf,
    object_root: PathBuf,
    plans: HashMap<String, PendingPlan>,
    sequence: u64,
}

impl PlanManager {
    pub fn new(runtime_root: PathBuf, object_root: PathBuf) -> Result<Self> {
        secure_dir(&runtime_root)?;
        secure_dir(&object_root)?;
        Ok(Self {
            runtime_root,
            object_root,
            plans: HashMap::new(),
            sequence: 0,
        })
    }

    pub fn create(&mut self, store: &MimicStore, request: PlanRequest) -> Result<PlanResponse> {
        self.reap();
        let text = units::normalize(&request.text);
        if text.is_empty() {
            return Err(MimicError::NotFound("empty text".into()));
        }
        self.sequence = self.sequence.wrapping_add(1);
        let now = now_secs();
        let plan_id = hash_hex(
            format!("{}:{}:{}:{}", now, std::process::id(), self.sequence, text).as_bytes(),
        );
        let staging = self.runtime_root.join(format!(".{plan_id}.tmp"));
        let manifest_dir = self.runtime_root.join(&plan_id);
        secure_dir(&staging)?;

        let words = units::tokenize(&text);
        let cache_domain = cache_domain(&request);
        let mut spans = Vec::new();
        let mut cached_chars = 0usize;
        let mut missing_words: Vec<String> = Vec::new();
        let mut ordinal = 0usize;
        let flush_missing =
            |spans: &mut Vec<ManifestSpan>, missing: &mut Vec<String>, ordinal: &mut usize| {
                if missing.is_empty() {
                    return;
                }
                let span_text = missing.join(" ");
                spans.push(ManifestSpan {
                    span_id: format!("span-{ordinal:04}"),
                    kind: "missing".into(),
                    text: span_text,
                    node_id: None,
                    link: None,
                    sha256: None,
                    bytes: 0,
                });
                *ordinal += 1;
                missing.clear();
            };

        let exact_phrase = store
            .lookup_exact(UnitLevel::Phrase, &text)
            .iter()
            .copied()
            .find(|id| {
                store.prop_string(*id, "voice").as_deref() == Some(request.voice_id.as_str())
                    && store.prop_string(*id, "cache_domain").as_deref()
                        == Some(cache_domain.as_str())
            });
        if let Some(id) = exact_phrase {
            let audio = store.get_audio(id)?;
            let bytes = audio::to_wav_bytes(&audio);
            let digest = hash_hex(&bytes);
            let object = self.object_root.join(format!("{digest}.wav"));
            if !object.exists() {
                atomic_write(&object, &bytes)?;
            }
            let link_name = format!("{ordinal:04}-{id}.wav");
            #[cfg(unix)]
            std::os::unix::fs::symlink(&object, staging.join(&link_name))?;
            cached_chars = text.chars().filter(|c| !c.is_whitespace()).count();
            spans.push(ManifestSpan {
                span_id: format!("span-{ordinal:04}"),
                kind: "cached".into(),
                text: text.clone(),
                node_id: Some(id.0),
                link: Some(link_name),
                sha256: Some(digest),
                bytes: bytes.len() as u64,
            });
        } else {
            for word in words {
                let id = store
                    .lookup_exact(UnitLevel::Word, &word)
                    .iter()
                    .copied()
                    .find(|id| {
                        store.prop_string(*id, "voice").as_deref()
                            == Some(request.voice_id.as_str())
                            && store.prop_string(*id, "cache_domain").as_deref()
                                == Some(cache_domain.as_str())
                    });
                if let Some(id) = id {
                    flush_missing(&mut spans, &mut missing_words, &mut ordinal);
                    let audio = store.get_audio(id)?;
                    let bytes = audio::to_wav_bytes(&audio);
                    let digest = hash_hex(&bytes);
                    let object = self.object_root.join(format!("{digest}.wav"));
                    if !object.exists() {
                        atomic_write(&object, &bytes)?;
                    }
                    let link_name = format!("{ordinal:04}-{id}.wav");
                    let link = staging.join(&link_name);
                    #[cfg(unix)]
                    std::os::unix::fs::symlink(&object, &link)?;
                    cached_chars += word.chars().count();
                    spans.push(ManifestSpan {
                        span_id: format!("span-{ordinal:04}"),
                        kind: "cached".into(),
                        text: word,
                        node_id: Some(id.0),
                        link: Some(link_name),
                        sha256: Some(digest),
                        bytes: bytes.len() as u64,
                    });
                    ordinal += 1;
                } else {
                    missing_words.push(word);
                }
            }
            flush_missing(&mut spans, &mut missing_words, &mut ordinal);
        }
        fs::rename(&staging, &manifest_dir)?;

        let total_chars = text.chars().filter(|c| !c.is_whitespace()).count();
        let missing_chars = total_chars.saturating_sub(cached_chars);
        let cached_bytes: u64 = spans.iter().map(|s| s.bytes).sum();
        let estimated_generated = missing_chars as u64 * 3_200;
        let response = PlanResponse {
            plan_id: plan_id.clone(),
            expires_at: now + PLAN_TTL.as_secs(),
            manifest_dir: manifest_dir.display().to_string(),
            total_chars,
            cached_chars,
            missing_chars,
            estimated_ram_bytes: RAM_HEADROOM + cached_bytes + estimated_generated * 2,
            estimated_storage_bytes: estimated_generated,
            spans,
        };
        self.plans.insert(
            plan_id,
            PendingPlan {
                request,
                response: response.clone(),
                injected: HashMap::new(),
            },
        );
        Ok(response)
    }

    pub fn inject_pcm(&mut self, plan_id: &str, span_id: &str, body: &[u8]) -> Result<()> {
        let plan = self.plan_mut(plan_id)?;
        let span = plan
            .response
            .spans
            .iter()
            .find(|s| s.span_id == span_id && s.kind == "missing")
            .ok_or_else(|| {
                MimicError::NotFound(format!(
                    "missing span {span_id}; create a new plan and retry"
                ))
            })?;
        if !body.len().is_multiple_of(2) {
            return Err(MimicError::Wav(
                "PCM body has odd byte length; submit complete i16 samples".into(),
            ));
        }
        let samples = body
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let audio = WavAudio::new(samples, SAMPLE_RATE);
        if let Some(existing) = plan.injected.get(span_id) {
            if existing != &audio {
                return Err(MimicError::Wav(
                    "idempotency conflict; create a new plan before changing audio".into(),
                ));
            }
            return Ok(());
        }
        let _ = span;
        plan.injected.insert(span_id.to_string(), audio);
        Ok(())
    }

    pub fn compose(
        &mut self,
        store: &mut MimicStore,
        g2p: &G2p,
        plan_id: &str,
        persist: bool,
    ) -> Result<(WavAudio, PlanResponse)> {
        self.reap();
        let plan = self.plans.remove(plan_id).ok_or_else(|| {
            MimicError::NotFound(format!("plan {plan_id}; create a new plan and retry"))
        })?;
        if now_secs() > plan.response.expires_at {
            return Err(MimicError::NotFound(
                "expired plan; create a new plan and retry".into(),
            ));
        }
        let mut parts = Vec::new();
        for span in &plan.response.spans {
            if span.kind == "cached" {
                let link =
                    Path::new(&plan.response.manifest_dir).join(span.link.as_deref().unwrap_or(""));
                let target = fs::canonicalize(&link)?;
                let root = fs::canonicalize(&self.object_root)?;
                if !target.starts_with(&root) {
                    return Err(MimicError::Wav(
                        "manifest link escaped object root; discard the plan and retry".into(),
                    ));
                }
                let bytes = fs::read(&target)?;
                if hash_hex(&bytes) != span.sha256.as_deref().unwrap_or("") {
                    return Err(MimicError::Wav(
                        "artifact checksum mismatch; discard the plan and retry".into(),
                    ));
                }
                parts.push(audio::parse_wav(&bytes)?);
            } else {
                let injected = plan
                    .injected
                    .get(&span.span_id)
                    .ok_or_else(|| {
                        MimicError::NotFound(format!(
                            "uninjected span {}; inject every missing span and try again",
                            span.span_id
                        ))
                    })?
                    .clone();
                if persist {
                    pipeline::ingest_with_options(
                        store,
                        &span.text,
                        &injected,
                        &plan.request.voice_id,
                        &cache_domain(&plan.request),
                        &pipeline::IngestOptions::p3_legacy(),
                        Some(g2p),
                    )?;
                }
                parts.push(injected);
            }
        }
        if persist {
            store.save()?;
        }
        let audio = audio::splice(&parts, crate::CROSSFADE_MS)?;
        let _ = fs::remove_dir_all(&plan.response.manifest_dir);
        Ok((audio, plan.response))
    }

    pub fn cancel(&mut self, plan_id: &str) -> bool {
        let removed = self.plans.remove(plan_id);
        if let Some(plan) = &removed {
            let _ = fs::remove_dir_all(&plan.response.manifest_dir);
        }
        removed.is_some()
    }

    fn plan_mut(&mut self, plan_id: &str) -> Result<&mut PendingPlan> {
        let plan = self.plans.get_mut(plan_id).ok_or_else(|| {
            MimicError::NotFound(format!("plan {plan_id}; create a new plan and retry"))
        })?;
        if now_secs() > plan.response.expires_at {
            return Err(MimicError::NotFound(
                "expired plan; create a new plan and retry".into(),
            ));
        }
        Ok(plan)
    }

    fn reap(&mut self) {
        let now = now_secs();
        let expired: Vec<String> = self
            .plans
            .iter()
            .filter(|(_, p)| p.response.expires_at < now)
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            self.cancel(&id);
        }
    }
}

fn cache_domain(request: &PlanRequest) -> String {
    let digest = hash_hex(format!("{}\0{}", request.model_id, request.settings_key).as_bytes());
    format!("voxd-elevenlabs:{}", &digest[..16])
}

fn secure_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&tmp, bytes)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn hash_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
