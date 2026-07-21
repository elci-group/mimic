//! MimicStore: audio-primitive cache persisted in a padagonia graph file.
//!
//! Each unit is an `AudioUnit` node whose properties carry text, level,
//! prosody metadata, and the wav file reference; the node embedding feeds a
//! padagonia HNSW index for acoustic-similarity retrieval. Consecutive units
//! of an utterance are linked with `follows` edges, preserving coarticulation
//! context for later transition-aware stitching.
//!
//! padagonia has no property-value index, so exact text lookup is served by
//! an in-memory side map rebuilt on load. padagonia is append-only — units
//! are write-once by design (a correction would be a new node).

use crate::audio::{self, WavAudio};
use crate::codec::AudioCodec;
use crate::features;
use crate::g2p::G2p;
use crate::units::UnitLevel;
use crate::{MimicError, Result};
use padagonia::{Distance, HnswIndex, Node, NodeId, Provenance, Scalar, Store, StringTableExt};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const HNSW_SEED: u64 = 42;
const HNSW_EF_SEARCH: usize = 50;

pub struct MimicStore {
    pub store: Store,
    hnsw: HnswIndex,
    exact: HashMap<(UnitLevel, String), Vec<NodeId>>,
    audio_dir: PathBuf,
    db_path: PathBuf,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StoreStats {
    pub phrases: usize,
    pub words: usize,
    pub morphemes: usize,
    pub diphones: usize,
    pub phonemes: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
}

impl MimicStore {
    /// Open an existing database (rebuilding the in-memory indexes) or start
    /// a fresh one. `db_path` is the .pad file; `audio_dir` holds unit wavs.
    pub fn open<P: Into<PathBuf>>(db_path: P, audio_dir: P) -> Result<Self> {
        let db_path = db_path.into();
        let audio_dir = audio_dir.into();
        let store = if db_path.exists() {
            Store::load(&db_path)?
        } else {
            Store::new()
        };
        let mut ms = Self {
            store,
            hnsw: HnswIndex::with_seed(Distance::Cosine, 16, 64, HNSW_EF_SEARCH, HNSW_SEED),
            exact: HashMap::new(),
            audio_dir,
            db_path,
        };
        ms.rebuild_indexes();
        Ok(ms)
    }

    fn rebuild_indexes(&mut self) {
        let mut rows: Vec<(NodeId, Option<Vec<f32>>, UnitLevel, String)> = Vec::new();
        for (id, node) in &self.store.nodes {
            if self.store.string_table.resolve_label(node.label) != Some("AudioUnit") {
                continue;
            }
            let level = self
                .prop_string(*id, "level")
                .and_then(|s| s.parse::<UnitLevel>().ok());
            let text = self.prop_string(*id, "text");
            if let (Some(level), Some(text)) = (level, text) {
                rows.push((*id, node.embedding.clone(), level, text));
            }
        }
        for (id, emb, level, text) in rows {
            if let Some(e) = emb {
                self.hnsw.insert(id, e);
            }
            self.exact.entry((level, text)).or_default().push(id);
        }
        for ids in self.exact.values_mut() {
            ids.sort();
        }
    }

    /// Store one audio unit: optional wav to `audio_dir` (legacy/reference
    /// path), optional codec token bytes inline in the node (P4 codec-native
    /// path), node + embedding to padagonia, entry in the exact-lookup map.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_unit(
        &mut self,
        level: UnitLevel,
        text: &str,
        phonemes: &str,
        voice: &str,
        audio: &WavAudio,
        tokens: Option<Vec<u8>>,
        write_wav: bool,
        context_prev: Option<&str>,
        context_next: Option<&str>,
        provider: &str,
    ) -> Result<NodeId> {
        std::fs::create_dir_all(&self.audio_dir)?;
        let filename = format!("{}.wav", self.store.next_node_id);
        if write_wav {
            audio::write_wav(audio, self.audio_dir.join(&filename))?;
        }

        let embedding = features::embed(text, audio);
        let voice_sig = features::voice_signature(audio);
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let prov = Provenance::new(
            "mimic",
            provider,
            1.0,
            text.len() as f32,
            ts,
            vec![filename.clone()],
        );
        let mut props: Vec<(&str, Scalar)> = vec![
            ("text", Scalar::String(text.to_string())),
            ("level", Scalar::String(level.as_str().to_string())),
            ("phonemes", Scalar::String(phonemes.to_string())),
            ("wav_path", Scalar::String(if write_wav { filename.clone() } else { String::new() })),
            ("duration_ms", Scalar::I64(audio.duration_ms().round() as i64)),
            ("sample_rate", Scalar::I64(audio.sample_rate as i64)),
            ("rms", Scalar::F64(audio.rms())),
            (
                "pitch_hz",
                Scalar::F64(audio.zero_crossing_rate() * audio.sample_rate as f64 / 2.0),
            ),
            ("voice", Scalar::String(voice.to_string())),
            (
                "context_prev",
                Scalar::String(context_prev.unwrap_or("").to_string()),
            ),
            (
                "context_next",
                Scalar::String(context_next.unwrap_or("").to_string()),
            ),
            ("voice_sig", Scalar::Embedding(voice_sig)),
        ];
        if let Some(t) = tokens {
            props.push(("codec", Scalar::String(crate::codec::MimicMct.name().to_string())));
            props.push(("frames", Scalar::I64(crate::codec::frames_for(audio.samples.len()) as i64)));
            props.push(("tokens", Scalar::Bytes(t)));
        }
        let id = self.store.add_node("AudioUnit", props, Some(embedding.clone()), prov);
        self.hnsw.insert(id, embedding);
        self.exact
            .entry((level, text.to_string()))
            .or_default()
            .push(id);
        Ok(id)
    }

    /// Link two consecutive units of one utterance.
    pub fn add_follows(&mut self, prev: NodeId, next: NodeId) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let prov = Provenance::new("mimic", "pipeline", 1.0, 0.0, ts, vec![]);
        self.store
            .add_edge(prev, next, "follows", vec![], None, prov);
    }

    pub fn lookup_exact(&self, level: UnitLevel, normalized_text: &str) -> &[NodeId] {
        self.exact
            .get(&(level, normalized_text.to_string()))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// All (text, ids) pairs at a level. Used by the phrase-matching pass.
    pub fn units_for_level(&self, level: UnitLevel) -> Vec<(String, Vec<NodeId>)> {
        self.exact
            .iter()
            .filter(|((l, _), _)| *l == level)
            .map(|((_, t), ids)| (t.clone(), ids.clone()))
            .collect()
    }

    /// Phoneme units whose base (stress-stripped) symbol matches, e.g.
    /// "AH" covers AH0/AH1/AH2. Used for diphone-chain gap filling.
    pub fn phonemes_by_base(&self, base: &str) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = self
            .exact
            .iter()
            .filter(|((l, t), _)| *l == UnitLevel::Phoneme && G2p::base(t) == base)
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect();
        out.sort();
        out
    }

    /// Pick among duplicate cached units by neighbor context match;
    /// deterministic tiebreak on lowest NodeId.
    pub fn best_context_match(
        &self,
        candidates: &[NodeId],
        prev: Option<&str>,
        next: Option<&str>,
    ) -> Option<NodeId> {
        let mut best: Option<(NodeId, i32)> = None;
        for &id in candidates {
            let mut score = 0;
            if let Some(p) = prev {
                if self.prop_string(id, "context_prev").as_deref() == Some(p) {
                    score += 1;
                }
            }
            if let Some(nx) = next {
                if self.prop_string(id, "context_next").as_deref() == Some(nx) {
                    score += 1;
                }
            }
            let better = match best {
                None => true,
                Some((bid, bs)) => score > bs || (score == bs && id < bid),
            };
            if better {
                best = Some((id, score));
            }
        }
        best.map(|(id, _)| id)
    }

    pub fn get_audio(&self, id: NodeId) -> Result<WavAudio> {
        let wav = self
            .prop_string(id, "wav_path")
            .ok_or_else(|| MimicError::NotFound(format!("node {id} has no wav_path")))?;
        if !wav.is_empty() {
            let path = self.audio_dir.join(wav);
            if path.exists() {
                return audio::read_wav(path);
            }
        }
        // codec-native fallback: decode inline tokens (P4 units carry no wav)
        let tokens = self.get_tokens(id)?;
        crate::codec::MimicMct.decode(&tokens)
    }

    /// Inline codec token stream of a unit (P4).
    pub fn get_tokens(&self, id: NodeId) -> Result<Vec<u8>> {
        let node = self
            .store
            .nodes
            .get(&id)
            .ok_or_else(|| MimicError::NotFound(format!("node {id}")))?;
        let kid = self
            .store
            .string_table
            .key_id("tokens")
            .ok_or_else(|| MimicError::NotFound(format!("node {id} has no tokens")))?;
        match node.properties.iter().find(|(k, _)| *k == kid).map(|(_, v)| v) {
            Some(Scalar::Bytes(b)) => Ok(b.clone()),
            _ => Err(MimicError::NotFound(format!("node {id} has no tokens"))),
        }
    }

    /// Voice signature embedding of a unit (native speaker-identity proxy).
    pub fn get_voice_sig(&self, id: NodeId) -> Option<Vec<f32>> {
        let node = self.store.nodes.get(&id)?;
        let kid = self.store.string_table.key_id("voice_sig")?;
        match node.properties.iter().find(|(k, _)| *k == kid).map(|(_, v)| v) {
            Some(Scalar::Embedding(v)) => Some(v.clone()),
            _ => None,
        }
    }

    /// Storage accounting across all AudioUnit nodes: (inline token bytes,
/// equivalent wav bytes at 44 + 2×samples). The P4 storage gate divides
/// the second by the first.
pub fn codec_storage(&self) -> (usize, usize) {
    let mut tokens = 0usize;
    let mut wav = 0usize;
    for (id, node) in &self.store.nodes {
        if self.store.string_table.resolve_label(node.label) != Some("AudioUnit") {
            continue;
        }
        let dur_ms: f64 = self
            .prop_string(*id, "duration_ms")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        wav += 44 + (dur_ms * 16.0) as usize * 2;
        if let Some(kid) = self.store.string_table.key_id("tokens") {
            if let Some(Scalar::Bytes(b)) = node
                .properties
                .iter()
                .find(|(k, _)| *k == kid)
                .map(|(_, v)| v)
            {
                tokens += b.len();
            }
        }
    }
    (tokens, wav)
}

/// Record a per-voice aggregate node ("Voice DNA") for an ingested
    /// utterance. Append-only like everything else: consumers use the node
    /// with the highest `samples` count for the voice.
    pub fn add_voice_dna(&mut self, voice: &str, audio: &WavAudio, provider: &str) -> NodeId {
        let sig = features::voice_signature(audio);
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let samples = audio.samples.len() as i64;
        let prov = Provenance::new("mimic", provider, 1.0, 0.0, ts, vec![]);
        self.store.add_node(
            "VoiceDNA",
            vec![
                ("voice", Scalar::String(voice.to_string())),
                ("text", Scalar::String(format!("voice:{voice}"))),
                ("samples", Scalar::I64(samples)),
                ("rms", Scalar::F64(audio.rms())),
            ],
            Some(sig),
            prov,
        )
    }

    pub fn get_text(&self, id: NodeId) -> Option<String> {
        self.prop_string(id, "text")
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.store.nodes.get(&id)
    }

    /// Read a property of a node as a display string, resolving keys through
    /// the padagonia string table.
    pub fn prop_string(&self, id: NodeId, key: &str) -> Option<String> {
        let node = self.store.nodes.get(&id)?;
        let kid = self.store.string_table.key_id(key)?;
        match node.properties.iter().find(|(k, _)| *k == kid)?.1 {
            Scalar::String(ref s) => Some(s.clone()),
            Scalar::I64(v) => Some(v.to_string()),
            Scalar::F64(v) => Some(v.to_string()),
            _ => None,
        }
    }

    /// Approximate nearest neighbors over unit embeddings.
    pub fn similar(&self, embedding: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        self.hnsw.search(embedding, k, HNSW_EF_SEARCH)
    }

    pub fn stats(&self) -> StoreStats {
        let mut s = StoreStats::default();
        for ((level, _), ids) in &self.exact {
            match level {
                UnitLevel::Phrase => s.phrases += ids.len(),
                UnitLevel::Word => s.words += ids.len(),
                UnitLevel::Morpheme => s.morphemes += ids.len(),
                UnitLevel::Diphone => s.diphones += ids.len(),
                UnitLevel::Phoneme => s.phonemes += ids.len(),
            }
        }
        s.total_nodes = self.store.nodes.len();
        s.total_edges = self.store.edges.len();
        s
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::create_dir_all(&self.audio_dir)?;
        self.store.save(&self.db_path)?;
        Ok(())
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn audio_dir(&self) -> &Path {
        &self.audio_dir
    }
}
