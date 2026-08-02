//! Mimic as a service: an axum HTTP server exposing compose (mimic-native
//! and OpenAI-compatible), ingest, and stats endpoints.
//!
//! The bundled provider is MockTts; wiring real providers is a config
//! question (see providers.rs — HTTP transport is http://-only until a TLS
//! stack lands). The server is intentionally stateful-simple: one
//! `MimicStore` behind a mutex, saved after each ingest/compose.

use crate::audio::{self, WavAudio};
use crate::g2p::G2p;
use crate::pipeline::{self, ComposeReport, IngestOptions};
use crate::ssml::{self, Segment};
use crate::store::MimicStore;
use crate::tts::{MockTts, TtsProvider};
use crate::CROSSFADE_MS;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::{Arc, Mutex};

pub struct AppState {
    pub store: Mutex<MimicStore>,
    pub tts: MockTts,
    pub g2p: G2p,
}

#[derive(Deserialize)]
struct ComposeIn {
    text: String,
    voice: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiSpeechIn {
    input: String,
    voice: Option<String>,
    #[allow(dead_code)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct IngestIn {
    text: String,
    voice: Option<String>,
}

fn compose_ssml(
    store: &mut MimicStore,
    tts: &dyn TtsProvider,
    g2p: Option<&G2p>,
    text: &str,
    voice: &str,
) -> crate::Result<(WavAudio, ComposeReport)> {
    let segments = ssml::parse(text);
    let mut parts: Vec<WavAudio> = Vec::new();
    let mut agg = ComposeReport::default();
    for seg in segments {
        match seg {
            Segment::Text(t) => {
                let (audio, rep) = crate::select::compose_v3_with_medium(
                    store,
                    tts,
                    &t,
                    voice,
                    g2p,
                    crate::select::Medium::Tokens,
                )?;
                parts.push(audio);
                agg.total_chars += rep.total_chars;
                agg.cached_chars += rep.cached_chars;
                agg.generated_chars += rep.generated_chars;
                agg.tts_calls.extend(rep.tts_calls);
                agg.hits.extend(rep.hits);
            }
            Segment::Break(ms) => parts.push(ssml::silence(ms, crate::SAMPLE_RATE)),
        }
    }
    let seams: Vec<f64> = parts
        .windows(2)
        .map(|w| audio::seam_discontinuity(&w[0], &w[1]))
        .collect();
    agg.mean_seam_discontinuity = if seams.is_empty() {
        0.0
    } else {
        seams.iter().sum::<f64>() / seams.len() as f64
    };
    let out = audio::splice(&parts, CROSSFADE_MS)?;
    Ok((out, agg))
}

fn wav_response(audio: &WavAudio, rep: &ComposeReport) -> Response {
    let bytes = audio::to_wav_bytes(audio);
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "audio/wav".parse().unwrap());
    headers.insert(
        "x-mimic-cache-hit-pct",
        format!("{:.1}", rep.cache_hit_pct()).parse().unwrap(),
    );
    headers.insert(
        "x-mimic-tts-calls",
        rep.tts_calls.len().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-mimic-seam",
        format!("{:.4}", rep.mean_seam_discontinuity)
            .parse()
            .unwrap(),
    );
    (StatusCode::OK, headers, bytes).into_response()
}

async fn health() -> &'static str {
    "ok"
}

async fn stats(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    let st = s.store.lock().unwrap().stats();
    Json(serde_json::json!({
        "phrases": st.phrases,
        "words": st.words,
        "morphemes": st.morphemes,
        "phonemes": st.phonemes,
        "nodes": st.total_nodes,
        "edges": st.total_edges,
        "provider": s.tts.name(),
    }))
}

async fn compose_ep(State(s): State<Arc<AppState>>, Json(inp): Json<ComposeIn>) -> Response {
    let voice = inp.voice.unwrap_or_else(|| "default".into());
    let mut store = s.store.lock().unwrap();
    match compose_ssml(&mut store, &s.tts, Some(&s.g2p), &inp.text, &voice) {
        Ok((audio, rep)) => {
            let _ = store.save();
            wav_response(&audio, &rep)
        }
        Err(e) => (StatusCode::BAD_REQUEST, format!("{e}")).into_response(),
    }
}

async fn openai_speech(
    State(s): State<Arc<AppState>>,
    Json(inp): Json<OpenAiSpeechIn>,
) -> Response {
    let voice = inp.voice.unwrap_or_else(|| "default".into());
    let mut store = s.store.lock().unwrap();
    match compose_ssml(&mut store, &s.tts, Some(&s.g2p), &inp.input, &voice) {
        Ok((audio, rep)) => {
            let _ = store.save();
            wav_response(&audio, &rep)
        }
        Err(e) => (StatusCode::BAD_REQUEST, format!("{e}")).into_response(),
    }
}

async fn ingest_ep(State(s): State<Arc<AppState>>, Json(inp): Json<IngestIn>) -> Response {
    let voice = inp.voice.unwrap_or_else(|| "default".into());
    let mut store = s.store.lock().unwrap();
    let audio = match s.tts.synthesize(&inp.text, &voice) {
        Ok(a) => a,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response(),
    };
    match pipeline::ingest_with_options(
        &mut store,
        &inp.text,
        &audio,
        &voice,
        s.tts.name(),
        &IngestOptions::default(),
        Some(&s.g2p),
    ) {
        Ok(rep) => {
            let _ = store.save();
            Json(serde_json::json!({
                "phrase_units": rep.phrase_units,
                "word_units": rep.word_units,
                "phoneme_units": rep.phoneme_units,
                "unresolved_words": rep.unresolved_words,
            }))
            .into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, format!("{e}")).into_response(),
    }
}

pub fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/stats", get(stats))
        .route("/v1/compose", post(compose_ep))
        .route("/v1/audio/speech", post(openai_speech))
        .route("/v1/ingest", post(ingest_ep))
        .with_state(state)
}

pub async fn serve(addr: &str, store: MimicStore, g2p: G2p) -> crate::Result<()> {
    let state = Arc::new(AppState {
        store: Mutex::new(store),
        tts: MockTts::new(),
        g2p,
    });
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("mimic serving on http://{addr} (provider: mock-tts)");
    axum::serve(listener, app(state))
        .await
        .map_err(|e| crate::MimicError::Wav(format!("server: {e}")))?;
    Ok(())
}
