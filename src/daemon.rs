//! Hardened, provider-free HTTP daemon used by voxd.

use crate::g2p::G2p;
use crate::plan::{PlanManager, PlanRequest};
use crate::store::MimicStore;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header::AUTHORIZATION, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use std::sync::{Arc, Mutex};

pub struct DaemonState {
    pub token: String,
    pub store: Mutex<MimicStore>,
    pub plans: Mutex<PlanManager>,
    pub g2p: G2p,
}

pub fn app(state: Arc<DaemonState>) -> Router {
    Router::new()
        .route(
            "/health",
            get(|| async { Json(serde_json::json!({"ok": true})) }),
        )
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .route("/v1/plans", post(create_plan))
        .route("/v1/plans/:plan/spans/:span", put(inject))
        .route("/v1/plans/:plan/compose", post(compose))
        .route("/v1/plans/:plan", delete(cancel))
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .with_state(state)
}

async fn auth(
    State(state): State<Arc<DaemonState>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    if req.uri().path() == "/health" {
        return Ok(next.run(req).await);
    }
    let expected = format!("Bearer {}", state.token);
    let ok = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .map(|h| h == expected)
        .unwrap_or(false);
    if ok {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

async fn ready(State(state): State<Arc<DaemonState>>) -> Response {
    let stats = state.store.lock().map(|s| s.stats());
    match stats {
        Ok(s) => {
            Json(serde_json::json!({"ok": true, "nodes": s.total_nodes, "edges": s.total_edges}))
                .into_response()
        }
        Err(_) => (StatusCode::SERVICE_UNAVAILABLE, "store lock poisoned").into_response(),
    }
}

async fn metrics(State(state): State<Arc<DaemonState>>) -> Response {
    let nodes = state
        .store
        .lock()
        .map(|s| s.stats().total_nodes)
        .unwrap_or(0);
    (
        [("content-type", "text/plain; version=0.0.4")],
        format!("mimic_nodes {nodes}\n"),
    )
        .into_response()
}

async fn create_plan(
    State(state): State<Arc<DaemonState>>,
    Json(req): Json<PlanRequest>,
) -> Response {
    let store = match state.store.lock() {
        Ok(v) => v,
        Err(_) => return internal("store lock poisoned"),
    };
    let mut plans = match state.plans.lock() {
        Ok(v) => v,
        Err(_) => return internal("plan lock poisoned"),
    };
    match plans.create(&store, req) {
        Ok(p) => Json(p).into_response(),
        Err(e) => bad(&e.to_string()),
    }
}

async fn inject(
    State(state): State<Arc<DaemonState>>,
    Path((plan, span)): Path<(String, String)>,
    body: Bytes,
) -> Response {
    let mut plans = match state.plans.lock() {
        Ok(v) => v,
        Err(_) => return internal("plan lock poisoned"),
    };
    match plans.inject_pcm(&plan, &span, &body) {
        Ok(()) => Json(serde_json::json!({"ok": true})).into_response(),
        Err(e) => bad(&e.to_string()),
    }
}

#[derive(Deserialize)]
struct ComposeQuery {
    #[serde(default = "default_true")]
    persist: bool,
}
fn default_true() -> bool {
    true
}

async fn compose(
    State(state): State<Arc<DaemonState>>,
    Path(plan): Path<String>,
    Query(q): Query<ComposeQuery>,
) -> Response {
    let mut store = match state.store.lock() {
        Ok(v) => v,
        Err(_) => return internal("store lock poisoned"),
    };
    let mut plans = match state.plans.lock() {
        Ok(v) => v,
        Err(_) => return internal("plan lock poisoned"),
    };
    match plans.compose(&mut store, &state.g2p, &plan, q.persist) {
        Ok((audio, report)) => {
            let bytes = crate::audio::to_wav_bytes(&audio);
            let headers = [
                ("content-type", "audio/wav".to_string()),
                (
                    "x-mimic-cache-hit-pct",
                    format!(
                        "{:.1}",
                        report.cached_chars as f64 * 100.0 / report.total_chars.max(1) as f64
                    ),
                ),
                ("x-mimic-provider-chars", report.missing_chars.to_string()),
            ];
            (StatusCode::OK, headers, bytes).into_response()
        }
        Err(e) => bad(&e.to_string()),
    }
}

async fn cancel(State(state): State<Arc<DaemonState>>, Path(plan): Path<String>) -> Response {
    let mut plans = match state.plans.lock() {
        Ok(v) => v,
        Err(_) => return internal("plan lock poisoned"),
    };
    Json(serde_json::json!({"deleted": plans.cancel(&plan)})).into_response()
}

fn bad(msg: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": msg})),
    )
        .into_response()
}
fn internal(msg: &str) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": msg})),
    )
        .into_response()
}
