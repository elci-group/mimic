use mimic::daemon::{self, DaemonState};
use mimic::g2p::G2p;
use mimic::plan::PlanManager;
use mimic::store::MimicStore;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
struct Config {
    #[serde(default = "default_bind")]
    bind: String,
    #[serde(default)]
    auth_token: String,
    #[serde(default = "default_state")]
    state_dir: PathBuf,
    #[serde(default = "default_runtime")]
    runtime_dir: PathBuf,
}

fn default_bind() -> String {
    "127.0.0.1:17844".into()
}
fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}
fn default_state() -> PathBuf {
    home().join(".local/share/mimic")
}
fn default_runtime() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".local/share/mimic/run"))
        .join("mimic")
}
fn default_config() -> PathBuf {
    home().join(".config/mimic/config.toml")
}

fn voxd_token() -> Result<String, Box<dyn std::error::Error>> {
    let path = home().join(".config/voxd/config.toml");
    let value: toml::Value = toml::from_str(&std::fs::read_to_string(&path)?)?;
    value
        .get("server")
        .and_then(|server| server.get("auth_token"))
        .and_then(toml::Value::as_str)
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            format!(
                "missing server.auth_token in {}; set it with `voxd-cli config set server.auth_token TOKEN`",
                path.display()
            )
            .into()
        })
}

fn expand(path: PathBuf) -> PathBuf {
    let raw = path.to_string_lossy();
    raw.strip_prefix("~/")
        .map(|rest| home().join(rest))
        .unwrap_or(path)
}

fn load(path: &Path) -> Result<Config, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Config {
            bind: default_bind(),
            auth_token: String::new(),
            state_dir: default_state(),
            runtime_dir: default_runtime(),
        });
    }
    Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let mut args = std::env::args().skip(1);
    let mut config_path = default_config();
    while let Some(arg) = args.next() {
        if arg == "--config" {
            config_path = PathBuf::from(
                args.next()
                    .ok_or("missing --config value; try again with --config PATH")?,
            );
        }
    }
    let mut cfg = load(&config_path)?;
    if cfg.auth_token.is_empty() {
        cfg.auth_token = voxd_token()?;
    }
    let state_dir = expand(cfg.state_dir);
    let runtime_dir = expand(cfg.runtime_dir);
    std::fs::create_dir_all(&state_dir)?;
    let store = MimicStore::open(state_dir.join("state.pad"), state_dir.join("audio"))?;
    let plans = PlanManager::new(runtime_dir.join("plans"), state_dir.join("objects"))?;
    let g2p = G2p::parse(include_str!("../../assets/cmudict.dict"));
    let state = Arc::new(DaemonState {
        token: cfg.auth_token,
        store: Mutex::new(store),
        plans: Mutex::new(plans),
        g2p,
    });
    let listener = tokio::net::TcpListener::bind(&cfg.bind).await?;
    tracing::info!(bind = %cfg.bind, "mimicd ready");
    axum::serve(listener, daemon::app(state))
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}
