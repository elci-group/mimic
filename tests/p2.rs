//! P2 tests: HTTP client, provider wire formats (against local mock
//! servers), SSML subset, the axum service end-to-end, and replay smoke.

use mimic::eval as meval;
use mimic::providers::{HttpProvider, ProviderKind};
use mimic::ssml::{self, Segment};
use mimic::tts::TtsProvider;
use mimic::SAMPLE_RATE;
use std::io::{Read, Write};
use std::net::TcpListener;

/// Spin a one-shot canned HTTP server; returns (base_url, captured request).
fn canned_server(response: Vec<u8>) -> (String, std::sync::mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = vec![0u8; 65536];
        let n = stream.read(&mut buf).unwrap();
        // read rest of body if content-length says more is coming
        let mut raw = buf[..n].to_vec();
        while let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..pos]).to_string();
            let lower = head.to_ascii_lowercase();
            let clen: usize = lower
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
            if raw.len() >= pos + 4 + clen {
                break;
            }
            let n2 = stream.read(&mut buf).unwrap();
            if n2 == 0 {
                break;
            }
            raw.extend_from_slice(&buf[..n2]);
        }
        tx.send(String::from_utf8_lossy(&raw).to_string()).unwrap();
        stream.write_all(&response).unwrap();
    });
    (format!("http://127.0.0.1:{port}"), rx)
}

fn http_response(content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut v = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    v.extend_from_slice(body);
    v
}

fn tiny_wav_bytes() -> Vec<u8> {
    mimic::audio::to_wav_bytes(&mimic::audio::WavAudio::new(vec![1, -1, 2, -2], SAMPLE_RATE))
}

#[test]
fn provider_wire_openai() {
    let (base, rx) = canned_server(http_response("audio/wav", &tiny_wav_bytes()));
    let p = HttpProvider::new(ProviderKind::OpenAi, base, "sk-test", "tts-1", "alloy");
    let audio = p.synthesize("hello there", "default").unwrap();
    assert_eq!(audio.samples, vec![1, -1, 2, -2]);

    let req = rx.recv().unwrap();
    assert!(req.starts_with("POST /v1/audio/speech "), "req: {req}");
    assert!(req.contains("Authorization: Bearer sk-test"));
    assert!(req.contains("\"input\":\"hello there\""));
    assert!(req.contains("\"voice\":\"alloy\""));
}

#[test]
fn provider_wire_elevenlabs() {
    // pcm_16000: raw s16le
    let pcm: Vec<u8> = [10i16, -10, 20, -20].iter().flat_map(|s| s.to_le_bytes()).collect();
    let (base, rx) = canned_server(http_response("audio/pcm", &pcm));
    let p = HttpProvider::new(ProviderKind::ElevenLabs, base, "xi-key", "eleven_multilingual_v2", "voice-id-1");
    let audio = p.synthesize("hi", "default").unwrap();
    assert_eq!(audio.samples, vec![10, -10, 20, -20]);
    assert_eq!(audio.sample_rate, SAMPLE_RATE);

    let req = rx.recv().unwrap();
    assert!(
        req.starts_with("POST /v1/text-to-speech/voice-id-1?output_format=pcm_16000 "),
        "req: {req}"
    );
    assert!(req.contains("xi-api-key: xi-key"));
    assert!(req.contains("\"text\":\"hi\""));
}

#[test]
fn provider_wire_cartesia() {
    let (base, rx) = canned_server(http_response("audio/wav", &tiny_wav_bytes()));
    let p = HttpProvider::new(ProviderKind::Cartesia, base, "ct-key", "sonic-2", "v-123");
    let audio = p.synthesize("yo", "default").unwrap();
    assert_eq!(audio.samples.len(), 4);

    let req = rx.recv().unwrap();
    assert!(req.starts_with("POST /tts/bytes "), "req: {req}");
    assert!(req.contains("X-API-Key: ct-key"));
    assert!(req.contains("Cartesia-Version:"));
    assert!(req.contains("\"transcript\":\"yo\""));
    assert!(req.contains("\"sample_rate\":16000"));
}

#[test]
fn provider_wire_gemini_base64() {
    // L16 PCM 24 kHz: 6 samples -> 4 after 3:2 decimation
    let pcm24: Vec<u8> = [1i16, 2, 3, 4, 5, 6].iter().flat_map(|s| s.to_le_bytes()).collect();
    let b64 = base64_encode(&pcm24);
    let body = format!(
        "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"inlineData\":{{\"mimeType\":\"audio/L16;rate=24000\",\"data\":\"{b64}\"}}}}]}}}}]}}"
    );
    let (base, rx) = canned_server(http_response("application/json", body.as_bytes()));
    let p = HttpProvider::new(ProviderKind::Gemini, base, "g-key", "gemini-2.5-flash-preview-tts", "Kore");
    let audio = p.synthesize("hello", "default").unwrap();
    assert_eq!(audio.samples, vec![1, 2, 4, 5]);
    assert_eq!(audio.sample_rate, SAMPLE_RATE);

    let req = rx.recv().unwrap();
    assert!(
        req.starts_with("POST /v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key=g-key "),
        "req: {req}"
    );
    assert!(req.contains("\"voiceName\":\"Kore\""));
}

fn base64_encode(data: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

#[test]
fn provider_https_gives_clear_error() {
    let p = HttpProvider::new(ProviderKind::OpenAi, "https://api.openai.com", "k", "m", "v");
    let err = p.synthesize("x", "default").unwrap_err().to_string();
    assert!(err.contains("http://"), "err: {err}");
}

#[test]
fn ssml_subset() {
    let segs = ssml::parse("<speak>hello <break time=\"500ms\"/> world &amp; all</speak>");
    assert_eq!(
        segs,
        vec![
            Segment::Text("hello".to_string()),
            Segment::Break(500.0),
            Segment::Text("world & all".to_string()),
        ]
    );
    // no tags: passthrough
    assert_eq!(ssml::parse("plain text"), vec![Segment::Text("plain text".to_string())]);
    // seconds + default break
    let s2 = ssml::parse("<speak>a<break time=\"2s\"/>b<break/>c</speak>");
    assert!(s2.contains(&Segment::Break(2000.0)));
    assert!(s2.contains(&Segment::Break(250.0)));
}

#[test]
fn server_end_to_end() {
    use mimic::g2p::G2p;
    use mimic::server::{app, AppState};
    use mimic::store::MimicStore;
    use std::sync::{Arc, Mutex};

    let dir = tempfile::tempdir().unwrap();
    let store = MimicStore::open(dir.path().join("s.pad"), dir.path().join("s.audio")).unwrap();
    let state = Arc::new(AppState {
        store: Mutex::new(store),
        tts: mimic::tts::MockTts::new(),
        g2p: G2p::from_str("hello HH AH0 L OW1\nworld W ER1 L D\nred R EH1 D\n"),
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let (addr, shutdown) = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        std::thread::spawn(move || {
            let rt2 = tokio::runtime::Runtime::new().unwrap();
            rt2.block_on(async move {
                axum::serve(listener, app(state))
                    .with_graceful_shutdown(async {
                        let _ = rx.await;
                    })
                    .await
                    .unwrap();
            });
        });
        (addr, tx)
    });
    // wait for readiness
    for attempt in 0..50 {
        if std::net::TcpStream::connect(addr).is_ok() {
            break;
        }
        assert!(attempt < 49, "server did not start");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    let http_post = |path: &str, body: &str| -> (u16, String, Vec<u8>) {
        let mut stream = std::net::TcpStream::connect(addr).unwrap();
        let req = format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(req.as_bytes()).unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = String::from_utf8_lossy(&raw[..split]).to_string();
        let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, head, raw[split + 4..].to_vec())
    };
    let http_get = |path: &str| -> (u16, Vec<u8>) {
        let mut stream = std::net::TcpStream::connect(addr).unwrap();
        stream
            .write_all(format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").as_bytes())
            .unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = String::from_utf8_lossy(&raw[..split]).to_string();
        let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, raw[split + 4..].to_vec())
    };

    let (s, body) = http_get("/health");
    assert_eq!(s, 200);
    assert_eq!(body, b"ok");

    // ingest a phrase
    let (s, _, _) = http_post("/v1/ingest", "{\"text\":\"hello world\"}");
    assert_eq!(s, 200);

    // openai-compatible shim: partial overlap -> headers show cache reuse
    let (s, head, wav) = http_post("/v1/audio/speech", "{\"input\":\"hello red world\",\"voice\":\"default\"}");
    assert_eq!(s, 200);
    assert!(wav.starts_with(b"RIFF"), "wav body");
    assert!(head.contains("x-mimic-cache-hit-pct:"), "headers: {head}");
    let lower_head = head.to_ascii_lowercase();
    let hit: f64 = lower_head
        .lines()
        .find_map(|l| l.strip_prefix("x-mimic-cache-hit-pct:"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(hit > 50.0, "hit {hit}");

    // stats shows units
    let (s, stats) = http_get("/v1/stats");
    assert_eq!(s, 200);
    let stats = String::from_utf8(stats).unwrap();
    assert!(stats.contains("\"words\":"), "{stats}");

    // SSML through the compose endpoint
    let (s, _, wav2) = http_post("/v1/compose", "{\"text\":\"<speak>hello <break time=\\\"100ms\\\"/> world</speak>\"}");
    assert_eq!(s, 200);
    assert!(wav2.starts_with(b"RIFF"));

    drop(shutdown); // stop accepting; server task finishes on pending
}

#[test]
fn replay_smoke() {
    let dir = tempfile::tempdir().unwrap();
    let corpora_dir = dir.path().join("corpora");
    std::fs::create_dir_all(&corpora_dir).unwrap();
    let mut support = String::new();
    for i in 0..14 {
        support.push_str(&format!("head phrase number {i} for support\n"));
    }
    std::fs::write(corpora_dir.join("support_repetitive.txt"), support).unwrap();
    std::fs::write(corpora_dir.join("long_tail.txt"), "unique tail one\ntail two here\n").unwrap();

    let g = mimic::g2p::G2p::from_str("head HH EH1 D\nphrase F R EY1 Z\nnumber N AH1 M B ER0\nfor F AO1 R\nsupport S AH0 P AO1 R T\nunique Y UW0 N IY1 K\ntail T EY1 L\none W AH1 N\ntwo T UW1\nhere HH IY1 R\n");
    let corpora = meval::load_corpora(&corpora_dir).unwrap();
    let rep = meval::run_replay(&corpora, &g, 120, "openai-tts").unwrap();

    // Zipf head repeats are fully cached after first sight -> high coverage
    assert!(
        rep.coverage_pct >= 80.0,
        "coverage {:.1}% (gen {} of {})",
        rep.coverage_pct,
        rep.generated_chars,
        rep.total_chars
    );
    assert_eq!(rep.requests, 120);
    assert!(rep.p99_ms > 0.0);
    assert!(rep.simulated_cloud_p99_ms > rep.mock_direct_p99_ms);
    // cost accounting is consistent
    assert!(rep.est_cost_usd <= rep.always_cloud_cost_usd);
    assert!(rep.always_cloud_cost_usd > 0.0);
}
