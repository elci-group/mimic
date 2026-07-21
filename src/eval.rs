//! The Mimic evaluation harness (P1).
//!
//! Runs the pipeline over the benchmark corpora in two modes —
//! `v1-baseline` (proportional segmentation, no phonemes: the pipeline as
//! originally shipped) and `p1-current` (energy-refined alignment, phoneme
//! inventory) — and reports coverage, RTF, seam discontinuity, and word
//! boundary error vs. MockTts ground truth. The v1-baseline row is the
//! historical reference every future phase must beat.
//!
//! Neural perceptual metrics (UTMOS/NISQA), Whisper WER, and SECS speaker
//! cosine need a Python ML stack; they are wired as an optional external
//! adapter (see scripts/eval_external.py) and reported as SKIPPED otherwise.

use crate::align::{self, AlignMode};
use crate::g2p::G2p;
use crate::pipeline::{compose, ingest_with_options, IngestOptions};
use crate::store::MimicStore;
use crate::tts::{mock_word_spans, MockTts, TtsProvider, GAP_MS};
use crate::units;
use crate::Result;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Corpus {
    pub name: String,
    pub utterances: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub struct ModeRow {
    pub mode: String,
    pub corpus: String,
    pub eval_utterances: usize,
    pub cache_hit_pct: f64,
    pub generated_chars: usize,
    pub total_chars: usize,
    pub rtf: f64,
    pub mean_seam: f64,
    pub unresolved_words: usize,
    pub units_total: usize,
    /// Codec-fidelity STOI over fully-cached utterances (token modes only).
    pub stoi: Option<f64>,
    /// Voice-signature fidelity over fully-cached utterances (token modes).
    pub voice_fidelity: Option<f64>,
    /// Inline token bytes vs equivalent wav bytes (token modes only).
    pub token_bytes: usize,
    pub wav_bytes: usize,
}

#[derive(Debug, Default, Clone)]
pub struct BoundaryStats {
    pub clips: usize,
    pub boundaries: usize,
    pub median_ms: f64,
    pub p90_ms: f64,
    pub max_ms: f64,
    pub baseline_median_ms: f64,
}

#[derive(Debug, Default)]
pub struct EvalReport {
    pub rows: Vec<ModeRow>,
    pub boundary: BoundaryStats,
    pub replay: Option<ReplayReport>,
    pub notes: Vec<String>,
}

/// Production-shaped workload replay (P2): a Zipf-ish request stream over
/// the support + long-tail corpora, measuring steady-state coverage,
/// compose latency, and cost vs. always-cloud.
#[derive(Debug, Default, Clone)]
pub struct ReplayReport {
    pub requests: usize,
    pub coverage_pct: f64,
    pub generated_chars: usize,
    pub total_chars: usize,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mock_direct_p99_ms: f64,
    pub simulated_cloud_p99_ms: f64,
    pub est_cost_usd: f64,
    pub always_cloud_cost_usd: f64,
    pub provider_profile: String,
}

/// Simulated cloud-TTS latency model (no live provider on this machine):
/// per-request base + per-char time, in milliseconds. Based on typical
/// hosted-TTS round trips; constants are documented and tunable.
pub const SIM_CLOUD_BASE_MS: f64 = 250.0;
pub const SIM_CLOUD_PER_CHAR_MS: f64 = 8.0;

fn percentile_ms(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[((xs.len() - 1) as f64 * p / 100.0).round() as usize]
}

/// Zipf-ish replay stream: 75% of requests from the top-10 head lines of
/// `support_repetitive`, 25% from the long tail. Deterministic (LCG).
pub fn replay_stream(corpora: &[Corpus], n: usize) -> Vec<String> {
    let support = corpora
        .iter()
        .find(|c| c.name == "support_repetitive")
        .map(|c| c.utterances.clone())
        .unwrap_or_default();
    let tail: Vec<String> = corpora
        .iter()
        .find(|c| c.name == "long_tail")
        .map(|c| c.utterances.clone())
        .unwrap_or_default()
        .into_iter()
        .chain(support.iter().skip(10).cloned())
        .collect();
    let head: Vec<String> = support.into_iter().take(10).collect();
    let mut out = Vec::with_capacity(n);
    let mut state: u64 = 7;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    for _ in 0..n {
        if !head.is_empty() && (next() % 100) < 75 {
            out.push(head[next() % head.len()].clone());
        } else if !tail.is_empty() {
            out.push(tail[next() % tail.len()].clone());
        }
    }
    out
}

pub fn run_replay(
    corpora: &[Corpus],
    g2p: &G2p,
    requests: usize,
    provider_profile: &str,
) -> Result<ReplayReport> {
    let stream = replay_stream(corpora, requests);
    let dir = temp_db("replay");
    let db = dir.join("m.pad");
    let adir = dir.join("m.audio");
    let mut rep = ReplayReport {
        requests: stream.len(),
        provider_profile: provider_profile.to_string(),
        ..Default::default()
    };
    {
        let mut store = MimicStore::open(db, adir)?;
        let tts = MockTts::new();
        let mut lat_ms = Vec::with_capacity(stream.len());
        let mut direct_ms = Vec::with_capacity(stream.len());
        let mut sim_ms = Vec::with_capacity(stream.len());
        for req in &stream {
            let t0 = Instant::now();
            let (_, r) = crate::select::compose_v3_with_medium(
                &mut store,
                &tts,
                req,
                "default",
                Some(g2p),
                crate::select::Medium::Tokens,
            )?;
            lat_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            rep.generated_chars += r.generated_chars;
            rep.total_chars += r.total_chars;

            let t1 = Instant::now();
            let _ = tts.synthesize(req, "default")?;
            direct_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
            let chars = units::tokenize(req).iter().map(|w| w.chars().count()).sum::<usize>();
            sim_ms.push(SIM_CLOUD_BASE_MS + SIM_CLOUD_PER_CHAR_MS * chars as f64);
        }
        rep.coverage_pct = if rep.total_chars > 0 {
            100.0 * (rep.total_chars - rep.generated_chars) as f64 / rep.total_chars as f64
        } else {
            0.0
        };
        rep.p50_ms = percentile_ms(lat_ms.clone(), 50.0);
        rep.p99_ms = percentile_ms(lat_ms, 99.0);
        rep.mock_direct_p99_ms = percentile_ms(direct_ms, 99.0);
        rep.simulated_cloud_p99_ms = percentile_ms(sim_ms, 99.0);
    }
    let _ = std::fs::remove_dir_all(&dir);
    let price = crate::providers::cost_per_million_chars(provider_profile);
    rep.est_cost_usd = rep.generated_chars as f64 / 1e6 * price;
    rep.always_cloud_cost_usd = rep.total_chars as f64 / 1e6 * price;
    Ok(rep)
}

pub fn load_corpora(dir: &Path) -> Result<Vec<Corpus>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    let mut corpora = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f)?;
        let utterances: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect();
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        corpora.push(Corpus { name, utterances });
    }
    Ok(corpora)
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((xs.len() - 1) as f64 * p / 100.0).round() as usize;
    xs[k.min(xs.len() - 1)]
}

fn temp_db(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("mimic-eval-{}-{}", std::process::id(), nanos) + tag)
}

/// Which composer a mode row uses (the greedy path is the frozen legacy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComposeMode {
    Greedy,
    ViterbiPcm,
    ViterbiTokens,
}

/// Run ingest+compose sweeps for one mode over all corpora.
fn run_mode(
    mode: &str,
    opts: &IngestOptions,
    compose_mode: ComposeMode,
    g2p: Option<&G2p>,
    corpora: &[Corpus],
) -> Result<Vec<ModeRow>> {
    let dir = temp_db(mode);
    let db = dir.join("m.pad");
    let adir = dir.join("m.audio");
    let mut rows = Vec::new();
    {
        let mut store = MimicStore::open(db, adir)?;
        for corpus in corpora {
            if corpus.name.starts_with("alignment_") {
                continue; // templates feed the boundary gate, not coverage sweeps
            }
            // deterministic 80/20 train/eval split (every 5th line is eval)
            let train: Vec<&String> = corpus
                .utterances
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 5 != 4)
                .map(|(_, u)| u)
                .collect();
            let eval: Vec<&String> = corpus
                .utterances
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 5 == 4)
                .map(|(_, u)| u)
                .collect();

            let tts_ingest = MockTts::new();
            let mut unresolved = 0usize;
            for u in &train {
                let audio = tts_ingest.synthesize(u, "default")?;
                let rep =
                    ingest_with_options(&mut store, u, &audio, "default", tts_ingest.name(), opts, g2p)?;
                unresolved += rep.unresolved_words.len();
            }

            let tts_eval = MockTts::new();
            let mut row = ModeRow {
                mode: mode.to_string(),
                corpus: corpus.name.clone(),
                eval_utterances: eval.len(),
                unresolved_words: unresolved,
                ..Default::default()
            };
            let mut compose_wall = 0.0f64;
            let mut out_secs = 0.0f64;
            let mut seams = 0.0f64;
            let mut stoi_vals: Vec<f64> = Vec::new();
            let mut voice_vals: Vec<f64> = Vec::new();
            for u in &eval {
                let start = Instant::now();
                let (audio, rep) = match compose_mode {
                    ComposeMode::Greedy => compose(&mut store, &tts_eval, u, "default", g2p)?,
                    ComposeMode::ViterbiPcm => {
                        crate::select::compose_v3(&mut store, &tts_eval, u, "default", g2p)?
                    }
                    ComposeMode::ViterbiTokens => crate::select::compose_v3_with_medium(
                        &mut store,
                        &tts_eval,
                        u,
                        "default",
                        g2p,
                        crate::select::Medium::Tokens,
                    )?,
                };
                compose_wall += start.elapsed().as_secs_f64();
                out_secs += audio.duration_ms() / 1000.0;
                seams += rep.mean_seam_discontinuity;
                row.generated_chars += rep.generated_chars;
                row.total_chars += rep.total_chars;
                // codec fidelity, isolated to fully-cached utterances
                if compose_mode == ComposeMode::ViterbiTokens && rep.generated_chars == 0 {
                    let reference = tts_eval.synthesize(u, "default")?;
                    stoi_vals.push(crate::metrics::stoi(&reference, &audio));
                    voice_vals.push(crate::metrics::cosine(
                        &crate::features::voice_signature(&reference),
                        &crate::features::voice_signature(&audio),
                    ));
                }
            }
            row.cache_hit_pct = if row.total_chars > 0 {
                100.0 * (row.total_chars - row.generated_chars) as f64 / row.total_chars as f64
            } else {
                0.0
            };
            row.rtf = if out_secs > 0.0 { compose_wall / out_secs } else { 0.0 };
            row.mean_seam = if !eval.is_empty() {
                seams / eval.len() as f64
            } else {
                0.0
            };
            row.units_total = store.stats().total_nodes;
            if compose_mode == ComposeMode::ViterbiTokens {
                let mean = |v: &Vec<f64>| {
                    if v.is_empty() {
                        None
                    } else {
                        Some(v.iter().sum::<f64>() / v.len() as f64)
                    }
                };
                row.stoi = mean(&stoi_vals);
                row.voice_fidelity = mean(&voice_vals);
                let (tok, wav) = store.codec_storage();
                row.token_bytes = tok;
                row.wav_bytes = wav;
            }
            rows.push(row);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(rows)
}

/// Deterministic {n} expansions for alignment templates: 13 per template.
pub fn expand_templates(templates: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut state: u64 = 42;
    for t in templates {
        for _ in 0..13 {
            // LCG -> digit string of length 3..=7
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let len = 3 + (state >> 59) % 5;
            let mut digits = String::new();
            for _ in 0..len {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                digits.push((b'0' + ((state >> 33) % 10) as u8) as char);
            }
            out.push(units::normalize(&t.replace("{n}", &digits)));
        }
    }
    out
}

/// Word boundary error vs. MockTts ground truth, per alignment mode.
pub fn boundary_errors(clips: &[String], mode: AlignMode) -> Vec<f64> {
    let tts = MockTts::new();
    let mut errs = Vec::new();
    for clip in clips {
        let words = units::tokenize(clip);
        if words.len() < 2 {
            continue;
        }
        let truth = mock_word_spans(clip); // (word, start_ms, end_ms)
        let audio = tts.synthesize(clip, "default").expect("mock synth");
        let spans = align::word_spans(&audio, &words, mode);
        for i in 0..words.len() - 1 {
            let truth_mid_ms = truth[i].2 + GAP_MS / 2.0;
            let got_ms = spans[i].end as f64 * 1000.0 / audio.sample_rate as f64;
            errs.push((got_ms - truth_mid_ms).abs());
        }
    }
    errs
}

pub fn run(corpora_dir: &Path, g2p: &G2p) -> Result<EvalReport> {
    let corpora = load_corpora(corpora_dir)?;
    let mut report = EvalReport::default();

    // --- coverage sweeps
    let mut rows = run_mode(
        "v1-baseline",
        &IngestOptions::v1(),
        ComposeMode::Greedy,
        None,
        &corpora,
    )?;
    rows.extend(run_mode(
        "p1-legacy",
        &IngestOptions::p1_legacy(),
        ComposeMode::Greedy,
        Some(g2p),
        &corpora,
    )?);
    rows.extend(run_mode(
        "p3-legacy",
        &IngestOptions::p3_legacy(),
        ComposeMode::ViterbiPcm,
        Some(g2p),
        &corpora,
    )?);
    rows.extend(run_mode(
        "p4-current",
        &IngestOptions::default(),
        ComposeMode::ViterbiTokens,
        Some(g2p),
        &corpora,
    )?);
    report.rows = rows;

    // --- boundary gate (alignment_templates corpus)
    if let Some(t) = corpora.iter().find(|c| c.name == "alignment_templates") {
        let clips = expand_templates(&t.utterances);
        let errs = boundary_errors(&clips, AlignMode::default());
        let base = boundary_errors(&clips, AlignMode::Proportional);
        report.boundary = BoundaryStats {
            clips: clips.len(),
            boundaries: errs.len(),
            median_ms: percentile(errs.clone(), 50.0),
            p90_ms: percentile(errs.clone(), 90.0),
            max_ms: errs.iter().cloned().fold(0.0, f64::max),
            baseline_median_ms: percentile(base, 50.0),
        };
    } else {
        report
            .notes
            .push("alignment_templates corpus not found; boundary gate skipped".into());
    }

    // --- production-shaped workload replay (P2)
    if corpora.iter().any(|c| c.name == "support_repetitive") {
        report.replay = Some(run_replay(&corpora, g2p, 300, "elevenlabs")?);
    } else {
        report
            .notes
            .push("support_repetitive corpus not found; replay skipped".into());
    }

    // --- optional external neural metrics (UTMOS / Whisper WER / SECS)
    if std::env::var("MIMIC_EVAL_EXTERNAL").ok().as_deref() == Some("1") {
        let script = Path::new("scripts/eval_external.py");
        if script.exists() {
            match std::process::Command::new("python3").arg(script).output() {
                Ok(out) => report.notes.push(format!(
                    "external adapter: {}",
                    String::from_utf8_lossy(&out.stdout).trim()
                )),
                Err(e) => report.notes.push(format!("external adapter failed to run: {e}")),
            }
        } else {
            report
                .notes
                .push("MIMIC_EVAL_EXTERNAL=1 but scripts/eval_external.py missing".into());
        }
    } else {
        report.notes.push(
            "external metrics (UTMOS/NISQA, Whisper WER, SECS): SKIPPED — \
             set MIMIC_EVAL_EXTERNAL=1 with scripts/eval_external.py deps installed"
                .into(),
        );
    }
    Ok(report)
}

#[derive(Debug)]
pub struct GateResult {
    pub pass: bool,
    pub messages: Vec<String>,
}

/// Check the P1+P2 exit gates. `gates_file` holds `key = value` lines
/// (`#` comments); missing file falls back to built-in defaults.
pub fn check_gates(report: &EvalReport, gates_file: &Path) -> GateResult {
    let mut max_median = 20.0;
    let mut max_p90 = 60.0;
    let mut min_coverage = 80.0;
    let mut min_longtail = 81.5;
    let mut storage_ratio_min = 10.0;
    let mut stoi_min = 0.95;
    let mut voice_min = 0.90;
    if let Ok(text) = std::fs::read_to_string(gates_file) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let v: f64 = v.trim().parse().unwrap_or(f64::NAN);
                match k.trim() {
                    "boundary_median_ms_max" => max_median = v,
                    "boundary_p90_ms_max" => max_p90 = v,
                    "replay_coverage_min_pct" => min_coverage = v,
                    "longtail_coverage_min_pct" => min_longtail = v,
                    "storage_ratio_min" => storage_ratio_min = v,
                    "stoi_min" => stoi_min = v,
                    "voice_fidelity_min" => voice_min = v,
                    _ => {}
                }
            }
        }
    }
    let b = &report.boundary;
    let mut messages = vec![
        format!("boundary clips: {}, boundaries measured: {}", b.clips, b.boundaries),
        format!(
            "boundary median: {:.2} ms (gate ≤ {:.0} ms; v1-baseline median {:.2} ms)",
            b.median_ms, max_median, b.baseline_median_ms
        ),
        format!("boundary p90: {:.2} ms (gate ≤ {:.0} ms), max {:.2} ms", b.p90_ms, max_p90, b.max_ms),
    ];
    let mut pass = b.boundaries > 0 && b.median_ms <= max_median && b.p90_ms <= max_p90;
    if b.boundaries == 0 {
        messages.push("no boundaries measured — boundary gate cannot pass".into());
    }
    if let Some(r) = &report.replay {
        messages.push(format!(
            "replay: {} requests, coverage {:.1}% (gate ≥ {:.0}%), p99 {:.1} ms vs simulated-cloud p99 {:.0} ms",
            r.requests, r.coverage_pct, min_coverage, r.p99_ms, r.simulated_cloud_p99_ms
        ));
        // The latency leg compares against the simulated cloud baseline —
        // there is no live provider on this machine; MockTts is near-free.
        let lat_ok = r.p99_ms < r.simulated_cloud_p99_ms;
        if r.coverage_pct < min_coverage || !lat_ok {
            pass = false;
            messages.push("replay gate FAILED (coverage or latency)".into());
        }
    } else {
        messages.push("replay missing — replay gate cannot pass".into());
        pass = false;
    }
    // P3/P4: long-tail coverage must clear v1's 66.5% by 15 points
    match report
        .rows
        .iter()
        .find(|r| r.mode == "p4-current" && r.corpus == "long_tail")
    {
        Some(row) => {
            messages.push(format!(
                "long-tail coverage (p4-current): {:.1}% (gate ≥ {:.1}%; v1-baseline row shows the pre-P3 number)",
                row.cache_hit_pct, min_longtail
            ));
            if row.cache_hit_pct < min_longtail {
                pass = false;
                messages.push("long-tail coverage gate FAILED".into());
            }
        }
        None => {
            messages.push("p4-current long_tail row missing — gate cannot pass".into());
            pass = false;
        }
    }
    // P4 codec gates: storage ratio, STOI fidelity, voice fidelity
    let p4: Vec<&ModeRow> = report.rows.iter().filter(|r| r.mode == "p4-current").collect();
    if !p4.is_empty() {
        let tok: usize = p4.iter().map(|r| r.token_bytes).sum();
        let wav: usize = p4.iter().map(|r| r.wav_bytes).sum();
        let ratio = if tok > 0 { wav as f64 / tok as f64 } else { 0.0 };
        messages.push(format!(
            "codec storage: {} B tokens vs {} B wav-equivalent = {:.1}× (gate ≥ {:.0}×)",
            tok, wav, ratio, storage_ratio_min
        ));
        if tok == 0 || ratio < storage_ratio_min {
            pass = false;
            messages.push("codec storage gate FAILED".into());
        }
        let stoi_vals: Vec<f64> = p4.iter().filter_map(|r| r.stoi).collect();
        if !stoi_vals.is_empty() {
            let m = stoi_vals.iter().sum::<f64>() / stoi_vals.len() as f64;
            messages.push(format!("codec STOI (fully-cached): {m:.3} (gate ≥ {stoi_min})"));
            if m < stoi_min {
                pass = false;
                messages.push("codec STOI gate FAILED".into());
            }
        } else {
            messages.push("codec STOI: no fully-cached utterances — skipped".into());
        }
        let voice_vals: Vec<f64> = p4.iter().filter_map(|r| r.voice_fidelity).collect();
        if !voice_vals.is_empty() {
            let m = voice_vals.iter().sum::<f64>() / voice_vals.len() as f64;
            messages.push(format!("voice fidelity (fully-cached): {m:.3} (gate ≥ {voice_min})"));
            if m < voice_min {
                pass = false;
                messages.push("voice fidelity gate FAILED".into());
            }
        } else {
            messages.push("voice fidelity: no fully-cached utterances — skipped".into());
        }
    }
    messages.push(if pass { "GATE: PASS".into() } else { "GATE: FAIL".into() });
    GateResult { pass, messages }
}

/// Markdown report (also valid to paste into PRs/docs).
pub fn to_markdown(report: &EvalReport) -> String {
    let mut s = String::from("# Mimic eval report\n\n");
    s.push_str("## Coverage & compose sweeps\n\n");
    s.push_str("| mode | corpus | eval utts | cache hit % | generated chars | RTF | seam | unresolved words | units |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for r in &report.rows {
        s.push_str(&format!(
            "| {} | {} | {} | {:.1} | {} | {:.3} | {:.4} | {} | {} |\n",
            r.mode,
            r.corpus,
            r.eval_utterances,
            r.cache_hit_pct,
            r.generated_chars,
            r.rtf,
            r.mean_seam,
            r.unresolved_words,
            r.units_total
        ));
    }
    let b = &report.boundary;
    s.push_str("\n## Alignment gate (vs MockTts ground truth)\n\n");
    s.push_str(&format!(
        "- clips: {}, boundaries: {}\n- p1 median: **{:.2} ms**, p90: {:.2} ms, max: {:.2} ms\n- v1-baseline median: {:.2} ms\n",
        b.clips, b.boundaries, b.median_ms, b.p90_ms, b.max_ms, b.baseline_median_ms
    ));
    if let Some(r) = &report.replay {
        s.push_str("\n## Production-workload replay\n\n");
        s.push_str(&format!(
            "- requests: {} (Zipf-ish: 75% head / 25% long tail)\n- coverage: **{:.1}%** ({} / {} chars generated)\n- compose latency: p50 {:.1} ms, p99 {:.1} ms\n- provider-direct mock p99: {:.1} ms; simulated cloud p99: {:.0} ms\n- estimated cost: ${:.4} vs always-cloud ${:.4} ({} profile)\n",
            r.requests,
            r.coverage_pct,
            r.generated_chars,
            r.total_chars,
            r.p50_ms,
            r.p99_ms,
            r.mock_direct_p99_ms,
            r.simulated_cloud_p99_ms,
            r.est_cost_usd,
            r.always_cloud_cost_usd,
            r.provider_profile
        ));
    }
    let p4: Vec<&ModeRow> = report.rows.iter().filter(|r| r.mode == "p4-current").collect();
    if !p4.is_empty() {
        let tok: usize = p4.iter().map(|r| r.token_bytes).sum();
        let wav: usize = p4.iter().map(|r| r.wav_bytes).sum();
        let ratio = if tok > 0 { wav as f64 / tok as f64 } else { 0.0 };
        let mean_opt = |vs: Vec<f64>| {
            if vs.is_empty() {
                "n/a".to_string()
            } else {
                format!("{:.3}", vs.iter().sum::<f64>() / vs.len() as f64)
            }
        };
        s.push_str("\n## Codec-native cache (P4)\n\n");
        s.push_str(&format!(
            "- storage: {} B tokens vs {} B wav-equivalent = **{:.1}×**\n- STOI (fully-cached, native): {}\n- voice fidelity (fully-cached, native sig): {}\n",
            tok,
            wav,
            ratio,
            mean_opt(p4.iter().filter_map(|r| r.stoi).collect()),
            mean_opt(p4.iter().filter_map(|r| r.voice_fidelity).collect()),
        ));
    }
    s.push_str("\n## Notes\n\n");
    for n in &report.notes {
        s.push_str(&format!("- {n}\n"));
    }
    s
}

/// Flat JSON (hand-rolled; the project carries no serde dependency).
pub fn to_json(report: &EvalReport) -> String {
    let b = &report.boundary;
    let mut s = String::from("{\n  \"rows\": [\n");
    for (i, r) in report.rows.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"mode\": \"{}\", \"corpus\": \"{}\", \"eval_utterances\": {}, \"cache_hit_pct\": {:.2}, \"generated_chars\": {}, \"total_chars\": {}, \"rtf\": {:.4}, \"mean_seam\": {:.5}, \"unresolved_words\": {}, \"units_total\": {}, \"stoi\": {}, \"voice_fidelity\": {}, \"token_bytes\": {}, \"wav_bytes\": {}}}{}\n",
            r.mode,
            r.corpus,
            r.eval_utterances,
            r.cache_hit_pct,
            r.generated_chars,
            r.total_chars,
            r.rtf,
            r.mean_seam,
            r.unresolved_words,
            r.units_total,
            r.stoi.map(|v| format!("{v:.4}")).unwrap_or_else(|| "null".into()),
            r.voice_fidelity.map(|v| format!("{v:.4}")).unwrap_or_else(|| "null".into()),
            r.token_bytes,
            r.wav_bytes,
            if i + 1 == report.rows.len() { "" } else { "," }
        ));
    }
    s.push_str(&format!(
        "  ],\n  \"boundary\": {{\"clips\": {}, \"boundaries\": {}, \"median_ms\": {:.3}, \"p90_ms\": {:.3}, \"max_ms\": {:.3}, \"baseline_median_ms\": {:.3}}}",
        b.clips, b.boundaries, b.median_ms, b.p90_ms, b.max_ms, b.baseline_median_ms
    ));
    if let Some(r) = &report.replay {
        s.push_str(&format!(
            ",\n  \"replay\": {{\"requests\": {}, \"coverage_pct\": {:.2}, \"generated_chars\": {}, \"total_chars\": {}, \"p50_ms\": {:.3}, \"p99_ms\": {:.3}, \"mock_direct_p99_ms\": {:.3}, \"simulated_cloud_p99_ms\": {:.1}, \"est_cost_usd\": {:.6}, \"always_cloud_cost_usd\": {:.6}, \"provider_profile\": \"{}\"}}",
            r.requests,
            r.coverage_pct,
            r.generated_chars,
            r.total_chars,
            r.p50_ms,
            r.p99_ms,
            r.mock_direct_p99_ms,
            r.simulated_cloud_p99_ms,
            r.est_cost_usd,
            r.always_cloud_cost_usd,
            r.provider_profile
        ));
    }
    s.push_str("\n}\n");
    s
}
