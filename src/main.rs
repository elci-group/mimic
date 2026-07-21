//! Mimic CLI: ingest, compose, lookup, stats, eval.

use mimic::audio;
use mimic::eval as meval;
use mimic::g2p::G2p;
use mimic::pipeline;
use mimic::store::MimicStore;
use mimic::tts::{MockTts, TtsProvider};
use mimic::units::{self, UnitLevel};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "mimic — phonetic memoization layer for TTS (padagonia-backed)

USAGE:
  mimic ingest --db PATH --text \"...\" [--wav FILE] [--voice V]
  mimic compose --db PATH --text \"...\" [--out FILE] [--voice V]
  mimic lookup --db PATH --text \"...\"
  mimic stats --db PATH
  mimic eval [--corpora DIR] [--gates FILE] [--out DIR] [--gate]
  mimic serve --db PATH [--addr 127.0.0.1:8787]

Without --wav, ingest synthesizes the \"original\" audio with the offline
MockTts (that's the TTS run being memoized). The unit wav cache lives in
<db with extension replaced by .audio>.

eval runs the benchmark harness (defaults: corpora assets/corpora, gates
eval/gates.txt, reports written to eval/reports). --gate exits non-zero if
the P1 alignment gate fails.";

/// CMUdict is embedded at compile time: no runtime path fragility.
fn default_g2p() -> G2p {
    G2p::from_str(include_str!("../assets/cmudict.dict"))
}

#[derive(Default)]
struct Args {
    db: Option<PathBuf>,
    text: Option<String>,
    wav: Option<PathBuf>,
    out: Option<PathBuf>,
    voice: Option<String>,
    corpora: Option<PathBuf>,
    gates: Option<PathBuf>,
    addr: Option<String>,
    gate: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gate" => {
                a.gate = true;
                i += 1;
            }
            flag => {
                let val = args
                    .get(i + 1)
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                match flag {
                    "--db" => a.db = Some(PathBuf::from(val)),
                    "--text" => a.text = Some(val.clone()),
                    "--wav" => a.wav = Some(PathBuf::from(val)),
                    "--out" => a.out = Some(PathBuf::from(val)),
                    "--voice" => a.voice = Some(val.clone()),
                    "--corpora" => a.corpora = Some(PathBuf::from(val)),
                    "--gates" => a.gates = Some(PathBuf::from(val)),
                    "--addr" => a.addr = Some(val.clone()),
                    other => return Err(format!("unknown flag: {other}")),
                }
                i += 2;
            }
        }
    }
    Ok(a)
}

fn open_store(a: &Args) -> Result<MimicStore, String> {
    let db = a.db.clone().ok_or("missing --db PATH")?;
    let audio_dir = db.with_extension("audio");
    MimicStore::open(db, audio_dir).map_err(|e| e.to_string())
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first().cloned() else {
        return Err(USAGE.to_string());
    };
    let a = parse_args(&argv[1..])?;
    let voice = a.voice.clone().unwrap_or_else(|| "default".to_string());

    match cmd.as_str() {
        "ingest" => {
            let text = a.text.clone().ok_or("missing --text")?;
            let mut store = open_store(&a)?;
            let g2p = default_g2p();
            let (audio, provider): (mimic::audio::WavAudio, String) = match &a.wav {
                Some(p) => (
                    audio::read_wav(p).map_err(|e| format!("read {}: {e}", p.display()))?,
                    "external".to_string(),
                ),
                None => {
                    let tts = MockTts::new();
                    (
                        tts.synthesize(&text, &voice).map_err(|e| e.to_string())?,
                        tts.name().to_string(),
                    )
                }
            };
            let report =
                pipeline::ingest(&mut store, &text, &audio, &voice, &provider, Some(&g2p))
                    .map_err(|e| e.to_string())?;
            store.save().map_err(|e| e.to_string())?;
            println!(
                "ingested \"{}\" ({:.0} ms): {} phrase + {} word + {} phoneme units ({} unresolved words)",
                units::normalize(&text),
                audio.duration_ms(),
                report.phrase_units,
                report.word_units,
                report.phoneme_units,
                report.unresolved_words.len()
            );
        }
        "compose" => {
            let text = a.text.clone().ok_or("missing --text")?;
            let mut store = open_store(&a)?;
            let g2p = default_g2p();
            let tts = MockTts::new();
            let (out_audio, report) = mimic::select::compose_v3_with_medium(
                &mut store,
                &tts,
                &text,
                &voice,
                Some(&g2p),
                mimic::select::Medium::Tokens,
            )
            .map_err(|e| e.to_string())?;
            store.save().map_err(|e| e.to_string())?;
            let out = a.out.clone().unwrap_or_else(|| PathBuf::from("mimic_out.wav"));
            audio::write_wav(&out_audio, &out).map_err(|e| e.to_string())?;
            println!("composed \"{}\"", units::normalize(&text));
            println!(
                "cache hit: {:.1}% ({} cached / {} generated of {} chars)",
                report.cache_hit_pct(),
                report.cached_chars,
                report.generated_chars,
                report.total_chars
            );
            for (t, level) in &report.hits {
                println!("  [{level:8}] {t}");
            }
            if report.tts_calls.is_empty() {
                println!("tts calls: none — fully served from cache");
            } else {
                println!("tts calls ({}):", report.tts_calls.len());
                for c in &report.tts_calls {
                    println!("  \"{c}\"");
                }
            }
            println!(
                "seam discontinuity: {:.4} (lower is smoother)",
                report.mean_seam_discontinuity
            );
            println!(
                "wrote {} ({:.0} ms, {} samples)",
                out.display(),
                out_audio.duration_ms(),
                out_audio.len()
            );
        }
        "lookup" => {
            let text = a.text.clone().ok_or("missing --text")?;
            let store = open_store(&a)?;
            let norm = units::normalize(&text);
            let mut found = 0usize;
            let mut show = |level: UnitLevel, key: &str| {
                for &id in store.lookup_exact(level, key) {
                    let dur = store.prop_string(id, "duration_ms").unwrap_or_default();
                    let wav = store.prop_string(id, "wav_path").unwrap_or_default();
                    println!("  [{level:8}] node {id} \"{key}\" {dur} ms -> {wav}");
                    found += 1;
                }
            };
            show(UnitLevel::Phrase, &norm);
            for w in units::tokenize(&text) {
                show(UnitLevel::Word, &w);
            }
            if found == 0 {
                println!("no cached units for \"{norm}\"");
            }
        }
        "stats" => {
            let store = open_store(&a)?;
            let s = store.stats();
            println!("db: {}", store.db_path().display());
            println!("audio dir: {}", store.audio_dir().display());
            println!(
                "units: {} phrases, {} words, {} morphemes, {} phonemes",
                s.phrases, s.words, s.morphemes, s.phonemes
            );
            println!("graph: {} nodes, {} edges", s.total_nodes, s.total_edges);
        }
        "serve" => {
            let store = open_store(&a)?;
            let g2p = default_g2p();
            let addr = a.addr.clone().unwrap_or_else(|| "127.0.0.1:8787".to_string());
            let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
            rt.block_on(mimic::server::serve(&addr, store, g2p))
                .map_err(|e| e.to_string())?;
        }
        "eval" => {
            let corpora = a.corpora.clone().unwrap_or_else(|| PathBuf::from("assets/corpora"));
            let gates = a.gates.clone().unwrap_or_else(|| PathBuf::from("eval/gates.txt"));
            let out_dir = a.out.clone().unwrap_or_else(|| PathBuf::from("eval/reports"));
            let g2p = default_g2p();
            let report = meval::run(&corpora, &g2p).map_err(|e| e.to_string())?;
            let md = meval::to_markdown(&report);
            println!("{md}");
            std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let md_path = out_dir.join(format!("eval-{ts}.md"));
            let json_path = out_dir.join(format!("eval-{ts}.json"));
            std::fs::write(&md_path, &md).map_err(|e| e.to_string())?;
            std::fs::write(&json_path, meval::to_json(&report)).map_err(|e| e.to_string())?;
            println!("report: {} / {}", md_path.display(), json_path.display());
            let gate = meval::check_gates(&report, &gates);
            for m in &gate.messages {
                println!("{m}");
            }
            if a.gate && !gate.pass {
                return Err("P1 gate failed".to_string());
            }
        }
        _ => return Err(USAGE.to_string()),
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
