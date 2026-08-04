//! Mimic CLI: ingest, compose, lookup, stats, eval.

use colored::Colorize;
use mimic::audio;
use mimic::eval as meval;
use mimic::g2p::G2p;
use mimic::pipeline;
use mimic::store::MimicStore;
use mimic::tts::{MockTts, TtsProvider};
use mimic::units::{self, UnitLevel};
use std::path::PathBuf;
use std::process::ExitCode;

use colored::control;

fn colored_usage() -> String {
    control::set_override(true);
    format!(
        "{header}\n│                    {title}           │\n│              {subtitle}               │\n{border}\n\n{description}\n\n{commands_header}\n  {ingest}      {ingest_desc}\n  {compose}     {compose_desc}\n  {lookup}      {lookup_desc}\n  {stats}       {stats_desc}\n  {eval}        {eval_desc}\n  {serve}       {serve_desc}\n\n{usage_header}\n  {ingest_cmd}\n  {compose_cmd}\n  {lookup_cmd}\n  {stats_cmd}\n  {eval_cmd}\n  {serve_cmd}\n\n{options_header}\n  {db_opt}         {db_desc}\n  {text_opt}       {text_desc}\n  {wav_opt}        {wav_desc}\n  {out_opt}        {out_desc}\n  {voice_opt}      {voice_desc}\n  {corpora_opt}    {corpora_desc}\n  {gates_opt}      {gates_desc}\n  {out_dir_opt}    {out_dir_desc}\n  {addr_opt}       {addr_desc}\n  {gate_opt}       {gate_desc}\n\n{examples_header}\n  {comment} Cache a TTS run (uses MockTts when --wav is omitted)\n  {ex1}\n\n  {comment} Ingest with external audio reference\n  {ex2}\n\n  {comment} Compose new speech, reusing cached units where possible\n  {ex3}\n\n  {comment} Compose with specific voice\n  {ex4}\n\n  {comment} Lookup cached units for inspection\n  {ex5}\n\n  {comment} View database statistics\n  {ex6}\n\n  {comment} Run evaluation with quality gate enforcement\n  {ex7}\n\n  {comment} Start HTTP server\n  {ex8}\n\n{notes_header}\n  {bullet} Unit audio cache lives in <db>.audio/ (e.g., demo.audio/)\n  {bullet} Without --wav, ingest synthesizes audio using the offline MockTts\n  {bullet} The compose algorithm uses target + join cost search over unit lattice\n  {bullet} Server endpoints: GET /health, GET /v1/stats, POST /v1/compose,\n                     POST /v1/audio/speech (OpenAI-compatible), POST /v1/ingest\n  {bullet} Cache metrics are exposed via response headers (x-mimic-cache-hit-pct, ...)\n\n{footer}\n",
        header = "╭─────────────────────────────────────────────────────────────────╮".bright_cyan(),
        border = "╰─────────────────────────────────────────────────────────────────╯".bright_cyan(),
        title = "Mimic — Phonetic Memoization Layer".bright_yellow().bold(),
        subtitle = "for Text-to-Speech (padagonia-backed)".bright_white(),
        description = "A semantic-aware speech cache with sub-word acoustic recomposition.\nGenerated speech is decomposed into reusable audio primitives (phrase,\nword, morpheme, phoneme) and stored as graph nodes. Future requests are\nresolved at the highest cached level, with missing spans synthesized\nand spliced using crossfades.".bright_white(),
        commands_header = "COMMANDS:".bright_green().bold(),
        ingest = "ingest".bright_cyan(),
        ingest_desc = "Cache a TTS run by decomposing audio into reusable units".bright_white(),
        compose = "compose".bright_cyan(),
        compose_desc = "Generate speech by reusing cached units at optimal levels".bright_white(),
        lookup = "lookup".bright_cyan(),
        lookup_desc = "Inspect cached units for a given text span".bright_white(),
        stats = "stats".bright_cyan(),
        stats_desc = "Display database statistics and cache metrics".bright_white(),
        eval = "eval".bright_cyan(),
        eval_desc = "Run benchmark harness with quality gates".bright_white(),
        serve = "serve".bright_cyan(),
        serve_desc = "Start HTTP server with mimic-native and OpenAI endpoints".bright_white(),
        usage_header = "USAGE:".bright_green().bold(),
        ingest_cmd = "mimic ingest --db PATH --text \"TEXT\" [--wav FILE] [--voice VOICE]".bright_white(),
        compose_cmd = "mimic compose --db PATH --text \"TEXT\" [--out FILE] [--voice VOICE]".bright_white(),
        lookup_cmd = "mimic lookup --db PATH --text \"TEXT\"".bright_white(),
        stats_cmd = "mimic stats --db PATH".bright_white(),
        eval_cmd = "mimic eval [--corpora DIR] [--gates FILE] [--out DIR] [--gate]".bright_white(),
        serve_cmd = "mimic serve --db PATH [--addr ADDRESS]".bright_white(),
        options_header = "OPTIONS:".bright_green().bold(),
        db_opt = "--db PATH".bright_cyan(),
        db_desc = "Path to padagonia database file (e.g., demo.pad)".bright_white(),
        text_opt = "--text TEXT".bright_cyan(),
        text_desc = "Input text to process (quoted string)".bright_white(),
        wav_opt = "--wav FILE".bright_cyan(),
        wav_desc = "Optional external audio file for ingest".bright_white(),
        out_opt = "--out FILE".bright_cyan(),
        out_desc = "Output audio file path (default: mimic_out.wav)".bright_white(),
        voice_opt = "--voice VOICE".bright_cyan(),
        voice_desc = "Voice identifier (default: \"default\")".bright_white(),
        corpora_opt = "--corpora DIR".bright_cyan(),
        corpora_desc = "Corpora directory for eval (default: assets/corpora)".bright_white(),
        gates_opt = "--gates FILE".bright_cyan(),
        gates_desc = "Quality gates file for eval (default: eval/gates.txt)".bright_white(),
        out_dir_opt = "--out DIR".bright_cyan(),
        out_dir_desc = "Output directory for eval reports (default: eval/reports)".bright_white(),
        addr_opt = "--addr ADDRESS".bright_cyan(),
        addr_desc = "Server bind address (default: 127.0.0.1:8787)".bright_white(),
        gate_opt = "--gate".bright_cyan(),
        gate_desc = "Enable quality gate enforcement (exits non-zero on failure)".bright_white(),
        examples_header = "EXAMPLES:".bright_green().bold(),
        comment = "#".bright_black(),
        ex1 = "mimic ingest --db demo.pad --text \"the quick brown fox jumps\"".bright_white(),
        ex2 = "mimic ingest --db demo.pad --text \"hello world\" --wav reference.wav".bright_white(),
        ex3 = "mimic compose --db demo.pad --text \"the quick red fox jumps\" --out output.wav".bright_white(),
        ex4 = "mimic compose --db demo.pad --text \"hello\" --voice clara --out hello.wav".bright_white(),
        ex5 = "mimic lookup --db demo.pad --text \"quick brown\"".bright_white(),
        ex6 = "mimic stats --db demo.pad".bright_white(),
        ex7 = "mimic eval --gate".bright_white(),
        ex8 = "mimic serve --db demo.pad --addr 127.0.0.1:8787".bright_white(),
        notes_header = "NOTES:".bright_green().bold(),
        bullet = "•".bright_cyan(),
        footer = "For more information, see README.md and ROADMAP.md".bright_black(),
    )
}

/// CMUdict is embedded at compile time: no runtime path fragility.
fn default_g2p() -> G2p {
    G2p::parse(include_str!("../assets/cmudict.dict"))
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
                let val = args.get(i + 1).ok_or_else(|| {
                    format!("missing value for {flag}; try again with {flag} VALUE")
                })?;
                match flag {
                    "--db" => a.db = Some(PathBuf::from(val)),
                    "--text" => a.text = Some(val.clone()),
                    "--wav" => a.wav = Some(PathBuf::from(val)),
                    "--out" => a.out = Some(PathBuf::from(val)),
                    "--voice" => a.voice = Some(val.clone()),
                    "--corpora" => a.corpora = Some(PathBuf::from(val)),
                    "--gates" => a.gates = Some(PathBuf::from(val)),
                    "--addr" => a.addr = Some(val.clone()),
                    other => {
                        return Err(format!(
                            "unknown flag: {other}; run `mimic`, then try again"
                        ))
                    }
                }
                i += 2;
            }
        }
    }
    Ok(a)
}

fn open_store(a: &Args) -> Result<MimicStore, String> {
    let db =
        a.db.clone()
            .ok_or("missing --db PATH; try again with --db PATH")?;
    let audio_dir = db.with_extension("audio");
    MimicStore::open(db, audio_dir).map_err(|e| e.to_string())
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first().cloned() else {
        return Err(colored_usage());
    };
    let a = parse_args(&argv[1..])?;
    let voice = a.voice.clone().unwrap_or_else(|| "default".to_string());

    match cmd.as_str() {
        "ingest" => {
            let text = a
                .text
                .clone()
                .ok_or("missing --text; try again with --text TEXT")?;
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
            let report = pipeline::ingest(&mut store, &text, &audio, &voice, &provider, Some(&g2p))
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
            let text = a
                .text
                .clone()
                .ok_or("missing --text; try again with --text TEXT")?;
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
            let out = a
                .out
                .clone()
                .unwrap_or_else(|| PathBuf::from("mimic_out.wav"));
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
            let text = a
                .text
                .clone()
                .ok_or("missing --text; try again with --text TEXT")?;
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
            let addr = a
                .addr
                .clone()
                .unwrap_or_else(|| "127.0.0.1:8787".to_string());
            let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
            rt.block_on(mimic::server::serve(&addr, store, g2p))
                .map_err(|e| e.to_string())?;
        }
        "eval" => {
            let corpora = a
                .corpora
                .clone()
                .unwrap_or_else(|| PathBuf::from("assets/corpora"));
            let gates = a
                .gates
                .clone()
                .unwrap_or_else(|| PathBuf::from("eval/gates.txt"));
            let out_dir = a
                .out
                .clone()
                .unwrap_or_else(|| PathBuf::from("eval/reports"));
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
                return Err(
                    "P1 gate failed; inspect the report, resolve each failed metric, and try again"
                        .to_string(),
                );
            }
        }
        _ => return Err(colored_usage()),
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
