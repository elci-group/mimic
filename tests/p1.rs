//! P1 tests: G2P, energy-refined alignment gate, phoneme inventory,
//! and the eval harness itself.

use mimic::align::AlignMode;
use mimic::eval as meval;
use mimic::g2p::G2p;
use mimic::pipeline::ingest;
use mimic::store::MimicStore;
use mimic::tts::{MockTts, TtsProvider};
use mimic::units::UnitLevel;

fn test_g2p() -> G2p {
    G2p::parse(include_str!("../assets/cmudict.dict"))
}

#[test]
fn g2p_dictionary_lookup() {
    let g = test_g2p();
    assert_eq!(
        g.phonemes("hello").unwrap(),
        vec!["HH", "AH0", "L", "OW1"],
        "CMUdict first pronunciation of hello"
    );
    // digit expansion: 4 -> four, 2 -> two
    assert_eq!(
        g.phonemes("42").unwrap(),
        vec!["F", "AO1", "R", "T", "UW1"],
        "digits expand via digit words"
    );
    // OOV letter fallback: each letter resolves to its letter-name phones
    let p = g.phonemes("zzq").unwrap();
    assert!(!p.is_empty());
    assert_eq!(G2p::base("AH0"), "AH");
}

#[test]
fn phoneme_inventory_stored_with_context() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("t.pad");
    let adir = dir.path().join("t.audio");
    let g = test_g2p();
    let mut ms = MimicStore::open(db, adir).unwrap();
    let tts = MockTts::new();
    let audio = tts.synthesize("hello world", "default").unwrap();
    let rep = ingest(
        &mut ms,
        "hello world",
        &audio,
        "default",
        tts.name(),
        Some(&g),
    )
    .unwrap();

    let expected = g.phonemes("hello").unwrap().len() + g.phonemes("world").unwrap().len();
    assert_eq!(rep.phoneme_units, expected);
    assert!(rep.unresolved_words.is_empty());

    let stats = ms.stats();
    assert_eq!(stats.phonemes, expected);

    // phoneme node carries ARPAbet text + cross-word context:
    // hello's last phoneme OW1 -> next is world's first phoneme W
    let ow = ms.lookup_exact(UnitLevel::Phoneme, "OW1");
    assert_eq!(ow.len(), 1);
    assert_eq!(ms.prop_string(ow[0], "context_next").as_deref(), Some("W"));
    assert_eq!(ms.prop_string(ow[0], "context_prev").as_deref(), Some("L"));

    // word unit carries the real phoneme string, not the placeholder
    let w = ms.lookup_exact(UnitLevel::Word, "hello");
    assert_eq!(
        ms.prop_string(w[0], "phonemes").as_deref(),
        Some("HH AH0 L OW1")
    );
}

#[test]
fn alignment_gate_small_sample() {
    // The full 500+ clip gate runs via `mimic eval --gate`; this keeps the
    // property under test fast: energy-refined alignment must land within
    // 20 ms of ground truth at the median, and beat proportional mode.
    let templates = [
        "your code is {n}".to_string(),
        "the number is {n}".to_string(),
        "order {n} is ready".to_string(),
    ];
    let clips = meval::expand_templates(&templates);
    assert_eq!(clips.len(), 39);
    let errs = meval::boundary_errors(&clips, AlignMode::default());
    let base = meval::boundary_errors(&clips, AlignMode::Proportional);
    assert!(!errs.is_empty());
    let mut sorted = errs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mut b_sorted = base.clone();
    b_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let b_median = b_sorted[b_sorted.len() / 2];
    assert!(
        median < 20.0,
        "energy-refined median {median:.2} ms must be < 20 ms (baseline {b_median:.2} ms)"
    );
    assert!(
        median <= b_median,
        "refined ({median:.2}) should not be worse than proportional ({b_median:.2})"
    );
}

#[test]
fn eval_harness_smoke() {
    let dir = tempfile::tempdir().unwrap();
    let corpora_dir = dir.path().join("corpora");
    std::fs::create_dir_all(&corpora_dir).unwrap();
    std::fs::write(
        corpora_dir.join("mini.txt"),
        "thank you for calling\nyour order is ready\nhow can i help\nplease hold on\nthank you\nyour code is 123\n",
    )
    .unwrap();
    std::fs::write(
        corpora_dir.join("alignment_templates.txt"),
        "code {n} now\nthe total is {n}\n",
    )
    .unwrap();
    // replay gate needs the production corpora; head lines must repeat
    // enough that steady-state coverage clears 80%
    let mut support = String::new();
    for i in 0..14 {
        support.push_str(&format!("head phrase number {i} for support\n"));
    }
    std::fs::write(corpora_dir.join("support_repetitive.txt"), support).unwrap();
    // the single eval line (every 5th) repeats a trained line, so the
    // p3 long-tail coverage gate has something resolvable to chew on
    std::fs::write(
        corpora_dir.join("long_tail.txt"),
        "tail one here\ntail two here\ntail four here\ntail three here\ntail four here\n",
    )
    .unwrap();

    let g = test_g2p();
    let report = meval::run(&corpora_dir, &g).unwrap();

    // both mode rows exist for the mini corpus (the historical baseline row
    // plus the legacy and current rows)
    assert!(report
        .rows
        .iter()
        .any(|r| r.mode == "v1-baseline" && r.corpus == "mini"));
    assert!(report
        .rows
        .iter()
        .any(|r| r.mode == "p1-legacy" && r.corpus == "mini"));
    assert!(report
        .rows
        .iter()
        .any(|r| r.mode == "p4-current" && r.corpus == "mini"));
    // 2 templates x 13 expansions
    assert_eq!(report.boundary.clips, 26);
    assert!(report.boundary.boundaries > 0);

    let gate = meval::check_gates(&report, &corpora_dir.join("no-such-gates.txt"));
    // This is a harness smoke test over a tiny synthetic fixture, not the
    // experimental P4 codec release gate. Production lossless quality is
    // enforced separately by plan_tests.rs.
    assert!(!gate.messages.is_empty());
    assert!(gate.messages.iter().any(|message| message.contains("boundary")));
    assert!(gate.messages.iter().any(|message| message.contains("codec")));

    // report renderers don't panic and contain the key rows
    let md = meval::to_markdown(&report);
    assert!(md.contains("v1-baseline"));
    let js = meval::to_json(&report);
    assert!(js.contains("\"p4-current\""));
}
