//! P3 tests: diphone inventory, Viterbi unit selection (target + join
//! cost search), context-based variant selection.

use mimic::g2p::G2p;
use mimic::pipeline::ingest;
use mimic::select::compose_v3;
use mimic::store::MimicStore;
use mimic::tts::{MockTts, TtsProvider};
use mimic::units::UnitLevel;

fn test_g2p() -> G2p {
    G2p::from_str(include_str!("../assets/cmudict.dict"))
}

fn fresh_store() -> (tempfile::TempDir, MimicStore) {
    let dir = tempfile::tempdir().unwrap();
    let ms = MimicStore::open(dir.path().join("t.pad"), dir.path().join("t.audio")).unwrap();
    (dir, ms)
}

fn ingest_text(ms: &mut MimicStore, text: &str, g2p: &G2p) {
    let tts = MockTts::new();
    let audio = tts.synthesize(text, "default").unwrap();
    ingest(ms, text, &audio, "default", tts.name(), Some(g2p)).unwrap();
}

#[test]
fn diphone_inventory_at_ingest() {
    let (_dir, mut ms) = fresh_store();
    let g = test_g2p();
    ingest_text(&mut ms, "hello world", &g);

    // 8 phonemes -> 7 diphones, including the cross-word transition
    let stats = ms.stats();
    assert_eq!(stats.phonemes, 8);
    assert_eq!(stats.diphones, 7);

    let ow_w = ms.lookup_exact(UnitLevel::Diphone, "OW+W");
    assert_eq!(ow_w.len(), 1, "cross-word diphone OW+W");
    assert_eq!(ms.prop_string(ow_w[0], "context_prev").as_deref(), Some("L"));
    assert_eq!(ms.prop_string(ow_w[0], "context_next").as_deref(), Some("ER1"));
    assert_eq!(ms.prop_string(ow_w[0], "phonemes").as_deref(), Some("OW1 W"));
}

#[test]
fn viterbi_prefers_word_hits_and_synthesizes_only_misses() {
    let (_dir, mut ms) = fresh_store();
    let g = test_g2p();
    ingest_text(&mut ms, "the quick brown fox jumps", &g);

    let tts = MockTts::new();
    let (_out, report) = compose_v3(&mut ms, &tts, "the quick red fox jumps", "default", Some(&g)).unwrap();
    assert_eq!(tts.calls.lock().unwrap().as_slice(), &["red".to_string()]);
    let levels: Vec<UnitLevel> = report.hits.iter().map(|(_, l)| *l).collect();
    assert_eq!(
        levels,
        vec![UnitLevel::Word, UnitLevel::Word, UnitLevel::Word, UnitLevel::Word],
        "known words resolve at word level: {:?}",
        report.hits
    );
    assert!(report.cached_chars > 0);
}

#[test]
fn viterbi_uses_diphone_chain_for_oov_word() {
    let (_dir, mut ms) = fresh_store();
    let g = test_g2p();
    ingest_text(&mut ms, "helmet", &g); // HH EH1 L M AH0 T

    let tts = MockTts::new();
    let (_out, report) = compose_v3(&mut ms, &tts, "helm", "default", Some(&g)).unwrap();
    // "helm" has no morpheme candidate in cache, but HH+EH, EH+L, L+M
    // diphones and HH/M phoneme edges all exist
    assert!(tts.calls.lock().unwrap().is_empty(), "no synthesis expected");
    assert!(
        report
            .hits
            .iter()
            .any(|(t, l)| t == "helm" && *l == UnitLevel::Diphone),
        "expected diphone hit: {:?}",
        report.hits
    );
}

#[test]
fn viterbi_falls_back_to_synth_for_novel_phonemes() {
    let (_dir, mut ms) = fresh_store();
    let g = test_g2p();
    ingest_text(&mut ms, "hello", &g);

    let tts = MockTts::new();
    let (_out, report) = compose_v3(&mut ms, &tts, "xqz", "default", Some(&g)).unwrap();
    assert_eq!(tts.calls.lock().unwrap().as_slice(), &["xqz".to_string()]);
    assert!(report.generated_chars > 0);
}

#[test]
fn viterbi_selects_context_matching_variant() {
    let (_dir, mut ms) = fresh_store();
    let g = test_g2p();
    ingest_text(&mut ms, "the cat sat", &g);
    ingest_text(&mut ms, "a cat sleeps", &g);

    let cats = ms.lookup_exact(UnitLevel::Word, "cat");
    assert_eq!(cats.len(), 2);
    let (cat_after_the, cat_after_a) = (cats[0], cats[1]);
    assert_eq!(
        ms.prop_string(cat_after_the, "context_prev").as_deref(),
        Some("the")
    );
    assert_eq!(ms.prop_string(cat_after_a, "context_prev").as_deref(), Some("a"));

    let tts = MockTts::new();
    let (_out, report) = compose_v3(&mut ms, &tts, "the cat sleeps", "default", Some(&g)).unwrap();
    assert!(tts.calls.lock().unwrap().is_empty());
    assert!(
        report.units.contains(&cat_after_the),
        "expected the context-matching cat variant in {:?}",
        report.units
    );
    assert!(
        !report.units.contains(&cat_after_a),
        "the mismatched variant should not be selected"
    );
}
