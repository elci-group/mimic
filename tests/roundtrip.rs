use mimic::audio::{self, splice, WavAudio};
use mimic::features;
use mimic::pipeline::{compose, ingest};
use mimic::store::MimicStore;
use mimic::tts::{MockTts, TtsProvider};
use mimic::units::UnitLevel;
use mimic::SAMPLE_RATE;

fn tone(freq: f64, ms: f64) -> WavAudio {
    let n = (SAMPLE_RATE as f64 * ms / 1000.0) as usize;
    let samples = (0..n)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE as f64;
            ((2.0 * std::f64::consts::PI * freq * t).sin() * 0.5 * i16::MAX as f64) as i16
        })
        .collect();
    WavAudio::new(samples, SAMPLE_RATE)
}

#[test]
fn wav_codec_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.wav");
    let audio = WavAudio::new(vec![0, 1000, -1000, i16::MAX, i16::MIN, 12345, -23456], SAMPLE_RATE);
    audio::write_wav(&audio, &path).unwrap();
    let back = audio::read_wav(&path).unwrap();
    assert_eq!(back, audio);
}

#[test]
fn splice_length_and_fade() {
    let a = tone(220.0, 100.0); // 1600 samples
    let b = tone(440.0, 100.0);
    let out = splice(&[a, b], 10).unwrap();
    // 1600 + 1600 - 160 (one 10 ms join)
    assert_eq!(out.len(), 3200 - 160);
    assert!(out.peak() <= i16::MAX as i32);
    assert!(out.peak() > 0);
}

#[test]
fn embedding_deterministic_and_discriminative() {
    let a1 = tone(220.0, 300.0);
    let a2 = tone(220.0, 300.0);
    let e1 = features::embed("hello world", &a1);
    let e2 = features::embed("hello world", &a2);
    assert_eq!(e1, e2, "embedding must be deterministic");
    assert!(features::cosine_distance(&e1, &e2) < 1e-6);

    let e3 = features::embed("zzq vorpal gimble", &tone(1370.0, 800.0));
    let d_same = features::cosine_distance(&e1, &e2);
    let d_diff = features::cosine_distance(&e1, &e3);
    assert!(d_diff > d_same + 0.01, "d_diff={d_diff} d_same={d_same}");
}

#[test]
fn store_persists_and_retrieves() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("t.pad");
    let adir = dir.path().join("t.audio");

    let tts = MockTts::new();
    let audio = tts.synthesize("hello", "default").unwrap();
    let id = {
        let mut ms = MimicStore::open(db.clone(), adir.clone()).unwrap();
        let id = ms
            .insert_unit(
                UnitLevel::Word,
                "hello",
                "hello",
                "default",
                &audio,
                None,
                true,
                None,
                None,
                tts.name(),
            )
            .unwrap();
        ms.save().unwrap();
        id
    };

    let ms2 = MimicStore::open(db, adir).unwrap();
    assert_eq!(ms2.lookup_exact(UnitLevel::Word, "hello"), &[id]);
    let emb = features::embed("hello", &audio);
    let hits = ms2.similar(&emb, 5);
    assert!(
        hits.iter().any(|(nid, _)| *nid == id),
        "HNSW should return the ingested unit among its nearest neighbors: {hits:?}"
    );
    // audio survives the round trip
    let back = ms2.get_audio(id).unwrap();
    assert_eq!(back, audio);
}

#[test]
fn compose_reuses_cache_and_synthesizes_only_misses() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("t.pad");
    let adir = dir.path().join("t.audio");
    let mut ms = MimicStore::open(db, adir).unwrap();

    let tts1 = MockTts::new();
    let full = tts1.synthesize("the quick brown fox jumps", "default").unwrap();
    ingest(&mut ms, "the quick brown fox jumps", &full, "default", tts1.name(), None).unwrap();

    let tts2 = MockTts::new();
    let (out, report) = compose(&mut ms, &tts2, "the quick red fox jumps", "default", None).unwrap();

    // only "red" was unknown: exactly one provider call for exactly that word
    assert_eq!(tts2.calls.lock().unwrap().as_slice(), &["red".to_string()]);
    assert_eq!(report.tts_calls, vec!["red".to_string()]);
    assert_eq!(report.generated_chars, 3);
    assert_eq!(report.cached_chars, 16); // the+quick+fox+jumps
    assert!(report.cache_hit_pct() > 80.0);

    // expected output length: sum of the 5 parts minus 4 crossfades
    let part_lens: usize = ["the", "quick", "red", "fox", "jumps"]
        .iter()
        .map(|w| {
            let id = ms.lookup_exact(UnitLevel::Word, w)[0];
            ms.get_audio(id).unwrap().len()
        })
        .sum();
    let fade = (SAMPLE_RATE as usize) * 10 / 1000;
    let expected = part_lens - 4 * fade;
    let diff = (out.len() as i64 - expected as i64).abs();
    assert!(diff <= 2, "len={} expected={expected}", out.len());
    assert!(out.peak() <= i16::MAX as i32, "no clipping");
    assert!(out.peak() > 0, "not silent");
}

#[test]
fn morpheme_fallback_segments_cached_word() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("t.pad");
    let adir = dir.path().join("t.audio");
    let mut ms = MimicStore::open(db, adir).unwrap();

    let tts = MockTts::new();
    let audio = tts.synthesize("running", "default").unwrap();
    let full_len = audio.len();
    ingest(&mut ms, "running", &audio, "default", tts.name(), None).unwrap();

    let tts2 = MockTts::new();
    let (out, report) = compose(&mut ms, &tts2, "run", "default", None).unwrap();

    assert!(tts2.calls.lock().unwrap().is_empty(), "no synthesis expected");
    assert!(
        report
            .hits
            .iter()
            .any(|(t, l)| t == "run" && *l == UnitLevel::Morpheme),
        "expected a morpheme hit: {:?}",
        report.hits
    );
    // "run" is a 3/7 char prefix of "running"
    let expected = full_len as f64 * 3.0 / 7.0;
    let diff = (out.len() as f64 - expected).abs() / expected;
    assert!(diff < 0.02, "len={} expected~{expected:.0}", out.len());
}
