//! P4 tests: codec roundtrip, compression, STOI, token-stream surgery,
//! token-mode compose, and voice signatures.

use mimic::audio::WavAudio;
use mimic::codec::{concat_tokens, frames_for, slice_tokens, AudioCodec, MimicMct, FRAME};
use mimic::features;
use mimic::g2p::G2p;
use mimic::metrics;
use mimic::pipeline::ingest;
use mimic::select::{compose_v3_with_medium, Medium};
use mimic::store::MimicStore;
use mimic::tts::{MockTts, TtsProvider};
use mimic::units::UnitLevel;
use mimic::SAMPLE_RATE;

fn tone(freq: f64, ms: f64) -> WavAudio {
    let n = (SAMPLE_RATE as f64 * ms / 1000.0) as usize;
    let samples = (0..n)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE as f64;
            let harm = (2.0 * std::f64::consts::PI * freq * t).sin()
                + 0.5 * (4.0 * std::f64::consts::PI * freq * t).sin()
                + 0.25 * (6.0 * std::f64::consts::PI * freq * t).sin();
            (harm * 0.35 * i16::MAX as f64) as i16
        })
        .collect();
    WavAudio::new(samples, SAMPLE_RATE)
}

fn test_g2p() -> G2p {
    G2p::parse(include_str!("../assets/cmudict.dict"))
}

#[test]
fn codec_roundtrip_high_fidelity() {
    let a = tone(220.0, 400.0);
    let tokens = MimicMct.encode(&a);
    let back = MimicMct.decode(&tokens).unwrap();
    assert_eq!(back.samples.len(), a.samples.len());
    let s = metrics::stoi(&a, &back);
    assert!(s > 0.95, "STOI on tonal content {s}");
}

#[test]
fn codec_compression_ratio() {
    // 1s of tone + 1s of silence: tonal frames compress, silence is 1 B/frame
    let mut samples = tone(300.0, 1000.0).samples;
    samples.extend(std::iter::repeat_n(0, SAMPLE_RATE as usize));
    let a = WavAudio::new(samples, SAMPLE_RATE);
    let tokens = MimicMct.encode(&a);
    let pcm_bytes = 44 + a.samples.len() * 2;
    let ratio = pcm_bytes as f64 / tokens.len() as f64;
    assert!(
        ratio >= 10.0,
        "ratio {ratio} ({} -> {} B)",
        pcm_bytes,
        tokens.len()
    );
}

#[test]
fn codec_token_surgery() {
    let a = tone(250.0, 300.0);
    let b = tone(350.0, 200.0);
    let ta = MimicMct.encode(&a);
    let tb = MimicMct.encode(&b);

    // slice: first half of a
    let half = slice_tokens(&ta, 0.0, 0.5).unwrap();
    let half_dec = MimicMct.decode(&half).unwrap();
    let expect = frames_for(a.samples.len()) / 2 * FRAME;
    assert!(
        (half_dec.samples.len() as i64 - expect as i64).abs() <= FRAME as i64,
        "sliced len {} vs ~{}",
        half_dec.samples.len(),
        expect
    );

    // concat: a + b == samples of both
    let cat = concat_tokens(&[&ta, &tb]).unwrap();
    let cat_dec = MimicMct.decode(&cat).unwrap();
    assert_eq!(cat_dec.samples.len(), a.samples.len() + b.samples.len());
    // and the boundary region is intact (OLA smoothed, not silent-glitch)
    assert!(cat_dec.peak() > 0);
}

#[test]
fn token_compose_matches_pcm_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let mut ms = MimicStore::open(dir.path().join("t.pad"), dir.path().join("t.audio")).unwrap();
    let g = test_g2p();
    let tts = MockTts::new();
    let audio = tts
        .synthesize("the quick brown fox jumps", "default")
        .unwrap();
    // P4-style ingest: tokens inline, no wav files
    ingest(
        &mut ms,
        "the quick brown fox jumps",
        &audio,
        "default",
        tts.name(),
        Some(&g),
    )
    .unwrap();

    // units carry inline tokens, and no wav files were written
    let first = ms.lookup_exact(UnitLevel::Word, "quick")[0];
    assert!(ms.get_tokens(first).is_ok());
    assert!(!dir
        .path()
        .join("t.audio")
        .join(format!("{}.wav", first.0))
        .exists());
    // but get_audio still serves audio by decoding tokens
    assert!(!ms.get_audio(first).unwrap().is_empty());

    let tts2 = MockTts::new();
    let (out, report) = compose_v3_with_medium(
        &mut ms,
        &tts2,
        "the quick red fox jumps",
        "default",
        Some(&g),
        Medium::Tokens,
    )
    .unwrap();
    assert_eq!(tts2.calls.lock().unwrap().as_slice(), &["red".to_string()]);
    assert_eq!(report.generated_chars, 3);
    assert_eq!(report.cached_chars, 16);
    assert!(out.peak() > 0);
    // and the decoded output is intelligible vs direct generation
    let reference = tts2
        .synthesize("the quick red fox jumps", "default")
        .unwrap();
    let s = metrics::stoi(&reference, &out);
    assert!(s > 0.9, "STOI composed vs direct {s}");
}

#[test]
fn voice_signatures_separate_voices() {
    let tts = MockTts::new();
    let a1 = tts.synthesize("hello world", "default").unwrap();
    let a2 = tts.synthesize("hello world", "default").unwrap();
    let other = tts.synthesize("hello world", "robot").unwrap();
    let s1 = features::voice_signature(&a1);
    let s2 = features::voice_signature(&a2);
    let s3 = features::voice_signature(&other);
    let same = metrics::cosine(&s1, &s2);
    let cross = metrics::cosine(&s1, &s3);
    assert!(same > 0.99, "same voice {same}");
    assert!(cross < same, "cross voice {cross} should be < {same}");
}
