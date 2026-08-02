use mimic::g2p::G2p;
use mimic::plan::{PlanManager, PlanRequest};
use mimic::store::MimicStore;
use mimic::tts::{MockTts, TtsProvider};

#[test]
fn injected_span_is_persisted_and_reused_by_voice() {
    let tmp = tempfile::tempdir().unwrap();
    let mut store =
        MimicStore::open(tmp.path().join("state.pad"), tmp.path().join("audio")).unwrap();
    let mut plans = PlanManager::new(tmp.path().join("plans"), tmp.path().join("objects")).unwrap();
    let request = PlanRequest {
        text: "hello world".into(),
        voice_id: "voice-a".into(),
        model_id: "model".into(),
        settings_key: "settings".into(),
    };
    let plan = plans.create(&store, request.clone()).unwrap();
    assert_eq!(plan.missing_chars, 10);
    let tts = MockTts::new();
    let audio = tts.synthesize("hello world", "voice-a").unwrap();
    let pcm: Vec<u8> = audio.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    plans
        .inject_pcm(&plan.plan_id, &plan.spans[0].span_id, &pcm)
        .unwrap();
    let g2p = G2p::parse("hello HH AH0 L OW1\nworld W ER1 L D\n");
    let (out, _) = plans
        .compose(&mut store, &g2p, &plan.plan_id, true)
        .unwrap();
    assert_eq!(out, audio, "all-miss compose must be lossless");
    assert!(mimic::metrics::stoi(&audio, &out) >= 0.999);

    let reused = plans.create(&store, request).unwrap();
    assert!(reused.cached_chars > 0);
    assert!(reused.spans.iter().any(|span| span.link.is_some()));
    let (cached, _) = plans
        .compose(&mut store, &g2p, &reused.plan_id, false)
        .unwrap();
    assert!(
        mimic::metrics::stoi(&audio, &cached) >= 0.95,
        "lossless cache recomposition must satisfy the production STOI gate"
    );

    let other_voice = plans
        .create(
            &store,
            PlanRequest {
                text: "hello world".into(),
                voice_id: "voice-b".into(),
                model_id: "model".into(),
                settings_key: "settings".into(),
            },
        )
        .unwrap();
    assert_eq!(other_voice.cached_chars, 0);
}
