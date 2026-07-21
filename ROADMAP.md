# Mimic — Roadmap to State of the Art

This roadmap takes Mimic from the v1 prototype (word-level PCM cache,
character-proportional segmentation, crossfade splicing) to a state-of-the-art
**semantic-aware TTS output cache with sub-word acoustic recomposition** —
in the endgame, an *acoustic CDN for AI agents*.

## 1. What "SOTA" means here (the measurable bar)

Mimic is not a TTS model; it is infrastructure *around* TTS models. SOTA is
therefore a compound claim, measured against three baselines on a fixed
benchmark suite (§6):

| Metric | Target | Baseline to beat |
|---|---|---|
| Naturalness of cached/composed speech (UTMOS + human MOS) | within 0.1 MOS of provider-direct generation | ElevenLabs/Cartesia direct (~4.7–4.8) |
| Cache coverage on production-like traffic | ≥95% of characters served from cache | naive word cache (v1): ~60–80% |
| Cost per 1M output characters | ≤5% of always-cloud | always-cloud: 100% |
| P90 latency, fully cached response | <100 ms | cloud TTS round trip: 300–800 ms |
| Voice consistency (SECS speaker-embedding cosine across sessions) | ≥0.95 | provider re-generation: 0.85–0.92 |
| Intelligibility (Whisper WER on composed audio) | ≤ provider-direct WER + 0.5% abs | provider-direct |

A claim of SOTA is only valid with the §6 harness green against all three
baselines. No harness, no claims — this is why P1 comes first.

## 2. Where v1 stands

Shipped (see README): padagonia-backed `AudioUnit` store (phrase/word levels +
morpheme fallback), exact-match side map, HNSW over handcrafted 64-dim
embeddings, `follows` context edges, crossfade splicer, MockTts, 6 tests.

Known gaps, honestly: character-proportional pseudo-alignment; PCM-level
stitching with audible seams on real voices; no prosody model; no real
provider integration; embeddings are handcrafted; single-file persistence;
no evaluation beyond unit tests.

## 3. The landscape Mimic must beat / absorb

- **Concatenative unit-selection TTS** (Hunt & Black 1996 lineage, Festival):
  the classic solution to our exact problem — diphone/triphone inventories +
  target/join-cost Viterbi search. Mimic's P3 is this, AI-era.
- **Neural codecs as speech tokenization**: EnCodec, DAC, and the 2024–26
  generation (WavTokenizer, Mimi, XCodec2, SemantiCodec) turn speech into
  discrete frames at ~0.25–8 kbps with high reconstruction quality. This is
  the storage *and* seam substrate for P4.
- **Open TTS 2025–26**: F5-TTS (flow-matching DiT), CosyVoice, Chatterbox,
  Kokoro-82M, Orpheus, Sesame CSM — zero-shot voice conditioning means the
  "10% cloud" fallback can also run *locally* (P5/P6 hybrid).
- **Voice-agent infra** (ElevenLabs Flash, Cartesia Sonic, Gemini 2.5 TTS):
  sets the latency bar (~100–300 ms first byte) Mimic must undercut via cache.
- **Semantic caches** (GPTCache et al.): text-level; Mimic's differentiator is
  operating on *acoustic* primitives with prosody/context metadata.

Mimic's moat: nobody ships a general, provider-agnostic, sub-word acoustic
recomposition cache with a graph/vector database backing. The pieces exist;
the integrated system does not.

## 4. Phases

Sizing assumes 1–2 engineers; gates must pass before moving on.

### P1 — Measurement & true alignment (2–4 weeks) — *gates everything*

**Status: DONE (2026-07-17), gate PASS.** Shipped: `mimic eval` harness
(v1-baseline frozen as historical row; coverage/RTF/seam sweeps over
`assets/corpora`); CMUdict G2P with digit/letter fallback; phoneme-level
units with cross-word context and `follows` edges; energy-refined alignment
replacing character-proportional segmentation. Gate result: **median
boundary error 1.25 ms, p90 1.25 ms** (gate: ≤20 ms) on 533 clips / 1222
boundaries vs. MockTts ground truth; v1-baseline median 16 ms. Seam
discontinuity roughly halved (0.137 → 0.056 on long_tail). Adaptations to
note: the gate is measured against deterministic MockTts ground truth, not
500 manually labeled human-speech clips (no real-speech corpus on this
machine — MFA/CTC alignment on human speech moves to P2's provider work);
UTMOS/Whisper-WER/SECS are wired as an optional adapter
(`scripts/eval_external.py`), skipped until the Python ML stack is
installed.

Original plan for reference:

- **Eval harness** (`mimic-eval` crate/module): benchmark corpora (repetitive
  support-style, long-tail dynamic, multilingual), metrics: UTMOS/NISQA,
  Whisper WER, SECS speaker cosine, cache hit %, RTF, cost model. CI gate.
- **Forced alignment** at ingest: Montreal Forced Aligner or CTC-segmentation
  (whisperX for quick integration) replaces character-proportional slicing.
  Store phoneme-level boundaries per unit in padagonia.
- **Phoneme units**: real Level-3 inventory with IPA/ARPAbet phoneme strings
  in the `phonemes` prop (no longer a placeholder).
- *Gate*: median boundary error <20 ms vs. manual labels on 500 clips;
  harness reproduces v1 numbers as the historical baseline row.

### P2 — Real providers & service shape (2–3 weeks, parallel with P1)

**Status: DONE (2026-07-17), gate PASS.** Shipped: `TtsProvider` HTTP wire
formats for OpenAI / ElevenLabs / Cartesia / Gemini (+ `SidecarTts`
contract for local open models) with a per-provider cost table;
`mimic serve` — axum service with `POST /v1/audio/speech`
(OpenAI-compatible), `POST /v1/compose`, `POST /v1/ingest`, `GET /v1/stats`,
cache metrics on response headers, and an SSML subset (`<speak>`, `<break>`);
production-shaped workload replay in the harness (Zipf-ish 75/25 stream,
300 requests). Gate result: **coverage 87.4%** (gate ≥80%), **p99 compose
15.3 ms vs simulated-cloud p99 674 ms** (and vs mock-direct p99 22.6 ms),
est. cost $0.35 vs $2.79 always-cloud (elevenlabs profile). Adaptations:
no API keys and no TLS stack on this machine — provider wire formats are
tested against local mock servers, not live endpoints; the latency leg uses
the documented simulated-cloud model (250 ms + 8 ms/char); enabling live
HTTPS (rustls) + keys is the P2.5 follow-up.

Original plan for reference:

- `TtsProvider` implementations: ElevenLabs, Cartesia, Gemini/OpenAI TTS, and
  one local open model (F5-TTS or Chatterbox via Python sidecar; Kokoro-82M
  for a CPU-friendly tier). Feature: provider abstraction incl. streaming
  synthesis and per-provider cost table (feeds the cost metric).
- Mimic as a service: HTTP/WebSocket compose endpoint, OpenAI-compatible
  `/v1/audio/speech` shim, SSML subset passthrough.
- *Gate*: replay a production-shaped workload; ≥80% char coverage on the
  repetitive corpus; p99 compose latency < provider-direct on cached spans.

### P3 — Unit selection done right (4–6 weeks)

**Status: DONE (2026-07-17), gate PASS.** Shipped: diphone inventory
(mid-phoneme → mid-phoneme transition slices, stress-stripped keys,
cross-word pairs, outer phonetic context in props, `follows` edges);
`src/select.rs` — lattice + Viterbi over phrase/word/morpheme/diphone/synth
edges with target + join (acoustic seam + phonetic context) costs and
λ-priced generation; duplicate-unit selection is now cost-based, not
heuristic. Gate result: **long-tail coverage 66.5% → 89.9%** (gate
≥81.5%), multilingual 16.2% → 90.5% (diphone chains cover OOV words),
support 69.2% → 91.9%, replay 87.4% → 93.0%, est. cost $0.195 vs $2.79
always-cloud (7.0% — §1's ≤5% target is in reach at ~95% coverage). Mean
seam roughly flat (0.056 → 0.059 on long-tail): diphone chains trade small
seams for big coverage — P5's neural seam work owns that next. Adaptation:
the roadmap's "ABX ≥70% preference" needs human ears; the mock domain
substitutes the objective gates above (coverage, seam, latency) plus unit
tests proving the DP picks context-matching variants and lower-cost paths.

Original plan for reference:

- Context-dependent inventory: diphones/triphones + word/phrase variants keyed
  by (left,right) phonetic context — the `context_prev/next` props and
  `follows` edges become transition records, not just metadata.
- **Target + join cost search** (the Festival formulation): candidate graph
  per request, costs from phonetic distance, prosody mismatch, spectral
  discontinuity at candidate joins; Viterbi for the cheapest path.
- Duplicate-unit selection upgrades from heuristics to cost-based.
- *Gate*: seam-audibility ABX vs. v1 word-cache: ≥70% preference; coverage on
  long-tail corpus +15 pts over v1 at equal quality.

### P4 — Codec-native cache (6–8 weeks) — *the pivotal architectural move*

- Store **neural codec tokens** (start: DAC or XCodec2; evaluate Mimi for
  streaming) per unit instead of PCM; padagonia nodes hold token-frame
  references + codec config; `Scalar::Bytes` already fits.
- Stitch in **token space** with 100–200 ms context padding, then a *single*
  decoder/vocoder pass over the whole stream — the decoder itself smooths
  joins. PCM splice path stays as reference implementation.
- Voice DNA: speaker embedding (ECAPA-TDNN/WeSpeaker) per unit + per-voice
  aggregate nodes; HNSW moves from handcrafted 64-dim to codec/content
  (HuBERT-derived) embeddings; per-voice index partitions.
- *Gate*: reconstruction parity (STOI ≥0.95, ViSQOL ≥4.0) vs. PCM pipeline;
  ≥10× storage reduction per speech-hour; SECS consistency ≥0.95 per voice.

### P5 — Neural seams & prosody (8–12 weeks)

- **Boundary inpainting**: small infilling model (Voicebox-style flow
  matching, or fine-tuned F5-TTS in infill mode) regenerates 50–150 ms around
  each join conditioned on both sides — coarticulation *synthesized*, not
  spliced. Fallback to P3 join-cost selection when confidence is low.
- Prosody transfer: FastSpeech2-style variance adapters (pitch/duration/
  energy) conditioned on surrounding sentence context; per-voice prosody
  priors as padagonia nodes updated from observed usage.
- Emotion/style tagging of units (valence/arousal from speech emotion
  embeddings) so compose can match the requested style.
- *Gate*: composed speech within 0.1 MOS of provider-direct on cached spans;
  ABX ≥50% preference vs. provider-direct on repetitive text (tie = win —
  at 5% of the cost).

### P6 — The acoustic CDN (ongoing)

- Content-addressed units (hash of text+voice+prosody context) → global
  dedup; tiered storage: RAM → padagonia file → object storage; eviction by
  value (frequency × replacement cost, Zipf head precomputed at deploy).
- Edge serving of unit audio + compose-at-edge; privacy tier: on-device cache
  with opt-in federated sharing (units are voice biometric-adjacent —
  treat accordingly: consent, retention, per-tenant isolation).
- Hybrid local endgame: a small local model conditioned on cached Voice DNA
  covers the uncached tail — the design's "90% cached / 10% cloud", with the
  10% optionally never leaving the machine.
- Multilingual: per-language phoneme inventories + cross-language unit reuse.
- *Gate*: the §1 table, all rows, on production traffic.

## 5. Critical path & parallelization

P1 → everything (no SOTA claims without the harness). P2 ∥ P1. P3 and P4 are
independent and can run in parallel; P5 needs both. P6 builds on P4's
storage layer. Fastest credible "SOTA-adjacent" demo: P1+P2+P4 ≈ 3 months.

## 6. Evaluation harness (the SOTA proof)

- **Corpora**: (a) repetitive support/agent corpus (Zipf head), (b) long-tail
  dynamic content (names, numbers, dates), (c) multilingual set, (d) an
  emotional/prosody-marked set for P5.
- **Baselines**: always-cloud (ElevenLabs/Cartesia direct), naive word cache
  (v1 frozen as a binary), full-local TTS (Kokoro/Chatterbox), and
  provider-direct as the quality ceiling.
- **Metrics pipeline**: UTMOS + human MOS panels (n≥20 raters for gate
  decisions), Whisper-large WER, SECS speaker cosine, STOI/ViSQOL/PESQ for
  codec gates, ABX for seam tests, latency/cost instrumentation. All gate
  numbers in §1/§4 are produced by this harness or they don't count.

## 7. Top risks

1. **Coarticulation long tail** — joins in rare phonetic contexts stay
   audible; mitigated by P3 join costs + P5 inpainting fallback to full
   regeneration when join cost is high (quality floor = provider).
2. **Voice drift / consistency** — provider model updates change a voice;
   units are keyed by (provider, model version, voice id) and old units are
   deprecated, not mixed (padagonia's provenance fields already carry this).
3. **Licensing/biometrics** — cached voices are voiceprints; P6's consent,
   retention, and tenant isolation are launch blockers, not nice-to-haves.
4. **Codec seam artifacts compounding** — token-space stitching can smear
   transients; gate on ViSQOL and keep PCM path as fallback.
5. **Eval gaming ourselves** — harness corpora must include held-out,
   regularly rotated test sets; human MOS panels for every quality gate.

## 8. Explicitly not building

Own foundation TTS model, own neural codec, own ASR (adopt: F5-TTS/
Chatterbox/Kokoro, DAC/XCodec2/Mimi, Whisper/MFA). Mimic's defensibility is
the cache architecture and the unit-selection/stitching stack, not models.

## References

- [CodeSOTA — Best open-source TTS models 2026](https://www.codesota.com/speech/best-open-source) (Sesame CSM, Fish S2 Pro, Orpheus, Kokoro MOS/rankings)
- [CodeSOTA — TTS vendors 2026](https://www.codesota.com/speech/tts-vendors) (ElevenLabs Turbo/Flash, Cartesia Sonic 2, Gemini 2.5 TTS latency/MOS bar)
- [gigagpu — Best TTS models 2026](https://gigagpu.com/best-tts-models-2026/) (F5-TTS, XTTS v2, StyleTTS 2 self-host tiers)
- [Local AI Master — Best local TTS 2026](https://localaimaster.com/blog/best-local-tts-models) (Kokoro-82M, Chatterbox licensing/quality)
- [F5-TTS paper (arXiv:2410.06885)](https://arxiv.org/html/2410.06885v3) (flow-matching DiT TTS; infill capability relevant to P5)
- [EntangleCodec (arXiv:2606.02739)](https://arxiv.org/html/2606.02739v1) and [PairAlign (arXiv:2605.06582)](https://arxiv.org/html/2605.06582v2) — 2026 codec/tokenizer landscape surveys (EnCodec, DAC, WavTokenizer, Mimi, XCodec2, SemantiCodec baselines)
