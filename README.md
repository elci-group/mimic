# Mimic

A phonetic memoization layer for TTS: a semantic-aware speech cache with
sub-word acoustic recomposition, backed by
[padagonia](https://github.com/elci-group/padagonia) (graph + HNSW vector
store).

Generated speech is decomposed into reusable audio primitives — phrase,
word, morpheme — stored as `AudioUnit` nodes (text, prosody metadata, wav
reference, embedding) with `follows` edges preserving sequential context.
Future requests are resolved at the highest cached level, only the missing
spans go to the TTS provider, and the pieces are spliced with crossfades.
Every newly synthesized span is itself decomposed into the cache, so the
library grows with use.

## Usage

```bash
cargo build

# memoize a TTS run (MockTts stands in for a real provider when --wav is absent)
cargo run -- ingest --db demo.pad --text "the quick brown fox jumps"

# compose new speech; only the missing spans hit the provider
cargo run -- compose --db demo.pad --text "the quick red fox jumps" --out out.wav

# inspect
cargo run -- lookup --db demo.pad --text "quick"
cargo run -- stats  --db demo.pad

# serve: mimic-native + OpenAI-compatible endpoints
cargo run -- serve --db demo.pad --addr 127.0.0.1:8787
curl -X POST localhost:8787/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"the quick brown fox jumps"}' -o speech.wav
# endpoints: GET /health, GET /v1/stats, POST /v1/compose,
#            POST /v1/audio/speech (OpenAI shim), POST /v1/ingest
# cache metrics ride on response headers (x-mimic-cache-hit-pct, ...)

# evaluate
cargo run --release -- eval --gate
```

Unit wavs live in `<db>.audio/` (e.g. `demo.audio/`). The database is a
single padagonia file; the HNSW index and exact-lookup map are rebuilt in
memory on open.

## Providers (P2)

`TtsProvider` is the seam. Implemented: `MockTts` (offline default), and
HTTP wire formats for OpenAI (`/v1/audio/speech`), ElevenLabs
(`pcm_16000`), Cartesia (`/tts/bytes`), and Gemini (`generateContent`,
base64 L16 with 3:2 decimation to 16 kHz), plus a `SidecarTts` contract for
local open models (`python3 script <text> <out.wav>` — Kokoro/F5/Chatterbox
scripts planned). The bundled HTTP client is `http://`-only: **live cloud
calls need a TLS stack (rustls) or an HTTP gateway** — that and API keys
are the P2.5 follow-up. Wire formats are tested against local mock servers.
Per-provider cost table in `src/providers.rs` (approximate list prices,
verify before use). A basic SSML subset (`<speak>`, `<break>`) is parsed
and honored at compose time.

## How compose decides

P3 replaced greedy matching with a **target + join cost search** (the
classic Festival unit-selection formulation) over a lattice of candidate
units; Viterbi picks the global cheapest path:

1. **Phrase** edges — every verbatim occurrence of a cached phrase.
2. **Word** edges — every cached variant; context fit is scored by the join
   cost (phonetic context mismatch + acoustic seam), not heuristic-picked.
3. **Morpheme** edges — a miss like `run` can reuse a segment of `running`.
4. **Diphone** edges — a miss whose phonemes were seen gets assembled from
   mid-phoneme-to-mid-phoneme transition slices (cross-word diphones
   included), with phoneme-half gap filling. This is sub-word acoustic
   recomposition proper, and it more than doubled long-tail coverage.
5. **Synth** edges — consecutive misses coalesce into one provider call;
   the result is ingested back as word + phoneme + diphone units.

Costs: `target` (phrase 0 < word 0.1 < morpheme 0.45 < diphone 0.85+gap
penalties) + `join` (0.5 × seam discontinuity + phonetic context mismatch)
+ λ 0.2/char for generation. The constants live at the top of
`src/select.rs`.

## v1 limitations / next steps

See [ROADMAP.md](ROADMAP.md) for the full phased plan to state of the art,
with measurable gates per phase. The short version:

- **P1 shipped**: CMUdict G2P (ARPAbet), phoneme-level units with
  cross-word context, energy-refined alignment (median boundary error
  **1.25 ms** vs. MockTts ground truth, gate ≤ 20 ms, on 533 clips), and
  the `mimic eval` harness with the v1 pipeline frozen as the baseline row.
  Next alignment step: real forced alignment (MFA/CTC) on human speech.
- Embeddings are 64-dim handcrafted features (duration/RMS/ZCR/Goertzel
  bands + hashed char trigrams), not MFCCs or learned acoustic embeddings.
- Only `MockTts` ships; `TtsProvider` is the seam for ElevenLabs/Gemini
  clients. 16 kHz mono i16 PCM only.
- Neural eval metrics (UTMOS/NISQA, Whisper WER, SECS) are an optional
  adapter (`scripts/eval_external.py`), skipped until the Python ML stack
  is installed.
- No HTTP server, prosody transfer, or transition-model stitching yet
  (the `follows` edges are the substrate for the latter).

## Evaluation

```bash
cargo run --release -- eval --gate
```

runs the harness over `assets/corpora`: coverage/RTF/seam sweeps for the
v1-baseline and p1-current modes, plus the alignment gate (500+ generated
clips vs. MockTts ground-truth word spans). Reports land in
`eval/reports/`; thresholds in `eval/gates.txt`.

## License

MIT OR Apache-2.0
