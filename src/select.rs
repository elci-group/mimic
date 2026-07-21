//! P3: unit selection done right — a target + join cost search over a
//! lattice of candidate units (the classic Festival formulation), replacing
//! v1's greedy left-to-right matching.
//!
//! Edge types per word span: cached phrase, cached word (every variant is a
//! separate edge — context fit is scored, not heuristic-picked), morpheme
//! segment, diphone chain (sub-word acoustic recomposition proper), and a
//! synth edge (provider call). Viterbi minimizes
//!   Σ(target cost) + Σ(join cost) + λ × generated chars
//! so the path trades cache cheapness against seam quality globally instead
//! of greedily. Consecutive synth edges coalesce into one provider call at
//! realization time, exactly like the greedy path did.

use crate::audio::{seam_discontinuity, splice, WavAudio};
use crate::codec::AudioCodec;
use crate::g2p::G2p;
use crate::pipeline::{segment_and_store_words, ComposeReport, IngestOptions};
use crate::store::MimicStore;
use crate::tts::TtsProvider;
use crate::units::{self, UnitLevel};
use crate::{MimicError, Result, CROSSFADE_MS};
use padagonia::NodeId;

// ---- cost model (documented constants; λ encodes "cache ≪ generation") ----
const PHRASE_TARGET: f64 = 0.0;
const WORD_TARGET: f64 = 0.1;
const MORPHEME_TARGET: f64 = 0.45;
const DIPHONE_TARGET: f64 = 0.85;
const GAP_PENALTY: f64 = 0.15; // per diphone substituted by phoneme halves
const SYNTH_BASE: f64 = 0.1;
const SYNTH_PER_CHAR: f64 = 0.2;
const JOIN_ACOUSTIC_W: f64 = 0.5;
const JOIN_PHONETIC_WORD: f64 = 0.08;
const JOIN_PHONETIC_DIPHONE: f64 = 0.05;
const JOIN_AFTER_SYNTH: f64 = 0.05;

/// How compose realizes its output: PCM splice (reference path) or
/// token-stream concatenation with a single decode pass (P4 codec-native).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Medium {
    Pcm,
    Tokens,
}

#[derive(Debug)]
struct Edge {
    start: usize,
    end: usize,
    level: UnitLevel,
    target: f64,
    /// None = synth edge (provider audio generated at realization time).
    audio: Option<WavAudio>,
    /// Codec token stream for this edge's span (Token medium only).
    tokens: Option<Vec<u8>>,
    nodes: Vec<NodeId>,
    chars: usize,
    text: String,
}

/// Assemble diphone-chain audio for one word:
/// [first half of p0] + diphone(p0,p1) + ... + [second half of p_last],
/// substituting phoneme halves for any missing diphone. None if a required
/// phoneme unit itself is missing.
fn diphone_chain(store: &MimicStore, phonemes: &[String]) -> Option<(WavAudio, usize, Vec<NodeId>)> {
    if phonemes.is_empty() {
        return None;
    }
    let mut pieces: Vec<WavAudio> = Vec::new();
    let mut nodes: Vec<NodeId> = Vec::new();
    let mut gaps = 0usize;

    let phon_unit = |base: &str| -> Option<(NodeId, WavAudio)> {
        let id = *store.phonemes_by_base(base).first()?;
        store.get_audio(id).ok().map(|a| (id, a))
    };

    let (id0, a0) = phon_unit(G2p::base(&phonemes[0]))?;
    pieces.push(a0.slice_frac(0.0, 0.5));
    nodes.push(id0);

    for k in 0..phonemes.len() - 1 {
        let lb = G2p::base(&phonemes[k]);
        let rb = G2p::base(&phonemes[k + 1]);
        let key = format!("{lb}+{rb}");
        if let Some(&did) = store.lookup_exact(UnitLevel::Diphone, &key).first() {
            if let Ok(a) = store.get_audio(did) {
                pieces.push(a);
                nodes.push(did);
                continue;
            }
        }
        // gap: phoneme halves of both sides
        gaps += 1;
        let (_, la) = phon_unit(lb)?;
        let (_, ra) = phon_unit(rb)?;
        pieces.push(la.slice_frac(0.5, 1.0));
        pieces.push(ra.slice_frac(0.0, 0.5));
    }

    let (idl, al) = phon_unit(G2p::base(&phonemes[phonemes.len() - 1]))?;
    pieces.push(al.slice_frac(0.5, 1.0));
    nodes.push(idl);

    // 2 ms joins inside the chain; the outer 10 ms crossfade handles edges
    let audio = splice(&pieces, 2).ok()?;
    Some((audio, gaps, nodes))
}

/// Assemble diphone-chain token stream for one word (Token medium):
/// frame-aligned token slices concatenated without any decode.
fn diphone_chain_tokens(
    store: &MimicStore,
    phonemes: &[String],
) -> Option<(Vec<u8>, usize, Vec<NodeId>)> {
    use crate::codec::{concat_tokens, slice_tokens};
    if phonemes.is_empty() {
        return None;
    }
    let mut pieces: Vec<Vec<u8>> = Vec::new();
    let mut nodes: Vec<NodeId> = Vec::new();
    let mut gaps = 0usize;

    let phon_tokens = |base: &str| -> Option<(NodeId, Vec<u8>)> {
        let id = *store.phonemes_by_base(base).first()?;
        store.get_tokens(id).ok().map(|t| (id, t))
    };

    let (id0, t0) = phon_tokens(G2p::base(&phonemes[0]))?;
    pieces.push(slice_tokens(&t0, 0.0, 0.5).ok()?);
    nodes.push(id0);

    for k in 0..phonemes.len() - 1 {
        let lb = G2p::base(&phonemes[k]);
        let rb = G2p::base(&phonemes[k + 1]);
        let key = format!("{lb}+{rb}");
        if let Some(&did) = store.lookup_exact(UnitLevel::Diphone, &key).first() {
            if let Ok(t) = store.get_tokens(did) {
                pieces.push(t);
                nodes.push(did);
                continue;
            }
        }
        gaps += 1;
        let (_, lt) = phon_tokens(lb)?;
        let (_, rt) = phon_tokens(rb)?;
        pieces.push(slice_tokens(&lt, 0.5, 1.0).ok()?);
        pieces.push(slice_tokens(&rt, 0.0, 0.5).ok()?);
    }

    let (idl, tl) = phon_tokens(G2p::base(&phonemes[phonemes.len() - 1]))?;
    pieces.push(slice_tokens(&tl, 0.5, 1.0).ok()?);
    nodes.push(idl);

    let refs: Vec<&[u8]> = pieces.iter().map(|p| p.as_slice()).collect();
    let stream = concat_tokens(&refs).ok()?;
    Some((stream, gaps, nodes))
}

fn join_cost(p: &Edge, e: &Edge, store: &MimicStore, words: &[String], g2p: Option<&G2p>) -> f64 {
    if e.audio.is_none() {
        return 0.0; // synth joins are generated coherently
    }
    let mut c = if p.audio.is_none() { JOIN_AFTER_SYNTH } else { 0.0 };
    if let (Some(a), Some(b)) = (&p.audio, &e.audio) {
        c += JOIN_ACOUSTIC_W * seam_discontinuity(a, b);
    }
    if e.start == 0 {
        return c;
    }
    match e.level {
        UnitLevel::Diphone => {
            // target: last base phoneme of the previous word; actual: the
            // first diphone's recorded left context
            if let (Some(g), Some(&nid)) = (g2p, e.nodes.first()) {
                let target = g
                    .phonemes(&words[e.start - 1])
                    .and_then(|p| p.last().map(|s| G2p::base(s).to_string()));
                let actual = store
                    .prop_string(nid, "context_prev")
                    .map(|s| G2p::base(&s).to_string());
                if let (Some(t), Some(a)) = (target, actual) {
                    if !a.is_empty() && t != a {
                        c += JOIN_PHONETIC_DIPHONE;
                    }
                }
            }
        }
        UnitLevel::Word | UnitLevel::Phrase | UnitLevel::Morpheme => {
            if let Some(&nid) = e.nodes.first() {
                let ctx = store.prop_string(nid, "context_prev").unwrap_or_default();
                if ctx.is_empty() {
                    c += JOIN_PHONETIC_WORD * 0.75; // unit was utterance-initial
                } else if ctx != words[e.start - 1] {
                    c += JOIN_PHONETIC_WORD;
                }
            }
        }
        _ => {}
    }
    c
}

fn build_edges(
    store: &MimicStore,
    words: &[String],
    g2p: Option<&G2p>,
    medium: Medium,
) -> Result<Vec<Edge>> {
    let n = words.len();
    let mut edges = Vec::new();
    let token_mode = medium == Medium::Tokens;
    // tokens for a unit: inline, or encoded on the fly for legacy PCM-only units
    let tokens_of = |id: NodeId| -> Result<Vec<u8>> {
        store
            .get_tokens(id)
            .or_else(|_| Ok(crate::codec::MimicMct.encode(&store.get_audio(id)?)))
    };

    // phrase edges: every verbatim occurrence of every cached phrase
    for (ptext, ids) in store.units_for_level(UnitLevel::Phrase) {
        let pw = units::tokenize(&ptext);
        if pw.is_empty() || pw.len() > n {
            continue;
        }
        for start in 0..=(n - pw.len()) {
            if words[start..start + pw.len()] == pw[..] {
                let audio = store.get_audio(ids[0])?;
                let tokens = if token_mode { Some(tokens_of(ids[0])?) } else { None };
                edges.push(Edge {
                    start,
                    end: start + pw.len(),
                    level: UnitLevel::Phrase,
                    target: PHRASE_TARGET,
                    audio: Some(audio),
                    tokens,
                    nodes: ids.clone(),
                    chars: words[start..start + pw.len()]
                        .iter()
                        .map(|w| w.chars().count())
                        .sum(),
                    text: ptext.clone(),
                });
            }
        }
    }

    for (i, w) in words.iter().enumerate() {
        let chars = w.chars().count();
        // word candidates: every cached variant is an edge
        for &id in store.lookup_exact(UnitLevel::Word, w) {
            edges.push(Edge {
                start: i,
                end: i + 1,
                level: UnitLevel::Word,
                target: WORD_TARGET,
                audio: Some(store.get_audio(id)?),
                tokens: if token_mode { Some(tokens_of(id)?) } else { None },
                nodes: vec![id],
                chars,
                text: w.clone(),
            });
        }
        // morpheme candidates
        for (cand, f0, f1) in units::morpheme_candidates(w) {
            if let Some(&cid) = store.lookup_exact(UnitLevel::Word, &cand).first() {
                let full = store.get_audio(cid)?;
                let tokens = if token_mode {
                    Some(crate::codec::slice_tokens(&tokens_of(cid)?, f0, f1)?)
                } else {
                    None
                };
                edges.push(Edge {
                    start: i,
                    end: i + 1,
                    level: UnitLevel::Morpheme,
                    target: MORPHEME_TARGET,
                    audio: Some(full.slice_frac(f0, f1)),
                    tokens,
                    nodes: vec![cid],
                    chars,
                    text: w.clone(),
                });
                break;
            }
        }
        // diphone chain
        if let Some(g) = g2p {
            if let Some(phonemes) = g.phonemes(w) {
                // (audio, tokens, gaps, nodes); decode failure drops the edge
                let chain: Option<(WavAudio, Option<Vec<u8>>, usize, Vec<NodeId>)> = match medium {
                    Medium::Pcm => diphone_chain(store, &phonemes)
                        .map(|(a, gaps, nodes)| (a, None, gaps, nodes)),
                    Medium::Tokens => diphone_chain_tokens(store, &phonemes).and_then(
                        |(s, gaps, nodes)| {
                            crate::codec::MimicMct
                                .decode(&s)
                                .ok()
                                .map(|a| (a, Some(s), gaps, nodes))
                        },
                    ),
                };
                if let Some((audio, tokens, gaps, nodes)) = chain {
                    edges.push(Edge {
                        start: i,
                        end: i + 1,
                        level: UnitLevel::Diphone,
                        target: DIPHONE_TARGET + GAP_PENALTY * gaps as f64,
                        audio: Some(audio),
                        tokens,
                        nodes,
                        chars,
                        text: w.clone(),
                    });
                }
            }
        }
        // synth fallback
        edges.push(Edge {
            start: i,
            end: i + 1,
            level: UnitLevel::Word,
            target: SYNTH_BASE + SYNTH_PER_CHAR * chars as f64,
            audio: None,
            tokens: None,
            nodes: Vec::new(),
            chars,
            text: w.clone(),
        });
    }
    Ok(edges)
}

/// Viterbi unit-selection composer (P3). Same contract as
/// [`crate::pipeline::compose`]. PCM realization (reference path).
pub fn compose_v3(
    store: &mut MimicStore,
    tts: &dyn TtsProvider,
    text: &str,
    voice: &str,
    g2p: Option<&G2p>,
) -> Result<(WavAudio, ComposeReport)> {
    compose_v3_with_medium(store, tts, text, voice, g2p, Medium::Pcm)
}

/// Viterbi composer with a realization medium. `Tokens` (P4) concatenates
/// unit token streams and decodes once — OLA smooths the joins — while
/// `Pcm` crossfade-splices waveforms (reference implementation).
pub fn compose_v3_with_medium(
    store: &mut MimicStore,
    tts: &dyn TtsProvider,
    text: &str,
    voice: &str,
    g2p: Option<&G2p>,
    medium: Medium,
) -> Result<(WavAudio, ComposeReport)> {
    let words = units::tokenize(text);
    if words.is_empty() {
        return Err(MimicError::NotFound("empty text".into()));
    }
    let n = words.len();
    let edges = build_edges(store, &words, g2p, medium)?;

    // index edges by start position
    let mut by_start: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    let mut by_end: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (idx, e) in edges.iter().enumerate() {
        by_start[e.start].push(idx);
        by_end[e.end].push(idx);
    }

    // DP: best cost to reach and *use* each edge
    let mut best = vec![f64::INFINITY; edges.len()];
    let mut back: Vec<Option<usize>> = vec![None; edges.len()];
    for pos in 0..=n {
        for &ei in &by_start[pos] {
            let e = &edges[ei];
            if pos == 0 {
                best[ei] = e.target;
                continue;
            }
            for &pi in &by_end[pos] {
                let c = best[pi] + join_cost(&edges[pi], e, store, &words, g2p);
                if c + e.target < best[ei] {
                    best[ei] = c + e.target;
                    back[ei] = Some(pi);
                }
            }
        }
    }
    let mut last = by_end[n]
        .iter()
        .copied()
        .min_by(|&a, &b| best[a].partial_cmp(&best[b]).unwrap())
        .ok_or_else(|| MimicError::NotFound("no path through lattice".into()))?;
    let mut path: Vec<usize> = Vec::new();
    loop {
        path.push(last);
        match back[last] {
            Some(p) => last = p,
            None => break,
        }
    }
    path.reverse();

    // realize: consecutive synth edges coalesce into one provider call.
    // Token medium accumulates token streams; PCM accumulates waveforms.
    let mut parts: Vec<WavAudio> = Vec::new();
    let mut part_tokens: Vec<Vec<u8>> = Vec::new();
    let mut report = ComposeReport {
        total_chars: words.iter().map(|w| w.chars().count()).sum(),
        ..Default::default()
    };
    let mut i = 0;
    while i < path.len() {
        let e = &edges[path[i]];
        if let Some(a) = &e.audio {
            report.cached_chars += e.chars;
            report.hits.push((e.text.clone(), e.level));
            report.units.extend(e.nodes.iter().copied());
            match medium {
                Medium::Pcm => parts.push(a.clone()),
                Medium::Tokens => part_tokens.push(match &e.tokens {
                    Some(t) => t.clone(),
                    None => crate::codec::MimicMct.encode(a),
                }),
            }
            i += 1;
            continue;
        }
        let mut j = i;
        while j < path.len() && edges[path[j]].audio.is_none() {
            j += 1;
        }
        let (w0, w1) = (edges[path[i]].start, edges[path[j - 1]].end);
        let span_text = words[w0..w1].join(" ");
        let audio = tts.synthesize(&span_text, voice)?;
        report.tts_calls.push(span_text.clone());
        report.generated_chars += words[w0..w1].iter().map(|w| w.chars().count()).sum::<usize>();
        let outer_prev = if w0 > 0 { Some(words[w0 - 1].as_str()) } else { None };
        let outer_next = words.get(w1).map(String::as_str);
        let span_phonemes: Vec<Option<Vec<String>>> = match g2p {
            Some(g) => words[w0..w1].iter().map(|w| g.phonemes(w)).collect(),
            None => vec![None; w1 - w0],
        };
        segment_and_store_words(
            store,
            &words[w0..w1],
            &span_phonemes,
            &audio,
            voice,
            tts.name(),
            outer_prev,
            outer_next,
            &IngestOptions::default(),
        )?;
        match medium {
            Medium::Pcm => parts.push(audio),
            Medium::Tokens => part_tokens.push(crate::codec::MimicMct.encode(&audio)),
        }
        i = j;
    }

    let (out, seams): (WavAudio, Vec<f64>) = match medium {
        Medium::Pcm => {
            let seams: Vec<f64> = parts
                .windows(2)
                .map(|w| seam_discontinuity(&w[0], &w[1]))
                .collect();
            (splice(&parts, CROSSFADE_MS)?, seams)
        }
        Medium::Tokens => {
            // one decode pass over the concatenated stream; OLA smooths joins
            let frame_counts: Vec<usize> = part_tokens
                .iter()
                .map(|t| crate::codec::token_frames(t).unwrap_or(0) as usize)
                .collect();
            let refs: Vec<&[u8]> = part_tokens.iter().map(|t| t.as_slice()).collect();
            let stream = crate::codec::concat_tokens(&refs)?;
            let out = crate::codec::MimicMct.decode(&stream)?;
            // seam measured on the decoded output at each join offset
            let mut seams = Vec::new();
            let mut off = 0usize;
            for fc in frame_counts.iter().take(frame_counts.len().saturating_sub(1)) {
                off += fc * crate::codec::FRAME;
                let w = out.sample_rate as usize / 100; // 10 ms
                if off >= w && off + w <= out.samples.len() {
                    let a = WavAudio::new(out.samples[off - w..off].to_vec(), out.sample_rate);
                    let b = WavAudio::new(out.samples[off..off + w].to_vec(), out.sample_rate);
                    seams.push(seam_discontinuity(&a, &b));
                }
            }
            (out, seams)
        }
    };
    report.mean_seam_discontinuity = if seams.is_empty() {
        0.0
    } else {
        seams.iter().sum::<f64>() / seams.len() as f64
    };
    Ok((out, report))
}
