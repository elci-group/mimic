//! Ingest and compose pipelines.
//!
//! Ingest: decompose a (text, audio) pair into a phrase unit plus word units
//! and (P1) phoneme units, using energy-refined alignment for boundaries.
//! Compose: resolve a request against the cache at the highest available
//! level (phrase longest-match > word exact > morpheme segment), synthesize
//! only the missing spans, splice everything, and grow the cache with what
//! was freshly generated.

use crate::align::{self, AlignMode};
use crate::audio::{splice, WavAudio};
use crate::codec::AudioCodec;
use crate::g2p::G2p;
use crate::store::MimicStore;
use crate::tts::TtsProvider;
use crate::units::{self, UnitLevel};
use crate::{MimicError, Result, CROSSFADE_MS};
use padagonia::NodeId;

#[derive(Debug)]
pub struct IngestReport {
    pub phrase_units: usize,
    pub word_units: usize,
    pub phoneme_units: usize,
    /// Words for which no phonemes could be resolved (OOV even after
    /// fallback) — visible so coverage gaps are measurable.
    pub unresolved_words: Vec<String>,
}

/// Ingest behavior knobs. `IngestOptions::v1()` reproduces the original
/// pipeline exactly (proportional segmentation, no phoneme units) and is
/// what the eval harness runs as the historical baseline row.
#[derive(Debug, Clone)]
pub struct IngestOptions {
    pub align: AlignMode,
    /// Insert phoneme-level units (requires a G2p).
    pub phonemes: bool,
    /// Insert diphone transition units (requires `phonemes`).
    pub diphones: bool,
    /// Store codec token streams inline in padagonia (P4 codec-native).
    pub codec_tokens: bool,
    /// Also write per-unit wav files (legacy/reference path; P4 skips them).
    pub write_wav: bool,
    /// Search window for intra-word phoneme refinement.
    pub phoneme_search_ms: f64,
}

impl Default for IngestOptions {
    /// P4 defaults: aligned, phonemes, diphones, codec tokens, no wav files.
    fn default() -> Self {
        Self {
            align: AlignMode::default(),
            phonemes: true,
            diphones: true,
            codec_tokens: true,
            write_wav: false,
            phoneme_search_ms: 25.0,
        }
    }
}

impl IngestOptions {
    pub fn v1() -> Self {
        Self {
            align: AlignMode::Proportional,
            phonemes: false,
            diphones: false,
            codec_tokens: false,
            write_wav: true,
            phoneme_search_ms: 0.0,
        }
    }

    /// P1/P2 behavior: alignment + phonemes, no diphone inventory, wav files.
    pub fn p1_legacy() -> Self {
        Self {
            diphones: false,
            codec_tokens: false,
            write_wav: true,
            ..Default::default()
        }
    }

    /// P3 behavior: diphone inventory, but PCM files instead of tokens.
    pub fn p3_legacy() -> Self {
        Self {
            codec_tokens: false,
            write_wav: true,
            ..Default::default()
        }
    }
}

pub fn ingest(
    store: &mut MimicStore,
    text: &str,
    audio: &WavAudio,
    voice: &str,
    provider: &str,
    g2p: Option<&G2p>,
) -> Result<IngestReport> {
    ingest_with_options(
        store,
        text,
        audio,
        voice,
        provider,
        &IngestOptions::default(),
        g2p,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ingest_with_options(
    store: &mut MimicStore,
    text: &str,
    audio: &WavAudio,
    voice: &str,
    provider: &str,
    opts: &IngestOptions,
    g2p: Option<&G2p>,
) -> Result<IngestReport> {
    let normalized = units::normalize(text);
    let words = units::tokenize(text);
    if words.is_empty() {
        return Err(MimicError::NotFound("empty text".into()));
    }
    let word_phonemes: Vec<Option<Vec<String>>> = match (opts.phonemes, g2p) {
        (true, Some(g)) => words.iter().map(|w| g.phonemes(w)).collect(),
        _ => vec![None; words.len()],
    };
    store.insert_unit(
        UnitLevel::Phrase,
        &normalized,
        &normalized,
        voice,
        audio,
        if opts.codec_tokens {
            Some(crate::codec::MimicMct.encode(audio))
        } else {
            None
        },
        opts.write_wav,
        None,
        None,
        provider,
    )?;
    let (ids, phoneme_units) = segment_and_store_words(
        store,
        &words,
        &word_phonemes,
        audio,
        voice,
        provider,
        None,
        None,
        opts,
    )?;
    // per-voice aggregate ("Voice DNA"): the utterance-level signature
    store.add_voice_dna(voice, audio, provider);
    Ok(IngestReport {
        phrase_units: 1,
        word_units: ids.len(),
        phoneme_units,
        unresolved_words: if opts.phonemes && g2p.is_some() {
            words
                .iter()
                .zip(&word_phonemes)
                .filter(|(_, p)| p.is_none())
                .map(|(w, _)| w.clone())
                .collect()
        } else {
            Vec::new()
        },
    })
}

/// Word spans come from the aligner (energy-refined by default). Each word
/// unit also emits phoneme units when a G2P pronunciation is available, and
/// (P3) diphone units: mid-phoneme-to-mid-phoneme transition slices for
/// every adjacent phoneme pair, including cross-word pairs (the real
/// coarticulation record). Phoneme context crosses word boundaries so unit
/// selection can score candidates by phonetic fit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn segment_and_store_words(
    store: &mut MimicStore,
    words: &[String],
    word_phonemes: &[Option<Vec<String>>],
    audio: &WavAudio,
    voice: &str,
    provider: &str,
    outer_prev: Option<&str>,
    outer_next: Option<&str>,
    opts: &IngestOptions,
) -> Result<(Vec<NodeId>, usize)> {
    let spans = align::word_spans(audio, words, opts.align);
    let mut ids: Vec<NodeId> = Vec::with_capacity(words.len());
    // Flat phoneme stream with absolute spans in the utterance audio.
    let mut stream: Vec<(String, align::Span)> = Vec::new();

    for (i, w) in words.iter().enumerate() {
        let span = spans[i];
        let seg = WavAudio::new(
            audio.samples[span.start..span.end].to_vec(),
            audio.sample_rate,
        );
        let prev = if i > 0 {
            Some(words[i - 1].as_str())
        } else {
            outer_prev
        };
        let next = if i + 1 < words.len() {
            Some(words[i + 1].as_str())
        } else {
            outer_next
        };
        let phon_str = word_phonemes[i]
            .as_ref()
            .map(|p| p.join(" "))
            .unwrap_or_else(|| w.clone());
        let id = store.insert_unit(
            UnitLevel::Word,
            w,
            &phon_str,
            voice,
            &seg,
            if opts.codec_tokens {
                Some(crate::codec::MimicMct.encode(&seg))
            } else {
                None
            },
            opts.write_wav,
            prev,
            next,
            provider,
        )?;
        if let Some(&p) = ids.last() {
            store.add_follows(p, id);
        }
        ids.push(id);

        if let Some(phonemes) = &word_phonemes[i] {
            let pspans = align::phoneme_spans(&seg, phonemes.len(), opts.phoneme_search_ms);
            for (j, phon) in phonemes.iter().enumerate() {
                let ps = pspans[j];
                stream.push((
                    phon.clone(),
                    align::Span {
                        start: span.start + ps.start.min(seg.samples.len()),
                        end: span.start + ps.end.min(seg.samples.len()),
                    },
                ));
            }
        }
    }

    // Phoneme + diphone units over the flat stream.
    let mut phoneme_count = 0usize;
    let mut prev_phoneme_id: Option<NodeId> = None;
    let mut prev_diphone_id: Option<NodeId> = None;
    for (k, (phon, abs)) in stream.iter().enumerate() {
        let pseg = WavAudio::new(
            audio.samples[abs.start..abs.end].to_vec(),
            audio.sample_rate,
        );
        let gprev = k
            .checked_sub(1)
            .and_then(|x| stream.get(x))
            .map(|(p, _)| p.as_str());
        let gnext = stream.get(k + 1).map(|(p, _)| p.as_str());
        let pid = store.insert_unit(
            UnitLevel::Phoneme,
            phon,
            phon,
            voice,
            &pseg,
            if opts.codec_tokens {
                Some(crate::codec::MimicMct.encode(&pseg))
            } else {
                None
            },
            opts.write_wav,
            gprev,
            gnext,
            provider,
        )?;
        if let Some(pp) = prev_phoneme_id {
            store.add_follows(pp, pid);
        }
        prev_phoneme_id = Some(pid);
        phoneme_count += 1;

        if opts.diphones && k > 0 {
            let (lphon, labs) = &stream[k - 1];
            // mid of left phoneme -> mid of right phoneme
            let start = (labs.start + labs.end) / 2;
            let end = (abs.start + abs.end) / 2;
            let daudio = WavAudio::new(
                audio.samples[start.min(audio.samples.len())..end.min(audio.samples.len())]
                    .to_vec(),
                audio.sample_rate,
            );
            let key = format!("{}+{}", G2p::base(lphon), G2p::base(phon));
            let full = format!("{lphon} {phon}");
            let dprev = k
                .checked_sub(2)
                .and_then(|x| stream.get(x))
                .map(|(p, _)| p.as_str());
            let did = store.insert_unit(
                UnitLevel::Diphone,
                &key,
                &full,
                voice,
                &daudio,
                if opts.codec_tokens {
                    Some(crate::codec::MimicMct.encode(&daudio))
                } else {
                    None
                },
                opts.write_wav,
                dprev,
                gnext,
                provider,
            )?;
            if let Some(pd) = prev_diphone_id {
                store.add_follows(pd, did);
            }
            prev_diphone_id = Some(did);
        }
    }
    Ok((ids, phoneme_count))
}

#[derive(Debug, Default)]
pub struct ComposeReport {
    pub total_chars: usize,
    pub cached_chars: usize,
    pub generated_chars: usize,
    pub tts_calls: Vec<String>,
    /// Resolved units in speaking order: (unit text, level).
    pub hits: Vec<(String, UnitLevel)>,
    /// Node ids of the cached units used (empty for the greedy path and
    /// for synthesized spans) — lets tests and callers verify *which*
    /// variants unit selection chose.
    #[allow(dead_code)]
    pub units: Vec<padagonia::NodeId>,
    /// Mean join discontinuity (amplitude + ZCR jump) across splices.
    pub mean_seam_discontinuity: f64,
}

impl ComposeReport {
    pub fn cache_hit_pct(&self) -> f64 {
        if self.total_chars == 0 {
            0.0
        } else {
            100.0 * self.cached_chars as f64 / self.total_chars as f64
        }
    }
}

/// One resolved output fragment.
struct Part {
    audio: WavAudio,
    text: String,
    level: UnitLevel,
}

pub fn compose(
    store: &mut MimicStore,
    tts: &dyn TtsProvider,
    text: &str,
    voice: &str,
    g2p: Option<&G2p>,
) -> Result<(WavAudio, ComposeReport)> {
    let words = units::tokenize(text);
    if words.is_empty() {
        return Err(MimicError::NotFound("empty text".into()));
    }
    let n = words.len();
    // None = unresolved; Some(None) = covered by an earlier unit;
    // Some(Some(part)) = unit starting at this word.
    let mut slots: Vec<Option<Option<Part>>> = (0..n).map(|_| None).collect();
    let mut cached_chars = 0usize;

    // --- phrase pass: longest non-overlapping verbatim matches first
    let mut candidates: Vec<(usize, usize, String, Vec<NodeId>)> = Vec::new();
    for (ptext, ids) in store.units_for_level(UnitLevel::Phrase) {
        let pw = units::tokenize(&ptext);
        if pw.is_empty() || pw.len() > n {
            continue;
        }
        for start in 0..=(n - pw.len()) {
            if words[start..start + pw.len()] == pw[..] {
                candidates.push((start, pw.len(), ptext.clone(), ids.clone()));
            }
        }
    }
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (start, len, ptext, ids) in candidates {
        if slots[start..start + len].iter().any(|s| s.is_some()) {
            continue;
        }
        let audio = store.get_audio(ids[0])?;
        cached_chars += words[start..start + len]
            .iter()
            .map(|w| w.chars().count())
            .sum::<usize>();
        slots[start] = Some(Some(Part {
            audio,
            text: ptext,
            level: UnitLevel::Phrase,
        }));
        for slot in slots.iter_mut().skip(start + 1).take(len - 1) {
            *slot = Some(None);
        }
    }

    // --- word / morpheme pass
    for i in 0..n {
        if slots[i].is_some() {
            continue;
        }
        let w = &words[i];
        let prev = if i > 0 {
            Some(words[i - 1].as_str())
        } else {
            None
        };
        let next = words.get(i + 1).map(String::as_str);

        let word_hits = store.lookup_exact(UnitLevel::Word, w);
        if !word_hits.is_empty() {
            let id = store
                .best_context_match(word_hits, prev, next)
                .unwrap_or(word_hits[0]);
            let audio = store.get_audio(id)?;
            cached_chars += w.chars().count();
            slots[i] = Some(Some(Part {
                audio,
                text: w.clone(),
                level: UnitLevel::Word,
            }));
            continue;
        }

        // morpheme fallback: reuse a segment of a longer cached word that
        // contains this word as a stem ("run" <- "running")
        for (cand, f0, f1) in units::morpheme_candidates(w) {
            let chits = store.lookup_exact(UnitLevel::Word, &cand);
            if chits.is_empty() {
                continue;
            }
            let full = store.get_audio(chits[0])?;
            cached_chars += w.chars().count();
            slots[i] = Some(Some(Part {
                audio: full.slice_frac(f0, f1),
                text: w.clone(),
                level: UnitLevel::Morpheme,
            }));
            break;
        }
    }

    // --- synthesize miss spans, one provider call per consecutive span
    let mut tts_calls = Vec::new();
    let mut generated_chars = 0usize;
    let mut i = 0;
    while i < n {
        if slots[i].is_some() {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && slots[j].is_none() {
            j += 1;
        }
        let span_text = words[i..j].join(" ");
        let audio = tts.synthesize(&span_text, voice)?;
        tts_calls.push(span_text.clone());
        // the freshly generated words join the cache
        let outer_prev = if i > 0 {
            Some(words[i - 1].as_str())
        } else {
            None
        };
        let outer_next = words.get(j).map(String::as_str);
        let span_phonemes: Vec<Option<Vec<String>>> = match g2p {
            Some(g) => words[i..j].iter().map(|w| g.phonemes(w)).collect(),
            None => vec![None; j - i],
        };
        segment_and_store_words(
            store,
            &words[i..j],
            &span_phonemes,
            &audio,
            voice,
            tts.name(),
            outer_prev,
            outer_next,
            &IngestOptions::default(),
        )?;
        generated_chars += words[i..j].iter().map(|w| w.chars().count()).sum::<usize>();
        // splice the span as one fragment to preserve its internal prosody
        slots[i] = Some(Some(Part {
            audio,
            text: span_text,
            level: UnitLevel::Word,
        }));
        for slot in slots.iter_mut().take(j).skip(i + 1) {
            *slot = Some(None);
        }
        i = j;
    }

    // --- finalize in speaking order
    let mut parts: Vec<Part> = Vec::with_capacity(n);
    for slot in slots.iter_mut() {
        if let Some(Some(part)) = slot.take() {
            parts.push(part);
        }
    }
    let mut report = ComposeReport {
        total_chars: words.iter().map(|w| w.chars().count()).sum(),
        cached_chars,
        generated_chars,
        tts_calls,
        hits: Vec::with_capacity(parts.len()),
        units: Vec::new(),
        mean_seam_discontinuity: 0.0,
    };
    for p in &parts {
        report.hits.push((p.text.clone(), p.level));
    }
    let audios: Vec<WavAudio> = parts.into_iter().map(|p| p.audio).collect();
    // Seam discontinuity proxy at each join, measured pre-splice: combines
    // amplitude and spectral-ish (ZCR) jumps across the boundary. Lower is
    // smoother; 0 for single-part output. This is the native stand-in until
    // the neural perceptual stack (UTMOS et al.) is wired in.
    let seams: Vec<f64> = audios
        .windows(2)
        .map(|w| crate::audio::seam_discontinuity(&w[0], &w[1]))
        .collect();
    report.mean_seam_discontinuity = if seams.is_empty() {
        0.0
    } else {
        seams.iter().sum::<f64>() / seams.len() as f64
    };
    let out = splice(&audios, CROSSFADE_MS)?;
    Ok((out, report))
}
