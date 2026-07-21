//! A deliberately small SSML subset.
//!
//! Supported: optional `<speak>` wrapper, `<break time="500ms"/>` /
//! `<break time="2s"/>` as explicit silences, and XML entity unescaping.
//! All other tags are stripped (their text content is kept), which keeps
//! provider SSML from breaking the cache key — text stays normalized.

#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    Text(String),
    Break(f64), // milliseconds
}

/// Parse SSML-ish input into text/break segments. If the input has no tags
/// at all, returns a single Text segment unchanged.
pub fn parse(input: &str) -> Vec<Segment> {
    if !input.contains('<') {
        return vec![Segment::Text(input.to_string())];
    }
    let mut out = Vec::new();
    let mut text = String::new();
    let mut rest = input;
    while let Some(lt) = rest.find('<') {
        text.push_str(&rest[..lt]);
        let gt = match rest.find('>') {
            Some(g) => g,
            None => {
                // unterminated tag: treat the rest as text
                text.push_str(&rest[lt..]);
                rest = "";
                break;
            }
        };
        let tag = &rest[lt + 1..gt];
        rest = &rest[gt + 1..];
        if let Some(ms) = break_ms(tag) {
            flush_text(&mut text, &mut out);
            out.push(Segment::Break(ms));
        }
        // all other tags (speak, emphasis, prosody, ...) are simply dropped
    }
    text.push_str(rest);
    flush_text(&mut text, &mut out);
    if out.is_empty() {
        out.push(Segment::Text(String::new()));
    }
    out
}

fn flush_text(text: &mut String, out: &mut Vec<Segment>) {
    let t = unescape(text.trim());
    if !t.is_empty() {
        out.push(Segment::Text(t));
    }
    text.clear();
}

/// `<break time="500ms"/>`, `<break time='2s'>`, `<break/>` (default 250 ms).
fn break_ms(tag: &str) -> Option<f64> {
    let t = tag.trim().trim_end_matches('/');
    if !t.starts_with("break") {
        return None;
    }
    let q = t.split('"').nth(1).or_else(|| t.split('\'').nth(1));
    match q {
        Some(v) => {
            if let Some(ms) = v.strip_suffix("ms") {
                ms.trim().parse().ok()
            } else if let Some(s) = v.strip_suffix('s') {
                s.trim().parse::<f64>().ok().map(|x| x * 1000.0)
            } else {
                v.trim().parse().ok()
            }
        }
        None => Some(250.0),
    }
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Milliseconds of silence as samples.
pub fn silence(ms: f64, sample_rate: u32) -> crate::audio::WavAudio {
    let n = (sample_rate as f64 * ms / 1000.0) as usize;
    crate::audio::WavAudio::new(vec![0; n], sample_rate)
}
