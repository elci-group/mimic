//! Minimal blocking HTTP/1.1 client (no external HTTP/TLS crates).
//!
//! Scope: `http://` URLs only — enough for local mock servers, sidecars, and
//! HTTP gateways fronting the real providers. `https://` URLs return a clear
//! error: enabling live cloud providers means adding a TLS stack (e.g.
//! rustls), which is deliberately not in the dependency tree yet.
//! Responses must be `Content-Length` framed or close-delimited; chunked
//! transfer encoding is not implemented.

use crate::{MimicError, Result};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

fn http_err(msg: impl Into<String>) -> MimicError {
    MimicError::Wav(format!("http: {}", msg.into()))
}

/// POST `body` to `url` with the given headers. `url` must be
/// `http://host[:port]/path?query`.
pub fn post(
    url: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    timeout: Duration,
) -> Result<HttpResponse> {
    let (host, port, path) = parse_url(url)?;
    let mut stream = TcpStream::connect((host, port))
        .map_err(|e| http_err(format!("connect {host}:{port}: {e}")))?;
    stream
        .set_read_timeout(Some(timeout))
        .and_then(|_| stream.set_write_timeout(Some(timeout)))
        .map_err(|e| http_err(format!("set timeout: {e}")))?;

    let mut req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    stream
        .write_all(req.as_bytes())
        .and_then(|_| stream.write_all(body))
        .map_err(|e| http_err(format!("write: {e}")))?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| http_err(format!("read: {e}")))?;
    parse_response(&raw)
}

fn parse_url(url: &str) -> Result<(&str, u16, &str)> {
    let rest = url.strip_prefix("http://").ok_or_else(|| {
        http_err(format!(
            "only http:// URLs supported (add TLS to enable https): {url}"
        ))
    })?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rfind(':') {
        Some(i) => {
            let p: u16 = authority[i + 1..]
                .parse()
                .map_err(|_| http_err(format!("bad port in {url}")))?;
            (&authority[..i], p)
        }
        None => (authority, 80),
    };
    if host.is_empty() {
        return Err(http_err(format!("empty host in {url}")));
    }
    Ok((host, port, path))
}

fn parse_response(raw: &[u8]) -> Result<HttpResponse> {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| http_err("malformed response (no header terminator)"))?;
    let head = String::from_utf8_lossy(&raw[..split]);
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| http_err("malformed status line"))?;
    for line in head.lines() {
        if line.to_ascii_lowercase().starts_with("transfer-encoding:")
            && line.to_ascii_lowercase().contains("chunked")
        {
            return Err(http_err("chunked responses not implemented"));
        }
    }
    Ok(HttpResponse {
        status,
        body: raw[split + 4..].to_vec(),
    })
}
