#!/bin/sh
set -eu
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
brandi genome validate --path .
brandi lint --path . --strict --fail-under 95
