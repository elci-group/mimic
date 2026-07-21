#!/usr/bin/env python3
"""Mimic eval harness — optional neural-metrics adapter.

Computes the P1+ quality metrics that need an ML stack:
  - UTMOS / NISQA   (predicted MOS of composed audio)
  - Whisper WER     (intelligibility of composed audio)
  - SECS            (speaker-embedding cosine similarity vs. reference voice)

The harness (mimic eval) calls this script only when MIMIC_EVAL_EXTERNAL=1.
It prints one JSON object per metric on stdout; anything it cannot compute
is reported as {"metric": ..., "status": "skipped", "reason": ...}.

Setup (deferred by default — this pulls in a large ML stack):

    python3 -m venv .venv
    .venv/bin/pip install torch torchaudio faster-whisper speechbrain utmos

Then: MIMIC_EVAL_EXTERNAL=1 cargo run -- eval --gate
"""

import json
import sys

METRICS = ["utmos", "nisqa", "whisper_wer", "secs"]


def main() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        for m in METRICS:
            print(json.dumps({
                "metric": m,
                "status": "skipped",
                "reason": "python ML stack not installed (see docstring)",
            }))
        sys.exit(0)

    # TODO(P2): implement metric computation against eval/reports wavs.
    for m in METRICS:
        print(json.dumps({"metric": m, "status": "skipped", "reason": "not yet implemented"}))


if __name__ == "__main__":
    main()
