#!/usr/bin/env python3
"""Parse Gmail RAW MIME (base64) and extract attachments to staging."""
from __future__ import annotations

import argparse
import base64
import email
import sys
from email import policy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GMAIL_STAGING = REPO_ROOT / "paper" / "_staging" / "gmail"
SOURCE_CONFIGS = REPO_ROOT / "docs" / "manuscripts" / "source_configs"


def extract_from_raw(raw_b64: str, out_dir: Path) -> list[str]:
    raw = base64.urlsafe_b64decode(raw_b64 + "==")
    msg = email.message_from_bytes(raw, policy=policy.default)
    saved: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in msg.walk():
        fn = part.get_filename()
        if not fn:
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        dest = out_dir / fn
        dest.write_bytes(payload)
        saved.append(str(dest))
        print(f"Saved {fn} ({len(payload)} bytes) -> {dest}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file", help="File containing base64 RAW MIME from Gmail get_message RAW")
    parser.add_argument("--out", default=str(GMAIL_STAGING))
    args = parser.parse_args()
    raw_b64 = Path(args.raw_file).read_text().strip()
    saved = extract_from_raw(raw_b64, Path(args.out))
    # Also copy config to source_configs
    for s in saved:
        p = Path(s)
        if p.name == "thiophene_poly_all_001_coreless_dpm.txt":
            SOURCE_CONFIGS.mkdir(parents=True, exist_ok=True)
            dest = SOURCE_CONFIGS / "thiophene_poly_all_001_coreless_dpm.gmail.txt"
            dest.write_bytes(p.read_bytes())
            print(f"Copied to {dest}")
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
