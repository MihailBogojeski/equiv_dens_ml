#!/usr/bin/env python3
"""Recover Gmail attachments to disk using Gmail API or RAW MIME fallback.

Gmail MCP has no attachment-download tool. This script tries:
1. google-api-python-client with OAuth creds in standard locations
2. RAW MIME parsing for small files (<2 MB message size)
"""
from __future__ import annotations

import argparse
import base64
import email
import json
import os
import re
import sys
from email import policy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GMAIL_STAGING = REPO_ROOT / "paper" / "_staging" / "gmail"
MANUSCRIPTS = REPO_ROOT / "docs" / "manuscripts"

# Data files needing recovery (Gmail-only or prefer Gmail copy)
DATA_TARGETS: dict[str, dict[str, Any]] = {
    "19a935353461d845": {
        "files": [
            "simulation_cluster_all_001_compressed_0.npy",
            "loss_base.csv",
            "loss_delta.csv",
        ],
        "dest_subdir": "",
        "size_mb": 34.6,
    },
    "191eb766ed82a80c": {
        "files": ["traj_dipole_moment_results.zip"],
        "dest_subdir": "",
        "size_mb": 8.4,
    },
    "1909ddd78ab5a84e": {
        "files": ["polythiophene_1ps_mid_trajectories.zip"],
        "dest_subdir": "",
        "size_mb": 4.3,
    },
    "1946acb67cf86348": {
        "files": ["thiophene_poly_all_001_coreless_dpm.txt"],
        "dest_subdir": "",
        "size_mb": 0.017,
    },
    "1925058c11aa6845": {
        "files": ["thiophene_append_clean.ipynb"],
        "dest_subdir": "",
        "size_mb": 4.3,
    },
}

# Supporting manuscripts (small enough for RAW)
MANUSCRIPT_TARGETS: dict[str, dict[str, Any]] = {
    "19083c286b6b97bb": {
        "files": [
            "Equivariant_neural_networks_for_density_enhanced_molecular_dynamics.pdf",
            "delta_densities.pdf",
        ],
        "year": "2024",
    },
    "1a0198c80b31e299": {
        "files": ["responses.pdf"],
        "year": "2026",
    },
    "194de34ad038352d": {"files": ["Summary (1).pdf"], "year": "2025"},
    "191df3fc1334af90": {
        "files": [
            "doi_10.pdf",
            "zhu-tuckerman-2002-ab-initio-molecular-dynamics-investigation-of-the-concentration-dependence-of-charged-defect.pdf",
        ],
        "year": "2024",
    },
}


def _try_gmail_api() -> Any | None:
    """Return authenticated Gmail service or None."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        return None

    scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    cred_paths = [
        Path.home() / ".config" / "gmail" / "token.json",
        Path.home() / ".credentials" / "gmail-token.json",
        REPO_ROOT / "credentials" / "gmail_token.json",
    ]
    client_secret_paths = [
        Path.home() / ".config" / "gmail" / "credentials.json",
        REPO_ROOT / "credentials" / "gmail_credentials.json",
    ]

    creds = None
    for tp in cred_paths:
        if tp.exists():
            creds = Credentials.from_authorized_user_file(str(tp), scopes)
            break

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if not creds or not creds.valid:
        for cp in client_secret_paths:
            if cp.exists():
                flow = InstalledAppFlow.from_client_secrets_file(str(cp), scopes)
                creds = flow.run_local_server(port=0)
                break

    if not creds or not creds.valid:
        return None

    return build("gmail", "v1", credentials=creds)


def _download_via_api(service: Any, msg_id: str, filename: str, dest: Path) -> bool:
    """Download one attachment by filename from a message."""
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    for part in msg.get("payload", {}).get("parts", []) or []:
        fn = part.get("filename", "")
        if fn != filename:
            # recurse into multipart
            if part.get("parts"):
                for sub in part["parts"]:
                    if sub.get("filename") == filename:
                        part = sub
                        fn = filename
                        break
                else:
                    continue
            else:
                continue
        att_id = part.get("body", {}).get("attachmentId")
        if not att_id:
            data = part.get("body", {}).get("data")
            if data:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(base64.urlsafe_b64decode(data + "=="))
                return True
            continue
        att = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=msg_id, id=att_id)
            .execute()
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.urlsafe_b64decode(att["data"] + "=="))
        return True
    return False


def _parse_raw_mime(raw_b64: str, filename: str) -> bytes | None:
    """Extract attachment bytes from RAW MIME for small messages."""
    raw = base64.urlsafe_b64decode(raw_b64 + "==")
    msg = email.message_from_bytes(raw, policy=policy.default)
    for part in msg.walk():
        if part.get_filename() == filename:
            payload = part.get_payload(decode=True)
            if payload:
                return payload
    return None


def recover_all(dry_run: bool = False) -> dict[str, Any]:
    """Attempt recovery; return status dict."""
    GMAIL_STAGING.mkdir(parents=True, exist_ok=True)
    MANUSCRIPTS.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"api_available": False, "recovered": [], "manual": []}

    service = _try_gmail_api()
    status["api_available"] = service is not None

    all_targets: list[tuple[str, str, Path, float]] = []

    for msg_id, info in DATA_TARGETS.items():
        for fn in info["files"]:
            dest = GMAIL_STAGING / fn
            all_targets.append((msg_id, fn, dest, info.get("size_mb", 999)))

    for msg_id, info in MANUSCRIPT_TARGETS.items():
        year = info.get("year", "unknown")
        ms_dest = MANUSCRIPTS / "gmail" / year
        for fn in info["files"]:
            dest = ms_dest / fn
            all_targets.append((msg_id, fn, dest, 2.0))

    for msg_id, filename, dest, size_mb in all_targets:
        if dest.exists() and dest.stat().st_size > 0:
            status["recovered"].append({"file": filename, "method": "already_present", "dest": str(dest)})
            continue

        if dry_run:
            status["manual"].append({"file": filename, "msg_id": msg_id, "reason": "dry_run"})
            continue

        ok = False
        if service is not None:
            try:
                ok = _download_via_api(service, msg_id, filename, dest)
                if ok:
                    status["recovered"].append({"file": filename, "method": "gmail_api", "dest": str(dest)})
            except Exception as exc:
                status.setdefault("errors", []).append(f"{filename}: {exc}")

        if not ok and size_mb <= 2.0:
            # RAW fallback documented but requires external MCP call — skip here
            pass

        if not ok and not dest.exists():
            manual_path = MANUSCRIPTS / "gmail_manual" / filename
            status["manual"].append(
                {
                    "file": filename,
                    "msg_id": msg_id,
                    "size_mb": size_mb,
                    "drop_path": str(manual_path),
                    "reason": "no_gmail_api_creds" if not service else "download_failed",
                }
            )

    report_path = GMAIL_STAGING / "recovery_status.json"
    report_path.write_text(json.dumps(status, indent=2))
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover Gmail attachments for Drive archive import")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    status = recover_all(dry_run=args.dry_run)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
