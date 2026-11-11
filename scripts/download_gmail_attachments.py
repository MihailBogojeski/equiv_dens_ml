#!/usr/bin/env python3
"""Download Gmail attachments via OAuth (separate from rclone Drive auth).

Gmail MCP can list messages but has no attachment-download tool. RAW MIME works
only for small messages; this script uses Gmail API attachments.get for any size.

Setup (once, on laptop + cluster like rclone):
  module load rclone/1.72.1   # optional; any terminal works
  .venv/bin/python scripts/download_gmail_attachments.py --authorize

Then download:
  .venv/bin/python scripts/download_gmail_attachments.py --message 1909ddd78ab5a84e
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = REPO_ROOT / "credentials" / "gmail_token.json"
CREDS_PATH = REPO_ROOT / "credentials" / "gmail_credentials.json"
STAGING = REPO_ROOT / "paper" / "_staging" / "gmail"

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# message_id -> list of filenames to fetch (empty = all attachments)
TARGETS: dict[str, list[str]] = {
    "19a935353461d845": [
        "simulation_cluster_all_001_compressed_0.npy",
        "loss_base.csv",
    ],
    "191eb766ed82a80c": ["traj_dipole_moment_results.zip"],
    "1909ddd78ab5a84e": ["polythiophene_1ps_mid_trajectories.zip"],
}


def get_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not CREDS_PATH.exists():
            print(
                "Missing OAuth client secrets at credentials/gmail_credentials.json\n"
                "Create a Desktop OAuth client in Google Cloud Console (Gmail API enabled),\n"
                "download JSON, save as credentials/gmail_credentials.json, then --authorize",
                file=sys.stderr,
            )
            sys.exit(1)
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
        creds = flow.run_local_server(port=0)
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def walk_parts(payload: dict, out: list[dict]) -> None:
    if payload.get("filename"):
        out.append(payload)
    for part in payload.get("parts") or []:
        walk_parts(part, out)


def download_message(service, msg_id: str, names: list[str] | None, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    parts: list[dict] = []
    walk_parts(msg.get("payload", {}), parts)
    saved: list[str] = []
    for part in parts:
        fn = part.get("filename") or ""
        if not fn:
            continue
        if names and fn not in names:
            continue
        att_id = part.get("body", {}).get("attachmentId")
        out_path = dest / fn
        if att_id:
            att = (
                service.users()
                .messages()
                .attachments()
                .get(userId="me", messageId=msg_id, id=att_id)
                .execute()
            )
            data = base64.urlsafe_b64decode(att["data"] + "==")
        else:
            data_b64 = part.get("body", {}).get("data")
            if not data_b64:
                continue
            data = base64.urlsafe_b64decode(data_b64 + "==")
        out_path.write_bytes(data)
        saved.append(str(out_path))
        print(f"Saved {fn} ({len(data)} bytes) -> {out_path}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize", action="store_true", help="Run OAuth flow only")
    parser.add_argument("--message", action="append", default=[], help="Message ID to download")
    parser.add_argument("--all", action="store_true", help="Download all TARGETS")
    args = parser.parse_args()

    service = get_service()
    if args.authorize:
        print(f"Gmail token saved to {TOKEN_PATH}")
        return 0

    msg_ids = list(TARGETS.keys()) if args.all else args.message
    if not msg_ids:
        parser.error("Specify --message ID or --all")

    for mid in msg_ids:
        names = TARGETS.get(mid, [])
        download_message(service, mid, names or None, STAGING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
