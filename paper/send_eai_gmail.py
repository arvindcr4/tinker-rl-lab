#!/usr/bin/env python3
"""Send EAI submission email via Gmail API using existing OAuth token."""
import base64
import mimetypes
import os
import sys
from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = "/Users/arvind/xes/email-pipeline/token.json"
CREDS_PATH = "/Users/arvind/xes/email-pipeline/credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
]

ATTACHMENTS = [
    "/Users/arvind/tinker-rl-lab/paper/main_eai.pdf",
    "/Users/arvind/tinker-rl-lab/paper/main_eai.tex",
    "/Users/arvind/tinker-rl-lab/paper/references.bib",
]

TO = "pesu.eclib@pes.edu"
CC = ["anwesh@greatlearning.in", "sudha.bg@greatlearning.in"]
SUBJECT = "EAI Format Submission -- Plagiarism Check Request -- TinkerRL Audit"

BODY = """Dear PESU e-Library Team,

Please find attached our manuscript prepared in the EAI Endorsed Transactions (LNICST/Vancouver) format for submission.

Paper:    Reward Contrast, Not Algorithm Labels: A Diagnostic Audit of Critic-Free Group-Relative RL for LLMs
Authors:  Arvind C R, Sandhya Jeyaraj, Madhu Kumara L, Mohammad Rafi, Dhruva N Murthy, Arumugam K,
          Anwesh Reddy Paduri, Narayana Darapaneni
Length:   52 pages (incl. appendices)
Format:   EAI Endorsed Transactions (single-column draft, Vancouver refs)

Request: Kindly perform a plagiarism check (Turnitin / iThenticate) before we submit to the EAI journal OJS portal. As per the journal's guidelines, plagiarism must be below 10%.

Attachments:
  1. main_eai.pdf    -- compiled manuscript
  2. main_eai.tex    -- LaTeX source
  3. references.bib  -- BibTeX bibliography

CC: Mr. Anwesh Reddy Paduri and Ms. Sudha BG (Great Learning) for visibility.

Thank you for your support.

Best regards,
Arvind C R (on behalf of the authors)
PES University
arvindcr4@gmail.com
"""


def load_creds() -> Credentials:
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_PATH, "w") as fh:
                fh.write(creds.to_json())
        else:
            sys.exit("Token invalid and cannot refresh. Re-run OAuth flow.")
    return creds


def build_message(sender: str, dry_run: bool) -> dict:
    msg = EmailMessage()
    msg["To"] = TO
    msg["Cc"] = ", ".join(CC)
    msg["From"] = sender
    msg["Subject"] = SUBJECT
    msg.set_content(BODY)

    for path in ATTACHMENTS:
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(path, "rb") as fh:
            data = fh.read()
        msg.add_attachment(
            data, maintype=maintype, subtype=subtype, filename=os.path.basename(path)
        )

    raw = base64.urlsafe_b64encode(bytes(msg)).decode()
    if dry_run:
        size_mb = len(raw) * 3 / 4 / 1024 / 1024
        print(f"[dry-run] encoded size ~ {size_mb:.2f} MB")
    return {"raw": raw}


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    send_as_draft = "--draft" in sys.argv

    creds = load_creds()
    service = build("gmail", "v1", credentials=creds)

    profile = service.users().getProfile(userId="me").execute()
    sender = profile["emailAddress"]
    print(f"authenticated as: {sender}")

    body = build_message(sender, dry_run)
    if dry_run:
        return

    if send_as_draft:
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": body})
            .execute()
        )
        print(f"draft created: id={draft.get('id')}")
    else:
        sent = (
            service.users()
            .messages()
            .send(userId="me", body=body)
            .execute()
        )
        print(f"sent: id={sent.get('id')} threadId={sent.get('threadId')}")


if __name__ == "__main__":
    main()
