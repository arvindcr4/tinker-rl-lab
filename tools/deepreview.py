#!/usr/bin/env python3
"""Client and CLI for Google's Deep Review API (deepreview.googleapis.com/v1).

Uploads a manuscript (or imports a Drive file), runs an asynchronous
multi-agent review, polls it, and writes the result: the review markdown,
anchored comments and citation-verification reports.

Auth: Application Default Credentials with the drive.file scope, e.g.

  gcloud auth application-default login \
    --scopes=https://www.googleapis.com/auth/drive.file,https://www.googleapis.com/auth/cloud-platform

or a raw token in DEEPREVIEW_ACCESS_TOKEN. A Bring-Your-Own-Key Gemini key is
taken from --gemini-key-env (default GEMINI_API_KEY) only when --byok is set.

  python tools/deepreview.py review thesis.pdf --topology HIGH --out reports/deepreview
  python tools/deepreview.py review thesis.pdf --engine agy --model gemini-3.8-flash-high
  python tools/deepreview.py list
  python tools/deepreview.py get documents/abc/reviews/xyz --out reports/deepreview
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import requests

API = "https://deepreview.googleapis.com"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
MAX_UPLOAD = 20 * 1024 * 1024  # discovery doc: mediaUpload.maxSize
TOPOLOGIES = ["EXTRA_HIGH", "HIGH", "MID", "LOW"]
TERMINAL = {"SUCCEEDED", "FAILED"}


class DeepReviewError(RuntimeError):
    pass


def _session(quota_project: str | None) -> requests.Session:
    token = os.environ.get("DEEPREVIEW_ACCESS_TOKEN")
    if token:
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {token}"
    else:
        import google.auth  # type: ignore[import-untyped]
        from google.auth.transport.requests import AuthorizedSession  # type: ignore[import-untyped]

        creds, project = google.auth.default(scopes=SCOPES)
        quota_project = quota_project or getattr(creds, "quota_project_id", None) or project
        s = AuthorizedSession(creds)
    if quota_project:
        s.headers["x-goog-user-project"] = quota_project
    return s


class DeepReview:
    def __init__(self, session: requests.Session, base: str = API) -> None:
        self.s = session
        self.base = base.rstrip("/")

    def _call(self, method: str, path: str, *, params=None, json_body=None, **kw) -> Any:
        r = self.s.request(
            method, f"{self.base}/{path}", params=params, json=json_body, timeout=120, **kw
        )
        if r.status_code >= 400:
            try:
                err = r.json()["error"]
                msg = f"{err.get('code')} {err.get('status')}: {err.get('message')}"
            except Exception:
                msg = f"{r.status_code}: {r.text[:500]}"
            raise DeepReviewError(f"{method} {path} -> {msg}")
        return r.json() if r.content and "json" in r.headers.get("content-type", "") else r.content

    # documents
    def upload(
        self, path: Path, display_name: str | None = None, document_id: str | None = None
    ) -> dict:
        data = path.read_bytes()
        if len(data) > MAX_UPLOAD:
            raise DeepReviewError(f"{path} is {len(data)} bytes; API limit is {MAX_UPLOAD}")
        meta = {"displayName": display_name or path.name}
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        boundary = uuid.uuid4().hex
        body = (
            (
                f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{json.dumps(meta)}\r\n"
                f"--{boundary}\r\nContent-Type: {mime}\r\n\r\n"
            ).encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        params = {"uploadType": "multipart", "requestId": str(uuid.uuid4())}
        if document_id:
            params["documentId"] = document_id
        return self._call(
            "POST",
            "upload/v1/documents",
            params=params,
            data=body,
            headers={"Content-Type": f"multipart/related; boundary={boundary}"},
        )

    def import_drive(
        self,
        file_id: str,
        display_name: str | None = None,
        resource_key: str | None = None,
        revision_id: str | None = None,
    ) -> dict:
        src = {"fileId": file_id}
        if resource_key:
            src["resourceKey"] = resource_key
        if revision_id:
            src["revisionId"] = revision_id
        doc: dict[str, Any] = {"driveSource": src}
        if display_name:
            doc["displayName"] = display_name
        return self._call(
            "POST", "v1/documents", params={"requestId": str(uuid.uuid4())}, json_body=doc
        )

    def _paged(self, path: str, key: str, params: dict) -> Iterator[dict]:
        params = {k: v for k, v in params.items() if v is not None}
        while True:
            page = self._call("GET", path, params=params)
            yield from page.get(key, [])
            if not page.get("nextPageToken"):
                return
            params["pageToken"] = page["nextPageToken"]

    def list_documents(
        self, filter: str | None = None, order_by: str | None = None
    ) -> Iterator[dict]:
        return self._paged("v1/documents", "documents", {"filter": filter, "orderBy": order_by})

    def get(self, name: str) -> dict:
        return self._call("GET", f"v1/{name}")

    def delete_document(self, name: str, force: bool = False) -> None:
        self._call("DELETE", f"v1/{name}", params={"force": str(force).lower()})

    def download(self, name: str) -> bytes:
        return self._call("GET", f"v1/{name}:download", params={"alt": "media"})

    # reviews
    def create_review(
        self, document: str, run_config: dict | None = None, review_id: str | None = None
    ) -> dict:
        params = {"requestId": str(uuid.uuid4())}
        if review_id:
            params["reviewId"] = review_id
        return self._call(
            "POST",
            f"v1/{document}/reviews",
            params=params,
            json_body={"runConfig": run_config} if run_config else {},
        )

    def list_reviews(self, document: str) -> Iterator[dict]:
        return self._paged(f"v1/{document}/reviews", "reviews", {})

    def cancel(self, operation: str) -> None:
        self._call("POST", f"v1/{operation}:cancel", json_body={})

    def wait(self, review_name: str, interval: float = 20, timeout: float = 3 * 3600) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while True:
            rev = self.get(review_name)
            state = rev.get("state", "STATE_UNSPECIFIED")
            status = (state, rev.get("progressPercent"))
            if status != last:
                print(
                    f"[deepreview] {review_name}: {state} {rev.get('progressPercent', 0)}%",
                    file=sys.stderr,
                )
                last = status
            if state in TERMINAL:
                return rev
            if time.monotonic() > deadline:
                raise DeepReviewError(f"timed out waiting for {review_name} (last state {state})")
            time.sleep(interval)


def _review_name_from_op(op: dict, document: str) -> str:
    # Operation names look like operations/documents/{d}/reviews/{r}; metadata/response may also carry it.
    for src in (op.get("response") or {}, op.get("metadata") or {}):
        if isinstance(src.get("name"), str) and "/reviews/" in src["name"]:
            return src["name"]
    name = op.get("name", "")
    if "documents/" in name and "/reviews/" in name:
        return name[name.index("documents/") :]
    raise DeepReviewError(
        f"cannot find review name in operation for {document}: {json.dumps(op)[:500]}"
    )


def _loc(c: dict) -> str:
    loc = c.get("location") or {}
    if "pdf" in loc:
        p = loc["pdf"]
        q = (p.get("quoteSelector") or {}).get("exact", "")
        return f"p.{p.get('pageNumber', '?')}" + (f' "{q[:80]}"' if q else "")
    if "text" in loc:
        t = loc["text"]
        return f"L{t.get('firstLine', '?')}-{t.get('lastLine', '?')}"
    return "-"


def write_outputs(review: dict, out: Path) -> Path:
    rid = review["name"].replace("/", "_")
    d = out / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "review.json").write_text(json.dumps(review, indent=2))
    res = review.get("result") or {}
    if res.get("error"):
        (d / "error.json").write_text(json.dumps(res["error"], indent=2))
    if res.get("reviewMarkdown"):
        (d / "review.md").write_text(res["reviewMarkdown"])

    comments = res.get("anchoredComments") or []
    order = {
        s: i
        for i, s in enumerate(["CRITICAL", "MAJOR", "MINOR", "SUGGESTION", "SEVERITY_UNSPECIFIED"])
    }
    comments.sort(key=lambda c: order.get(c.get("severity", "SEVERITY_UNSPECIFIED"), 9))
    lines = ["# Anchored comments", ""]
    for c in comments:
        lines.append(f"## [{c.get('severity', '?')}] {c.get('summary', '')}  ({_loc(c)})")
        lines += ["", c.get("bodyMarkdown", "")]
        if c.get("suggestedEdit"):
            lines += ["", "**Suggested edit:**", "", "```", c["suggestedEdit"], "```"]
        lines.append("")
    (d / "comments.md").write_text("\n".join(lines))

    cites = [c["citationReport"] for c in comments if c.get("citationReport")]
    if cites:
        cites.sort(key=lambda r: r.get("refNum", 0))
        rows = ["| # | verdict | lookup | search | reference |", "|---|---|---|---|---|"]
        for r in cites:
            ref = (r.get("rawReference") or "").replace("|", "\\|").replace("\n", " ")[:160]
            rows.append(
                f"| {r.get('refNum', '')} | {r.get('finalVerdict', '')} | {r.get('lookupVerdict', '')} "
                f"| {r.get('searchVerdict', '')} | {ref} |"
            )
        (d / "citations.md").write_text("\n".join(["# Citation verification", ""] + rows) + "\n")
        bib = [r["retrievedBibtex"] for r in cites if r.get("retrievedBibtex")]
        if bib:
            (d / "retrieved.bib").write_text("\n\n".join(bib) + "\n")

    sev = Counter(c.get("severity", "?") for c in comments)
    verdicts = Counter(r.get("finalVerdict", "?") for r in cites)
    print(f"[deepreview] {review.get('state')} -> {d}", file=sys.stderr)
    print(f"[deepreview] comments: {dict(sev)}; citations: {dict(verdicts)}", file=sys.stderr)
    return d


AGY_PROMPT = """You are a senior reviewer for a top ML venue (NeurIPS/ICLR). Review the manuscript
below (PDF pages are marked `<!-- page N -->`). It is complete; do not use any tools.

Write the review in markdown with these sections: Summary; Strengths; Weaknesses;
Anchored comments (one bullet each: `[CRITICAL|MAJOR|MINOR|SUGGESTION] p.N: "short quote" -
issue, then a concrete fix`); Questions for the authors; Citations to verify (references that
look wrong, unverifiable or possibly fabricated); Overall score (1-10) and confidence (1-5).
Ground every claim in the text; do not invent results. Output only the review.
"""


def _manuscript_text(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return path.read_text()
    import fitz  # type: ignore[import-not-found]  # Optional PyMuPDF

    with fitz.open(path) as doc:
        return "\n\n".join(f"<!-- page {i} -->\n{p.get_text()}" for i, p in enumerate(doc, 1))


def review_with_agy(
    path: Path, out: Path, model: str | None, prompt: str | None, timeout: float
) -> Path:
    """Fallback reviewer on Antigravity CLI credits. Single agent: no anchors or citation lookups."""
    import shutil
    import subprocess
    import tempfile

    agy = shutil.which("agy")
    if not agy:
        raise DeepReviewError("agy CLI not found on PATH")
    focus = f"\nAdditional reviewer focus:\n{prompt}\n" if prompt else ""
    content = f"{AGY_PROMPT}{focus}\n<manuscript>\n{_manuscript_text(path)}\n</manuscript>\n"
    # Manuscript goes in via stream-json stdin (argv is too small for a thesis); plan mode = read-only.
    msg = json.dumps({"event": "user", "message": {"content": content}}) + "\n"
    cmd = [
        agy,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--mode",
        "plan",
        "--print-timeout",
        f"{int(timeout)}s",
    ]
    if model:
        cmd += ["--model", model]
    cmd.append("-p=")
    print(f"[deepreview] agy review of {path.name} ({model or 'default model'})", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="deepreview-agy-") as ws:
        r = subprocess.run(
            cmd, cwd=ws, input=msg, capture_output=True, text=True, timeout=timeout + 60
        )
    result = None
    for line in r.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "result":
            result = ev["result"]
    if not result or result.get("status") != "SUCCESS" or not result.get("response", "").strip():
        err = (result or {}).get("error") or (r.stderr or r.stdout)[-500:]
        raise DeepReviewError(f"agy failed ({r.returncode}): {err}")
    d = out / f"agy_{path.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "review.md").write_text(result["response"].strip() + "\n")
    meta = {
        "engine": "agy",
        "model": model,
        "source": str(path),
        "usage": result.get("usage"),
        "conversation_id": result.get("conversation_id"),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[deepreview] agy -> {d}", file=sys.stderr)
    return d


def _run_config(a: argparse.Namespace) -> dict:
    cfg: dict[str, Any] = {}
    if a.topology:
        cfg["topology"] = a.topology
    if a.model:
        cfg["model"] = a.model
    if a.prompt_file:
        cfg["userPrompt"] = Path(a.prompt_file).read_text()
    elif a.prompt:
        cfg["userPrompt"] = a.prompt
    if a.byok:
        key = os.environ.get(a.gemini_key_env)
        if not key:
            raise DeepReviewError(f"--byok set but ${a.gemini_key_env} is empty")
        cfg["geminiApiKey"] = key
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--quota-project", help="sets x-goog-user-project (needed for some user ADC setups)"
    )
    p.add_argument("--endpoint", default=API)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser(
        "review", help="upload/import a manuscript, run a review, wait, write outputs"
    )
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", type=Path, help="PDF / .tex / .md / .txt (<= 20 MiB)")
    src.add_argument("--drive-file-id")
    src.add_argument("--document", help="reuse an existing documents/{id}")
    r.add_argument("--resource-key")
    r.add_argument("--display-name")
    r.add_argument(
        "--engine",
        choices=["deepreview", "agy"],
        default="deepreview",
        help="agy = single-agent fallback on Antigravity CLI credits (local file only)",
    )
    r.add_argument("--topology", choices=TOPOLOGIES)
    r.add_argument("--model")
    r.add_argument("--prompt", help="focus areas / instructions for the reviewers")
    r.add_argument("--prompt-file")
    r.add_argument(
        "--byok", action="store_true", help="bill to your Gemini key (see --gemini-key-env)"
    )
    r.add_argument("--gemini-key-env", default="GEMINI_API_KEY")
    r.add_argument("--no-wait", action="store_true")
    r.add_argument("--interval", type=float, default=20)
    r.add_argument("--timeout", type=float, default=3 * 3600)
    r.add_argument("--out", type=Path, default=Path("reports/deepreview"))

    g = sub.add_parser("get", help="fetch a document or review; for a review, write outputs")
    g.add_argument("name")
    g.add_argument("--wait", action="store_true")
    g.add_argument("--out", type=Path, default=Path("reports/deepreview"))

    ls = sub.add_parser("list", help="list documents, or a document's reviews")
    ls.add_argument("document", nargs="?")
    ls.add_argument("--filter")

    dl = sub.add_parser("download", help="download a document's original content")
    dl.add_argument("document")
    dl.add_argument("dest", type=Path)

    rm = sub.add_parser("delete", help="delete a document (and with --force its reviews)")
    rm.add_argument("document")
    rm.add_argument("--force", action="store_true")

    cn = sub.add_parser("cancel", help="cancel a review operation")
    cn.add_argument("operation")

    a = p.parse_args(argv)
    try:
        if a.cmd == "review" and a.engine == "agy":
            if not a.file:
                raise DeepReviewError("--engine agy needs a local file")
            prompt = Path(a.prompt_file).read_text() if a.prompt_file else a.prompt
            review_with_agy(a.file, a.out, a.model, prompt, a.timeout)
            return 0
    except DeepReviewError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    api = DeepReview(_session(a.quota_project), a.endpoint)
    try:
        if a.cmd == "review":
            cfg = _run_config(a)
            if a.document:
                doc = a.document
            elif a.drive_file_id:
                doc = api.import_drive(a.drive_file_id, a.display_name, a.resource_key)["name"]
            else:
                doc = api.upload(a.file, a.display_name)["name"]
            print(f"[deepreview] document: {doc}", file=sys.stderr)
            op = api.create_review(doc, cfg)
            name = _review_name_from_op(op, doc)
            print(f"[deepreview] review: {name}  (operation {op.get('name')})", file=sys.stderr)
            if a.no_wait:
                print(name)
                return 0
            rev = api.wait(name, a.interval, a.timeout)
            write_outputs(rev, a.out)
            return 0 if rev.get("state") == "SUCCEEDED" else 1
        if a.cmd == "get":
            if "/reviews/" in a.name:
                rev = api.wait(a.name) if a.wait else api.get(a.name)
                write_outputs(rev, a.out)
            else:
                print(json.dumps(api.get(a.name), indent=2))
        elif a.cmd == "list":
            items = api.list_reviews(a.document) if a.document else api.list_documents(a.filter)
            for it in items:
                state = it.get("state") or (it.get("latestReviewSummary") or {}).get("state", "")
                print(
                    f"{it['name']}\t{state}\t{it.get('displayName', '')}\t{it.get('createTime', '')}"
                )
        elif a.cmd == "download":
            a.dest.write_bytes(api.download(a.document))
        elif a.cmd == "delete":
            api.delete_document(a.document, a.force)
        elif a.cmd == "cancel":
            api.cancel(a.operation)
    except DeepReviewError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
