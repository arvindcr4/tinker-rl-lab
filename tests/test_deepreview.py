import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import deepreview  # noqa: E402


class FakeResp:
    def __init__(self, body):
        self.status_code, self._b = 200, body
        self.content, self.headers = b"x", {"content-type": "application/json"}

    def json(self):
        return self._b


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return FakeResp({"name": "documents/d1"})


def test_upload_is_multipart_with_metadata(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 hi")
    s = FakeSession()
    assert deepreview.DeepReview(s).upload(pdf)["name"] == "documents/d1"
    method, url, kw = s.calls[0]
    assert (method, url) == ("POST", "https://deepreview.googleapis.com/upload/v1/documents")
    assert kw["params"]["uploadType"] == "multipart"
    assert kw["headers"]["Content-Type"].startswith("multipart/related; boundary=")
    assert b'"displayName": "paper.pdf"' in kw["data"] and b"%PDF-1.4 hi" in kw["data"]


def test_review_name_from_operation():
    op = {"name": "operations/documents/d1/reviews/r1", "done": False}
    assert deepreview._review_name_from_op(op, "documents/d1") == "documents/d1/reviews/r1"


def test_write_outputs(tmp_path):
    review = {
        "name": "documents/d1/reviews/r1",
        "state": "SUCCEEDED",
        "result": {
            "reviewMarkdown": "# Review",
            "anchoredComments": [
                {"severity": "MINOR", "summary": "typo", "location": {"text": {"firstLine": 3, "lastLine": 3}}},
                {
                    "severity": "CRITICAL",
                    "summary": "bad ref",
                    "location": {"pdf": {"pageNumber": 9, "quoteSelector": {"exact": "[12]"}}},
                    "citationReport": {"refNum": 12, "finalVerdict": "HALLUCINATED", "rawReference": "Foo 2031",
                                       "retrievedBibtex": "@misc{foo}"},
                },
            ],
        },
    }
    d = deepreview.write_outputs(review, tmp_path)
    assert (d / "review.md").read_text() == "# Review"
    comments = (d / "comments.md").read_text()
    assert comments.index("CRITICAL") < comments.index("MINOR") and 'p.9 "[12]"' in comments
    assert "HALLUCINATED" in (d / "citations.md").read_text()
    assert (d / "retrieved.bib").read_text().startswith("@misc{foo}")
