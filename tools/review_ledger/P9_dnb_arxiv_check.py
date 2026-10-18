"""P9 Deep Review: print arXiv title/authors (and abstract with a 2nd arg) for comma-separated IDs."""

import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from xml.parsers import expat

ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
MAX_FEED_BYTES = 8 * 1024 * 1024


def parse_feed(data: bytes) -> ET.Element:
    """Parse an Atom response without allowing DTDs or entity declarations.

    Expat rejects declarations through its callbacks before entity expansion,
    including in non-UTF-8 XML. TreeBuilder only receives ordinary XML events.
    """
    if len(data) > MAX_FEED_BYTES:
        raise ValueError("arXiv feed exceeds the response size limit")
    builder = ET.TreeBuilder()
    parser = expat.ParserCreate(namespace_separator="}")

    def qualified(name: str) -> str:
        return "{" + name if "}" in name else name

    def start(name: str, attrs: dict[str, str]) -> None:
        builder.start(qualified(name), {qualified(k): v for k, v in attrs.items()})

    def end(name: str) -> None:
        builder.end(qualified(name))

    def reject_declaration(*args: object) -> None:
        raise ValueError("DTD and entity declarations are not allowed in arXiv feeds")

    def reject_external_entity(*args: object) -> int:
        raise ValueError("External entities are not allowed in arXiv feeds")

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = builder.data
    parser.StartDoctypeDeclHandler = reject_declaration
    parser.EntityDeclHandler = reject_declaration
    parser.ExternalEntityRefHandler = reject_external_entity
    parser.Parse(data, True)
    return builder.close()


def required_text(element: ET.Element, name: str) -> str:
    text = element.findtext(name, namespaces=ATOM_NS)
    if text is None or not text.strip():
        raise ValueError(f"arXiv entry is missing required text: {name}")
    return text


def format_entries(data: bytes, include_abstract: bool = False) -> list[str]:
    lines = []
    for entry in parse_feed(data).findall("a:entry", ATOM_NS):
        identifier = required_text(entry, "a:id").split("/")[-1]
        title = " ".join(required_text(entry, "a:title").split())
        authors = [required_text(author, "a:name") for author in entry.findall("a:author", ATOM_NS)]
        lines.append(f"{identifier} | {title} | {', '.join(authors[:10])}")
        if include_abstract:
            lines.append(" ".join(required_text(entry, "a:summary").split()))
    return lines


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit("usage: P9_dnb_arxiv_check.py <comma-separated IDs> [abstract]")
    query = urllib.parse.urlencode({"max_results": 40, "id_list": args[0]})
    with urllib.request.urlopen("https://export.arxiv.org/api/query?" + query, timeout=60) as feed:
        data = feed.read(MAX_FEED_BYTES + 1)
    for line in format_entries(data, include_abstract=len(args) > 1):
        print(line)


if __name__ == "__main__":
    main()
