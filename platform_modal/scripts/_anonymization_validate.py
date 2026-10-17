"""Fail-closed byte/name scan of a staged anonymous package (review23-v2).

This checks known identifiers only, not a guarantee against semantic identification.
Symlinks are rejected so a package cannot expose unscanned external content.
"""
from pathlib import Path
import sys

IDENTIFIERS = (b'arvindcr4', b'arvindcr', b'pes-llm-research', b'pes llm research')

def validate(root):
    root = Path(root)
    failures = []
    for path in root.rglob('*'):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            failures.append(f'symlink: {relative}')
            continue
        if any(token in relative.lower().encode() for token in IDENTIFIERS):
            failures.append(f'identifier in filename: {relative}')
        if path.is_file() and any(token in path.read_bytes().lower() for token in IDENTIFIERS):
            failures.append(f'identifier in content: {relative}')
    return failures

if __name__ == '__main__':
    problems = validate(sys.argv[1])
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        raise SystemExit(2)
