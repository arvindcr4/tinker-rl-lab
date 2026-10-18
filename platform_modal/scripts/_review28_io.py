"""Atomic, schema-bearing outputs for revised exploratory analyses."""
import csv
import os
import tempfile
from pathlib import Path


def write_table(path, rows, fields=None, comment=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else ['status'])
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name+'.')
    try:
        with os.fdopen(fd,'w') as fh:
            if comment:
                for line in comment.splitlines():fh.write('# '+line.lstrip('# ')+'\n')
            writer=csv.DictWriter(fh,fieldnames=fields,delimiter='\t',extrasaction='raise')
            writer.writeheader();writer.writerows(rows)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
