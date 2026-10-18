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


def read_calibration(path):
    """Read iter102's versioned named schema, never positional column offsets."""
    import math
    path = Path(path)
    if not path.exists():
        return []
    columns = {'p': 'p_difficulty', 'zvf_emp': 'zvf_emp',
               'zvf_iid': 'zvf_iid_pred', 'delta': 'delta_calibration',
               'rho': 'rho_overdispersion', 'collapse_rate': 'collapse_rate',
               'converged_rate': 'converged_rate'}
    rows = []
    with path.open() as fh:
        reader = csv.DictReader((line for line in fh if not line.startswith('#')), delimiter='\t')
        required = {'library', 'model', 'G', *columns.values()}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing named iter102 calibration columns')
        for row in reader:
            values = {key: float(row[column]) for key, column in columns.items()}
            g = float(row['G'])
            if not all(math.isfinite(v) for v in [g, *values.values()]) or g < 1 or not g.is_integer():
                raise ValueError('Invalid calibration observation')
            rows.append(dict(library=row['library'], model=row['model'], G=int(g),
                             **values, evidence_path=row.get('evidence_path', ''),
                             estimator=row.get('estimator', ''), interpretation=row.get('interpretation', '')))
    return rows
