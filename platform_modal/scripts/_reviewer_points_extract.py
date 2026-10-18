#!/usr/bin/env python3
"""Validate complete YAML before emitting scorer rows; no partial fallback."""
import sys


def main(path: str) -> int:
    try:
        import yaml
        with open(path) as stream:
            doc=yaml.safe_load(stream)
        if not isinstance(doc,dict) or not isinstance(doc.get("weaknesses"),list):
            raise ValueError("missing weaknesses list")
        rows=[]; ids=set()
        for row in doc["weaknesses"]:
            if not isinstance(row,dict) or any(not isinstance(row.get(k),str) or not row[k].strip() for k in ("id","marker")):
                raise ValueError("every weakness needs nonempty id and marker")
            if row["id"] in ids: raise ValueError("duplicate weakness id")
            ids.add(row["id"])
            severity=row.get("severity","medium")
            values=(row["id"],row["marker"],severity)
            if any(not isinstance(v,str) or any(c in v for c in "|\n\r") for v in values):
                raise ValueError("invalid scorer field")
            rows.append("|".join(values))
        print("\n".join(rows))
        return 0
    except (ImportError,OSError,ValueError,TypeError,KeyError) as exc:
        print(f"reviewer YAML invalid or parser unavailable: {exc}",file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"reviewer YAML parse failed: {exc}",file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
