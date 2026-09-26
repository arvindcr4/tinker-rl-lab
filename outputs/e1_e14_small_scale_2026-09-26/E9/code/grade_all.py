"""Grade each E9 submission with the official mlebench grader (mlebench.grade.grade_csv). Missing submission => graded as no-submission.
Usage (mlebench venv): grade_all.py <data_dir> <raw_dir> <comp> ..."""
import json, os, sys
from pathlib import Path
from mlebench.grade import grade_csv
from mlebench.registry import registry

data, raw, comps = sys.argv[1], sys.argv[2], sys.argv[3:]
reg = registry.set_data_dir(Path(data))
reports = []
for c in comps:
    sub = Path(raw) / c / "submission.csv"
    rep = grade_csv(sub, reg.get_competition(c)).to_dict()
    reports.append(rep)
    print(c, rep.get("score"), "any_medal", rep.get("any_medal"), "above_median", rep.get("above_median"), "valid", rep.get("valid_submission"))
json.dump(reports, open(Path(raw) / "grades.json", "w"), indent=1, default=str)
