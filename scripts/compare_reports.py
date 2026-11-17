"""Side-by-side numbers from evaluation report.json files.

Usage: python scripts/compare_reports.py <label=dir> [<label=dir> ...]
"""
import json
import sys

print(f"{'run':28s} {'benign FA':>16s} {'catch(any)':>11s} {'catch(correct)':>15s}  recall per category (grooming, bullying, isolation, personal_info, platform_migration, threats)")
for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    r = json.load(open(f"{d}/report.json"))
    fa = r["benign_false_alarms"]
    cc, cr = r["catch_rate"], r["catch_rate_category"]
    rec = []
    for c in ("grooming", "bullying", "isolation", "personal_info", "platform_migration", "threats"):
        h, n = r["per_category"][c]["recall"]
        rec.append(f"{h / n:.0%}" if n else "n/a")
    print(f"{label:28s} {fa['hits'] / max(1, fa['support']):>7.1%} ({fa['hits']:3d}/{fa['support']:3d}) {cc['value']:>10.1%} {cr['hits'] / max(1, cr['support']):>14.1%}   {' '.join(rec)}")
pol = json.load(open(f"{sys.argv[-1].split('=', 1)[1]}/report.json")).get("threshold_policy")
