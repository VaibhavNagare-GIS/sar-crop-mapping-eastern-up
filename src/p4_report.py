"""Phase 4b: readable summary of the single test evaluation (no new test look; it only reads results/).
Usage: python src/p4_report.py [--out results]  -> prints tables and writes results/test_summary.md (for the README and the site)."""
import argparse
import json
from pathlib import Path

import pandas as pd


def f(x, ci=None):
    return f"{x:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def report(out):
    out = Path(out)
    r = json.loads((out / "test_metrics.json").read_text())
    L = ["## Test results (95% intervals from 2,000 field-level bootstrap resamples)", "",
         "| Model | Level | Macro-F1 | Balanced acc. | OA (always-biggest-class baseline) | AUC |", "|---|---|---|---|---|---|"]
    for n, m in r["field"].items():
        L.append(f"| {n} | field (n={m['n_fields']}) | {f(m['f1'], m['ci']['f1'])} | {f(m['bal'], m['ci']['bal'])} | {m['oa']:.3f} ({m['baseline_oa']:.3f}) | {f(m['auc'], m['ci']['auc'])} |")
    m = r["pixel"]
    L.append(f"| pixel model (map) | pixel (n={m['n_pixels']}, {m['n_fields']} fields) | {f(m['f1'], m['ci']['f1'])} | {f(m['bal'], m['ci']['bal'])} | {m['oa']:.3f} ({m['baseline_oa']:.3f}) | {f(m['auc'], m['ci']['auc'])} |")
    v = r["pixel_model_field_vote"]
    L.append(f"| pixel model, averaged per field | field | {v['f1']:.3f} | {v['bal']:.3f} | {v['oa']:.3f} | {v['auc']:.3f} |")
    L += ["", "### Primary (two-orbit Sentinel-1) minus each alternative, paired field bootstrap", "", "| Alternative | Metric | Difference | 95% interval | Share of resamples where primary is better |", "|---|---|---|---|---|"]
    for n, d in r["field_diff_primary_minus"].items():
        for k in ("f1", "auc"):
            L.append(f"| {n} | {k} | {d[k]['diff']:+.3f} | [{d[k]['ci'][0]:+.3f}, {d[k]['ci'][1]:+.3f}] | {d[k]['p_better']:.2f} |")
    pc = pd.read_csv(out / "per_class.csv")
    L += ["", "### Per class (producer's accuracy = recall, user's accuracy = precision)", "", "```", pc.round(3).to_string(index=False), "```"]
    for p in sorted(out.glob("confusion_*.csv")):
        cm = pd.read_csv(p, index_col=0)
        L += ["", f"### {p.stem} (rows = truth, columns = prediction; counts, then row share)", "", "```", cm.to_string(), "", cm.div(cm.sum(axis=1), axis=0).round(2).to_string(), "```"]
    text = "\n".join(L)
    (out / "test_summary.md").write_text(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    print(report(ap.parse_args().out))
