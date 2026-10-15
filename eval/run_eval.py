"""Evaluation harness. Run from the project root:  python eval/run_eval.py"""
import sys, hashlib, subprocess
sys.path.insert(0, "src")
import pandas as pd
from run import build
from notes import load_notes
from verdict import decide

# ---- 1. Hand-labeled expected verdicts for every route-week the detector flags ----
# Labeled from reading the notes myself, NOT from the pipeline's output.
EXPECTED = {
    ("Delhi-Jaipur",      "2024-11-11"): ("Yes", ""),
    ("Delhi-Jaipur",      "2024-11-18"): ("Yes", ""),
    ("Ahmedabad-Mumbai",  "2025-01-20"): ("No (justified)", "N002"),
    ("Chennai-Bangalore", "2025-02-24"): ("No (justified)", "N001"),
    ("Chennai-Bangalore", "2025-03-03"): ("No (justified)", "N001"),
    ("Chennai-Bangalore", "2025-03-10"): ("Yes", ""),   # N001 window ended Mar 8
    ("Chennai-Bangalore", "2025-03-17"): ("Yes", ""),   # N009: back to normal
    ("Mumbai-Pune",       "2025-09-15"): ("Yes", ""),   # N006: stable demand
    ("Mumbai-Pune",       "2025-11-17"): ("Yes", ""),
}

# ---- 2. Adversarial cases fed straight to the verdict logic ----
# (route, week_of, expected_flagged, expected_note, why this case exists)
ADVERSARIAL = [
    ("Chennai-Bangalore",   "2025-03-03", "No (justified)", "N001", "inside N001 window"),
    ("Chennai-Bangalore",   "2025-03-10", "Yes", "",     "right route, window just ended"),
    ("Delhi-Jaipur",        "2025-01-20", "Yes", "",     "N002 is a different route"),
    ("Ahmedabad-Mumbai",    "2025-01-20", "No (justified)", "N002", "exact N002 match"),
    ("Mumbai-Delhi",        "2024-07-29", "Yes", "",     "N005 says costs not affected"),
    ("Delhi-Jaipur",        "2024-05-20", "Yes", "",     "N007 describes improved roads"),
    ("Delhi-Jaipur",        "2024-03-11", "Yes", "",     "N004 affects routes not in data"),
    ("Mumbai-Pune",         "2025-10-27", "Yes", "",     "N010: no rate change"),
    ("Delhi-Jaipur",        "2025-05-12", "No (justified)", "N003", "all-routes diesel rise, in window"),
    ("Delhi-Jaipur",        "2025-08-04", "Yes", "",     "after the assumed N003 window"),
    ("Kolkata-Bhubaneswar", "2025-06-09", "No (justified)", "N003", "N008 is no-effect, N003 still applies"),
]

def main():
    notes = load_notes()
    ok = True

    # --- candidate-week accuracy ---
    out = build().fillna("")
    got = {(r.route, r.week_of): (r.flagged, r.matched_note_id) for r in out.itertuples()}
    missing = set(EXPECTED) - set(got); extra = set(got) - set(EXPECTED)
    verdict_ok = sum(got.get(k) is not None and got[k][0] == v[0] for k, v in EXPECTED.items())
    note_ok = sum(got.get(k) is not None and got[k][1] == v[1] for k, v in EXPECTED.items())
    false_just = sum(1 for k, v in EXPECTED.items()
                     if k in got and v[0] == "Yes" and got[k][0] != "Yes")
    print(f"Detector: expected {len(EXPECTED)} weeks, got {len(got)}; missing={sorted(missing)} extra={sorted(extra)}")
    print(f"Verdict accuracy   : {verdict_ok}/{len(EXPECTED)}")
    print(f"matched_note_id acc: {note_ok}/{len(EXPECTED)}")
    print(f"FALSE JUSTIFICATIONS (must be 0): {false_just}")
    ok &= (not missing and not extra and verdict_ok == len(EXPECTED)
           and note_ok == len(EXPECTED) and false_just == 0)

    # --- adversarial cases ---
    print("\nAdversarial cases:")
    adv_pass = 0
    for route, wk, exp_flag, exp_note, why in ADVERSARIAL:
        flag, nid, _ = decide(notes, route, wk)
        good = (flag == exp_flag and nid == exp_note); adv_pass += good
        print(f"  [{'PASS' if good else 'FAIL'}] {route:<19}{wk}  {flag:<15}{nid or '-':<5} ({why})")
    print(f"Adversarial: {adv_pass}/{len(ADVERSARIAL)}")
    ok &= adv_pass == len(ADVERSARIAL)

    # --- reproducibility: 3 fresh runs, compare hashes ---
    print("\nReproducibility (3 separate runs):")
    hashes = []
    for i in (1, 2, 3):
        p = f"outputs/run{i}.csv"
        subprocess.run([sys.executable, "src/run.py", p], check=True, capture_output=True)
        hashes.append(hashlib.sha256(open(p, "rb").read()).hexdigest())
        print(f"  run{i}: {hashes[-1][:16]}")
    same = len(set(hashes)) == 1
    print("  identical" if same else "  DIFFERENT")
    ok &= same

    print("\nOVERALL:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main() 