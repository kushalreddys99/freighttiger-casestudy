import sys, json, time
import pandas as pd
from metrics import load, weekly, add_baselines
from detect import candidates
from notes import load_notes
from verdict import decide, retrieval_trace
from retrieval import Retriever

COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history",
           "vs_similar_routes", "flagged", "matched_note_id", "reason"]

def build_full(shipments="data/shipment_records.csv", notes_path="data/context_notes.csv"):
    t0 = time.perf_counter()
    w = add_baselines(weekly(load(shipments)))
    notes = load_notes(notes_path)
    retriever = Retriever(notes)
    rows, trace = [], []
    for _, r in candidates(w).iterrows():
        trace += retrieval_trace(notes, retriever, r["route"], r["week_of"])
        flagged, nid, reason = decide(notes, r["route"], r["week_of"], retriever)
        rows.append({
            "route": r["route"],
            "week_of": r["week_of"].strftime("%Y-%m-%d"),
            "cost_per_tonne_km": f"{r['cpt']:.2f}",
            "vs_own_history": f"{r['vs_own']:+.1f}% vs this route's past average",
            "vs_similar_routes": f"{r['vs_peers']:+.1f}% vs similar-length routes this week",
            "flagged": flagged, "matched_note_id": nid, "reason": reason})
    df = (pd.DataFrame(rows, columns=COLUMNS)
            .sort_values(["week_of", "route"]).reset_index(drop=True))
    trace_df = pd.DataFrame(trace)
    usage = {  # deterministic fields only, so reruns stay byte-identical
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "llm_cost_inr": 0.0,
        "retrieval_method": "TF-IDF (scikit-learn), local, no API",
        "retrieval_queries": retriever.queries, "rows_written": len(df),
        "retrieval_proposals_rejected_by_filters": int((~trace_df["passes_hard_filters"]).sum()),
    }
    timing = {"pipeline_seconds": round(time.perf_counter() - t0, 3)}
    return df, trace_df, usage, timing

def build(*a, **k):
    return build_full(*a, **k)[0]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs/output.csv"
    df, trace_df, usage, timing = build_full()
    df.to_csv(out, index=False)
    trace_df.to_csv("outputs/retrieval_trace.csv", index=False)
    json.dump(usage, open("outputs/usage_log.json", "w"), indent=2)
    print(f"wrote {len(df)} rows to {out}")
    print(f"timing (not part of the hashed output): {timing}")