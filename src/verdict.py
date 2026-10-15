import pandas as pd
from notes import applicable_notes

NEAR_MISS_DAYS = 30

def _gap_days(note, week_of):
    """Days between the week and the note's window (0 = overlap)."""
    wk_start = pd.Timestamp(week_of); wk_end = wk_start + pd.Timedelta(days=6)
    if note["end"] < wk_start: return (wk_start - note["end"]).days
    if note["start"] > wk_end: return (note["start"] - wk_end).days
    return 0

def closest_note(notes, route, week_of, scores=None):
    """Nearest in-scope note within NEAR_MISS_DAYS; retrieval score breaks the tie."""
    pool = notes[notes["scope"].isin([route, "All Routes"])].copy()
    pool["gap"] = pool.apply(lambda n: _gap_days(n, week_of), axis=1)
    pool = pool[pool["gap"] <= NEAR_MISS_DAYS]
    if pool.empty:
        return None
    pool["score"] = pool["note_id"].map(scores or {}).fillna(0.0)
    pool = pool.sort_values(["score", "gap", "note_id"], ascending=[False, True, True])
    return pool.iloc[0]

def _near_miss_text(n, week_of):
    if n["effect"] == "increases_cost":
        return (f"The closest note ({n['note_id']}) covers {n['start'].date()} to {n['end'].date()}, "
                f"which does not include this week, so it cannot explain it.")
    return (f"The closest note ({n['note_id']}, {n['date']}) does not describe a reason for a cost "
            f"rise on this route.")

def decide(notes, route, week_of, retriever=None):
    """Return (flagged, matched_note_id, reason). The verdict is decided by code only:
    retrieval may rank candidates, but a note must pass scope + window + effect checks."""
    scores = retriever.score(route, week_of) if retriever else {}
    hits = applicable_notes(notes, route, week_of)
    if not hits.empty:
        # route-specific beats all-routes; then higher retrieval score; then note_id
        hits = hits.assign(_r=(hits["scope"] == "All Routes").astype(int),
                           _s=-hits["note_id"].map(scores).fillna(0.0)
                           ).sort_values(["_r", "_s", "note_id"])
        n = hits.iloc[0]
        reason = (f"Matches note {n['note_id']} dated {n['date']}: {n['note']} "
                  f"The cost rise has a clear explanation.")
        return "No (justified)", n["note_id"], reason
    reason = "No matching note found for this route or date range. "
    near = closest_note(notes, route, week_of, scores)
    if near is not None:
        reason = _near_miss_text(near, week_of) + " No genuine justification found; flagged for review. "
    reason += "Cost rise looks unexplained and worth a human review."
    return "Yes", "", reason.strip()

def retrieval_trace(notes, retriever, route, week_of, k=3):
    """For each top-k retrieved note: did it survive the hard filters?"""
    ok_ids = set(applicable_notes(notes, route, week_of)["note_id"])
    return [{"route": route, "week_of": str(week_of)[:10], "rank": rank, "note_id": nid,
             "similarity": round(sc, 3), "passes_hard_filters": nid in ok_ids}
            for rank, (nid, sc) in enumerate(retriever.search(route, week_of, k), 1)]