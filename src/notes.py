import pandas as pd

# Hand-labeled from data/context_notes.csv. This is your ground truth.
#   scope:  a route name or "All Routes"
#   start/end: the window in which the note's effect applies (inclusive)
#   effect: "increases_cost" is the ONLY value that can justify a flag
NOTES = [
    dict(id="N001", scope="Chennai-Bangalore",   start="2025-02-24", end="2025-03-08", effect="increases_cost"),
    dict(id="N002", scope="Ahmedabad-Mumbai",    start="2025-01-20", end="2025-01-26", effect="increases_cost"),
    dict(id="N003", scope="All Routes",          start="2025-05-05", end="2025-06-29", effect="increases_cost"),
    dict(id="N004", scope="All Routes",          start="2024-03-11", end="2024-03-17", effect="not_applicable"),  # affected routes not in dataset
    dict(id="N005", scope="Mumbai-Delhi",        start="2024-07-29", end="2024-08-04", effect="no_effect"),       # "costs not significantly affected"
    dict(id="N006", scope="All Routes",          start="2025-09-22", end="2025-12-31", effect="no_effect"),       # "stable demand"
    dict(id="N007", scope="Delhi-Jaipur",        start="2024-05-20", end="2024-05-26", effect="no_effect"),       # road improved, not a cost rise
    dict(id="N008", scope="Kolkata-Bhubaneswar", start="2025-06-09", end="2025-06-30", effect="no_effect"),
    dict(id="N009", scope="Chennai-Bangalore",   start="2025-03-17", end="2025-03-23", effect="no_effect"),       # back to normal
    dict(id="N010", scope="All Routes",          start="2025-10-27", end="2025-10-27", effect="no_effect"),       # absorbed, no rate change
]

def load_notes(path="data/context_notes.csv"):
    """Merge hand labels with the source CSV, and fail if they ever disagree."""
    src = pd.read_csv(path)
    lab = pd.DataFrame(NOTES)
    assert set(src["note_id"]) == set(lab["id"]), "note ids differ from source CSV"
    m = src.merge(lab, left_on="note_id", right_on="id")
    assert (m["applies_to"] == m["scope"]).all(), "scope differs from source CSV"
    assert (pd.to_datetime(m["date"]) == pd.to_datetime(m["start"])).all(), "start differs from note date"
    m["start"] = pd.to_datetime(m["start"])
    m["end"] = pd.to_datetime(m["end"])
    return m.drop(columns=["id"])

def applicable_notes(notes, route, week_of):
    """Notes that could justify a rise: right route, window overlaps the week, effect is a cost rise."""
    wk_start = pd.Timestamp(week_of)
    wk_end = wk_start + pd.Timedelta(days=6)
    scope_ok = notes["scope"].isin([route, "All Routes"])
    window_ok = (notes["start"] <= wk_end) & (notes["end"] >= wk_start)
    effect_ok = notes["effect"] == "increases_cost"
    return notes[scope_ok & window_ok & effect_ok]