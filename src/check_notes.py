from metrics import load, weekly, add_baselines
from detect import candidates
from notes import load_notes, applicable_notes

w = add_baselines(weekly(load()))
notes = load_notes()
for _, r in candidates(w).iterrows():
    hits = applicable_notes(notes, r["route"], r["week_of"])["note_id"].tolist()
    print(f'{r["route"]:<18} {r["week_of"].date()}  ->  {hits or "UNEXPLAINED"}')