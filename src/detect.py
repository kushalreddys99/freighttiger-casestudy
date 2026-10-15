import sys
import pandas as pd
from metrics import load, weekly, add_baselines

OWN_THRESHOLD = 8.0     # % above own trailing 8-week average
PEER_THRESHOLD = 15.0   # % above same-route_type peers that week

def candidates(w, own=OWN_THRESHOLD, peer=PEER_THRESHOLD):
    """Route-weeks that are rising AND out of the ordinary on both baselines."""
    ok = w["vs_own"].notna() & w["vs_peers"].notna()
    return w[ok & (w["vs_own"] > own) & (w["vs_peers"] > peer)].copy()

if __name__ == "__main__":
    w = add_baselines(weekly(load()))

    print("=== Distribution (to justify thresholds) ===")
    print(w[["vs_own", "vs_peers"]].describe().round(2))

    print("\n=== Sensitivity: how many flags at different thresholds ===")
    for own in (5, 8, 10, 15, 20):
        for peer in (10, 15, 20):
            print(f"own>{own:>2}%  peers>{peer}%  ->  {len(candidates(w, own, peer))} flags")

    print("\n=== Candidates at chosen thresholds ===")
    c = candidates(w)
    print(c[["route", "week_of", "cpt", "vs_own", "vs_peers"]].round(2).to_string(index=False))