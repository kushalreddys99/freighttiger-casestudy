import pandas as pd

def load(path="data/shipment_records.csv"):
    d = pd.read_csv(path, parse_dates=["shipment_date"])
    assert d.notna().all().all()
    assert (d[["quantity_tonnes", "distance_km", "freight_cost_inr"]] > 0).all().all()
    d["route"] = d.origin + "-" + d.destination
    assert d.groupby("route").route_type.nunique().max() == 1
    return d

def weekly(d):
    d = d.copy()
    d["week_of"] = d.shipment_date - pd.to_timedelta(d.shipment_date.dt.weekday, unit="D")
    d["tonne_km"] = d.quantity_tonnes * d.distance_km
    w = (d.groupby(["route", "route_type", "week_of"], as_index=False)
           .agg(cost=("freight_cost_inr", "sum"), tonne_km=("tonne_km", "sum")))
    w["cpt"] = w.cost / w.tonne_km          # ratio of sums, as the brief defines it
    return w.sort_values(["route", "week_of"]).reset_index(drop=True)

def add_baselines(w):
    # own history: trailing 8 prior weeks, strictly before current week
    w["own_avg"] = w.groupby("route")["cpt"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    w["vs_own"] = (w["cpt"] / w["own_avg"] - 1) * 100

    # peers: same route_type, same week, excluding the route itself
    g = w.groupby(["route_type", "week_of"])["cpt"]
    w["peer_avg"] = (g.transform("sum") - w["cpt"]) / (g.transform("count") - 1)
    w["vs_peers"] = (w["cpt"] / w["peer_avg"] - 1) * 100
    return w
if __name__ == "__main__":
    w = add_baselines(weekly(load()))
    print(w[(w["route"] == "Delhi-Jaipur") & (w["week_of"] == "2024-11-11")])