
"""Where does a controller make or lose money with the battery?
Usage: python analyze_battery.py runs/RUN/llm_memoryless [family_prefix]
battery_trade_pnl is GROSS: discharge kWh x export price minus charge kWh x import price. It overstates the cost of
charging that actually came from surplus solar, so use it to compare controllers, not as exact accounting."""
import glob, json, os, sys

def analyze(ctrl_dir, fam=None):
    summ = {r["episode_id"]: r for r in json.load(open(os.path.join(ctrl_dir, "summary.json")))["rows"]}
    out = []
    for path in sorted(glob.glob(os.path.join(ctrl_dir, "*.public.jsonl"))):
        ep = os.path.basename(path).replace(".public.jsonl", "")
        if fam and not ep.startswith(fam): continue
        rows = [json.loads(l) for l in open(path)]
        ch = [(r["executed"]["battery_charge_kw"], r["obs"]["import_price_per_kwh"]) for r in rows if r["executed"]["battery_charge_kw"] > 1e-9]
        dis = [(r["executed"]["battery_discharge_kw"], r["obs"]["export_price_per_kwh"]) for r in rows if r["executed"]["battery_discharge_kw"] > 1e-9]
        ckwh, dkwh = sum(k for k, _ in ch), sum(k for k, _ in dis)
        out.append(dict(
            episode=ep, score=summ[ep]["score"], idle_hours=sum(r["modes"]["battery_mode"] == "IDLE" for r in rows),
            charged_kwh=round(ckwh, 1), avg_buy=round(sum(k * p for k, p in ch) / ckwh, 3) if ckwh else None,
            discharged_kwh=round(dkwh, 1), avg_sell=round(sum(k * p for k, p in dis) / dkwh, 3) if dkwh else None,
            battery_trade_pnl=round(sum(k * p for k, p in dis) - sum(k * p for k, p in ch), 2),
            fees=round(sum(r["reward_components"]["injection_fee"] for r in rows), 2),
            curtailed_kwh=round(sum(r["executed"]["solar_curtail_kw"] for r in rows), 1),
            terminal_adj=summ[ep]["terminal_adj"], thermal_events=summ[ep]["thermal_events"]))
    return out

if __name__ == "__main__":
    res = analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    if res:
        keys = list(res[0]); print(" | ".join(keys))
        for r in res: print(" | ".join(str(r[k]) for k in keys))
