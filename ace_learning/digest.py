"""Episode digest: facts computed IN CODE from the public ledger, so the Reflector never has to do arithmetic.
Input is public_view() records only (no private simulator state, no decision_summary)."""
import re

def _avg(pairs):
    tot = sum(w for w, _ in pairs)
    return (sum(w * v for w, v in pairs) / tot) if tot > 1e-9 else None

def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"

def build_digest(summary, ledger, max_rows=18):
    rows = ledger
    ch = [(r["battery_charge_kw"], r["import_price_per_kwh"]) for r in rows if r["battery_charge_kw"] > 1e-9]
    ch_pct = [(r["battery_charge_kw"], r.get("import_price_percentile")) for r in rows if r["battery_charge_kw"] > 1e-9 and r.get("import_price_percentile") is not None]
    dis = [(r["battery_discharge_kw"], r["export_price_per_kwh"]) for r in rows if r["battery_discharge_kw"] > 1e-9]
    dis_pct = [(r["battery_discharge_kw"], r.get("import_price_percentile")) for r in rows if r["battery_discharge_kw"] > 1e-9 and r.get("import_price_percentile") is not None]
    ckwh, dkwh = sum(w for w, _ in ch), sum(w for w, _ in dis)
    pnl = sum(w * p for w, p in dis) - sum(w * p for w, p in ch)
    fee_rows = [r for r in rows if r["reward_components"].get("injection_fee", 0) > 1e-9]
    fee_total = sum(r["reward_components"]["injection_fee"] for r in rows)
    fee_export = sum(max(-r["grid_kw"], 0.0) for r in fee_rows)
    sold_in_fee = sum(r["battery_discharge_kw"] for r in fee_rows)
    idle = summary.get("idle_baseline_score")
    L = ["DAY RESULT",
         f"- score {summary['score']:.2f} (cash {summary['cash']:.2f}, end-of-day battery value {summary['terminal_adj']:.2f}); "
         f"battery energy {summary.get('initial_energy_kwh', 'n/a')} -> {summary['final_energy_kwh']} kWh",
         (f"- a do-nothing day (battery idle, EVs on deadline pace) would have scored {idle:.2f}; this day was {summary['score'] - idle:+.2f} versus that"
          if idle is not None else "- do-nothing reference not available"),
         f"- unmet EV energy {summary['unmet_ev_kwh']} kWh; thermal warnings {summary['thermal_events']}; fallbacks {summary.get('fallbacks', 0)}",
         "BATTERY",
         f"- non-idle hours {sum(r['battery_mode'] != 'IDLE' for r in rows)} of {len(rows)}",
         f"- charged {ckwh:.1f} kWh at average import price {_f(_avg(ch))} (average price percentile {_f(_avg(ch_pct), 2)})",
         f"- discharged {dkwh:.1f} kWh at average export price {_f(_avg(dis))} (average price percentile {_f(_avg(dis_pct), 2)})",
         f"- gross trade result (discharge kWh x export price - charge kWh x import price) {pnl:+.2f}",
         "CHARGES AND EVENTS SEEN IN THE REWARD LINES"]
    if fee_rows:
        L.append(f"- injection_fee (a CHARGE subtracted from reward, paid on exported energy) in hours {[r['hour'] for r in fee_rows]}: total {fee_total:.2f}"
                 + (f", about {fee_total / fee_export:.3f} per exported kWh" if fee_export > 1e-9 else "")
                 + f"; battery energy sold in those hours {sold_in_fee:.1f} kWh")
    else:
        L.append("- no injection fee charged")
    pen_rows = [r for r in rows if r["reward_components"].get("ev_unmet_penalty", 0) > 1e-9]
    if pen_rows:
        L.append(f"- ev_unmet_penalty (a CHARGE subtracted from reward, paid on exported energy) in hours {[r['hour'] for r in pen_rows]}: total {sum(r['reward_components']['ev_unmet_penalty'] for r in pen_rows):.2f}")
    for r in rows:
        if "ThermalDegradationWarning" in r["warnings"]:
            L.append(f"- hour {r['hour']}: ThermalDegradationWarning (battery {r['battery_temperature_c']} C, discharged {r['battery_discharge_kw']:.0f} kW, mode {r['battery_mode']})")
    lim = [r["hour"] for r in rows if "discharge_power_limit" in r["physical_clips"]]
    if lim: L.append(f"- discharge power was limited by the system in hours {lim} (modes requested: {sorted({r['battery_mode'] for r in rows if r['hour'] in lim})})")
    cut = [r["hour"] for r in rows if "export_limit_reduce_discharge" in r["interventions"] and r["battery_mode"].startswith("DISCHARGE")]
    if cut: L.append(f"- discharge requests reduced by the grid export limit in hours {cut}")
    used = {}
    for r in rows:
        for rid in r.get("applied_rule_ids", []): used.setdefault(rid, []).append(r)
    L.append("PLAYBOOK RULE USAGE")
    L.extend([f"- {rid} cited in hours {[x['hour'] for x in rs]}; reward in those hours {sum(x['reward'] for x in rs):+.2f}" for rid, rs in sorted(used.items())]
             or ["- no rule cited"])
    notable = [r for r in rows if r["battery_mode"] != "IDLE" or r["warnings"] or r["physical_clips"]
               or r["reward_components"].get("injection_fee", 0) > 1e-9 or r["interventions"]]
    L.append(f"NOTABLE HOURS ({len(notable)} of {len(rows)}; kW > 0 charging, < 0 discharging)")
    L.append("hour | price pct | import price | temp C | battery kW | grid kW | fee | reward | mode | flags")
    for r in notable[:max_rows]:
        kw = r["battery_charge_kw"] - r["battery_discharge_kw"]
        flags = ",".join(r["warnings"] + r["physical_clips"] + r["interventions"]) or "-"
        L.append(f"{r['hour']:>4} | {_f(r.get('import_price_percentile'), 2):>9} | {r['import_price_per_kwh']:.3f} | "
                f"{r['battery_temperature_c']:>6.1f} | {kw:+7.1f} | {r['grid_kw']:+7.1f} | "
                f"{r['reward_components'].get('injection_fee', 0):.2f} | {r['reward']:+7.2f} | {r['battery_mode']} | {flags}")
    for r in rows:                                   # add next to the existing warning lines
        if "ThermalDegradationWarning" in r["warnings"]:
            nxt = [x for x in rows if r["hour"] < x["hour"] <= r["hour"] + 3]
            L.append(f"- after the warning in hour {r['hour']}: " + "; ".join(
                f"hour {x['hour']} mode {x['battery_mode']} executed discharge {x['battery_discharge_kw']:.0f} kW" for x in nxt))
    if len(notable) > max_rows: L.append(f"... {len(notable) - max_rows} more notable hours omitted")
    return "\n".join(L)
