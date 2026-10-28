"""Mode definitions, deterministic translation, and feasibility validation (shared by all controllers)."""
BATTERY_MODES = {"IDLE": 0.0, "CHARGE_LOW": 0.25, "CHARGE_HIGH": 0.75, "DISCHARGE_LOW": -0.25, "DISCHARGE_HIGH": -0.75}
EV_MODES = ["MAX_AVAILABLE", "DEADLINE_AWARE", "DEFER"]

def available_battery_modes(obs, assets):
    can_ch = obs["battery_energy_kwh"] < obs["battery_max_energy_kwh"] - 1e-6
    can_dis = obs["battery_energy_kwh"] > obs["battery_min_energy_kwh"] + 1e-6
    return [m for m, f in BATTERY_MODES.items() if f == 0 or (f > 0 and can_ch) or (f < 0 and can_dis)]

def translate(battery_mode, ev_mode, obs):
    f = BATTERY_MODES[battery_mode]
    ch = f * obs["battery_charge_limit_kw"] if f > 0 else 0.0
    dis = -f * obs["battery_discharge_limit_kw"] if f < 0 else 0.0
    ev, budget = {}, obs["ev_site_limit_kw"]
    for g in sorted(obs["ev_groups"], key=lambda g: g["departure_hour"]):
        cap = min(g["max_charge_kw"], g["remaining_energy_kwh"] / obs["step_duration_hours"])
        if ev_mode == "DEADLINE_AWARE":
            rem_h = max(1, g["departure_hour"] - obs["hour"])
            want = min(cap, g["remaining_energy_kwh"] / rem_h)
        elif ev_mode == "MAX_AVAILABLE":
            want = cap
        else:
            want = 0.0
        ev[g["id"]] = min(want, budget); budget -= ev[g["id"]]
    return dict(battery_charge_kw=ch, battery_discharge_kw=dis, ev_kw=ev, solar_curtail_kw=0.0)

def validate(sp, obs, assets):
    sp = {**sp, "ev_kw": dict(sp["ev_kw"])}; iv = []; dt = obs["step_duration_hours"]
    eta_c, eta_d = assets["charge_efficiency"], assets["discharge_efficiency"]
    ch_max = max(0.0, (obs["battery_max_energy_kwh"] - obs["battery_energy_kwh"]) / (eta_c * dt))
    dis_max = max(0.0, (obs["battery_energy_kwh"] - obs["battery_min_energy_kwh"]) * eta_d / dt)
    if sp["battery_charge_kw"] > ch_max: sp["battery_charge_kw"] = ch_max; iv.append("charge_energy_bound")
    if sp["battery_discharge_kw"] > dis_max: sp["battery_discharge_kw"] = dis_max; iv.append("discharge_energy_bound")
    grid = lambda: (sum(sp["ev_kw"].values()) + sp["battery_charge_kw"] - sp["battery_discharge_kw"]
                    - (obs["solar_kw"] - sp["solar_curtail_kw"]))
    g = grid()
    if g > obs["grid_import_limit_kw"]:
        cut = min(sp["battery_charge_kw"], g - obs["grid_import_limit_kw"]); sp["battery_charge_kw"] -= cut
        g = grid(); iv.append("import_limit_reduce_charge")
        if g > obs["grid_import_limit_kw"]:
            tot = sum(sp["ev_kw"].values()); k = max(0.0, (tot - (g - obs["grid_import_limit_kw"])) / tot) if tot else 0.0
            sp["ev_kw"] = {i: v * k for i, v in sp["ev_kw"].items()}; iv.append("import_limit_reduce_ev")
    g = grid()
    if g < -obs["grid_export_limit_kw"] - 1e-9:
        excess = -g - obs["grid_export_limit_kw"]
        cut = min(sp["battery_discharge_kw"], excess)
        if cut > 0:
            sp["battery_discharge_kw"] -= cut; iv.append("export_limit_reduce_discharge")
        g = grid()
        if g < -obs["grid_export_limit_kw"] - 1e-9:
            sp["solar_curtail_kw"] = min(obs["solar_kw"], sp["solar_curtail_kw"] + (-g - obs["grid_export_limit_kw"]))
            iv.append("export_limit_curtail_solar")
    return sp, iv
