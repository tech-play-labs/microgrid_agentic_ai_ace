"""Deterministic microgrid simulator. All physics and accounting live here (never in the LLM)."""
import bisect

class MicrogridEnv:
    def __init__(self, assets, economics, private_rules, scenario, price_reference=None):
        self.a, self.eco, self.rules, self.sc = assets, economics, private_rules, scenario
        self.dt = assets["step_hours"]
        self.price_ref = price_reference

    def reset(self):
        self.hour = 0
        e0 = self.sc.get("initial_energy_kwh")
        self.E = self.a["initial_energy_kwh"] if e0 is None else e0
        self.E0 = self.E
        self.T = self.a["initial_temperature_c"]
        self.pen_left = 0
        self.ev_rem = {g["id"]: g["required_energy_kwh"] for g in self.sc["ev_groups"]}
        self.unmet = {}
        return self._obs()

    def _connected(self, h):
        return [g for g in self.sc["ev_groups"] if g["arrival_hour"] <= h < g["departure_hour"]]

    def _obs(self):
        h, a, sc = self.hour, self.a, self.sc
        obs = dict(
            episode_id=f"{sc['family']}_{sc['seed']}", hour=h, hours_remaining=24 - h, step_duration_hours=self.dt,
            solar_kw=sc["solar_kw"][h], battery_energy_kwh=self.E, battery_capacity_kwh=a["capacity_kwh"],
            battery_min_energy_kwh=a["min_energy_kwh"], battery_max_energy_kwh=a["max_energy_kwh"],
            battery_temperature_c=round(self.T, 3), ambient_temperature_c=sc["ambient_c"][h],
            battery_charge_limit_kw=a["max_charge_kw"], battery_discharge_limit_kw=a["max_discharge_kw"],
            grid_import_limit_kw=a["grid"]["max_import_kw"], grid_export_limit_kw=a["grid"]["max_export_kw"],
            ev_site_limit_kw=a["grid"]["ev_site_limit_kw"],
            import_price_per_kwh=sc["import_price"][h], export_price_per_kwh=sc["export_price"][h],
            ev_groups=[dict(id=g["id"], remaining_energy_kwh=self.ev_rem[g["id"]], max_charge_kw=g["max_charge_kw"],
                            departure_hour=g["departure_hour"], hours_left=g["departure_hour"] - h,
                            min_kw_needed_now=self.ev_rem[g["id"]] / max(1, g["departure_hour"] - h)) for g in self._connected(h)])
        if self.price_ref:
            obs["import_price_percentile"] = round(
                bisect.bisect_right(self.price_ref, sc["import_price"][h]) / len(self.price_ref), 3)
        return obs

    def step(self, sp):
        a, eco, h, dt = self.a, self.eco, self.hour, self.dt
        tp, fee_cfg = self.rules["thermal_penalty"], self.rules["injection_fee"]
        solar = self.sc["solar_kw"][h]; amb = self.sc["ambient_c"][h]
        ch, dis = max(0.0, sp.get("battery_charge_kw", 0.0)), max(0.0, sp.get("battery_discharge_kw", 0.0))
        if ch > 1e-9 and dis > 1e-9:
            raise ValueError("simultaneous charge and discharge")
        eta_c = a["charge_efficiency"]
        eta_d = a["discharge_efficiency"] * (tp.get("efficiency_multiplier", 1.0) if self.pen_left > 0 else 1.0)
        clips = []
        ch_max = max(0.0, (a["max_energy_kwh"] - self.E) / (eta_c * dt))
        dis_max = max(0.0, (self.E - a["min_energy_kwh"]) * eta_d / dt)
        if ch > ch_max + 1e-9: ch = ch_max; clips.append("charge_energy_bound")
        if dis > dis_max + 1e-9: dis = dis_max; clips.append("discharge_energy_bound")
        cap_frac = tp.get("discharge_cap_fraction")
        if self.pen_left > 0 and cap_frac is not None:
            cap_kw = cap_frac * a["max_discharge_kw"]
            if dis > cap_kw + 1e-9: dis = cap_kw; clips.append("discharge_power_limit")
        E_new = self.E + eta_c * ch * dt - dis / eta_d * dt
        ev_total, ev_exec = 0.0, {}
        for g in self._connected(h):
            kw = min(max(0.0, sp.get("ev_kw", {}).get(g["id"], 0.0)), g["max_charge_kw"], self.ev_rem[g["id"]] / dt)
            ev_exec[g["id"]] = kw; ev_total += kw; self.ev_rem[g["id"]] -= kw * dt
        curtail = min(max(0.0, sp.get("solar_curtail_kw", 0.0)), solar)
        grid = ev_total + ch - dis - (solar - curtail)
        imp, exp = max(grid, 0.0), max(-grid, 0.0)
        violation = imp > a["grid"]["max_import_kw"] + 1e-6 or exp > a["grid"]["max_export_kw"] + 1e-6
        fee = fee_cfg["fee_per_exported_kwh"] * exp * dt if h in fee_cfg["applicable_hours"] else 0.0
        revenue = self.sc["export_price"][h] * exp * dt; cost = self.sc["import_price"][h] * imp * dt
        unmet_kwh = 0.0
        for g in self.sc["ev_groups"]:
            if g["departure_hour"] == h + 1:
                u = self.ev_rem[g["id"]]; self.unmet[g["id"]] = u; unmet_kwh += u
        ev_pen = eco["unmet_ev_penalty_per_kwh"] * unmet_kwh
        reward = revenue - cost - fee - ev_pen
        trigger = (self.T >= tp["temperature_threshold_c"] and dis >= tp["discharge_fraction_threshold"] * a["max_discharge_kw"] - 1e-9 and dis > 0)
        th = a["thermal"]
        T_new = self.T + th["k_heat"] * ((ch / a["max_charge_kw"]) ** 2 + (dis / a["max_discharge_kw"]) ** 2) - th["k_cool"] * (self.T - amb)
        eff_used = eta_d
        if self.pen_left > 0: self.pen_left -= 1
        warnings = []
        if trigger: self.pen_left = tp["duration_steps"]; warnings.append("ThermalDegradationWarning")
        info = dict(hour=h, executed=dict(battery_charge_kw=ch, battery_discharge_kw=dis, ev_kw=ev_exec, solar_curtail_kw=curtail),
                    grid_kw=grid, physical_clips=clips, grid_limit_violation=violation,
                    reward_components=dict(export_revenue=revenue, import_cost=cost, injection_fee=fee, ev_unmet_penalty=ev_pen),
                    warnings=warnings, discharge_efficiency_used=round(eff_used, 4),
                    private=dict(penalty_steps_left=self.pen_left, trigger=trigger))
        self.E, self.T = E_new, T_new; self.hour += 1
        done = self.hour >= 24
        return (None if done else self._obs()), reward, done, info

    def terminal_adjustment(self):
        mult = self.rules["thermal_penalty"].get("efficiency_multiplier", 1.0) if self.pen_left > 0 else 1.0
        return self.eco["terminal_value_per_kwh"] * (self.E * mult - self.E0)
