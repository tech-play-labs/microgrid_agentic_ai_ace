"""Controllers return modes only."""
class DoNothing:
    name = "do_nothing"
    last_meta = {}
    def select(self, obs, history=None):
        return dict(battery_mode="IDLE", ev_mode="DEADLINE_AWARE", applied_rule_ids=[])

class ThresholdHeuristic:
    name = "heuristic"
    def __init__(self, high_price=0.30, low_price=0.13, soc_floor=0.25, soc_ceiling=0.9):
        self.hp, self.lp, self.floor, self.ceil = high_price, low_price, soc_floor, soc_ceiling
        self.last_meta = {}
    def select(self, obs, history=None):
        soc = obs["battery_energy_kwh"] / obs["battery_capacity_kwh"]
        ev_demand = sum(g["max_charge_kw"] for g in obs["ev_groups"])
        if obs["import_price_per_kwh"] >= self.hp and soc > self.floor: bm = "DISCHARGE_HIGH"
        elif (obs["solar_kw"] > ev_demand or obs["import_price_per_kwh"] <= self.lp) and soc < self.ceil:
            bm = "CHARGE_HIGH" if obs["solar_kw"] > ev_demand else "CHARGE_LOW"
        else: bm = "IDLE"
        return dict(battery_mode=bm, ev_mode="DEADLINE_AWARE", applied_rule_ids=[])

class ThresholdHeuristicTrapAware(ThresholdHeuristic):
    """ORACLE reference (knows the hidden 35 C threshold)."""
    name = "heuristic_trap_aware"
    def __init__(self, hot_c=35.0, hot_high_price=0.60, **kw):
        super().__init__(**kw); self.hot_c, self.hot_high_price = hot_c, hot_high_price
    def select(self, obs, history=None):
        a = super().select(obs, history)
        if (a["battery_mode"] == "DISCHARGE_HIGH" and obs["battery_temperature_c"] >= self.hot_c
                and obs["import_price_per_kwh"] < self.hot_high_price):
            a["battery_mode"] = "DISCHARGE_LOW"
        return a
