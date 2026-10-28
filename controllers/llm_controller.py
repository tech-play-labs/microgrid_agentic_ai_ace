"""LLM controller. Memoryless by default; the ACE controller subclasses it and supplies a playbook."""
import json, re
from typing import Literal
from pydantic import BaseModel, ConfigDict
from env.actions import available_battery_modes, BATTERY_MODES, EV_MODES
from llm.model_client import LLMError
from llm.budget import RoleBudget, complete_with_budget, BudgetExhausted

class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    battery_mode: Literal["IDLE", "CHARGE_LOW", "CHARGE_HIGH", "DISCHARGE_LOW", "DISCHARGE_HIGH"]
    ev_mode: Literal["MAX_AVAILABLE", "DEADLINE_AWARE", "DEFER"]
    applied_rule_ids: list[str]
    decision_summary: str

ACTION_SCHEMA = Action.model_json_schema()

def _round(x, n=2):
    if isinstance(x, float): return round(x, n)
    if isinstance(x, dict): return {k: _round(v, n) for k, v in x.items()}
    if isinstance(x, list): return [_round(v, n) for v in x]
    return x

def build_system_prompt(assets, eco, playbook=None):
    pen = eco["unmet_ev_penalty_per_kwh"]
    tv = eco["terminal_value_per_kwh"]
    loss = 1 - assets["charge_efficiency"] * assets["discharge_efficiency"]
    export_gap = 1 - eco["export_price_ratio"]
    battery_lines = "".join(
        f"  - {m}: {'do nothing' if f == 0 else ('charge' if f > 0 else 'discharge')}"
        f"{'' if f == 0 else f' at {abs(f)*100:.0f}% of rated power'}\n"
        for m, f in BATTERY_MODES.items())
    has_pb = bool(playbook and playbook.strip())
    playbook_block = (f"PLAYBOOK (lessons from earlier days; they may be incomplete or wrong; "
                      f"cite the ids of the rules you rely on in applied_rule_ids)\n{playbook.strip()}\n\n" if has_pb else "")
    rule_ids_text = ("applied_rule_ids (ids of the playbook rules you relied on this hour; empty list if none)" if has_pb
                     else "applied_rule_ids (always an empty list)")
    return (
"SITUATION\n"
"You operate a small commercial microgrid for one 24-hour day, making one decision per hour. "
"It has solar panels, a battery, EV chargers, and a connection to the public grid, where you can buy electricity (import price) and sell it (export price). "
"Prices, sunshine, and EV demand change from hour to hour.\n\n"
"GOAL\n"
"Maximize total net money over the whole day: the sum of the 24 hourly rewards (export revenue minus import cost, charges, and penalties), "
f"plus the value of the energy left in the battery at the end ({tv} per kWh).\n\n"
"POINTS TO CONSIDER (not a complete list)\n"
f"- EV groups: each group has an energy target and a departure_hour. Each kWh still missing at departure costs {pen}. "
"Each group shows hours_left and min_kw_needed_now (remaining energy / hours_left).\n"
"- Battery: it can store energy now and release it later. Storing pays off only when the energy is later used or sold "
f"for more than it cost to store; a charge-then-discharge round trip loses about {loss:.0%} of the energy.\n"
f"- Prices: they change through the day, and exports are paid about {export_gap:.0%} less than imports cost. "
"import_price_percentile compares this hour's import price with typical prices (0 = among the cheapest, 1 = among the most expensive).\n"
"- Time: hours_remaining counts the hours left in the day, including the current one.\n"
"- Solar and grid: solar can serve the site, charge the battery, or be exported. Export is limited, and solar above the limit is wasted.\n"
"- Timing: an action that costs money now can pay off in a later hour, and the reverse.\n"
"- Feedback: the recent history shows what was actually executed (power, interventions, clips) and the resulting reward_components, warnings, and efficiency.\n\n"
"OTHER FACTORS\n"
"The list above is not complete. The environment has other rules and effects that are not described here. "
"Infer them from the observation and the recent history, and adapt your choices.\n\n"
+ playbook_block +
"CHOICES (pick one of each, every hour)\n"
"- battery_mode:\n" + battery_lines +
"- ev_mode:\n"
"  - MAX_AVAILABLE: charge EV groups as fast as allowed\n"
"  - DEADLINE_AWARE: charge each group at its min_kw_needed_now rate\n"
"  - DEFER: no EV charging this hour (the energy is still needed later)\n"
"Known physical limits are enforced by code.\n\n"
"OUTPUT\n"
f"Reply with one JSON object only: battery_mode, ev_mode, {rule_ids_text}, decision_summary (one short sentence).")

OBS_KEYS = ("hour", "hours_remaining", "solar_kw", "battery_energy_kwh", "battery_capacity_kwh", "battery_temperature_c",
            "ambient_temperature_c", "import_price_per_kwh", "export_price_per_kwh", "import_price_percentile",
            "grid_import_limit_kw", "grid_export_limit_kw")

def build_user_prompt(obs, avail, recent):
    o = {k: obs[k] for k in OBS_KEYS if k in obs}
    o["ev_groups"] = obs["ev_groups"]
    return ("Current observation:\n" + json.dumps(_round(o)) + "\nAvailable battery modes: " + json.dumps(avail) +
            "\nRecent hours (oldest first):\n" + (json.dumps(_round(recent)) if recent else "[]"))

def extract_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i: raise ValueError("no JSON object found")
    return t[i:j + 1]

class LLMController:
    def __init__(self, client, assets, economics, budget=None, name="llm_memoryless", history_window=3, max_retries=1, playbook=None):
        self.client, self.assets, self.economics, self.name = client, assets, economics, name
        self.budget = budget if budget is not None else RoleBudget.from_config()
        self.window, self.max_retries = history_window, max_retries
        self.rule_ids = set()                       # ids the Generator may cite (empty => always [])
        self.set_playbook(playbook)
        self.usage = dict(calls=0, fallbacks=0, retries=0, prompt_tokens=0, completion_tokens=0, latency_s=0.0)
        self.last_meta = {}

    def set_playbook(self, text):
        self.playbook = text
        self.system = build_system_prompt(self.assets, self.economics, text)

    def select(self, obs, history=None):
        avail = available_battery_modes(obs, self.assets)
        user = build_user_prompt(obs, avail, (history or [])[-self.window:])
        meta = dict(attempts=0, fallback=False, error=None, latency_s=0.0, prompt_tokens=0, completion_tokens=0,
                    raw_text="", decision_summary="")
        prompt, action = user, None

        def _acc(r):                                                   # accounting moved here so it also
            meta["latency_s"] += r.latency_s                           # covers a call repeated after escalation
            meta["prompt_tokens"] += r.prompt_tokens
            meta["completion_tokens"] += r.completion_tokens
            meta["raw_text"] = r.text[:1000]

        for _ in range(self.max_retries + 1):
            meta["attempts"] += 1
            getattr(self.client, "set_context", lambda **k: None)(
                episode_id=obs["episode_id"], hour=obs["hour"], attempt=meta["attempts"])
            try:
                res = complete_with_budget(self.client, self.budget, "generator",
                                        self.system, prompt, ACTION_SCHEMA, on_result=_acc)
            except BudgetExhausted as e:
                meta["error"] = f"context: {e}"[:300]; break
            except LLMError as e:
                meta["error"] = f"backend: {e}"[:300]; continue
            try:
                a = Action.model_validate_json(extract_json(res.text))
                if a.battery_mode not in avail: raise ValueError(f"battery_mode {a.battery_mode} not in available {avail}")
                action, meta["error"] = a, None; break
            except ValueError as e:
                meta["error"] = f"invalid: {e}"[:300]
                prompt = user + f"\n\nYour previous reply was rejected ({meta['error'][:150]}). Reply again with valid JSON only."
        
        u = self.usage
        u["calls"] += meta["attempts"]; u["retries"] += meta["attempts"] - 1
        u["prompt_tokens"] += meta["prompt_tokens"]; u["completion_tokens"] += meta["completion_tokens"]; u["latency_s"] += meta["latency_s"]
        if action is None:
            meta["fallback"] = True; u["fallbacks"] += 1; self.last_meta = meta
            return dict(battery_mode="IDLE", ev_mode="DEADLINE_AWARE", applied_rule_ids=[])
        meta["decision_summary"] = action.decision_summary[:200]; self.last_meta = meta
        ids = [i for i in dict.fromkeys(action.applied_rule_ids) if i in self.rule_ids]    # unknown ids dropped
        return dict(battery_mode=action.battery_mode, ev_mode=action.ev_mode, applied_rule_ids=ids)
