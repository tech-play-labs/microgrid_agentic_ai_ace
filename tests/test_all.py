import sys, os, math, json, copy, argparse, bisect, statistics, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from run import (load, load_price_reference, run_episode, public_view, write_manifest, episode_order, run_series, MAX_EVAL_SEED)
from env.simulator import MicrogridEnv
from env.scenarios import generate_scenario, FAMILIES
from controllers.baselines import DoNothing, ThresholdHeuristic, ThresholdHeuristicTrapAware
from env.actions import translate, validate
from controllers.llm_controller import LLMController, build_system_prompt
from llm.model_client import MockClient
from tools.make_price_reference import REFERENCE_SEEDS
from validation.compare_runs import compare

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "config")
A, ECO, R = load(CFG); PREF = load_price_reference(CFG)
def sc(seed=0, fam="clear_hot"): return generate_scenario(seed, fam, A["solar_peak_kw"], ECO["export_price_ratio"])
IDLE = dict(battery_charge_kw=0, battery_discharge_kw=0, ev_kw={}, solar_curtail_kw=0)
def rules_with(**kw):
    r = copy.deepcopy(R); r["thermal_penalty"].update(kw); return r
TUNING_SEEDS = list(range(200, 210))     # reserved for environment design only

# ---------- scenarios / seeds ----------
def test_old_families_unchanged_golden():
    s = sc(0, "clear_hot")
    assert (s["ambient_c"][0], s["import_price"][0], s["import_price"][8], s["solar_kw"][7]) == (28.26, 0.1082, 0.4357, 38.823)
    assert [(g["id"], g["required_energy_kwh"], g["departure_hour"]) for g in s["ev_groups"]][2] == ("EV_GROUP_03", 48.4, 16)
    assert "initial_energy_kwh" not in s

def spike_info(fam, seed):
    s = sc(seed, fam); env = MicrogridEnv(A, ECO, R, s); env.reset(); temps = []
    for _ in range(24): temps.append(env.T); env.step(IDLE)
    h = max(range(24), key=lambda i: s["import_price"][i]); return s, h, temps[h]

def test_hot_spike_coincides_with_hot_battery_and_low_solar():
    for seed in range(30):
        s, h, T = spike_info("hot_spike", seed)
        assert 17 <= h <= 19 and s["import_price"][h] >= 0.40 and T >= 35.0 and s["solar_kw"][h] <= 40.0
        assert s["initial_energy_kwh"] == 160.0

def test_mild_spike_is_control():
    for seed in range(30):
        s, h, T = spike_info("mild_spike", seed)
        assert s["import_price"][h] >= 0.40 and T < 35.0

def test_scenario_ev_feasible():
    for fam in FAMILIES:
        for s in range(20):
            for g in sc(s, fam)["ev_groups"]:
                assert g["required_energy_kwh"] <= g["max_charge_kw"] * (g["departure_hour"] - g["arrival_hour"])

def test_reserved_seed_ranges_do_not_overlap_evaluation_seeds():
    assert max(TUNING_SEEDS) >= MAX_EVAL_SEED and REFERENCE_SEEDS.start >= MAX_EVAL_SEED
    assert not set(TUNING_SEEDS) & set(REFERENCE_SEEDS)

def test_episode_order_is_round_robin():
    o = episode_order(["a", "b", "c"], range(2))
    assert o == [("a", 0), ("b", 0), ("c", 0), ("a", 1), ("b", 1), ("c", 1)] and len(o) == 3 * 2

# ---------- core physics / accounting ----------
def test_determinism():
    a = run_episode(ThresholdHeuristic(), A, ECO, R, sc(0, "hot_spike"), PREF)[0]
    assert a == run_episode(ThresholdHeuristic(), A, ECO, R, sc(0, "hot_spike"), PREF)[0]

@pytest.mark.parametrize("ctrl", [DoNothing, ThresholdHeuristic, ThresholdHeuristicTrapAware])
def test_invariants_all_scenarios(ctrl):
    for fam in FAMILIES:
        for s in range(8):
            _, led = run_episode(ctrl(), A, ECO, R, sc(s, fam), PREF)
            for l in led:
                assert not l["grid_limit_violation"]
                assert not (l["executed"]["battery_charge_kw"] > 0 and l["executed"]["battery_discharge_kw"] > 0)
                assert A["min_energy_kwh"] - 1e-6 <= l["obs"]["battery_energy_kwh"] <= A["max_energy_kwh"] + 1e-6

def test_energy_balance_and_reward():
    _, led = run_episode(ThresholdHeuristic(), A, ECO, R, sc(1, "cloudy"), PREF)
    for l in led:
        e, o = l["executed"], l["obs"]
        g = sum(e["ev_kw"].values()) + e["battery_charge_kw"] - e["battery_discharge_kw"] - (o["solar_kw"] - e["solar_curtail_kw"])
        assert math.isclose(g, l["grid_kw"], abs_tol=1e-9)
        c = l["reward_components"]
        assert math.isclose(l["reward"], c["export_revenue"] - c["import_cost"] - c["injection_fee"] - c["ev_unmet_penalty"], abs_tol=1e-9)

# ---------- thermal power cap ----------
def hot_env(rules):
    env = MicrogridEnv(A, ECO, rules, sc(0, "hot_spike"), PREF); env.reset(); env.T = 36.0; return env

def test_power_cap_window_blocks_then_releases():
    env = hot_env(R)
    _, _, _, i0 = env.step({**IDLE, "battery_discharge_kw": 75.0})
    assert "ThermalDegradationWarning" in i0["warnings"] and i0["executed"]["battery_discharge_kw"] == 75.0
    cap_kw = R["thermal_penalty"]["discharge_cap_fraction"] * A["max_discharge_kw"]; caps = []
    for _ in range(3):
        env.T = 20.0
        _, _, _, i = env.step({**IDLE, "battery_discharge_kw": 50.0}); caps.append((i["executed"]["battery_discharge_kw"], i["physical_clips"]))
    assert caps == [(cap_kw, ["discharge_power_limit"])] * 3
    env.T = 20.0
    _, _, _, i = env.step({**IDLE, "battery_discharge_kw": 50.0})
    assert i["executed"]["battery_discharge_kw"] == 50.0 and i["physical_clips"] == []

def test_power_cap_fraction_is_configurable():
    env = hot_env(rules_with(discharge_cap_fraction=0.25))
    env.step({**IDLE, "battery_discharge_kw": 75.0}); env.T = 20.0
    _, _, _, i = env.step({**IDLE, "battery_discharge_kw": 50.0})
    assert i["executed"]["battery_discharge_kw"] == 25.0

def test_low_discharge_never_triggers():
    _, _, _, i = hot_env(R).step({**IDLE, "battery_discharge_kw": 25.0})
    assert i["warnings"] == []

def test_terminal_adjustment_haircut_only_for_efficiency_penalty():
    env = MicrogridEnv(A, ECO, R, sc(0, "hot_spike")); env.reset(); base = env.terminal_adjustment(); env.pen_left = 2
    assert env.terminal_adjustment() == base
    env2 = MicrogridEnv(A, ECO, rules_with(efficiency_multiplier=0.8), sc(0, "hot_spike")); env2.reset(); b2 = env2.terminal_adjustment(); env2.pen_left = 2
    assert env2.terminal_adjustment() < b2

# ---------- validator / observation ----------
def obs_with(solar, energy=180.0, hour=9):
    o = MicrogridEnv(A, ECO, R, sc(0, "clear_mild")).reset()
    return {**o, "solar_kw": solar, "battery_energy_kwh": energy, "hour": hour, "ev_groups": []}

def test_export_limit_reduces_discharge_before_curtailing_solar():
    o = obs_with(60.0); sp, iv = validate(translate("DISCHARGE_HIGH", "DEFER", o), o, A)
    assert iv == ["export_limit_reduce_discharge"] and sp["solar_curtail_kw"] == 0.0 and abs(sp["battery_discharge_kw"] - 40.0) < 1e-9
    o = obs_with(106.066); sp, iv = validate(translate("DISCHARGE_HIGH", "DEFER", o), o, A)
    assert iv == ["export_limit_reduce_discharge", "export_limit_curtail_solar"] and sp["battery_discharge_kw"] == 0.0

def test_observation_new_fields_and_percentile():
    env = MicrogridEnv(A, ECO, R, sc(1, "hot_spike"), PREF); env.reset(); env.hour = 10
    o = env._obs()
    assert o["hours_remaining"] == 14 and 0.0 <= o["import_price_percentile"] <= 1.0
    for g in o["ev_groups"]:
        assert g["hours_left"] == g["departure_hour"] - 10
        assert abs(g["min_kw_needed_now"] - g["remaining_energy_kwh"] / g["hours_left"]) < 1e-9
    assert "import_price_percentile" not in MicrogridEnv(A, ECO, R, sc(1, "hot_spike")).reset()
    pct = lambda p: bisect.bisect_right(PREF, p) / len(PREF)
    assert PREF == sorted(PREF) and pct(0.10) < pct(0.20) < pct(0.30) < pct(0.50) and 0.85 <= pct(0.30) <= 0.97

def test_public_view_contents():
    _, led = run_episode(ThresholdHeuristic(), A, ECO, R, sc(0, "hot_spike"), PREF)
    v = public_view(led[18])
    for k in ("import_price_per_kwh", "import_price_percentile", "solar_kw", "grid_kw", "physical_clips", "interventions", "warnings"):
        assert k in v
    assert all("private" not in public_view(l) and "decision_summary" not in public_view(l) for l in led)

# ---------- prompt ----------
HIDDEN = ["thermal", "temperature_threshold", "derat", "0.08", "injection", "9 to 15", "35"]
def test_prompt_is_balanced_and_leak_free():
    p = build_system_prompt(A, ECO)
    for bad in HIDDEN: assert bad not in p, bad
    for good in ("hours_remaining", "import_price_percentile", "not a complete list", "OTHER FACTORS"): assert good in p
    for emph in ("PRIORITY", "never", "avoid", "main tool"): assert emph not in p

def test_playbook_is_only_difference():
    base = build_system_prompt(A, ECO); with_pb = build_system_prompt(A, ECO, "- R001 [candidate]: Example rule.")
    header = "PLAYBOOK (lessons from earlier days; they may be incomplete or wrong; cite the ids of the rules you rely on in applied_rule_ids)\n- R001 [candidate]: Example rule.\n\n"
    out_with = "applied_rule_ids (ids of the playbook rules you relied on this hour; empty list if none)"
    out_base = "applied_rule_ids (always an empty list)"
    assert "PLAYBOOK" not in base and with_pb.replace(header, "").replace(out_with, out_base) == base

# ---------- the ACE-style learning loop ----------
GOOD = json.dumps(dict(battery_mode="IDLE", ev_mode="DEADLINE_AWARE", applied_rule_ids=[], decision_summary="x"))

class Learner(LLMController):
    """Stand-in for the ACE controller: after each day it writes a note into the playbook."""
    def end_of_episode(self, summary, public_ledger):
        assert all("private" not in h for h in public_ledger)
        self.seen = getattr(self, "seen", []) + [summary["episode_id"]]
        self.set_playbook((self.playbook or "") + f"- lesson from {summary['episode_id']}\n")

def test_playbook_update_is_picked_up_by_the_next_episode_only(tmp_path):
    mock = MockClient(lambda s, u: GOOD); c = Learner(mock, A, ECO)
    order = episode_order(["clear_mild", "hot_spike"], range(2))          # 4 episodes
    rows = run_series(c, A, ECO, R, order, PREF, out_dir=str(tmp_path))
    assert [r["episode_index"] for r in rows] == [0, 1, 2, 3] and c.seen == [r["episode_id"] for r in rows]
    per_ep = {}
    for s, u in mock.calls: per_ep.setdefault(len(per_ep) // 24, s)        # system prompt of each episode's first call
    systems = [mock.calls[i * 24][0] for i in range(4)]
    assert "PLAYBOOK" not in systems[0]
    assert "lesson from clear_mild_0" in systems[1] and "lesson from hot_spike_0" not in systems[1]
    assert "lesson from hot_spike_0" in systems[2] and "lesson from clear_mild_1" not in systems[2]
    assert all(s == systems[i] for i in range(4) for s in [mock.calls[i * 24 + 23][0]])   # constant within a day
    assert (tmp_path / "playbook_after_clear_mild_0.txt").exists()

def test_memoryless_and_learner_identical_on_first_episode():
    base = LLMController(MockClient(lambda s, u: GOOD), A, ECO)
    ace = Learner(MockClient(lambda s, u: GOOD), A, ECO)
    order = episode_order(["hot_spike"], range(1))
    assert run_series(base, A, ECO, R, order, PREF)[0]["score"] == run_series(ace, A, ECO, R, order, PREF)[0]["score"]

def test_full_llm_episode_has_no_leak_and_new_fields():
    mock = MockClient(lambda s, u: GOOD); c = LLMController(mock, A, ECO)
    run_series(c, A, ECO, R, episode_order(["hot_spike"], range(1)), PREF)
    for s, u in mock.calls:
        for k in ["private", "penalty_steps_left", "temperature_threshold", "discharge_cap_fraction"]: assert k not in s and k not in u
    assert any("hours_remaining" in u and "import_price_percentile" in u for _, u in mock.calls)

# ---------- comparison script ----------
def fake_run(tmp_path, name, scores):
    d = tmp_path / name; d.mkdir(parents=True)
    fams = ["a", "b"]
    rows = [dict(episode_id=f"{fams[i % 2]}_{i // 2}", family=fams[i % 2], score=s, unmet_ev_kwh=0.0, thermal_events=0, episode_index=i)
            for i, s in enumerate(scores)]
    json.dump(dict(rows=rows, usage=dict(prompt_tokens=100)), open(d / "summary.json", "w")); return str(d)

def test_compare_runs_paired(tmp_path):
    a = fake_run(tmp_path, "A", [10, 10, 10, 10, 10, 10]); b = fake_run(tmp_path, "B", [10, 10, 12, 11, 14, 13])
    r = compare(a, b, n_boot=500)
    assert r["n"] == 6 and r["first_episode_same"] and r["wins"] == 4 and r["ties"] == 2 and r["losses"] == 0
    assert abs(r["mean_diff"] - 10 / 6) < 1e-9 and r["by_family"] == {"a": 2.0, "b": pytest.approx(4 / 3)}
    assert r["cumulative_advantage_by_round"] == [0.0, 3.0, 10.0] and r["first_half_diff"] < r["second_half_diff"]

# ---------- manifest / CLI ----------
def test_manifest_written(tmp_path):
    c = LLMController(MockClient(), A, ECO)
    write_manifest(str(tmp_path / "r"), argparse.Namespace(seeds=2), [c, DoNothing()], CFG)
    m = json.load(open(tmp_path / "r" / "manifest.json"))
    assert m["controllers"]["llm_memoryless"]["system_prompt"].startswith("SITUATION") and m["price_reference_loaded"] is True

def test_cli_rejects_reserved_seeds():
    r = subprocess.run([sys.executable, "run.py", "--controllers", "do_nothing", "--seeds", "201"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "reserved" in (r.stderr + r.stdout)

# ---------- acceptance test for the penalty design (reserved tuning seeds) ----------
def test_trap_aware_beats_naive_on_hot_spike_and_ties_on_control():
    seeds = TUNING_SEEDS
    mean = lambda ctrl, fam: statistics.mean(run_episode(ctrl, A, ECO, R, sc(s, fam))[0]["score"] for s in seeds)
    assert mean(ThresholdHeuristicTrapAware(), "hot_spike") >= mean(ThresholdHeuristic(), "hot_spike") * 1.10
    assert abs(mean(ThresholdHeuristicTrapAware(), "mild_spike") - mean(ThresholdHeuristic(), "mild_spike")) < 1e-9
