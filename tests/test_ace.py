import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
import llm.model_client as model_client
from llm.model_client import MockClient, LLMError
from run import (load, load_price_reference, run_series, episode_order, public_view, run_episode, make_controller, write_manifest)
from env.scenarios import generate_scenario
from controllers.baselines import DoNothing, ThresholdHeuristic
from controllers.llm_controller import LLMController, build_system_prompt
from ace_learning.playbook import Playbook, MAX_RULES, MAX_TEXT, MAX_RENDER_CHARS
from ace_learning.digest import build_digest
from controllers.ace_controller import ACEController

CFG = os.path.join(os.path.dirname(__file__), "..", "config")
A, ECO, R = load(CFG); PREF = load_price_reference(CFG)
RULE = "When the price percentile is high and the battery is hot, discharge at low power until the price peak."
def sc(seed=0, fam="hot_spike"): return generate_scenario(seed, fam, A["solar_peak_kw"], ECO["export_price_ratio"])

def act(ids=()): return json.dumps(dict(battery_mode="IDLE", ev_mode="DEADLINE_AWARE", applied_rule_ids=list(ids), decision_summary="x"))
REFLECT_OK = json.dumps(dict(insufficient_evidence=False, lessons=[dict(observation="o", evidence="e", confidence="high", suggested_rule=RULE)]))
REFLECT_NONE = json.dumps(dict(insufficient_evidence=True, lessons=[]))
def curate(*ops): return json.dumps(dict(ops=[dict(op=o, id=i, text=t) for o, i, t in ops]))
def router(gen=act(), refl=REFLECT_OK, cur=curate(("ADD", "", RULE))):
    pick = lambda x: x
    def f(system, user):
        if system.startswith("You review one finished day"): return pick(refl)
        if system.startswith("You maintain a short playbook"): return pick(cur)
        return gen
    return f
def ace(tmp_path, responder=None, **kw):
    mock = MockClient(responder or router())
    c = ACEController(mock, A, ECO, memory_file=str(tmp_path / "mem" / "memory.json"), log_path=str(tmp_path / "ace_log.jsonl"), **kw)
    return c, mock
def log(tmp_path):
    p = tmp_path / "ace_log.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []
ORDER2 = episode_order(["clear_mild", "hot_spike"], range(1))        # 2 episodes

# =============== playbook ===============
def test_playbook_lifecycle_and_ids_never_reused():
    pb = Playbook()
    ap, rj = pb.apply([dict(op="ADD", id="", text="Rule one text about prices.")], "e0"); assert not rj and pb.get("R001").status == "candidate"
    pb.apply([dict(op="SUPPORT", id="R001", text="")], "e1"); assert pb.get("R001").status == "supported"
    pb.apply([dict(op="UPDATE", id="R001", text="Rule one improved.")], "e2"); assert pb.get("R001").text == "Rule one improved."
    pb.apply([dict(op="DEPRECATE", id="R001", text="")], "e3"); assert pb.active() == [] and pb.render() == ""
    pb.apply([dict(op="ADD", id="", text="A different rule about hours of the day.")], "e4"); assert pb.get("R002") is not None
    assert [r.id for r in pb.rules] == ["R001", "R002"]

@pytest.mark.parametrize("bad", ["", "x" * (MAX_TEXT + 1), "In clear_hot_3 do this", "Because of seed 4 do that", "On episode 12 it failed"])
def test_playbook_rejects_bad_text(bad):
    pb = Playbook(); ap, rj = pb.apply([dict(op="ADD", id="", text=bad)], "e0")
    assert not ap and len(rj) == 1 and pb.rules == []

def test_playbook_rejects_unknown_ids_ops_and_never_raises():
    pb = Playbook(); pb.apply([dict(op="ADD", id="", text="First rule text here.")], "e0")
    ap, rj = pb.apply([dict(op="UPDATE", id="R999", text="Hello there friend."), dict(op="SUPPORT", id="R404", text=""),
                       dict(op="FLY", id="", text=""), dict(), dict(op="ADD", id="", text=None)], "e1")
    assert not ap and len(rj) == 5 and [r.id for r in pb.rules] == ["R001"]

def test_playbook_merges_near_duplicates():
    pb = Playbook(); pb.apply([dict(op="ADD", id="", text="Discharge at low power when the battery is hot.")], "e0")
    ap, rj = pb.apply([dict(op="ADD", id="", text="Discharge at low power when the battery is hot!")], "e1")
    assert len(pb.rules) == 1 and pb.get("R001").support == 2 and "merged" in ap[0]["note"]

def test_playbook_capacity_and_text_budget():
    pb = Playbook(); texts = [("Distinct long rule number %d " % i) + ("abcdefghij" * 20) for i in range(MAX_RULES + 2)]
    ap, rj = pb.apply([dict(op="ADD", id="", text=t[:MAX_TEXT]) for t in texts], "e0")
    assert len(pb.active()) <= MAX_RULES and len(pb.render()) <= MAX_RENDER_CHARS
    assert any("budget" in r["reason"] or "full" in r["reason"] for r in rj)

def test_playbook_version_changes_only_when_prompt_text_changes():
    pb = Playbook(); pb.apply([dict(op="ADD", id="", text="A rule about prices at night.")], "e0"); v1 = pb.version
    pb.apply([dict(op="SUPPORT", id="R001", text="")], "e1"); v2 = pb.version      # candidate -> supported changes the text
    pb.apply([dict(op="SUPPORT", id="R001", text="")], "e2"); v3 = pb.version      # still supported: text identical
    assert (v1, v2, v3) == (1, 2, 2)

def test_playbook_auto_deprecates_contradicted_rules():
    pb = Playbook(); pb.apply([dict(op="ADD", id="", text="A rule that will turn out wrong.")], "e0")
    pb.apply([dict(op="CONTRADICT", id="R001", text="")], "e1"); assert pb.get("R001").status == "candidate"
    pb.apply([dict(op="CONTRADICT", id="R001", text="")], "e2"); assert pb.get("R001").deprecated

def test_playbook_persistence_roundtrip_and_errors(tmp_path):
    pb = Playbook(); pb.apply([dict(op="ADD", id="", text="Persisted rule text number one.")], "e0")
    p = str(tmp_path / "d" / "m.json"); pb.save(p)
    assert not os.path.exists(p + ".tmp") and Playbook.load(p).render() == pb.render() and Playbook.load(p).next_id == 2
    open(p, "w").write("")
    with pytest.raises(ValueError): Playbook.load(p)
    open(p, "w").write(json.dumps({"version": 1}))
    with pytest.raises(ValueError): Playbook.load(p)

# =============== digest ===============
def real_ledger(ctrl, fam="hot_spike", seed=0):
    s, led = run_episode(ctrl, A, ECO, R, sc(seed, fam), PREF)
    s["idle_baseline_score"] = run_episode(DoNothing(), A, ECO, R, sc(seed, fam), PREF)[0]["score"]
    return s, [public_view(l) for l in led]

def test_digest_contains_computed_facts_and_nothing_private():
    s, led = real_ledger(ThresholdHeuristic())
    d = build_digest(s, led)
    for k in ("DAY RESULT", "BATTERY", "do-nothing day", "injection_fee in hours", "per exported kWh", "NOTABLE HOURS"): assert k in d, k
    assert "ThermalDegradationWarning" in d and "discharge power was limited" in d
    for bad in ("decision_summary", "private", "penalty_steps_left", "hot_spike_0", "seed"): assert bad not in d, bad
    assert len(d) < 6000
    assert "about 0.080 per exported kWh" in d            # fee rate derived from public reward lines only

def test_digest_rule_usage_section():
    s, led = real_ledger(DoNothing(), "clear_mild"); led[5]["applied_rule_ids"] = ["R001"]; led[6]["applied_rule_ids"] = ["R001"]
    d = build_digest(s, led); assert "R001 cited in hours [5, 6]" in d
    assert "no rule cited" in build_digest(s, real_ledger(DoNothing(), "clear_mild")[1])

# =============== ACE loop ===============
def test_playbook_is_learned_saved_and_used_by_the_next_episode_only(tmp_path):
    c, mock = ace(tmp_path)
    rows = run_series(c, A, ECO, R, ORDER2, PREF, out_dir=str(tmp_path / "out"))
    assert [r["playbook_version"] for r in rows] == [0, 1] and "idle_baseline_score" in rows[0]
    first = [s for s, u in mock.calls if not s.startswith(("You review", "You maintain"))]
    assert first[0] == build_system_prompt(A, ECO)                       # day 1 prompt identical to the memoryless prompt
    assert "PLAYBOOK" not in first[0] and "PLAYBOOK" in first[-1] and "R001" in first[-1] and "ids of the playbook rules" in first[-1]
    assert Playbook.load(str(tmp_path / "mem" / "memory.json")).get("R001").text == RULE
    assert (tmp_path / "out" / "playbook_after_hot_spike_0.txt").exists()
    stages = [(r["stage"], r["status"]) for r in log(tmp_path)]
    assert ("reflect", "ok") in stages and ("apply", "ok") in stages
    assert c.usage["ace_calls"] == 4 and c.usage["ace_failures"] == 0 and c.usage["ace_prompt_tokens"] == 40

def test_generator_rule_ids_are_filtered_to_existing_rules(tmp_path):
    c, _ = ace(tmp_path, router(gen=act(["R001", "R999"])))
    rows_ids = []
    orig = c.select
    def spy(obs, hist=None):
        out = orig(obs, hist); rows_ids.append((obs["episode_id"], out["applied_rule_ids"])); return out
    c.select = spy
    run_series(c, A, ECO, R, ORDER2, PREF)
    assert {i for e, i in rows_ids if e == "clear_mild_0"} == {()} if False else all(i == [] for e, i in rows_ids if e == "clear_mild_0")
    assert all(i == ["R001"] for e, i in rows_ids if e == "hot_spike_0")

def test_memoryless_controller_always_returns_empty_rule_ids():
    c = LLMController(MockClient(lambda s, u: act(["R001"])), A, ECO)
    assert "always an empty list" in c.system and "PLAYBOOK" not in c.system
    rows = run_series(c, A, ECO, R, episode_order(["clear_mild"], range(1)), PREF)
    assert rows[0]["fallbacks"] == 0

@pytest.mark.parametrize("refl", ["garbage", None])
def test_reflector_failure_keeps_old_playbook_and_run_continues(tmp_path, refl):
    resp = router(refl=(LLMError("backend down") if refl is None else refl))
    c, mock = ace(tmp_path, resp)
    rows = run_series(c, A, ECO, R, ORDER2, PREF)
    assert len(rows) == 2 and c.memory.version == 0 and c.memory.rules == [] and "PLAYBOOK" not in c.system
    errs = [r for r in log(tmp_path) if r["status"] == "error"]
    assert errs and errs[0]["stage"] == "reflect" and errs[0]["action"] == "kept old playbook" and c.usage["ace_failures"] == 2
    assert not any(s.startswith("You maintain") for s, u in mock.calls)         # curator not reached

def test_curator_failure_keeps_old_playbook_and_run_continues(tmp_path):
    c, _ = ace(tmp_path, router(cur="{not json"))
    rows = run_series(c, A, ECO, R, ORDER2, PREF)
    assert len(rows) == 2 and c.memory.rules == [] and not os.path.exists(c.memory_file)
    assert any(r["stage"] == "curate" and r["status"] == "error" for r in log(tmp_path))

def test_curator_partially_invalid_ops_apply_the_valid_ones(tmp_path):
    cur = curate(("ADD", "", RULE), ("ADD", "", "Because of seed 3 always sell."), ("UPDATE", "R999", "Whatever text here."))
    c, _ = ace(tmp_path, router(cur=cur)); run_series(c, A, ECO, R, episode_order(["clear_mild"], range(1)), PREF)
    ap = [r for r in log(tmp_path) if r["stage"] == "apply"][0]
    assert len(ap["applied"]) == 1 and len(ap["rejected"]) == 2 and [r.id for r in c.memory.rules] == ["R001"]

def test_insufficient_evidence_skips_the_curator(tmp_path):
    c, mock = ace(tmp_path, router(refl=REFLECT_NONE)); run_series(c, A, ECO, R, ORDER2, PREF)
    assert not any(s.startswith("You maintain") for s, u in mock.calls) and c.memory.rules == []
    assert any(r["stage"] == "curate" and r["status"] == "skipped" for r in log(tmp_path))

def test_memory_save_failure_is_logged_and_playbook_still_used(tmp_path):
    blocker = tmp_path / "afile"; blocker.write_text("x")
    c = ACEController(MockClient(router()), A, ECO, memory_file=str(blocker / "memory.json"), log_path=str(tmp_path / "ace_log.jsonl"))
    rows = run_series(c, A, ECO, R, ORDER2, PREF)
    assert len(rows) == 2 and "R001" in c.system and any(r["stage"] == "save" and r["status"] == "error" for r in log(tmp_path))

def test_unexpected_exception_in_learning_never_propagates(tmp_path):
    c, _ = ace(tmp_path)
    c.end_of_episode({"episode_id": "x"}, [{}])                 # malformed input
    assert c.usage["ace_failures"] == 1 and log(tmp_path)[0]["stage"] == "end_of_episode"

def test_separate_reflector_client_is_used_only_for_reflection(tmp_path):
    gen = MockClient(lambda s, u: act()); refl = MockClient(router())
    c = ACEController(gen, A, ECO, reflector_client=refl, memory_file=str(tmp_path / "m.json"), log_path=str(tmp_path / "l.jsonl"))
    run_series(c, A, ECO, R, ORDER2, PREF)
    assert all(not s.startswith(("You review", "You maintain")) for s, u in gen.calls)
    assert refl.calls and all(s.startswith(("You review", "You maintain")) for s, u in refl.calls)
    assert c.describe()["reflector_same_client_as_generator"] is False and refl.context["stage"].startswith("ace_")

def test_default_reflector_is_the_generator_client(tmp_path):
    c, _ = ace(tmp_path); assert c.reflector_client is c.client and c.describe()["reflector_same_client_as_generator"] is True

def test_existing_memory_file_is_loaded_and_used_from_the_first_episode(tmp_path):
    c1, _ = ace(tmp_path); run_series(c1, A, ECO, R, ORDER2, PREF)
    c2 = ACEController(MockClient(act()), A, ECO, memory_file=c1.memory_file, log_path=str(tmp_path / "l2.jsonl"))
    assert c2.rules_loaded == 1 and "R001" in c2.system and c2.playbook_version == c1.memory.version

def test_corrupted_memory_file_fails_fast_at_startup(tmp_path):
    p = tmp_path / "bad.json"; p.write_text("{oops")
    with pytest.raises(ValueError): ACEController(MockClient(act()), A, ECO, memory_file=str(p))

# =============== wiring (make_controller / CLI / manifest) ===============
def fake_loader(monkeypatch):
    calls = []
    def load_client(path="config/model.json", profile=None):
        c = MockClient(router()); c.profile = profile; calls.append(c); return c
    monkeypatch.setattr(model_client, "load_client", load_client); return calls

def test_make_controller_default_and_custom_memory_paths(monkeypatch, tmp_path):
    calls = fake_loader(monkeypatch); monkeypatch.chdir(tmp_path)
    c = make_controller("llm_ace", A, ECO, run_name="myrun")
    assert c.name == "llm_ace" and c.memory_file == "runs/myrun/llm_ace/memory.json" and c.log_path == "runs/myrun/llm_ace/ace_log.jsonl"
    c2 = make_controller("llm_ace", A, ECO, memory_file=str(tmp_path / "custom.json"), run_name="myrun")
    assert c2.memory_file == str(tmp_path / "custom.json")

def test_make_controller_reflector_profile(monkeypatch, tmp_path):
    calls = fake_loader(monkeypatch); monkeypatch.chdir(tmp_path)
    same = make_controller("llm_ace", A, ECO, profile="ollama_qwen3-8b", run_name="r1")
    assert same.reflector_client is same.client and [c.profile for c in calls] == ["ollama_qwen3-8b"]
    calls.clear()
    sep = make_controller("llm_ace", A, ECO, profile="ollama_qwen3-8b", reflector_profile="groq", run_name="r2")
    assert sep.reflector_client is not sep.client and [c.profile for c in calls] == ["ollama_qwen3-8b", "groq"]

def test_make_controller_bad_memory_file_exits_cleanly(monkeypatch, tmp_path):
    fake_loader(monkeypatch); p = tmp_path / "bad.json"; p.write_text("nope")
    with pytest.raises(SystemExit) as e: make_controller("llm_ace", A, ECO, memory_file=str(p), run_name="r")
    assert "ACE memory error" in str(e.value)

def test_make_controller_memoryless_unchanged(monkeypatch):
    fake_loader(monkeypatch); c = make_controller("llm", A, ECO)
    assert c.name == "llm_memoryless" and not hasattr(c, "end_of_episode")

def test_cli_help_lists_new_options():
    import subprocess
    out = subprocess.run([sys.executable, "run.py", "--help"], cwd=os.path.join(os.path.dirname(__file__), ".."), capture_output=True, text=True).stdout
    assert "--reflector-profile" in out and "--memory-file" in out and "llm_ace" in out

def test_manifest_records_ace_details(tmp_path):
    c, _ = ace(tmp_path); import argparse
    write_manifest(str(tmp_path / "r"), argparse.Namespace(seeds=1), [c], CFG)
    m = json.load(open(tmp_path / "r" / "manifest.json"))["controllers"]["llm_ace"]
    assert m["system_prompt"].startswith("SITUATION") and m["rules_loaded_at_start"] == 0 and m["reflector_same_client_as_generator"] is True

# =============== regressions ===============
def test_all_controllers_respect_invariants_with_idle_baseline():
    order = episode_order(["clear_mild", "clear_hot", "cloudy", "low_solar", "hot_spike", "mild_spike"], range(2))
    for ctrl in (DoNothing(), ThresholdHeuristic()):
        rows = run_series(ctrl, A, ECO, R, order, PREF)
        assert all(r["grid_violations"] == 0 and "idle_baseline_score" in r for r in rows)
    assert all(abs(r["score"] - r["idle_baseline_score"]) < 1e-9 for r in run_series(DoNothing(), A, ECO, R, order, PREF))

def test_public_view_new_fields():
    _, led = run_episode(ThresholdHeuristic(), A, ECO, R, sc(), PREF)
    v = public_view(led[3]); assert "export_price_per_kwh" in v and "applied_rule_ids" not in v
    led[3]["modes"]["applied_rule_ids"] = ["R001"]; assert public_view(led[3])["applied_rule_ids"] == ["R001"]

def test_golden_values_unchanged():
    s = sc(0, "clear_hot")
    assert (s["ambient_c"][0], s["import_price"][0], s["import_price"][8], s["solar_kw"][7]) == (28.26, 0.1082, 0.4357, 38.823)
