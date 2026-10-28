"""Episode runner and CLI. Controllers only ever see public_view() records.
Episodes = families x seeds, run in ROUND-ROBIN order so a learning controller sees every family early and late."""
import argparse, datetime, json, os, statistics, subprocess, sys
from env.simulator import MicrogridEnv
from env.scenarios import generate_scenario, FAMILIES
from env.actions import translate, validate
from controllers.baselines import DoNothing, ThresholdHeuristic, ThresholdHeuristicTrapAware
from llm.budget import RoleBudget

MAX_EVAL_SEED = 200      # seeds >= 200 are reserved (penalty tuning 200-209, price reference 1000-1199)

def load(root="config"):
    r = lambda n: json.load(open(f"{root}/{n}.json"))
    return r("assets"), r("economics"), r("private_environment_rules")

def load_price_reference(root="config"):
    p = os.path.join(root, "price_reference.json")
    return json.load(open(p))["quantiles"] if os.path.exists(p) else None

def public_view(rec):
    """Whitelist of what a controller may see about a past hour. Never includes rec['private'] or decision_summary."""
    rc = {k: round(v, 4) for k, v in rec["reward_components"].items()}
    o = rec["obs"]
    v = dict(hour=rec["hour"], import_price_per_kwh=o["import_price_per_kwh"], export_price_per_kwh=o["export_price_per_kwh"],
             solar_kw=o["solar_kw"], battery_temperature_c=o["battery_temperature_c"],
             battery_mode=rec["modes"]["battery_mode"], ev_mode=rec["modes"]["ev_mode"],
             battery_charge_kw=round(rec["executed"]["battery_charge_kw"], 2),
             battery_discharge_kw=round(rec["executed"]["battery_discharge_kw"], 2),
             grid_kw=round(rec["grid_kw"], 2), reward_components=rc, reward=round(rec["reward"], 4),
             warnings=rec["warnings"], discharge_efficiency_used=rec["discharge_efficiency_used"],
             interventions=rec["interventions"], physical_clips=rec["physical_clips"])
    if "import_price_percentile" in o: v["import_price_percentile"] = o["import_price_percentile"]
    if rec["modes"].get("applied_rule_ids"): v["applied_rule_ids"] = rec["modes"]["applied_rule_ids"]
    return v

def episode_order(families, seeds):
    return [(f, s) for s in seeds for f in families]

def run_episode(controller, assets, eco, rules, scenario, price_reference=None):
    env = MicrogridEnv(assets, eco, rules, scenario, price_reference); obs = env.reset()
    ledger, hist, total, done = [], [], 0.0, False
    while not done:
        modes = controller.select(obs, hist)
        req = translate(modes["battery_mode"], modes["ev_mode"], obs)
        sp, iv = validate(req, obs, assets)
        nxt, r, done, info = env.step(sp); total += r
        rec = dict(obs=obs, modes=modes, requested=req, validated=sp, interventions=iv, reward=r,
                   controller_meta=dict(getattr(controller, "last_meta", {})), **info)
        ledger.append(rec); hist.append(public_view(rec)); obs = nxt
    term = env.terminal_adjustment()
    summary = dict(episode_id=f"{scenario['family']}_{scenario['seed']}", family=scenario["family"], controller=controller.name,
                   cash=round(total, 4), terminal_adj=round(term, 4), score=round(total + term, 4),
                   thermal_events=sum("ThermalDegradationWarning" in l["warnings"] for l in ledger),
                   unmet_ev_kwh=round(sum(env.unmet.values()), 3), final_energy_kwh=round(env.E, 2),
                   initial_energy_kwh=round(env.E0, 2),
                   validator_interventions=sum(len(l["interventions"]) for l in ledger),
                   fallbacks=sum(bool(l["controller_meta"].get("fallback", False)) for l in ledger),
                   grid_violations=sum(l["grid_limit_violation"] for l in ledger))
    return summary, ledger

def write_episode(out_dir, summary, ledger):
    os.makedirs(out_dir, exist_ok=True); ep = summary["episode_id"]
    with open(f"{out_dir}/{ep}.public.jsonl", "w") as pub, open(f"{out_dir}/{ep}.private.jsonl", "w") as prv:
        for l in ledger:
            l = dict(l); p = l.pop("private"); pub.write(json.dumps(l) + "\n"); prv.write(json.dumps(dict(hour=l["hour"], **p)) + "\n")

def run_series(ctrl, assets, eco, rules, order, price_reference=None, out_dir=None, verbose=False):
    """Run episodes in order. After each day a controller exposing end_of_episode(summary, public_ledger) may update itself
    (ACE: Reflector + Curator update the playbook); the NEXT episode then uses the new playbook.
    Every summary also carries idle_baseline_score: the same day played by DoNothing (a reference number, nothing else)."""
    rows = []
    for idx, (fam, seed) in enumerate(order):
        sc = generate_scenario(seed, fam, assets["solar_peak_kw"], eco["export_price_ratio"])
        s, led = run_episode(ctrl, assets, eco, rules, sc, price_reference)
        s["episode_index"] = idx
        s["playbook_version"] = getattr(ctrl, "playbook_version", None)          # version USED during this episode
        s["idle_baseline_score"] = run_episode(DoNothing(), assets, eco, rules, sc, price_reference)[0]["score"]
        if out_dir: write_episode(out_dir, s, led)
        rows.append(s)
        if hasattr(ctrl, "end_of_episode"):
            ctrl.end_of_episode(dict(s), [public_view(l) for l in led])          # public data only
            if out_dir and getattr(ctrl, "playbook", None):
                open(os.path.join(out_dir, f"playbook_after_{s['episode_id']}.txt"), "w").write(ctrl.playbook)
        if verbose: print(s, flush=True)
    return rows

def write_manifest(run_dir, args, controllers, config_root="config"):
    """Everything needed to reproduce / audit a run (kept out of any prompt)."""
    os.makedirs(run_dir, exist_ok=True)
    cfg = {}
    for n in sorted(os.listdir(config_root)):
        if not n.endswith(".json") or n == "price_reference.json":
            continue
        try:
            cfg[n[:-5]] = json.load(open(os.path.join(config_root, n)))
        except json.JSONDecodeError:
            cfg[n[:-5]] = "UNREADABLE_OR_EMPTY"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    man = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), argv=sys.argv, args=vars(args),
               git_commit=commit, python=sys.version.split()[0], configs=cfg,
               price_reference_loaded=load_price_reference(config_root) is not None,
               controllers={c.name: dict(system_prompt=getattr(c, "system", None), **getattr(c, "describe", lambda: {})())
                            for c in controllers})
    json.dump(man, open(os.path.join(run_dir, "manifest.json"), "w"), indent=1)

def make_controller(name, assets, eco, profile=None, reflector_profile=None, memory_file=None, run_name="run"):
    budget = RoleBudget.from_config("config/model.json")
    if name == "do_nothing": return DoNothing()
    if name == "heuristic": return ThresholdHeuristic()
    if name == "heuristic_trap_aware": return ThresholdHeuristicTrapAware()
    if name in ("llm", "llm_ace"):
        from llm.model_client import load_client
        gen = load_client("config/model.json", profile)
        if name == "llm":
            from controllers.llm_controller import LLMController
            return LLMController(gen, assets, eco, budget)
        from controllers.ace_controller import ACEController
        # separate Reflector/Curator model only when asked; otherwise the generator's client is reused
        refl = gen if not reflector_profile else load_client("config/model.json", reflector_profile)
        run_dir = f"runs/{run_name}/llm_ace"
        mem = memory_file or f"{run_dir}/memory.json"                          # default: one memory per run, starts empty
        try:
            return ACEController(gen, assets, eco, budget, reflector_client=refl, memory_file=mem, log_path=f"{run_dir}/ace_log.jsonl")
        except ValueError as e:
            raise SystemExit(f"ACE memory error: {e}")
    raise SystemExit(f"unknown controller {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", default="heuristic", help="comma list of: do_nothing, heuristic, heuristic_trap_aware, llm, llm_ace")
    ap.add_argument("--profile", default=None, help="model profile for the Generator (and the Reflector/Curator unless --reflector-profile)")
    ap.add_argument("--reflector-profile", default=None, help="separate model profile for Reflector/Curator (default: the Generator's)")
    ap.add_argument("--memory-file", default=None, help="playbook memory file for llm_ace (default: runs/<run-name>/llm_ace/memory.json)")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds per family (episodes = families x seeds)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--run-name", default="run"); a = ap.parse_args()
    if a.seed_start + a.seeds > MAX_EVAL_SEED:
        raise SystemExit(f"seeds must stay below {MAX_EVAL_SEED} (higher ranges are reserved)")
    names = a.controllers.split(",")
    if (a.memory_file or a.reflector_profile) and "llm_ace" not in names:
        print("warning: --memory-file / --reflector-profile are only used by llm_ace", file=sys.stderr)
    assets, eco, rules = load(); pref = load_price_reference()
    fams = a.families.split(","); order = episode_order(fams, range(a.seed_start, a.seed_start + a.seeds))
    ctrls = [make_controller(c, assets, eco, a.profile, a.reflector_profile, a.memory_file, a.run_name) for c in names]
    for c in ctrls:
        if hasattr(c, "memory"):
            print(f"ACE memory: {c.memory_file} ({'loaded %d rules' % c.rules_loaded if c.rules_loaded else 'starting empty'}); "
                  f"reflector model: {'same as generator' if c.reflector_client is c.client else a.reflector_profile}")
    write_manifest(f"runs/{a.run_name}", a, ctrls); results = {}
    for ctrl in ctrls:
        out = f"runs/{a.run_name}/{ctrl.name}"
        rows = run_series(ctrl, assets, eco, rules, order, pref, out, verbose=True); results[ctrl.name] = rows
        json.dump(dict(rows=rows, usage=getattr(ctrl, "usage", None),
                    budget=ctrl.budget.snapshot() if hasattr(ctrl, "budget") else None),
                open(f"{out}/summary.json", "w"), indent=1)
    print("\nMean score by family (%d episodes per controller):" % len(order))
    for cname, rows in results.items():
        print(cname, {f: round(statistics.mean(r["score"] for r in rows if r["family"] == f), 2) for f in fams},
              "fallbacks:", sum(r["fallbacks"] for r in rows))

if __name__ == "__main__":
    main()
