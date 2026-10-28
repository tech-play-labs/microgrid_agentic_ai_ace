"""Paired comparison of two runs over the SAME episodes.
Usage: python compare_runs.py runs/RUN_A/llm_memoryless runs/RUN_B/llm_ace [metric]"""
import json, random, statistics, sys

def load(path):
    d = json.load(open(f"{path}/summary.json"))
    return {r["episode_id"]: r for r in d["rows"]}, d.get("usage")

def compare(a_dir, b_dir, metric="score", n_boot=2000):
    A, ua = load(a_dir); B, ub = load(b_dir)
    ids = [i for i in A if i in B]
    if not ids: raise SystemExit("no common episodes")
    diffs = [B[i][metric] - A[i][metric] for i in ids]
    rng = random.Random(0)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    res = dict(n=len(ids), missing=len(set(A) ^ set(B)),
               mean_a=statistics.mean(A[i][metric] for i in ids), mean_b=statistics.mean(B[i][metric] for i in ids),
               mean_diff=statistics.mean(diffs), ci95=(boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]),
               wins=sum(d > 1e-9 for d in diffs), ties=sum(abs(d) <= 1e-9 for d in diffs), losses=sum(d < -1e-9 for d in diffs))
    fams = list(dict.fromkeys(A[i]["family"] for i in ids))
    res["by_family"] = {f: statistics.mean(d for i, d in zip(ids, diffs) if A[i]["family"] == f) for f in fams}
    k = len(fams); half = len(ids) // 2
    res["first_half_diff"] = statistics.mean(diffs[:half]) if half else None
    res["second_half_diff"] = statistics.mean(diffs[half:]) if half else None
    cum, run = [], 0.0
    for r in range(0, len(ids), k):
        run += sum(diffs[r:r + k]); cum.append(round(run, 2))
    res["cumulative_advantage_by_round"] = cum
    res["first_episode_same"] = abs(diffs[0]) <= 1e-9
    res["unmet_ev_kwh"] = (sum(A[i]["unmet_ev_kwh"] for i in ids), sum(B[i]["unmet_ev_kwh"] for i in ids))
    res["thermal_events"] = (sum(A[i]["thermal_events"] for i in ids), sum(B[i]["thermal_events"] for i in ids))
    res["generator_prompt_tokens"] = (ua or {}).get("prompt_tokens"), (ub or {}).get("prompt_tokens")
    res["ace_reflector_curator_prompt_tokens"] = (ub or {}).get("ace_prompt_tokens")
    res["ace_failures"] = (ub or {}).get("ace_failures")
    pbv = [B[i].get("playbook_version") for i in ids]
    res["playbook_version_by_episode"] = pbv if any(v is not None for v in pbv) else None
    return res

if __name__ == "__main__":
    r = compare(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "score")
    for k, v in r.items(): print(f"{k:>38}: {v}")
