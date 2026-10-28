"""Pick the best Ollama model for the Reflector/Curator by replaying ONE reference request many times.

Self-contained: needs only `requests` (and, if run from the project root, the project's schemas for validation).

The reference is the LAST LINE of a debug log (a logged ace_reflect or ace_curate call). Every candidate profile gets the
same prompt, N repeats (different seeds). Each reply is checked automatically:

  GATES (fail = score 0)   server/HTTP error or missing model | derailment (chat markers, off-task text) | invalid JSON
                           for the schema | slower than --max-seconds
  SCORE (0-100)            grounding 30   hours and numbers cited in the reply must exist in the digest (hallucination proxy)
                           correctness 30 fee read as income, charge/discharge vs price direction inverted (-15 each)
                           specificity 20 rules need a condition, an action, a number or hour, and <= 240 characters
                           concision 10   no near-duplicate rules, at most 4 lessons
                           compliance 10  no episode ids or seeds in the text
  (hidden-rule hints are reported but NOT scored: this one day may not contain them)

Usage (project root):
  python tools/select_reflector.py --log runs/debug/latest_run/llm_calls.jsonl --prefix r_ --repeats 2
  python tools/select_reflector.py --log ... --profiles r_qwen3_8b_nothink,r_gemma3_12b,groq --cap 2500
A profile may set "repro_cap" in config/model.json to override --cap (e.g. 6000 for a thinking model).
Heuristics are crude: always read the reported rules of the top candidates yourself.
"""
import argparse, difflib, json, os, re, statistics, time
import requests

DOMAIN = ("hour", "price", "discharge", "charge", "battery", "playbook", "lesson", "reward", "export", "digest")
MARKERS = re.compile(r"\n(?:user|assistant|system)\n|<\uff5c|Review of the Finished", re.I)
ACTION = re.compile(r"\b(discharg\w*|charg\w*|idle|defer\w*|export\w*|import\w*|curtail\w*|hold|prefer|avoid)\b", re.I)
COND = re.compile(r"\b(when|if|during|while|above|below|at least|over|under|between|before|after|in hours?)\b|[<>\u2265\u2264]", re.I)
OBSERVABLE = re.compile(r"\bhours?\s*\d|midday|morning|evening|afternoon|percentile|temperature|solar|battery (energy|level)|state of charge|\d+\s*(%|kw)", re.I)
FEE_INCOME = re.compile(r"(earn|income|revenue|profit|gain)\w*[^.;]{0,40}(fee|injection)|(fee|injection)[^.;]{0,40}(earn|income|revenue|profit)", re.I)
DIS_LOW = re.compile(r"\bdischarg\w*[^.;]{0,60}\b(low (import )?price|price percentile (is |are )?(low|below|under)|(low|small) percentile|cheap\w*)", re.I)
CH_HIGH = re.compile(r"\bcharg\w*[^.;]{0,60}\b(high (import )?price|price percentile (is |are )?(high|above|over)|(high|large) percentile|expensive)", re.I)
NEG = re.compile(r"(avoid|not|never|don't|do not|stop)\W+\w*\W*$", re.I)
BANNED = re.compile(r"(?i)\b(?:clear_mild|clear_hot|cloudy|low_solar|hot_spike|mild_spike)_\d+\b|\bseed\s*\d+\b|\bepisode\s*\d+\b")
HINTS = dict(mentions_temperature=r"temperature|thermal|\u00b0c|overheat", mentions_fee=r"fee|inject",
             mentions_fee_hours=r"hours?\s*\d+\s*(-|to|\u2013)\s*\d+|midday|noon")


def last_line(path):
    """Return the last non-empty line of a (possibly huge) file without reading all of it."""
    with open(path, "rb") as f:
        f.seek(0, 2); pos = f.tell(); buf = b""
        while pos > 0:
            step = min(65536, pos); pos -= step; f.seek(pos); buf = f.read(step) + buf
            lines = buf.rstrip(b"\r\n").split(b"\n")
            if len(lines) > 1 or pos == 0: return lines[-1].decode("utf-8")
    raise SystemExit("empty log file")


def pick_schema(stage):
    """Project schema for the logged stage (Reflector or Curator); None if the project is not importable."""
    try:
        if "curat" in (stage or ""):
            from controllers.ace_controller import CuratorOut as M
        else:
            from controllers.ace_controller import ReflectorOut as M
        return M, M.model_json_schema()
    except Exception:
        return None, None


def derailed(text):
    if MARKERS.search(text): return "chat-marker/off-topic text"
    tail = text[-1200:]
    if len(text) > 1800 and not any(w in tail.lower() for w in DOMAIN): return "no task vocabulary in the last 1200 chars"
    return None


def append(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def stream_run(host, payload, cap, deadline_s):
    """Stream an Ollama /api/chat call; stop early on derailment or deadline and keep what was produced."""
    payload = {**payload, "stream": True}
    payload["options"] = {**payload["options"], "num_predict": cap}
    t0, think, content, chunks, final, why, err = time.time(), [], [], 0, {}, None, None
    with requests.post(f"{host}/api/chat", json=payload, stream=True, timeout=(10, 600)) as r:
        if r.status_code != 200: return dict(error=f"HTTP {r.status_code}: {r.text[:200]}")
        for line in r.iter_lines():
            if not line: continue
            d = json.loads(line)
            if d.get("error"): err = str(d["error"])[:300]; break
            m = d.get("message") or {}
            think.append(m.get("thinking", "")); content.append(m.get("content", "")); chunks += 1
            if d.get("done"): final = d; break
            if chunks % 25 == 0:
                why = derailed("".join(think) + "".join(content))
                if why: break
            if time.time() - t0 > deadline_s: why = "deadline"; break
    return dict(think="".join(think), content="".join(content), chunks=chunks, final=final, stop_reason=why,
                server_error=err, seconds=round(time.time() - t0, 3))


def digest_facts(user):
    hours = set()
    for m in re.finditer(r"^\s*(\d{1,2})\s*\|", user, re.M): hours.add(int(m.group(1)))
    for m in re.finditer(r"hours?\s*\[([^\]]*)\]", user, re.I): hours |= {int(x) for x in re.findall(r"\d+", m.group(1))}
    for m in re.finditer(r"\bhour\s+(\d{1,2})\b", user, re.I): hours.add(int(m.group(1)))
    decs = [float(x) for x in re.findall(r"-?\d+\.\d+", user)]
    return hours, decs


def parse_reply(ct, model_cls):
    s, e = ct.find("{"), ct.rfind("}")
    if s < 0 or e <= s: return None
    try:
        obj = model_cls.model_validate_json(ct[s:e + 1]) if model_cls else json.loads(ct[s:e + 1])
    except Exception:
        return None
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def reply_text(d):
    parts = []
    for l in d.get("lessons", []): parts += [l.get("observation", ""), l.get("evidence", ""), l.get("suggested_rule", "")]
    for o in d.get("ops", []): parts += [o.get("text", "")]
    return " ".join(p for p in parts if p)


def rules_of(d):
    return [r for r in [l.get("suggested_rule") for l in d.get("lessons", [])] + [o.get("text") for o in d.get("ops", [])] if r]


def score(d, user):
    hours, decs = digest_facts(user)
    text, rules = reply_text(d), rules_of(d)
    cited_h = [int(x) for m in re.finditer(r"\bhours?\s+(\d{1,2}(?:\s*(?:,|and|to|-|\u2013)\s*\d{1,2})*)", text, re.I) for x in re.findall(r"\d+", m.group(1))]
    cited_n = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", text)]
    ok_h = sum(h in hours for h in cited_h)
    ok_n = sum(any(abs(abs(x) - abs(y)) <= max(0.006, 0.02 * abs(y)) for y in decs) for x in cited_n)
    total_c = len(cited_h) + len(cited_n)
    ratio = (ok_h + ok_n) / total_c if total_c else None
    grounding = 30 * ratio if ratio is not None else 10
    flags = []
    adv = " ".join(rules)
    if FEE_INCOME.search(adv): flags.append("fee_read_as_income")
    for pat, name in ((DIS_LOW, "discharge_when_price_low"), (CH_HIGH, "charge_when_price_high")):
        for m in pat.finditer(text):
            if not NEG.search(text[max(0, m.start() - 25):m.start()]): flags.append(name); break
    correctness = max(0, 30 - 15 * len(flags))
    good = [r for r in rules if len(r) <= 240 and ACTION.search(r) and COND.search(r) and OBSERVABLE.search(r)]
    specificity = 20 * len(good) / len(rules) if rules else (10 if d.get("insufficient_evidence") else 0)
    dups = sum(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.8 for i, a in enumerate(rules) for b in rules[i + 1:])
    concision = max(0, 10 - 5 * dups - (5 if len(d.get("lessons", [])) > 4 or len(d.get("ops", [])) > 4 else 0))
    compliance = 0 if BANNED.search(text) else 10
    total = round(grounding + correctness + specificity + concision + compliance, 1)
    hints = {k: bool(re.search(p, text.lower())) for k, p in HINTS.items()}
    return dict(total=total, grounding=round(grounding, 1), grounded=f"{ok_h + ok_n}/{total_c}", correctness=correctness, flags=flags,
                specificity=round(specificity, 1), concision=concision, compliance=compliance, hints=hints, rules=rules)


def unload(host, model):
    try: requests.post(f"{host}/api/generate", json=dict(model=model, keep_alive=0), timeout=60)
    except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="debug log; its LAST LINE is the reference request")
    ap.add_argument("--config", default="config/model.json")
    ap.add_argument("--profiles"); ap.add_argument("--prefix", default="r_", help="use every profile whose name starts with this")
    ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, help="override every profile's temperature")
    ap.add_argument("--cap", type=int, default=2500, help="max generated tokens per call")
    ap.add_argument("--ctx", type=int, default=8192, help="num_ctx for every call (large enough that the window never shifts)")
    ap.add_argument("--max-seconds", type=int, default=900); ap.add_argument("--min-score", type=float, default=60)
    ap.add_argument("--out", default="runs/debug/repro/llm_calls.jsonl"); ap.add_argument("--report", default="runs/debug/repro/selection_report.json")
    a = ap.parse_args()
    rec = json.loads(last_line(a.log))
    if not (rec.get("system") and rec.get("user")): raise SystemExit("the last line has no prompt: " + str(list(rec)))
    system, user, stage = rec["system"], rec["user"], rec.get("stage") or "ace_reflect"
    model_cls, sch = pick_schema(stage)
    cfgs = json.load(open(a.config))["profiles"]
    names = a.profiles.split(",") if a.profiles else [n for n in cfgs if n.startswith(a.prefix)]
    if not names: raise SystemExit("no profiles selected")
    print(f"reference: stage={stage} episode={rec.get('episode_id')} | {len(names)} profiles x {a.repeats} repeats | cap {a.cap}, ctx {a.ctx}"
          + ("" if sch else " | project schema not importable: JSON is only checked for syntax"))
    seq, report = 0, {}
    for pname in names:
        cfg = cfgs[pname]; ptype = cfg.get("type"); cap = cfg.get("repro_cap") or a.cap
        print(f"\n##### {pname} ({ptype}, {cfg.get('model')}, cap {cap})", flush=True)
        results = []
        for i in range(a.repeats):
            seed = a.seed + i; seq += 1; row = dict(profile=pname, repeat=i, seed=seed, gate=None, total=0)
            call = dict(seq=seq, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), profile=pname, model=cfg.get("model"), stage=stage,
                        episode_id=rec.get("episode_id"), attempt=1, req_num_ctx=a.ctx, req_max_tokens=cap, system=system, user=user,
                        repro_run=f"select_{i}", seed=seed)
            try:
                if ptype == "ollama":
                    host = cfg.get("host", "http://localhost:11434").rstrip("/")
                    temp = a.temperature if a.temperature is not None else cfg.get("temperature", 0.0)
                    payload = dict(model=cfg["model"], messages=[dict(role="system", content=system), dict(role="user", content=user)],
                                   options=dict(temperature=temp, seed=seed, num_ctx=a.ctx), **dict(cfg.get("extra_body") or {}))
                    if sch: payload["format"] = sch
                    r = stream_run(host, payload, cap, a.max_seconds)
                    if r.get("error"):
                        row["gate"] = "error: " + r["error"] + ("  (ollama pull %s ?)" % cfg["model"] if "404" in r["error"] else "")
                        call["error"] = r["error"]
                    else:
                        fin, th, ct = r["final"], r["think"], r["content"]
                        d = parse_reply(ct, model_cls)
                        why = r["stop_reason"] or r["server_error"] or (derailed(th + ct) if (fin and d is None) else None)
                        if not fin and not why: why = "stream ended without a final chunk"
                        gen = fin.get("eval_count") or r["chunks"]
                        raw = dict(fin) if fin else dict(done=False, stopped_by=why)
                        raw["message"] = dict(role="assistant", content=ct, thinking=th)
                        call.update(response_text=ct, prompt_tokens=fin.get("prompt_eval_count"), completion_tokens=gen, latency_s=r["seconds"], raw=raw)
                        row.update(seconds=round(r["seconds"]), tokens=gen, thinking_chars=len(th))
                        if why: row["gate"] = f"stopped: {why}"
                        elif d is None: row["gate"] = "invalid JSON for the schema"
                        elif r["seconds"] > a.max_seconds: row["gate"] = f"too slow ({round(r['seconds'])}s)"
                        else: row.update(score(d, user))
                elif ptype == "openai_compat":
                    from llm.model_client import build_client          # needs the project; API profiles are skipped otherwise
                    res = build_client(cfg).complete(system, user, sch or {}, num_ctx=None, max_tokens=cap)
                    d = parse_reply(res.text, model_cls)
                    call.update(response_text=res.text, prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, latency_s=res.latency_s, raw=res.raw)
                    row.update(seconds=round(res.latency_s), tokens=res.completion_tokens)
                    if d is None: row["gate"] = "invalid JSON for the schema"
                    else: row.update(score(d, user))
                else:
                    row["gate"] = f"unsupported profile type {ptype}"
            except Exception as e:                                       # network, import or LLMError
                row["gate"] = f"{type(e).__name__}: {str(e)[:200]}"; call["error"] = row["gate"]
            call.update(gate=row["gate"], score=row.get("total"), score_detail={k: row.get(k) for k in ("grounding", "grounded", "correctness", "flags", "specificity", "concision", "compliance")})
            append(a.out, call); results.append(row)
            print("  ", {k: v for k, v in row.items() if k not in ("rules", "hints")}, flush=True)
            for rule in row.get("rules", [])[:4]: print("      rule:", rule)
            if str(row.get("gate") or "").startswith("error"): break
        if ptype == "ollama": unload(cfg.get("host", "http://localhost:11434").rstrip("/"), cfg["model"])
        report[pname] = results
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    json.dump(report, open(a.report, "w"), indent=1)

    rank = []
    for pname, rows in report.items():
        passed = [r for r in rows if not r["gate"]]
        scores = [r["total"] for r in passed]
        rank.append(dict(profile=pname, passed=f"{len(passed)}/{len(rows)}", mean=round(statistics.mean(scores), 1) if scores else 0,
                         worst=min(scores) if scores else 0, seconds=round(statistics.mean([r["seconds"] for r in passed])) if passed else None,
                         gates=sorted({str(r["gate"]) for r in rows if r["gate"]})))
    rank.sort(key=lambda r: (r["passed"].split("/")[0] == r["passed"].split("/")[1], r["mean"], r["worst"]), reverse=True)
    print("\n=== ranking (all repeats must pass the gates to rank first) ===")
    for r in rank: print(f"{r['profile']:<28} passed {r['passed']:<5} mean {r['mean']:<6} worst {r['worst']:<6} avg {r['seconds']}s  {r['gates'] or ''}")
    ok = [r for r in rank if r["passed"].split("/")[0] == r["passed"].split("/")[1] and r["mean"] >= a.min_score]
    print(f"\nrecommended: {ok[0]['profile']}" if ok else f"\nno profile passed every repeat with mean >= {a.min_score}; read the rules of the best ones before deciding")
    print(f"calls logged to {a.out}; report in {a.report}")


if __name__ == "__main__":
    main()
