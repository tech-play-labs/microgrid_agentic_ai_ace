"""ACE controller: Generator (inherited from LLMController) + Reflector + Curator + persistent playbook.
After each day, end_of_episode() runs Reflector then Curator. ANY failure there is logged and the old playbook is kept."""
import json, os
from typing import Literal
from pydantic import BaseModel, ConfigDict
from ace_learning.digest import build_digest
from controllers.llm_controller import LLMController, extract_json
from llm.model_client import LLMError
from ace_learning.playbook import Playbook, MAX_RULES, MAX_TEXT
from llm.budget import complete_with_budget, BudgetExhausted

# ---------------- structured outputs (no length constraints in the schema: limits are enforced in code) ----------------
class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: str
    evidence: str
    confidence: Literal["low", "medium", "high"]
    suggested_rule: str                      # "" if the lesson should not become a rule

class ReflectorOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insufficient_evidence: bool
    lessons: list[Lesson]

class Op(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["ADD", "UPDATE", "SUPPORT", "CONTRADICT", "DEPRECATE"]
    id: str                                   # "" for ADD
    text: str                                 # "" for SUPPORT / CONTRADICT / DEPRECATE

class CuratorOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ops: list[Op]

REFLECTOR_SYSTEM = (
"You review one finished day of a microgrid operation and extract lessons for the operator's playbook.\n"
"You receive a DIGEST of the day (facts computed by code) and the CURRENT PLAYBOOK.\n\n"
"Guidelines:\n"
"- Base every lesson on evidence in the digest. Describe what happened; do not assume mechanisms the digest does not show.\n"
"- The environment has rules that are not documented. Charges, warnings, and limited power in the digest are the clues. "
"If the evidence is thin or ambiguous, give low confidence or set insufficient_evidence to true.\n"
"Every charge, fee, and penalty in the digest reduces the reward. They are costs, never income."
"- Compare the day with the do-nothing reference: if the day scored below it, find which battery actions lost money; if above, which gained.\n"
"- A lesson must be general: describe conditions the operator can observe (hour of day, price percentile, battery temperature, solar level, warnings, executed power, charges) and what to do. "
"Never mention episode ids, seeds, or dates.\n"
"- Say whether the day supports or contradicts existing playbook rules, citing their ids.\n"
"- At most 4 lessons. Use an empty suggested_rule when a lesson should not become a rule.\n\n"
"Reply with one JSON object only: {\"insufficient_evidence\": bool, \"lessons\": [{\"observation\": str, \"evidence\": str, "
"\"confidence\": \"low\"|\"medium\"|\"high\", \"suggested_rule\": str}]}")

CURATOR_SYSTEM = (
"You maintain a short playbook of operating rules for a microgrid operator. You receive the reflector's lessons from the latest day and the CURRENT PLAYBOOK.\n\n"
"Return a small list of operations (at most 4):\n"
"- ADD: a new rule (id empty, text required)\n"
"- UPDATE: replace the text of a rule (id and text required)\n"
"- SUPPORT: the latest day agrees with a rule (id required, text empty)\n"
"- CONTRADICT: the latest day disagrees with a rule (id required, text empty)\n"
"- DEPRECATE: remove a wrong or useless rule (id required, text empty)\n\n"
f"Rule requirements: one sentence, imperative, general (observable conditions + action), under {MAX_TEXT} characters, no episode ids or seeds, no duplicates. "
"Conditions must use only what the operator sees when deciding: hour of day or an hour range, import/export price or price percentile, battery energy or temperature, solar output, EV deadlines, hours remaining, and warnings in the last hours. Never use 'injection fee applied', flags, interventions, clips or curtailment as conditions; express a fee period as an hour range."
f"The playbook holds at most {MAX_RULES} rules and has a text budget; deprecate a rule before adding when full.\n"
"Prefer SUPPORT or UPDATE over ADD. Add a rule only when a lesson has medium or high confidence or repeats an earlier candidate. "
"If nothing should change, return an empty list.\n\n"
"Reply with one JSON object only: {\"ops\": [{\"op\": str, \"id\": str, \"text\": str}]}")

class StageError(Exception):
    def __init__(self, msg, raw=""):
        super().__init__(msg); self.raw = raw

class ACEController(LLMController):
    def __init__(self, client, assets, economics, budget=None, reflector_client=None, memory_file="memory/playbook.json",
                 log_path=None, name="llm_ace", **kw):
        super().__init__(client, assets, economics, budget, name=name, **kw)
        self.reflector_client = reflector_client if reflector_client is not None else client
        self.memory_file = memory_file
        self.log_path = log_path or os.path.join(os.path.dirname(memory_file) or ".", "ace_log.jsonl")
        self.memory = Playbook.load(memory_file) if os.path.exists(memory_file) else Playbook()
        self.rules_loaded = len(self.memory.active())
        self.usage.update(ace_calls=0, ace_prompt_tokens=0, ace_completion_tokens=0, ace_latency_s=0.0, ace_failures=0)
        self._sync_prompt()

    # ---- generator side ----
    @property
    def playbook_version(self): return self.memory.version

    def _sync_prompt(self):
        self.rule_ids = self.memory.active_ids()
        self.set_playbook(self.memory.render())

    def describe(self):
        return dict(memory_file=self.memory_file, rules_loaded_at_start=self.rules_loaded, log_path=self.log_path,
                    reflector_same_client_as_generator=self.reflector_client is self.client,
                    budget_ladders=self.budget.describe())

    # ---- learning side ----
    def end_of_episode(self, summary, public_ledger):
        ep = summary.get("episode_id", "?")
        try:
            self._learn(summary, public_ledger, ep)
        except BaseException as e:                      # a bad reply must never kill the run
            if isinstance(e, (KeyboardInterrupt, SystemExit)): raise
            self.usage["ace_failures"] += 1
            self._log(dict(episode=ep, stage="end_of_episode", status="error", error=f"{type(e).__name__}: {e}"[:500]))

    def _learn(self, summary, ledger, ep):
        digest = build_digest(summary, ledger)
        # --- Reflector ---
        reflect_user = f"DIGEST OF THE FINISHED DAY\n{digest}\n\nCURRENT PLAYBOOK\n{self.memory.render_for_curator()}"
        try:
            refl = self._call(REFLECTOR_SYSTEM, reflect_user, ReflectorOut, ep, "reflect")
        except StageError as e:
            self._fail(ep, "reflect", e); return
        lessons = refl.lessons[:4]
        self._log(dict(episode=ep, stage="reflect", status="ok", insufficient_evidence=refl.insufficient_evidence,
                       lessons=[l.model_dump() for l in lessons]))
        if refl.insufficient_evidence and not lessons:
            self._log(dict(episode=ep, stage="curate", status="skipped", reason="insufficient evidence")); return
        # --- Curator ---
        curate_user = ("LESSONS FROM THE LATEST DAY\n" + json.dumps([l.model_dump() for l in lessons], indent=1)
                       + f"\n\nCURRENT PLAYBOOK\n{self.memory.render_for_curator()}")
        try:
            cur = self._call(CURATOR_SYSTEM, curate_user, CuratorOut, ep, "curate")
        except StageError as e:
            self._fail(ep, "curate", e); return
        applied, rejected = self.memory.apply([o.model_dump() for o in cur.ops[:4]], ep)
        self._log(dict(episode=ep, stage="apply", status="ok", applied=applied, rejected=rejected, version=self.memory.version))
        if applied:
            try: self.memory.save(self.memory_file)
            except OSError as e:
                self.usage["ace_failures"] += 1
                self._log(dict(episode=ep, stage="save", status="error", error=str(e)[:300]))   # memory still lives in RAM
            self._sync_prompt()

    def _call(self, system, user, model_cls, ep, stage):
        client = self.reflector_client
        role = "reflector" if stage == "reflect" else "curator"
        getattr(client, "set_context", lambda **k: None)(episode_id=ep, stage=f"ace_{stage}", attempt=1)
        schema, prompt, last, raw = model_cls.model_json_schema(), user, "", ""
        
        def _acc(r):                                                     # runs for every response,
            nonlocal raw                                                 # including one repeated after escalation
            u = self.usage; u["ace_calls"] += 1; u["ace_prompt_tokens"] += r.prompt_tokens
            u["ace_completion_tokens"] += r.completion_tokens; u["ace_latency_s"] += r.latency_s
            raw = r.text[:2000]
        
        for attempt in (1, 2):
            try:
                res = complete_with_budget(client, self.budget, role,
                                    system, prompt, schema, on_result=_acc)
            except BudgetExhausted as e:
                raise StageError(str(e), raw)
            except LLMError as e:
                last = f"backend: {e}"              # in select(): meta["error"] = f"backend: {e}"[:300]
                if "timeout after" in str(e): break # don't repeat a call that already used the whole timeout
                continue
            try:
                return model_cls.model_validate_json(extract_json(res.text))
            except ValueError as e:
                last = f"invalid: {e}"[:300]
                prompt = user + f"\n\nYour previous reply was rejected ({last[:150]}). Reply again with valid JSON only."
                continue
        raise StageError(last or "no reply", raw) 

    def _fail(self, ep, stage, e):
        self.usage["ace_failures"] += 1
        self._log(dict(episode=ep, stage=stage, status="error", error=str(e)[:300], raw=e.raw, action="kept old playbook"))

    def _log(self, rec):
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a") as f: f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
