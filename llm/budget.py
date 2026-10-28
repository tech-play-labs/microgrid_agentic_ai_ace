"""Per-role adaptive context / output budgets (sticky escalation). Ladders live in config/model.json."""
import json
from dataclasses import dataclass

ROLES = ("generator", "reflector", "curator")


class BudgetExhausted(RuntimeError):
    """Raised when the limit that overflowed is already at its top step."""


@dataclass(frozen=True)
class Rung:
    num_ctx: int
    max_tokens: int


def _validate(ladders):
    if not isinstance(ladders, dict):
        raise ValueError("budget config must be an object keyed by role")
    extra = set(ladders) - set(ROLES)
    missing = set(ROLES) - set(ladders)
    if extra or missing:
        raise ValueError(f"budget config: missing roles {sorted(missing)}, unknown roles {sorted(extra)}")
    for role, l in ladders.items():
        if set(l) != {"num_ctx", "max_tokens"}:
            raise ValueError(f"budget[{role}] needs exactly the keys num_ctx and max_tokens")
        for key, steps in l.items():
            if (not isinstance(steps, list) or not steps
                    or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in steps)):
                raise ValueError(f"budget[{role}][{key}] must be a non-empty list of positive integers")
            if any(b <= a for a, b in zip(steps, steps[1:])):
                raise ValueError(f"budget[{role}][{key}] must be strictly increasing: {steps}")
        if max(l["max_tokens"]) >= max(l["num_ctx"]):
            raise ValueError(f"budget[{role}]: largest max_tokens must be smaller than largest num_ctx")
    return ladders


class RoleBudget:
    def __init__(self, ladders):
        self.ladders = _validate(ladders)
        self.ctx_level = {r: 0 for r in ROLES}
        self.out_level = {r: 0 for r in ROLES}
        self.history = []

    @classmethod
    def from_config(cls, path="config/model.json"):
        cfg = json.load(open(path)).get("budget")
        if cfg is None:
            raise ValueError(f"{path}: missing 'budget' section")
        return cls(cfg)

    def current(self, role):
        l = self.ladders[role]
        return Rung(l["num_ctx"][self.ctx_level[role]], l["max_tokens"][self.out_level[role]])

    def escalate(self, role, kind, reason):
        """kind: 'ctx' raises num_ctx only, 'out' raises max_tokens only."""
        levels, key = (self.ctx_level, "num_ctx") if kind == "ctx" else (self.out_level, "max_tokens")
        if levels[role] + 1 >= len(self.ladders[role][key]):
            return False
        levels[role] += 1
        r = self.current(role)
        self.history.append(dict(role=role, raised=key, reason=reason, num_ctx=r.num_ctx, max_tokens=r.max_tokens))
        return True

    def snapshot(self):
        return dict(ctx_level=dict(self.ctx_level), out_level=dict(self.out_level), history=list(self.history))

    def describe(self):
        return self.ladders


def was_cut_off(res):
    raw = res.raw or {}
    choices = raw.get("choices") or [{}]
    return raw.get("done_reason") == "length" or choices[0].get("finish_reason") == "length"


def overflow_reason(res, rung, ctx_aware=True):
    """None if fine, else (kind, reason): kind 'ctx' = context too small, 'out' = reply cap too small."""
    if ctx_aware and res.prompt_tokens >= rung.num_ctx - 8:
        return "ctx", f"prompt filled context ({res.prompt_tokens}/{rung.num_ctx})"
    if was_cut_off(res):
        if ctx_aware and res.prompt_tokens + res.completion_tokens >= 0.97 * rung.num_ctx:
            return "ctx", "prompt+reply exhausted context"
        return "out", f"reply hit max_tokens={rung.max_tokens}"
    return None


def complete_with_budget(client, budget, role, system, prompt, schema, on_result=None):
    """Call the client; on overflow raise ONLY the limit that failed (sticky) and repeat the call at once."""
    ctx_aware = bool(getattr(client, "supports_num_ctx", False))
    while True:
        rung = budget.current(role)
        res = client.complete(system, prompt, schema, num_ctx=rung.num_ctx, max_tokens=rung.max_tokens)
        if on_result:
            on_result(res)
        why = overflow_reason(res, rung, ctx_aware)
        if why is None:
            return res
        kind, text = why
        if not budget.escalate(role, kind, text):
            raise BudgetExhausted(f"{role}: {text}; already at top step")