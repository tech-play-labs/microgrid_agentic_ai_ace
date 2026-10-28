"""Persistent, validated playbook: the Curator proposes operations, this module checks and applies them."""
import copy, difflib, json, os, re
from dataclasses import dataclass, field, asdict
from env.scenarios import FAMILIES

MAX_RULES = 10            # active rules
MAX_TEXT = 240            # characters per rule
MAX_RENDER_CHARS = 2400   # what the Generator sees (~600 tokens): keeps prompts well inside the context window
DUP_RATIO = 0.85
OPS = ("ADD", "UPDATE", "SUPPORT", "CONTRADICT", "DEPRECATE")
_EPISODE_REF = re.compile(r"(?i)\b(?:%s)_\d+\b|\bseed\s*\d+\b|\bepisode\s*\d+\b" % "|".join(FAMILIES))

@dataclass
class Rule:
    id: str
    text: str
    support: int = 1
    contra: int = 0
    deprecated: bool = False
    evidence: list = field(default_factory=list)      # episode ids (metadata only, never shown to the Generator)
    created: str = ""
    updated: str = ""

    @property
    def status(self):
        if self.deprecated: return "deprecated"
        families = {e.rsplit("_", 1)[0] for e in self.evidence}
        return "supported" if self.support >= 3 and len(families) >= 2 and self.support > self.contra else "candidate"

_HOUR_WINDOW = re.compile(r"\bhours?\s*\d{1,2}\s*(?:-|\u2013|to)\s*\d{1,2}\b|\b(?:between|from)\s+\d{1,2}(?::00)?\s*(?:h\s*)?(?:and|to|-)\s*\d{1,2}", re.I)
_OBSERVABLE = re.compile(
    r"\bhours?\s*\d|\bhour of (?:the )?day\b|\b(?:morning|midday|noon|afternoon|evening|night)\b"
    r"|\bprice\b|\bpercentile\b|\btariff\b"
    r"|\bbattery (?:energy|level|temperature|temp)\b|\bstate of charge\b|\bsoc\b|\b(?:full|empty)\b|\btemperature\b"
    r"|\bsolar\b|\bhours? (?:left|remaining)\b|\bdeparture\b|\bdeadline\b|\bev\b|\bgrid (?:import|export) limit\b"
    r"|\b(?:previous|last|recent) hours?\b|\bwarnings?\b|\d+(?:\.\d+)?\s*(?:kwh?|%|\u00b0\s?c)\b", re.I)
_NOT_OBSERVABLE = re.compile(
    r"\bexport_limit_\w+|\binterventions?\b|\bphysical[_ ]clips?\b|\bclip(?:s|ped)?\b|\bflags?\b|\bcurtail\w*"
    r"|\bexport limit (?:reduc\w+|active|binding|reached|hit)\b"
    r"|\binjection[_ ]fees?\s+(?:is |are )?(?:not )?(?:applied|applicable|active|present|charged|incurred|apply|applies)\b"
    r"|\b(?:when|if|while)\s+(?:there (?:is|are) )?(?:an? )?injection[_ ]fees?\b|\bno injection[_ ]fees?\b|\bwithout (?:an? )?injection[_ ]fees?\b", re.I)

def text_problem(text):
    t = (text or "").strip()
    if not t: return "empty text"
    if len(t) > MAX_TEXT: return f"text longer than {MAX_TEXT} characters"
    if _EPISODE_REF.search(t): return "mentions an episode id or seed (rules must be general)"
    if not _OBSERVABLE.search(t):
        return "no condition on something the operator sees when deciding (hour, price or percentile, battery level or temperature, solar, EV deadline)"
    if _NOT_OBSERVABLE.search(t) and not _HOUR_WINDOW.search(t):
        return "condition depends on something not visible before acting (fee applied, flags, interventions, clips, curtailment); use an hour range or an observable threshold"
    return None

class Playbook:
    def __init__(self):
        self.rules, self.version, self.next_id = [], 0, 1

    # ---- views ----
    def active(self): return [r for r in self.rules if not r.deprecated]
    def get(self, rid): return next((r for r in self.rules if r.id == rid), None)
    def active_ids(self): return {r.id for r in self.active()}

    def render(self):
        """Text injected into the Generator prompt."""
        rs = sorted(self.active(), key=lambda r: (r.status != "supported", -r.support, r.id))
        return "\n".join(f"- {r.id} [{r.status}]: {r.text}" for r in rs)

    def render_for_curator(self):
        rs = sorted(self.active(), key=lambda r: r.id)
        body = "\n".join(f"- {r.id} [{r.status}] +{r.support}/-{r.contra}: {r.text}" for r in rs) or "(empty)"
        return (f"{body}\nSlots left: {MAX_RULES - len(rs)} of {MAX_RULES}. "
                f"Characters left in the Generator view: {max(0, MAX_RENDER_CHARS - len(self.render()))}.")

    # ---- operations ----
    def apply(self, ops, episode_id):
        """ops: list of dicts {op, id, text}. Returns (applied, rejected); never raises on bad ops."""
        applied, rejected = [], []
        before = self.render()
        for raw in ops:
            snapshot = (copy.deepcopy(self.rules), self.next_id)
            try:
                note = self._apply_one(raw, episode_id)
                if len(self.render()) > MAX_RENDER_CHARS:
                    self.rules, self.next_id = snapshot
                    raise ValueError("playbook text budget exceeded")
                applied.append(dict(op=raw, note=note))
            except Exception as e:
                self.rules, self.next_id = snapshot
                rejected.append(dict(op=raw, reason=str(e)))
        for r in self.active():                      # automatic retirement of contradicted rules
            if r.contra >= 2 and r.contra > r.support: r.deprecated = True
        if self.render() != before: self.version += 1
        return applied, rejected

    def _apply_one(self, raw, ep):
        op, rid, text = str(raw.get("op", "")).upper(), str(raw.get("id", "") or "").strip(), str(raw.get("text", "") or "").strip()
        if op not in OPS: raise ValueError(f"unknown op {op!r}")
        if op == "ADD":
            problem = text_problem(text)
            if problem: raise ValueError(problem)
            for r in self.active():
                if difflib.SequenceMatcher(None, r.text.lower(), text.lower()).ratio() >= DUP_RATIO:
                    r.support += 1; r.updated = ep; self._evidence(r, ep)
                    return f"merged into {r.id} (near-duplicate)"
            if len(self.active()) >= MAX_RULES: raise ValueError("playbook full: deprecate a rule first")
            r = Rule(id=f"R{self.next_id:03d}", text=text, created=ep, updated=ep, evidence=[ep]); self.next_id += 1
            self.rules.append(r); return f"added {r.id}"
        r = self.get(rid)
        if r is None or r.deprecated: raise ValueError(f"unknown or deprecated rule id {rid!r}")
        if op == "UPDATE":
            problem = text_problem(text)
            if problem: raise ValueError(problem)
            r.text = text
        elif op == "SUPPORT": r.support += 1
        elif op == "CONTRADICT": r.contra += 1
        elif op == "DEPRECATE": r.deprecated = True
        r.updated = ep; self._evidence(r, ep)
        return f"{op.lower()} {r.id}"

    @staticmethod
    def _evidence(r, ep):
        if ep not in r.evidence: r.evidence = (r.evidence + [ep])[-5:]

    # ---- persistence (atomic) ----
    def to_dict(self): return dict(version=self.version, next_id=self.next_id, rules=[asdict(r) for r in self.rules])

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f: json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        try:
            d = json.load(open(path))
            pb = cls(); pb.version, pb.next_id = int(d["version"]), int(d["next_id"])
            pb.rules = [Rule(**r) for r in d["rules"]]
            return pb
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"cannot read playbook memory file {path!r}: {e}") from e
