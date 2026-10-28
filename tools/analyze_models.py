import json, sys
from run import load, public_view
from llm.model_client import load_client, MockClient
from controllers.ace_controller import ACEController

assets, eco, _ = load()
src, profile = sys.argv[1], sys.argv[2]        # e.g. runs/first/heuristic  ollama_reflector
out = f"runs/bakeoff/{profile}"
rows = json.load(open(f"{src}/summary.json"))["rows"]
ctrl = ACEController(MockClient(), assets, eco,
                     reflector_client=load_client("config/model.json", profile),
                     memory_file=f"{out}/memory.json", log_path=f"{out}/ace_log.jsonl")
for row in rows:
    ledger = [public_view(json.loads(l)) for l in open(f"{src}/{row['episode_id']}.public.jsonl")]
    ctrl.end_of_episode(dict(row), ledger)
print(ctrl.memory.render())