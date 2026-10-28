# microgrid-ace

A hands-on, runnable example of **Agentic Context Engineering (ACE)**: a language model operates a simulated microgrid one hour at a time, and a Reflector and Curator maintain a small playbook of rules that is fed back to the next day's decisions. Everything can run locally.

This is an educational project. The results below come from runs with small local models, and they are an observation, not a verdict on ACE.

- Blog post: [LINK_TO_BLOG](LINK_TO_BLOG): the full walkthrough
- Scenario details: [scenario_details.md](scenario_details.md): the energy system, scenario generation, and assumptions

## What you can learn here

- How to put a language model inside a decision loop with an environment, validation, retries, and a safe fallback.
- How ACE's three roles (Generator, Reflector, Curator) and a persistent playbook fit together, and where code, not the model, should make the decisions.
- How to compare controllers fairly, with paired days, intervals, and per-family and over-time views.

## Requirements

- Python 3.10 or newer
- One of:
  - [Ollama](https://ollama.com) with a model pulled (default: `ollama pull qwen3:8b`, a 5.2 GB download that takes about 6 GB once loaded at a 4096-token context). A GPU with 8 GB or more holds it fully. On a 4 GB GPU it runs partly on the CPU, is slower, and needs about 16 GB of RAM.
  - an OpenAI-compatible API key (`OPENROUTER_API_KEY`, `GROQ_API_KEY`, etc.)
  - nothing, with the `mock` profile (pipeline checks only)

**Cost and time.** A simulated day needs at least 24 model calls. The default experiment (6 families × 5 seeds) is 30 days per LLM controller, about 720 calls before retries; `llm_ace` adds up to two calls per day (Reflector and Curator). Check provider prices first. Locally, with capacity stated above, a call took about 15 seconds at best, which is roughly 6 minutes per day for the Generator alone.

## Quick start

```bash
git clone LINK_TO_REPO
cd microgrid_ace
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. No model needed: check the pipeline with the baselines
python run.py --controllers do_nothing,heuristic,heuristic_trap_aware --seeds 1 --run-name smoke

# 2. Real run: one seed per family (6 days per controller)
ollama pull qwen3:8b
python run.py --controllers do_nothing,heuristic,llm,llm_ace --profile ollama_qwen3-8b --seeds 1 --run-name first

# 3. Real run with a different model for the Reflector and Curator
#    (add a matching profile for the model to config/model.json first)
ollama pull phi4:14b
python run.py --controllers llm_ace --profile ollama_qwen3-8b --reflector-profile ollama_phi4-14b --seeds 1 --run-name second

# 4. Compare two controllers over the same days
python tools/compare_runs.py runs/first/llm_memoryless runs/first/llm_ace
```

**Reusing a baseline.** At temperature 0 the memoryless run is repeatable, so you can compare a new `llm_ace` run with an earlier `llm` run instead of repeating it. Keep `--families`, `--seeds` and `--seed-start` the same, change nothing in `config/` or the prompts, and use the same Ollama version and settings. Check by re-running one day and diffing the hourly actions; the first ACE day, played with an empty playbook, should match the baseline exactly.

## Controllers

| Name | What it is |
|---|---|
| `do_nothing` | Battery idle, EVs charged at deadline pace. The reference. |
| `heuristic` | Threshold rules on price, solar, and battery level. |
| `heuristic_trap_aware` | The same heuristic, informed about a hidden rule (an oracle reference). |
| `llm` | The memoryless LLM controller: one model call per hour. |
| `llm_ace` | The same controller plus a playbook that a Reflector and Curator update after each day. |

## Useful options

| Option | Meaning |
|---|---|
| `--controllers` | Comma-separated list of the names above |
| `--seeds`, `--seed-start` | Seeds per family and where to start (seeds must be below 200) |
| `--families` | Subset of the six day types |
| `--profile`, `--reflector-profile` | Model profile for the Generator, and optionally for the Reflector and Curator |
| `--memory-file` | Playbook file for `llm_ace` (default: a fresh one per run) |
| `--run-name` | Output folder under `runs/` |

## Configuration

`config/model.json` holds the backends and the token budgets:

- `profiles`: `ollama_qwen3-8b` (default), `openrouter`, `groq`, and `mock`. Any model on Ollama or any OpenAI-compatible provider is just another profile. Add a new kind of backend by subclassing `LLMClient`, registering it with `@register_backend`, and adding a profile.
- `budget`: per-role (generator, reflector, curator) lists of context sizes and reply limits. When a call is cut off or the context fills, only the limit that failed is raised one step for that role.
- `debug_log`: path for a call-by-call log of every prompt and reply.

The environment lives in `config/assets.json`, `economics.json`, `private_environment_rules.json`, and `price_reference.json`.

## Outputs

Everything is written under `runs/<run-name>/`: a `manifest.json`, per-controller ledgers and `summary.json`, and for `llm_ace` the playbook (`memory.json`), a log of every Reflector and Curator step (`ace_log.jsonl`), and a playbook snapshot after each day.

## Project layout

```
run.py              command-line entry point
config/             environment parameters and model profiles
env/                scenarios, simulator, actions and validation (no LLM code)
controllers/        baselines, LLM controller, ACE controller
llm/                model backends and adaptive token/context budgets
ace_learning/       digest of a finished day, and the playbook
tools/              price reference, run comparison, battery analysis, Reflector model selection
tests/              pytest suite
runs/               outputs (not versioned)
```

## Troubleshooting

- **Calls get slower during a long run, CPUs are all busy, swap fills up.** I saw this after an Ollama update: the newer runner keeps a cache of earlier prompts in system memory, and a simulation's prompts share little, so the cache grows without being reused. I added Ollama settings to turn that cache off and to keep one model loaded at a time (`LLAMA_ARG_CACHE_RAM`, `OLLAMA_MAX_LOADED_MODELS=1`). Check `ollama ps` to see how much of the model is on the GPU, and `free -h` for memory.

## Two observations (two runs, not a verdict)

Setup: `qwen3:8b` via Ollama as Generator, temperature 0, 6 families × 5 seeds = 30 days per controller. Run A used `qwen3:8b` for the Reflector and Curator as well. Run B used `phi4:14b` for the Reflector and Curator.

| Controller | Mean score | Note |
|---|---|---|
| `heuristic` | 87.1 | |
| `llm` (memoryless) | 67.6 | −19.5 vs. heuristic (95% interval −26.8 to −12.7) |
| `llm_ace`, run A (`qwen3:8b` reflects) | 61.2 | −6.4 vs. memoryless (95% interval −11.7 to −1.6) |
| `llm_ace`, run B (`phi4:14b` reflects) | XX.X | XX.X vs. memoryless (95% interval XX.X to XX.X) |

**What the two runs show.** In run A, ACE got worse over time, and its playbook held five near-duplicate rules, all of the form "avoid discharging" or "stay idle", one of them backwards (it said to avoid discharging when fees are low and prices high). Run B's playbook had more varied rules, two of them clearly useful (do not charge when the import price is high; avoid exporting when solar is high and the export limit binds), and others that were vague or harmful.

**The hidden rule was never found, because it was never triggered.** The thermal rule fired twice in 30 days for the memoryless controller and not at all in the ACE runs, so there was nothing for the Reflector to learn from. One possible reason is that the Generator is cautious and rarely uses the highest discharge power when the battery is hot. Whether a stronger model would trigger it, and then learn it, is untested.

**Lessons so far.**

- **Code should control what enters the playbook.** The rules a model writes cannot be trusted as they are: they included an inverted rule, a fee read as income, references to a baseline score the Generator never sees, and conditions the Generator cannot observe. Plain code can reject such rules before they enter the playbook, or hold them for a person to approve. I have not applied either in these runs, so the playbooks above are unfiltered.
- **Rule quality depends on what the model is shown.** When the digest left out a sign, the fee hours, or the battery temperature, the Reflector could not get them right, however good the model.
- **It also depends on model strength.** The same digest produced very different rules from `qwen3:8b` and `phi4:14b`, and a reasoning model was too slow and still wrong.

[experiments.md](experiments.md) lists experiments you can try.

## Tests

```bash
python -m pytest tests
```

## Limitations

One simulated site with a toy price model, few seeds, small local models, lightly tuned prompts, and a simplified, single-pass version of ACE. Only two Reflector models were compared in full runs. The hidden rule was never triggered in the ACE runs, so its detection was not tested. Treat the numbers as a worked example.

## Credits

ACE comes from Zhang et al., *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models* (ICLR 2026), [arXiv:2510.04618](https://arxiv.org/abs/2510.04618). This project is an independent, simplified adaptation for learning purposes and is not affiliated with the authors. Their reference implementation is at [github.com/ace-agent/ace](https://github.com/ace-agent/ace).
