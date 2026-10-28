## What went wrong along the way, and what I changed

The first complete runs did not work as planned. Most of the lessons in this post came from reading logs after things broke, so this section lists them. If you build something similar, you will probably hit some of the same problems.

### Problems that are fixed in the code or the tooling

**1. A Reflector reply cut off in the middle of its JSON.**
The first symptom was a validation error on the Reflector's output: the JSON ended partway through a list. The model had not written bad JSON. It hit the 300-token reply limit. The retry used the same prompt and failed the same way.
*What changed:* the code now reads the backend's stop reason (`done_reason: length` on Ollama, `finish_reason: length` on API providers) and tells apart "the model wrote something invalid" from "the model ran out of room". Failed replies are logged with up to 2,000 characters, so you can see where they stopped.

**2. Raising the limits for everything made everything slow.**
The obvious fix was to raise `max_tokens` and `num_ctx` globally. But each hourly decision needs only a few hundred tokens, while the Reflector's reply is much longer. A big context for every call made all of them slower, and on Ollama a change of context size between requests can make the server reload the model.
*What changed:* each role (Generator, Reflector, Curator) has its own limits and starts small. When a call overflows, only that role is raised one step, the call is repeated at once, and the new value is kept for the rest of the run.

**3. The two limits are different problems.**
My first version raised the context size and the reply limit together. They fail differently. A reply cut off with plenty of context left needs a higher `max_tokens`, not more memory. A prompt that fills the context needs a bigger `num_ctx`, not a longer reply.
*What changed:* each role has two independent counters, one per limit, and the code raises only the one that failed. The steps live in `config/model.json`:

```json
"budget": {
  "generator": {"num_ctx": [4096, 8192, 16384], "max_tokens": [300, 600, 1200]},
  "reflector": {"num_ctx": [4096, 8192, 16384], "max_tokens": [800, 1500, 2500]},
  "curator":   {"num_ctx": [4096, 8192, 16384], "max_tokens": [800, 1500, 2500]}
}
```

The config is validated at start-up, so a malformed ladder stops the run before any model call. After each run, `summary.json` records every escalation and the final step per role. In my first full run all counters stayed at zero, which means the defaults were enough for `qwen3:8b`.

**4. API backends do not behave like Ollama.**
A hosted model ignores `num_ctx`, so comparing a prompt's size against it produced false "context full" alarms and pointless escalations. And one of the API code paths was not passing the ladder's `max_tokens` at all.
*What changed:* only clients that actually honor `num_ctx` get the context checks, and the ladder's reply limit is the one that is sent. A profile's own `max_tokens` is only a fallback for calls made without a budget.

**5. Ordinary bugs, found by reading the code.**
Reviewing the files after the changes turned up several plain mistakes: a controller that did not pass the budget to its parent class, an exception raised by the budget code that nothing caught inside the Reflector call, an attribute used before it was defined, and imports that still used the old flat module names after the reorganization. None of these is about language models, and all would have broken a run. Fake clients in the tests are a cheap way to exercise failure paths like these without a model.

**6. The project layout.**
The code started as a flat folder. As it grew, it became hard to see which file did what. It is now split by role: `env/` for the simulated world, `controllers/` for anything that picks actions, `llm/` for model access, `ace_learning/` for the digest and the playbook, and `tools/` for offline helpers.

**7. Seeing the problem at all.**
A single average score hid what was happening. The comparison tool pairs each controller's days, reports intervals, wins and losses, per-family differences, first-half against second-half, and the cumulative gap by round. That is what showed ACE drifting downward as its playbook grew. The debug log now also records the limits each request was sent with.

**8. Choosing the Reflector model with a script, not by feel.**
Picking a Reflector by trying it for a few days is slow and anecdotal. I wrote a script that replays one saved day through each candidate model, twice with different seeds, and scores the replies: do the numbers it cites exist in the digest, does it read a fee as income or say to charge when the price is high, are its rules specific, are they short, and is the JSON valid.
*What I found:* `deepseek-r1:8b` failed both repeats (invalid JSON after 2,500 tokens, about 9,000 to 10,000 characters of reasoning, 531 to 565 seconds per call). `qwen3:8b` was fast, but both of its replies told the operator to charge when the import price was high. `gemma3:12b` gave the best single reply, the only one that named the fee hours (9 to 15), but it varied between repeats and hedged a lot. `phi4:14b` wrote the most concrete rules, with price-percentile thresholds. Calls took between about 80 and 330 seconds on my laptop.
*A caution:* the automatic score was not always right. It flagged `phi4:14b` for errors that its rules did not contain, because it read the whole reply, including the description of what happened, as advice. I ended up reading the rules themselves. A script narrows the field. It does not replace reading.

**9. Ollama became slow after an update.**
Partway through, the same run that took about 15 seconds per call slowed to about 90 seconds after roughly 100 calls, with every CPU core busy. The model was running 55% on the CPU and 45% on a 4 GB GPU, memory was full, swap was full, and the server process held about 10 GB. My reading is that the newer Ollama runner keeps a cache of earlier prompts in system memory. My prompts share only their first few hundred tokens, so the cache fills up with entries that are never reused, and the machine starts swapping. The verbose server log that appeared at the same time also came from the runner, not from my code.
*What changed:* I added Ollama settings to turn off that prompt cache and to keep only one model loaded at a time, to make the requests faster. Two things to remember: a different Ollama version or a different CPU/GPU split can change results slightly even at temperature 0, so record the version and run the compared controllers under the same setup.

### Problems that were found, with fixes still to apply or test

**10. The Reflector reads each day against a do-nothing day.**
It is a cheap, label-free reference, but it seems to nudge the rules toward inaction. The playbook written with `qwen3:8b` as Reflector had five rules, all of the form "avoid discharging" or "keep the battery idle", two of them explicitly "compared to the do-nothing baseline", a number the Generator never sees. I have not proven this is the cause. The options are a harder reference such as the heuristic's score, hour-by-hour comparisons, or no reference at all. All of these can be tested by replaying saved days through the Reflector and Curator without running new simulations.

**11. A fee was read as income.**
With a reasoning model as Reflector, the playbook after three days contained a rule telling the operator to discharge "to earn injection_fee". In the simulator that fee is a charge subtracted from the reward. The digest named the number but did not say which way it counts. The planned fix is to state the sign explicitly in the digest ("a charge subtracted from reward") and to add a line to the Reflector prompt saying all fees and penalties are costs.

**12. Code can reject harmful rules, or ask a human. I have not applied either.**
The playbook is just text that a model proposes, so plain code can sit between the Curator and the playbook. Cheap automatic checks would catch several of the problems above: reject a rule that mentions the baseline score, reject a rule whose condition the Generator cannot see when it decides (for example "when injection fees are applied", with no hours given), and reject a rule that reads a fee or penalty as income. A stronger option is to hold every new or changed rule until a person approves it, which is slow but safe for a system that edits its own prompt. In the runs described here, models produced rules are filtered with minimal checks.

**13. A model cannot learn from a situation it never meets.**
The hidden thermal rule fired twice in 30 days for the memoryless controller and never for ACE. I also compared two Reflector models (`qwen3:8b` and `phi4:14b`) with the same Generator, and neither playbook contained a thermal rule, simply because no thermal event happened in those runs, so there was nothing to reflect on. One possible reason is that the Generator is cautious and rarely picks the highest discharge power when the battery is hot. Replaying the simulator also showed that only the hot families (`clear_hot`, `hot_spike`) can trigger the rule for a typical controller, which is 10 of the 30 days, and the mild families never can. Whether a stronger model would behave differently is open: it might discharge harder, trigger the rule, and learn from it, or it might avoid heat problems from general knowledge and never see the rule fire. I have not tested this. Another option is to run the Reflector on days played by another controller that does trigger the rule, then test on fresh seeds.

**14. Reasoning models need a different budget, and may not be worth it.**
With thinking left on, `deepseek-r1:8b` used its whole 2,500-token reply budget before finishing, and the stage failed with a clear error. Turning thinking off in the request did not stop it from reasoning. I tried the recommended temperature of 0.6 and adjusted the context and output limits, and neither helped: the model kept producing wrong content, including the fee-as-income mistake, and was too slow to be practical, taking about 9 minutes per call on my machine. I dropped it. If you still want a reasoning model, give the Reflector and Curator much bigger ladders (for example 3,000, 6,000, and 10,000 tokens) and a long timeout.

**15. Small GPUs mean model swapping.**
With 4 GB of VRAM, two different models cannot both sit on the GPU. Ollama unloads one to load the other, so using a different Reflector model costs about two reloads per simulated day, and a 14B model runs mostly on the CPU. Using one model for both, or an API model for the Reflector only, avoids that.

### A short checklist for your own runs

- Check whether a failed reply was invalid or merely cut off.
- Record the limits each call was sent with, and the escalations at the end of a run.
- Compare controllers on paired days, and look at drift over time, not only the mean.
- Pick the Reflector by replaying a saved day through each candidate, then read the rules it wrote.
- Read the playbook after the first three days, and ask of each rule: is the sign right, does it name an observable condition, and does it duplicate another?
- Treat a rule's "support" count as the model's opinion until something checks outcomes.
- Check the server too: record the Ollama version, cap its prompt cache, and watch memory and swap during long runs. 
  ```
  # For example, disable caching in Ollama to speed up requests that do not benefit from prompt caching
  [Service]
  Environment="LLAMA_ARG_CACHE_RAM=0"
  Environment="OLLAMA_MAX_LOADED_MODELS=1"
  ``` 
- Make sure the situation you want the system to learn from actually happens in your runs.
- Test failure paths with fake clients before spending hours of model time.
