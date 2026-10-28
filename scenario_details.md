# Scenario details

This document describes the simulated energy system, how scenarios are generated, and the simplifying assumptions behind them. It says nothing about how controllers decide; for that, see the README.

Everything here is implemented in `env/simulator.py`, `env/scenarios.py`, `env/actions.py` and the files in `config/`. If this document and the code disagree, the code is right.

Conventions: energy in kWh, power in kW, temperature in °C, money in plain currency units (cu). Grid power is positive when importing and negative when exporting.

## 1. The site

A small commercial microgrid with four parts: solar panels, a battery, EV chargers, and a grid connection where energy can be bought (import price) and sold (export price). The site has no other electrical load. One episode is one day: 24 steps of 1 hour, one decision per step. The battery configuration is identified as `bess_A` in `config/assets.json`.

| Parameter | Value | Where |
|---|---|---|
| Step length | 1 h | `assets.json` |
| Solar peak | 150 kW (times a per-family scale) | `assets.json` |
| Battery capacity | 200 kWh | `assets.json` |
| Usable energy window | 20 to 180 kWh (10% to 90%) | `assets.json` |
| Battery rated power | 100 kW charge, 100 kW discharge | `assets.json` |
| Efficiency | 0.95 charging, 0.95 discharging | `assets.json` |
| Initial battery energy | 100 kWh (160 kWh in the two spike families) | `assets.json`, `scenarios.py` |
| Initial battery temperature | 28 °C | `assets.json` |
| Battery thermal coefficients | heating 4.0, cooling 0.2 | `assets.json` |
| Grid import / export limit | 150 kW / 100 kW | `assets.json` |
| EV site limit (all chargers together) | 80 kW | `assets.json` |
| Export price ratio | export price = 0.9 × import price | `economics.json` |
| Unmet EV energy penalty | 0.6 cu per kWh missing at departure | `economics.json` |
| Terminal battery value | 0.10 cu per kWh | `economics.json` |

There is no load besides EV charging. Everything the site produces or stores is either used by the EVs, stored, or traded with the grid.

## 2. What happens in one hour

1. The simulator produces an observation for the current hour (section 6).
2. The controller picks one battery mode and one EV mode (section 3).
3. Deterministic code translates the two modes into kW setpoints.
4. A validator adjusts the setpoints so they respect physical and grid limits (section 4).
5. The simulator executes the step: it updates battery energy and temperature and the EV energy still needed, computes the grid flow, and applies the money terms (section 5).
6. The reward is returned and the next hour starts. After hour 23 the day ends.

Controllers never produce kW directly. They choose modes, and code does the rest. This keeps the physics out of the model's responsibilities.

## 3. Decisions: the modes

**Battery mode**, as a fraction of rated power:

| Mode | Effect |
|---|---|
| `IDLE` | Nothing |
| `CHARGE_LOW` | Charge at 25% (25 kW) |
| `CHARGE_HIGH` | Charge at 75% (75 kW) |
| `DISCHARGE_LOW` | Discharge at 25% (25 kW) |
| `DISCHARGE_HIGH` | Discharge at 75% (75 kW) |

Charging is offered only when the battery is below its upper energy bound, and discharging only when it is above its lower bound.

**EV mode.** Each EV group is connected from its arrival hour until its departure hour. For each connected group, earliest departure first:

- `cap = min(max_charge_kw, remaining_kwh / step_hours)`
- `MAX_AVAILABLE`: request `cap`
- `DEADLINE_AWARE`: request `min(cap, remaining_kwh / max(1, departure_hour - hour))`, the pace that just finishes in time
- `DEFER`: request 0

Each request is limited by what is left of the 80 kW site budget, so groups with earlier departures get served first. The controller never chooses solar curtailment; only the validator uses it.

## 4. Physical limits and the validator

The validator runs before the simulator and records each change as an *intervention*:

| Intervention | When |
|---|---|
| `charge_energy_bound` | Charge request exceeds what fits below the upper energy bound |
| `discharge_energy_bound` | Discharge request exceeds what the battery can deliver above the lower bound |
| `import_limit_reduce_charge` | Grid import would exceed 150 kW: battery charging is reduced first |
| `import_limit_reduce_ev` | Still above the import limit: EV charging is scaled down proportionally |
| `export_limit_reduce_discharge` | Grid export would exceed 100 kW: battery discharge is reduced first |
| `export_limit_curtail_solar` | Still above the export limit: solar is curtailed (wasted) |

The simulator checks the energy bounds again and reports them as *physical clips*. Charging and discharging in the same step is impossible by construction. The grid balance is `grid_kw = EV charging + battery charge - battery discharge - (solar - curtailed solar)`.

## 5. Reward and score

Each hour:

```
reward = export_revenue - import_cost - injection_fee - ev_unmet_penalty
export_revenue   = export_price * exported_kWh
import_cost      = import_price * imported_kWh
ev_unmet_penalty = 0.6 * kWh still missing, charged in the departure hour's step
```

At the end of the day a terminal adjustment values the energy left in the battery relative to the start:

```
terminal_adjustment = terminal_value_per_kwh * (final_energy * m - initial_energy)
score = sum of the 24 hourly rewards + terminal_adjustment
```

With a terminal value of 0.10 cu/kWh, each kWh left in the battery above the starting level is worth 0.10 cu, and each kWh below it costs 0.10 cu. Buying a kWh stores only 0.95 kWh, worth 0.095 cu, so charging for the end-of-day value alone pays off only when the import price is below 0.095 cu/kWh. In the price reference that is below the 0.5th percentile (0.0975 cu/kWh), so well under 1% of hours.

`m` is 1, except when a penalty period is still active at the end of the day (see section 11). The terminal value stops a controller from winning by emptying the battery before the day ends.

Every episode also records an *idle baseline*: the score of the same day played with the battery idle and EVs charged at deadline pace. It is a reference number for comparison and is never shown to a controller.

**Worked example** (a logged `clear_hot` day, 28 °C start, battery at 100 kWh):

- Hour 1, `CHARGE_LOW`: 25 kW bought at 0.1161 cu/kWh costs 2.90 cu. With 0.95 efficiency, 23.75 kWh is stored, so the battery goes from 100 to 123.75 kWh.
- Hour 2, `CHARGE_HIGH` (75 kW requested): only `(180 - 123.75) / 0.95 = 59.2` kW fits, so the validator cuts the request to 59.2 kW (`charge_energy_bound`).
- Hour 6, `DISCHARGE_HIGH`: 75 kW exported at 0.137 cu/kWh earns 10.28 cu; the battery drops by `75 / 0.95 = 78.9` kWh.

**Rule of thumb.** Buying 1 kWh stores 0.95 kWh, which delivers 0.95 × 0.95 = 0.9025 kWh back to the grid, sold at 0.9 × the later import price. A charge-then-discharge cycle only breaks even when the later import price is about 23% higher than the price paid (1 / 0.9025 / 0.9 = 1.23), before any fee or terminal value.

## 6. What a controller can see

Each prompt contains the current observation, the list of available battery modes, and the last 3 hours of history.

| Field | Meaning |
|---|---|
| `hour`, `hours_remaining` | Position in the day (remaining includes the current hour) |
| `solar_kw` | Solar output now |
| `battery_energy_kwh`, `battery_capacity_kwh` | State of charge and capacity |
| `battery_temperature_c`, `ambient_temperature_c` | Battery and outdoor temperature |
| `import_price_per_kwh`, `export_price_per_kwh` | Current prices |
| `import_price_percentile` | Where the current import price sits among typical prices, 0 = cheapest, 1 = most expensive (only if a price reference exists) |
| `grid_import_limit_kw`, `grid_export_limit_kw` | Grid limits |
| `ev_groups` | For each connected group: id, `remaining_energy_kwh`, `max_charge_kw`, `departure_hour`, `hours_left`, `min_kw_needed_now` |

History records show, per past hour: prices, solar, battery temperature, the modes chosen, the executed battery power, grid power, the reward and its components, warnings, interventions, physical clips, and the discharge efficiency actually used.

**There is no forecast.** A controller sees only the current hour's prices and solar and its own past hours. It does not see private simulator state, its own earlier decision summaries, or any other episode.

## 7. How scenarios are generated

A scenario is a function of `(seed, family)`. The generator uses a random source seeded with `"{seed}:{family}"`, so the same pair always gives the same day, and scenarios do not depend on controller actions.

To look at one:

```python
from env.scenarios import generate_scenario
sc = generate_scenario(seed=0, family="clear_hot")
print(sc["import_price"][:6], sc["ev_groups"])
```

**Families**

| Family | Cloud | Temp range (°C) | Solar scale | Price spike | Start energy |
|---|---|---|---|---|---|
| `clear_mild` | none | 16 to 27 | 1.0 | random | 100 |
| `clear_hot` | none | 26 to 40 | 1.0 | random | 100 |
| `cloudy` | 0.5 | 20 to 32 | 1.0 | random | 100 |
| `low_solar` | 0.2 | 18 to 30 | 0.4 | random | 100 |
| `hot_spike` | none | 27 to 41 | 1.0 | forced, large | 160 |
| `mild_spike` | none | 16 to 27 | 1.0 | forced, large | 160 |

**Solar.** A half-sine from 6:00 to 18:00 scaled to the peak, multiplied each hour by `1 - cloud * U(0, 1)`.

**Ambient temperature.** A daily sine wave between the family's minimum and maximum, peaking at 15:00, plus noise of 0.3 °C.

**Prices.** Per hour, import price in cu/kWh:

```
p = 0.12
  + 0.15 * exp(-((h - 8) / 1.5)^2)            # morning bump
  + 0.25 * exp(-((h - 18) / 1.5)^2)           # evening bump
  + spike * exp(-((h - spike_hour) / width)^2)
  + noise(0, 0.01), floored at 0.02
export price = 0.9 * p
```

In the spike families `spike_hour` is 17, 18, or 19, `spike = 0.30`, and `width = 2`. In the others the spike hour is one of 7, 8, 9, 17, 18, 19 and the height is drawn from `{0, 0, 0.15}`, so two days in three have no extra spike. Width is 1.

**EV groups.** Two or three groups per day with ids `EV_GROUP_01` and so on:

- arrival hour uniform in 7 to 10, departure hour uniform in 16 to 20
- maximum charge power drawn from 20, 30, or 40 kW
- required energy `min(U(40, 120), 0.8 * kW * (departure - arrival))`, rounded to 0.1 kWh, so one group could always finish alone, though groups share the 80 kW site limit

## 8. Episodes, seeds and the evaluation setup

- An **episode** is one `(family, seed)` pair. The experiment runs the Cartesian product of families and seeds.
- Episodes run in round-robin order, all families for seed 0, then seed 1, and so on, so a learning controller sees every family early and late.
- Default: 6 families × 5 seeds = 30 episodes.
- Seeds from 200 up are reserved (200 to 209 for penalty tuning, 1000 to 1199 for the price reference), so evaluation seeds must stay below 200.
- The price reference (`config/price_reference.json`) holds 201 quantile points (every 0.5%) of the 28,800 hourly import prices from seeds 1000 to 1199, all 6 families, 24 hours each. It is generated offline by `python -m tools.make_price_reference` from the project root, and never used for evaluation. The percentile in the observation is the number of quantile points at or below the current price, divided by 201. If the file is missing, the field is omitted.

Each episode produces a summary with: cash, terminal adjustment, score, thermal events, unmet EV energy, validator interventions, fallbacks, grid-limit violations, and the idle baseline score.

## 9. Simplifying assumptions

- Hourly resolution; power is constant within an hour.
- One electrical node; no network losses, voltage, or reactive power.
- Battery: fixed efficiencies, five discrete power levels, no cycle or calendar aging. The only degradation is the thermal rule in section 11.
- EV groups are aggregated: each group is one charger with one maximum power. Charging only, no vehicle-to-grid, continuous power, integer arrival and departure hours.
- Prices are synthetic and deterministic apart from small noise; the export price is always 0.9 × the import price. There are no forecasts.
- Weather, prices, and EV demand are independent of each other except through the family's settings.
- Grid limits are hard limits enforced by the validator. A violation flag exists but a validated action should not trigger it.
- The same site parameters apply to every family; only weather, spikes, and the starting energy differ.
- The terminal value is a simplification that stands in for the value of the stored energy after the day ends.

## 10. Reading a run's output

A run writes, per controller and episode, `<episode>.public.jsonl` (observation, chosen modes, requested and validated setpoints, interventions, executed power, grid power, reward components, warnings) and `<episode>.private.jsonl` (simulator-internal state that controllers never see). It also writes `summary.json` and a `manifest.json` that records the config files used.

## 11. Spoiler: the hidden rules

Controllers are told that the environment has rules they have not been told. Stop reading if you want to find them yourself. The rules live in `config/private_environment_rules.json`.

**Thermal lockout.** The battery temperature follows:

```
T_next = T + k_heat * ((charge / max_charge)^2 + (discharge / max_discharge)^2) - k_cool * (T - ambient)
k_heat = 4.0, k_cool = 0.2
```

A step triggers the rule when the battery temperature at the start of the step is at least 35 °C and the executed discharge power is at least 60% of rated power (60 kW). `DISCHARGE_HIGH` (75 kW) qualifies; `DISCHARGE_LOW` (25 kW) does not. A `ThermalDegradationWarning` is raised, and for the next 3 steps the discharge power is capped at 0% of rated power, which means the battery cannot discharge at all. A new trigger during a lockout resets the 3-step timer. The configuration also has a discharge-efficiency multiplier, but it is set to 1.0, so it has no effect here, and the terminal value is therefore unaffected.

The rule mostly matters in the hot families, where daytime ambient temperature reaches 40 to 41 °C. The cost is indirect: a lockout during an expensive evening hour removes the opportunity to discharge exactly when it pays most.

**Injection fee.** Energy exported in hours 9 to 15 (inclusive) pays a fee of 0.08 cu per kWh. It shows up as `injection_fee` in the reward components. At a typical base import price of about 0.12 cu/kWh the export price is about 0.108 cu/kWh, so in those hours the fee takes roughly three quarters of the export revenue. Exporting at full power around midday is much less attractive than the price alone suggests.

Neither rule appears in the prompt. A controller can only learn about them through the reward components, warnings, and clips in the history. One baseline, `heuristic_trap_aware`, knows the 35 °C threshold and serves as an oracle reference for what is achievable when the thermal rule is known.