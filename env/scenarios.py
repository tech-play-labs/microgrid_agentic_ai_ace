"""Seeded synthetic 24-hour scenarios. External conditions are independent of controller actions."""
import math, random

_SPIKE = dict(force_spike=True, spike_hours=[17, 18, 19], spike_height=0.30, spike_width=2.0, initial_energy_kwh=160.0)
FAMILIES = {
    "clear_mild":   dict(cloud=0.0, t_min=16, t_max=27, solar_scale=1.0),
    "clear_hot":    dict(cloud=0.0, t_min=26, t_max=40, solar_scale=1.0),
    "cloudy":       dict(cloud=0.5, t_min=20, t_max=32, solar_scale=1.0),
    "low_solar":    dict(cloud=0.2, t_min=18, t_max=30, solar_scale=0.4),
    "hot_spike":    dict(cloud=0.0, t_min=27, t_max=41, solar_scale=1.0, **_SPIKE),
    "mild_spike":   dict(cloud=0.0, t_min=16, t_max=27, solar_scale=1.0, **_SPIKE),
}

def generate_scenario(seed, family, solar_peak_kw=150.0, export_ratio=0.9):
    f = FAMILIES[family]
    rng = random.Random(f"{seed}:{family}")
    solar, amb, imp, exp = [], [], [], []
    if f.get("force_spike"):
        spike_hour = rng.choice(f["spike_hours"]); spike = f["spike_height"]
    else:
        spike_hour = rng.choice([7, 8, 9, 17, 18, 19])
        spike = rng.choice([0.0, 0.0, 0.15])
    width = f.get("spike_width", 1.0)
    for h in range(24):
        base = max(0.0, math.sin(math.pi * (h - 6) / 12)) if 6 <= h < 18 else 0.0
        cloud = 1.0 - f["cloud"] * rng.random() if base > 0 else 1.0
        solar.append(round(solar_peak_kw * f["solar_scale"] * base * cloud, 3))
        mid, amp = (f["t_min"] + f["t_max"]) / 2, (f["t_max"] - f["t_min"]) / 2
        amb.append(round(mid + amp * math.sin(2 * math.pi * (h - 9) / 24) + rng.gauss(0, 0.3), 2))
        p = (0.12 + 0.15 * math.exp(-((h - 8) / 1.5) ** 2) + 0.25 * math.exp(-((h - 18) / 1.5) ** 2)
             + spike * math.exp(-((h - spike_hour) / width) ** 2) + rng.gauss(0, 0.01))
        p = round(max(0.02, p), 4)
        imp.append(p); exp.append(round(p * export_ratio, 4))
    ev = []
    for i in range(rng.randint(2, 3)):
        arr = rng.randint(7, 10); dep = rng.randint(16, 20); kw = rng.choice([20.0, 30.0, 40.0])
        energy = round(min(rng.uniform(40, 120), 0.8 * kw * (dep - arr)), 1)
        ev.append(dict(id=f"EV_GROUP_{i+1:02d}", arrival_hour=arr, departure_hour=dep,
                       required_energy_kwh=energy, max_charge_kw=kw))
    sc = dict(seed=seed, family=family, solar_kw=solar, ambient_c=amb,
              import_price=imp, export_price=exp, ev_groups=ev)
    if "initial_energy_kwh" in f:
        sc["initial_energy_kwh"] = f["initial_energy_kwh"]
    return sc
