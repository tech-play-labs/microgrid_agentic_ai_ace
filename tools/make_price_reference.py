"""Build config/price_reference.json: quantiles of typical import prices from fixed reserved seeds (never used for evaluation)."""
import json
from run import load
from env.scenarios import generate_scenario, FAMILIES

REFERENCE_SEEDS = range(1000, 1200)

def main(n_quantiles=201):
    assets, eco, _ = load()
    prices = sorted(p for fam in FAMILIES for s in REFERENCE_SEEDS
                    for p in generate_scenario(s, fam, assets["solar_peak_kw"], eco["export_price_ratio"])["import_price"])
    q = [prices[min(len(prices) - 1, round(i * (len(prices) - 1) / (n_quantiles - 1)))] for i in range(n_quantiles)]
    json.dump(dict(seeds=[REFERENCE_SEEDS.start, REFERENCE_SEEDS.stop - 1], families=list(FAMILIES), n_prices=len(prices), quantiles=q),
              open("config/price_reference.json", "w"))
    print("n_prices", len(prices), "p50", q[100], "p90", q[180], "p95", q[190], "max", q[-1])

if __name__ == "__main__":
    main()
