"""Performance and optimization quality test suite for OptiStock API.

Generates realistic inventory data, tests all tiers, validates profit
maximization logic, and benchmarks performance.
"""

import sys
import os
import time
import tracemalloc
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import ItemInput, HistoricalDataPoint, OptimizeRequest
from api.routes import run_single_item, run_batch_psl_optimization, run_psl_optimization

# ─── Realistic Data Generator ───────────────────────────────────────────────

def generate_historical_data(
    item_id: str,
    base_demand: float,
    lead_time_days: float,
    cost: float,
    sale_price: float,
    days: int = 400,
    seasonal: bool = True,
    stockout_periods: list | None = None,
    seed: int = 42,
) -> list[HistoricalDataPoint]:
    """Generate realistic daily inventory history."""
    rng = np.random.RandomState(seed)
    data = []
    start_date = np.datetime64("2025-01-01")

    for d in range(days):
        date = start_date + np.timedelta64(d, "D")
        day_of_year = d % 365

        # Seasonal demand multiplier
        if seasonal:
            # Holiday spike (Nov-Dec)
            if 305 <= day_of_year <= 365:
                mult = 1.8
            # Summer dip
            elif 150 <= day_of_year <= 240:
                mult = 0.7
            # Spring bump
            elif 60 <= day_of_year <= 120:
                mult = 1.2
            else:
                mult = 1.0
        else:
            mult = 1.0

        # Daily demand with noise
        mean_demand = base_demand * mult
        if mean_demand > 1:
            demand = max(0, rng.poisson(mean_demand))
        else:
            demand = max(0, rng.poisson(mean_demand * 10) / 10) if rng.rand() < mean_demand else 0

        # Available stock (simulate occasional stockouts)
        is_stockout = False
        if stockout_periods:
            for start, end in stockout_periods:
                if start <= d <= end:
                    is_stockout = True
                    break

        available = 0.0 if is_stockout else max(0, rng.randint(int(base_demand * 7), int(base_demand * 30 + 1)))
        mercury_qty = float(rng.randint(0, max(1, int(base_demand * 2))))

        data.append(HistoricalDataPoint(
            date=str(date),
            quantity=float(demand),
            available=available,
            mercury_order_quantity=mercury_qty,
        ))

    return data


def generate_lead_times(base_lt: float, n: int = 30, seed: int = 42) -> list[float]:
    """Generate realistic lead time history with variance."""
    rng = np.random.RandomState(seed)
    return [max(1.0, rng.normal(base_lt, base_lt * 0.15)) for _ in range(n)]


def create_item_profile(name: str, base_demand: float, lt: float, cost: float, sale_price: float,
                        stockout_periods=None, seed=42) -> ItemInput:
    """Create a fully-populated ItemInput with realistic history."""
    hist = generate_historical_data(name, base_demand, lt, cost, sale_price,
                                    stockout_periods=stockout_periods, seed=seed)
    lt_history = generate_lead_times(lt, seed=seed)
    current_avail = int(base_demand * 14)  # ~2 weeks of stock

    return ItemInput(
        item_id=name,
        current_available=current_avail,
        on_order_qty=0,
        back_order_qty=0,
        order_frequency_days=7,
        cost=cost,
        sale_price=sale_price,
        length=1.0, width=1.0, height=1.0,
        payment_terms_days=30,
        sales_terms_days=30,
        lead_time_days=lt,
        historical_lead_times=lt_history,
        historical_data=hist,
    )


# ─── Test Items ──────────────────────────────────────────────────────────────

ITEMS = {
    "high_volume":   dict(base_demand=50, lt=14, cost=8.0,  sale_price=12.0),   # 50% margin
    "medium_volume": dict(base_demand=10, lt=17, cost=15.0, sale_price=25.0),   # 40% margin
    "low_volume":    dict(base_demand=2,  lt=21, cost=30.0, sale_price=60.0),   # 50% margin
    "high_margin":   dict(base_demand=5,  lt=14, cost=10.0, sale_price=50.0),   # 80% margin
    "low_margin":    dict(base_demand=20, lt=14, cost=18.0, sale_price=20.0),   # 10% margin
}

STOCKOUT_ITEMS = {
    "with_stockouts": dict(base_demand=15, lt=14, cost=12.0, sale_price=22.0,
                           stockout_periods=[(100, 115), (250, 260)]),
}


def make_item(key, seed=42):
    if key in ITEMS:
        p = ITEMS[key]
        return create_item_profile(key, p["base_demand"], p["lt"], p["cost"], p["sale_price"], seed=seed)
    p = STOCKOUT_ITEMS[key]
    return create_item_profile(key, p["base_demand"], p["lt"], p["cost"], p["sale_price"],
                               stockout_periods=p.get("stockout_periods"), seed=seed)


# ─── Test Functions ──────────────────────────────────────────────────────────

def test_tier_comparison():
    """Compare basic/premium/elite tier results for the same items."""
    print("\n" + "=" * 70)
    print("TIER COMPARISON TEST")
    print("=" * 70)

    item = make_item("medium_volume")
    results = {}

    for tier in ["basic", "premium", "elite"]:
        t0 = time.perf_counter()
        result = run_single_item(item, tier, 0.14)
        elapsed = (time.perf_counter() - t0) * 1000
        results[tier] = (result, elapsed)

        print(f"\n  {tier.upper()} tier ({elapsed:.0f}ms):")
        print(f"    PSL={result.optimal_psl}, Order={result.recommended_order_qty}, "
              f"Profit=${result.expected_profit:.2f}/day")
        print(f"    ADS={result.ads:.2f}, Var={result.variance:.2f}, Source={result.demand_source}")

    # Validate: all tiers should produce positive PSL for positive demand
    passed = all(r.optimal_psl > 0 for r, _ in results.values())
    print(f"\n  ✅ All tiers produce positive PSL" if passed else "  ❌ Some tier produced zero PSL")
    return passed


def test_batch_optimization():
    """Test batch optimization with 3, 5, 10 items."""
    print("\n" + "=" * 70)
    print("BATCH OPTIMIZATION TEST")
    print("=" * 70)

    all_keys = list(ITEMS.keys()) + list(STOCKOUT_ITEMS.keys())
    batch_sizes = [3, 5, 10]
    passed = True

    # We need 10 items, so create extras
    extra_items = []
    for i in range(5):
        extra_items.append(create_item_profile(f"extra_{i}", base_demand=3 + i * 5, lt=14 + i,
                                                cost=10.0 + i * 5, sale_price=20.0 + i * 10, seed=100 + i))

    for size in batch_sizes:
        items = [make_item(k) for k in all_keys[:min(size, len(all_keys))]]
        while len(items) < size:
            items.append(extra_items[len(items) - len(all_keys)])

        # Warm up numba for first batch call
        t0 = time.perf_counter()
        results = run_batch_psl_optimization(items, "basic", 0.14)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n  Batch size {size} ({elapsed:.0f}ms):")
        for r in results:
            print(f"    {r.item_id}: PSL={r.optimal_psl}, Order={r.recommended_order_qty}, "
                  f"Profit=${r.expected_profit:.2f}/day")

        # Validate: all items with demand should have positive PSL
        nonzero = [r for r in results if r.ads > 0]
        if not all(r.optimal_psl > 0 for r in nonzero):
            print(f"  ❌ Some items with demand got zero PSL in batch {size}")
            passed = False
        else:
            print(f"  ✅ All items with demand have positive PSL")

    return passed


def test_profit_maximization():
    """Validate that optimization makes business sense."""
    print("\n" + "=" * 70)
    print("PROFIT MAXIMIZATION VALIDATION")
    print("=" * 70)

    passed = True

    # Test 1: Higher margin → higher PSL (relative to demand)
    hm_item = make_item("high_margin")   # 80% margin, demand=5
    lm_item = make_item("low_margin")    # 10% margin, demand=20

    hm_result = run_single_item(hm_item, "basic", 0.14)
    lm_result = run_single_item(lm_item, "basic", 0.14)

    # Compare PSL/demand ratio (how many days of cover)
    hm_cover = hm_result.optimal_psl / max(hm_result.ads, 0.1)
    lm_cover = lm_result.optimal_psl / max(lm_result.ads, 0.1)

    print(f"\n  High margin (80%): PSL={hm_result.optimal_psl}, ADS={hm_result.ads:.1f}, "
          f"cover={hm_cover:.0f} days")
    print(f"  Low margin (10%):  PSL={lm_result.optimal_psl}, ADS={lm_result.ads:.1f}, "
          f"cover={lm_cover:.0f} days")

    # High margin should get relatively more coverage
    if hm_cover > lm_cover:
        print(f"  ✅ High-margin item gets more days of coverage")
    else:
        print(f"  ⚠️  Margin effect on coverage unclear (hm_cover={hm_cover:.1f}, lm_cover={lm_cover:.1f})")

    # Test 2: Cost of capital impact
    low_cap = run_single_item(make_item("medium_volume"), "basic", 0.05)
    high_cap = run_single_item(make_item("medium_volume"), "basic", 0.30)

    print(f"\n  Cost of capital 5%:  PSL={low_cap.optimal_psl}, Profit=${low_cap.expected_profit:.2f}")
    print(f"  Cost of capital 30%: PSL={high_cap.optimal_psl}, Profit=${high_cap.expected_profit:.2f}")

    if high_cap.optimal_psl < low_cap.optimal_psl:
        print(f"  ✅ Higher cost of capital → leaner inventory")
    elif high_cap.optimal_psl == low_cap.optimal_psl:
        print(f"  ⚠️  No difference (may be within tolerance)")
    else:
        print(f"  ❌ Higher cost of capital should reduce PSL")
        passed = False

    # Test 3: Profit is positive for all normal items
    print(f"\n  Profit positivity check:")
    for key in ITEMS:
        r = run_single_item(make_item(key), "basic", 0.14)
        status = "✅" if r.expected_profit > 0 else "❌"
        print(f"    {key}: ${r.expected_profit:.2f}/day {status}")
        if r.expected_profit <= 0:
            passed = False

    return passed


def test_edge_cases():
    """Test edge cases: zero demand, negative margin, extreme demand."""
    print("\n" + "=" * 70)
    print("EDGE CASE TESTS")
    print("=" * 70)

    passed = True

    # Zero demand item
    zero_item = ItemInput(
        item_id="zero_demand", current_available=10, cost=5.0, sale_price=10.0,
        lead_time_days=14, historical_data=[
            HistoricalDataPoint(date=f"2025-01-{d:02d}", quantity=0.0, available=100.0)
            for d in range(1, 30)
        ],
    )
    zr = run_single_item(zero_item, "basic", 0.14)
    print(f"\n  Zero demand: PSL={zr.optimal_psl}, Order={zr.recommended_order_qty}, "
          f"Warnings={zr.warnings}")
    if zr.optimal_psl == 0 or "No demand" in str(zr.warnings):
        print(f"  ✅ Zero demand handled correctly")
    else:
        print(f"  ❌ Zero demand should produce zero PSL or warning")
        passed = False

    # Negative margin
    neg_item = ItemInput(
        item_id="neg_margin", current_available=10, cost=20.0, sale_price=15.0,
        lead_time_days=14, historical_data=[
            HistoricalDataPoint(date=f"2025-01-{d:02d}", quantity=float(d % 3), available=100.0)
            for d in range(1, 30)
        ],
    )
    nr = run_single_item(neg_item, "basic", 0.14)
    print(f"\n  Negative margin: PSL={nr.optimal_psl}, Order={nr.recommended_order_qty}, "
          f"Warnings={nr.warnings}")
    if nr.optimal_psl == 0 or "Negative" in str(nr.warnings):
        print(f"  ✅ Negative margin handled correctly")
    else:
        print(f"  ❌ Negative margin should produce zero PSL or warning")
        passed = False

    # Very high demand
    high_item = create_item_profile("very_high", base_demand=200, lt=14, cost=5.0, sale_price=8.0, seed=99)
    hr = run_single_item(high_item, "basic", 0.14)
    print(f"\n  Very high demand (200/day): PSL={hr.optimal_psl}, Order={hr.recommended_order_qty}")
    if hr.optimal_psl > 0 and hr.expected_profit > 0:
        print(f"  ✅ High demand handled correctly")
    else:
        print(f"  ❌ High demand should produce positive PSL and profit")
        passed = False

    return passed


def test_performance_benchmarks():
    """Benchmark cold start, single item, batch optimization."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)

    # Memory baseline
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]

    # Cold start: import + first call
    t0 = time.perf_counter()
    from simulation.psl_optimizer import calc_opti_psl_3
    item = make_item("medium_volume")
    result = run_single_item(item, "basic", 0.14)
    cold_start = (time.perf_counter() - t0) * 1000
    print(f"\n  Cold start (import + first optimize): {cold_start:.0f}ms")

    # Single item (warmed up)
    times = []
    for tier in ["basic", "premium", "elite"]:
        t0 = time.perf_counter()
        run_single_item(make_item("medium_volume"), tier, 0.14)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Single item ({tier}): {elapsed:.0f}ms")

    # Batch optimization
    all_keys = list(ITEMS.keys()) + list(STOCKOUT_ITEMS.keys())
    extras = [create_item_profile(f"bench_{i}", base_demand=3 + i * 4, lt=14 + i,
                                  cost=10.0 + i * 5, sale_price=20.0 + i * 10, seed=200 + i)
              for i in range(8)]

    for size in [3, 5, 10]:
        items = [make_item(k) for k in all_keys[:min(size, len(all_keys))]]
        while len(items) < size:
            items.append(extras[len(items) - len(all_keys)])

        t0 = time.perf_counter()
        run_batch_psl_optimization(items, "basic", 0.14)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  Batch ({size} items): {elapsed:.0f}ms")

    # Memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n  Memory: baseline={baseline / 1e6:.1f}MB, peak={peak / 1e6:.1f}MB, "
          f"delta={(peak - baseline) / 1e6:.1f}MB")

    # Performance thresholds
    passed = True
    if cold_start > 30000:
        print(f"  ❌ Cold start too slow ({cold_start:.0f}ms > 30s)")
        passed = False
    else:
        print(f"  ✅ Cold start acceptable")

    avg_single = sum(times) / len(times)
    if avg_single > 10000:
        print(f"  ❌ Single item too slow ({avg_single:.0f}ms > 10s)")
        passed = False
    else:
        print(f"  ✅ Single item latency acceptable")

    return passed


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          OptiStock API — Performance & Quality Test Suite          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    results = {}
    tests = [
        ("Tier Comparison", test_tier_comparison),
        ("Batch Optimization", test_batch_optimization),
        ("Profit Maximization", test_profit_maximization),
        ("Edge Cases", test_edge_cases),
        ("Performance Benchmarks", test_performance_benchmarks),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  ❌ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print(f"\nOverall: {'ALL PASSED ✅' if all_pass else 'SOME FAILED ❌'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
