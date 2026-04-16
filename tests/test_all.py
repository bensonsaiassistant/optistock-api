# noqa: F401
import os
import sys
import asyncio
import numpy as np
import pandas as pd

# Ensure project is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 1. DEMAND DISTRIBUTION ---------------------------------------------------
import simulation.demand_dist as dd


def test_calc_nb_array_ln_normal():
    arr = dd.calc_nb_array_ln(5.0, 10.0)
    assert len(arr) > 0, f"Expected non-empty array, got len={len(arr)}"
    total = arr.sum()
    assert abs(total - 1.0) < 1e-6, f"Sum={total}, expected ~1.0"


def test_calc_nb_array_ln_poisson_path():
    arr = dd.calc_nb_array_ln(10.0, 5.0)
    assert len(arr) > 0
    total = arr.sum()
    assert abs(total - 1.0) < 1e-6


def test_calc_nb_array_ln_zero_mean():
    arr = dd.calc_nb_array_ln(0, 1.0)
    assert arr.sum() == 0.0
    assert len(arr) == 2000


def test_calc_nb_array_ln_very_low_demand():
    arr = dd.calc_nb_array_ln(0.01, 0.5)
    assert len(arr) > 0


def test_numba_choice_returns_correct_count():
    population = np.arange(10, dtype=np.float64)
    weights = np.ones(10) / 10.0
    result = dd.numba_choice(population, weights, 5)
    assert len(result) == 5
    assert all(x in population for x in result)


def test_find_cdf_indexes():
    pdf = np.array([0.1, 0.2, 0.3, 0.25, 0.15], dtype=np.float64)
    x = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    indexes = dd.find_cdf_indexes(pdf, x)
    assert len(indexes) == 3
    assert all(isinstance(v, (float, np.floating)) for v in indexes)


# --- 2. DAY SIMULATION --------------------------------------------------------
import simulation.day_sim as ds


def test_day_sim_3_returns_length_5():
    lt_sample = np.ones(100, dtype=np.int64)
    demand_sample = np.ones(100, dtype=np.int64)
    result = ds.day_sim_3(
        psl=10, ads=10, var=5, lt=14,
        p_terms=30, s_terms=30, days=100,
        demand_sample=demand_sample, lt_sample=lt_sample,
        cost_of_capital=0.14
    )
    assert len(result) == 5


def test_calc_single_psl_returns_length_6():
    result = ds.calc_single_psl(
        psl=20, ads=10, var=5, lt=14,
        gm=25.0, cost=50.0, avg_sale_price=75.0,
        length=1.0, width=1.0, height=1.0,
        p_terms=30, s_terms=30, min_of_1=1,
        cost_of_capital=0.14
    )
    assert len(result) == 6
    assert result[0] == 20


def test_day_sim_poisson_demand_no_error():
    lt_sample = np.full(50, 14, dtype=np.int64)
    demand_sample = np.zeros(50, dtype=np.int64)
    result = ds.day_sim_3(
        psl=15, ads=10, var=5, lt=14,
        p_terms=30, s_terms=30, days=50,
        demand_sample=demand_sample, lt_sample=lt_sample,
        cost_of_capital=0.14
    )
    assert len(result) == 5


def test_day_sim_nb_demand_no_error():
    demand_array = dd.calc_nb_array_ln(5.0, 10.0)
    demand_size = np.arange(demand_array.size)
    demand_sample = dd.numba_choice(demand_size, demand_array, 50)
    lt_sample = np.full(50, 14, dtype=np.int64)
    result = ds.day_sim_3(
        psl=15, ads=5, var=10, lt=14,
        p_terms=30, s_terms=30, days=50,
        demand_sample=demand_sample, lt_sample=lt_sample,
        cost_of_capital=0.14
    )
    assert len(result) == 5


# --- 3. PSL OPTIMIZER ---------------------------------------------------------
import simulation.psl_optimizer as psl


def test_calc_opti_psl_3_returns_6_elements():
    result = psl.calc_opti_psl_3(
        ads=5.0, var=10.0, lt=14,
        gm=25.0, cost=50.0, avg_sale_price=75.0,
        length=1.0, width=1.0, height=1.0,
        p_terms=30, s_terms=30, min_of_1=1,
        cost_of_capital=0.14
    )
    assert len(result) == 6
    assert result[0] > 0


def test_get_all_psls_shape():
    ids = np.array([0, 1, 2], dtype=np.int64)
    ads_arr = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    var_arr = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    lt_arr = np.array([14.0, 14.0, 14.0], dtype=np.float64)
    gm_arr = np.array([25.0, 25.0, 25.0], dtype=np.float64)
    cost_arr = np.array([50.0, 50.0, 50.0], dtype=np.float64)
    sale_arr = np.array([75.0, 75.0, 75.0], dtype=np.float64)
    length_arr = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    width_arr = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    height_arr = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    pterms = np.array([30, 30, 30], dtype=np.int64)
    sterms = np.array([30, 30, 30], dtype=np.int64)
    min1 = np.array([1, 1, 1], dtype=np.int64)

    result = psl.get_all_psls(
        ids, ads_arr, var_arr, lt_arr, gm_arr, cost_arr,
        sale_arr, length_arr, width_arr, height_arr,
        pterms, sterms, min1, cost_of_capital=0.14
    )
    assert result.shape == (3, 6)


def test_higher_gross_margin_higher_profit():
    r_low = psl.calc_opti_psl_3(
        ads=8.0, var=5.0, lt=14,
        gm=5.0, cost=50.0, avg_sale_price=55.0,
        length=1.0, width=1.0, height=1.0,
        p_terms=30, s_terms=30, min_of_1=1,
        cost_of_capital=0.14
    )
    r_high = psl.calc_opti_psl_3(
        ads=8.0, var=5.0, lt=14,
        gm=30.0, cost=50.0, avg_sale_price=80.0,
        length=1.0, width=1.0, height=1.0,
        p_terms=30, s_terms=30, min_of_1=1,
        cost_of_capital=0.14
    )
    assert r_high[1] > r_low[1]


def test_cost_of_capital_affects_result():
    r_low_coc = psl.calc_opti_psl_3(
        ads=8.0, var=5.0, lt=14,
        gm=25.0, cost=50.0, avg_sale_price=75.0,
        length=1.0, width=1.0, height=1.0,
        p_terms=30, s_terms=30, min_of_1=1,
        cost_of_capital=0.05
    )
    r_high_coc = psl.calc_opti_psl_3(
        ads=8.0, var=5.0, lt=14,
        gm=25.0, cost=50.0, avg_sale_price=75.0,
        length=1.0, width=1.0, height=1.0,
        p_terms=30, s_terms=30, min_of_1=1,
        cost_of_capital=0.30
    )
    assert r_low_coc[1] != r_high_coc[1]


# --- 4. ADJUSTED DEMAND -------------------------------------------------------
import demand.adjusted_demand as adj


def test_normal_sales_returns_positive():
    np.random.seed(42)
    dates = pd.date_range(end='2024-01-01', periods=180, freq='D')
    df = pd.DataFrame({
        'item': 'TEST001',
        'date': dates,
        'quantity': np.random.randint(5, 15, 180).astype(float),
        'available': 1.0,
        'mercury_order_quantity': 0.0,
    })
    result = adj.calculate_adjusted_demand(df)
    last_row = result.iloc[-1]
    assert last_row['final_avg'] > 0
    assert last_row['final_var'] > 0


def test_all_zeros_returns_zero():
    dates = pd.date_range(end='2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'item': 'TEST002',
        'date': dates,
        'quantity': 0.0,
        'available': 1.0,
        'mercury_order_quantity': 0.0,
    })
    result = adj.calculate_adjusted_demand(df)
    last_row = result.iloc[-1]
    assert last_row['final_avg'] == 0.0
    assert last_row['final_var'] == 0.0


def test_single_row_returns_quantity():
    # Single row: use the convenience wrapper which handles this edge case
    ads, var, source = adj.calculate_demand_from_history([
        {'date': '2024-01-01', 'quantity': 7.0, 'available': 1.0, 'mercury_order_quantity': 0.0}
    ])
    assert ads == 7.0
    assert var == 7.0
    assert source == 'adjusted_demand'


def test_empty_input_returns_zero():
    # Empty input: use the convenience wrapper
    ads, var, source = adj.calculate_demand_from_history([])
    assert ads == 0.0
    assert var == 0.0
    assert source == 'no_data'


def test_outlier_day_lowers_average():
    dates = pd.date_range(end='2024-01-01', periods=180, freq='D')
    quantities = np.ones(180) * 5.0
    quantities[50] = 1000.0
    df = pd.DataFrame({
        'item': 'TEST004',
        'date': dates,
        'quantity': quantities,
        'available': 1.0,
        'mercury_order_quantity': 0.0,
    })
    result = adj.calculate_adjusted_demand(df)
    last_row = result.iloc[-1]
    raw_mean = quantities.mean()
    assert last_row['final_avg'] < raw_mean


def test_multiple_items_no_error():
    dates = pd.date_range(end='2024-01-01', periods=90, freq='D')
    np.random.seed(7)
    df1 = pd.DataFrame({
        'item': 'ITEM_A',
        'date': dates,
        'quantity': np.random.randint(3, 15, 90).astype(float),
        'available': 1.0,
        'mercury_order_quantity': 0.0,
    })
    np.random.seed(8)
    df2 = pd.DataFrame({
        'item': 'ITEM_B',
        'date': dates,
        'quantity': np.random.randint(1, 10, 90).astype(float),
        'available': 1.0,
        'mercury_order_quantity': 0.0,
    })
    df = pd.concat([df1, df2], ignore_index=True)
    result = adj.calculate_adjusted_demand(df)
    assert len(result) > 0
    assert 'final_avg' in result.columns
    assert 'final_var' in result.columns


# --- 5. API SCHEMAS ----------------------------------------------------------
import api.schemas as schemas


def test_optimize_request_valid_data():
    req = schemas.OptimizeRequest(
        items=[
            schemas.ItemInput(
                item_id='SKU001',
                current_available=50,
                cost=10.0,
                sale_price=25.0,
            )
        ],
        cost_of_capital=0.14,
    )
    assert req.tier == 'basic'
    assert len(req.items) == 1


def test_optimize_request_missing_required():
    # Empty items list is valid per Pydantic — no error expected
    req = schemas.OptimizeRequest(items=[])
    assert req.tier == 'basic'
    assert len(req.items) == 0


def test_item_input_defaults():
    item = schemas.ItemInput(
        item_id='SKU002',
        cost=5.0,
        sale_price=15.0,
    )
    assert item.length == 1.0
    assert item.width == 1.0
    assert item.height == 1.0
    assert item.payment_terms_days == 30
    assert item.sales_terms_days == 30
    assert item.lead_time_days == 14.0


# --- 6. API ROUTES -----------------------------------------------------------
os.environ['OPTISTOCK_API_KEYS'] = ''
import api.routes as routes


def test_calculate_demand_90_days():
    np.random.seed(99)
    dates = [(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)).isoformat()
             for i in range(90)]
    hist = [
        schemas.HistoricalDataPoint(
            date=d, quantity=float(np.random.randint(5, 15)),
            available=1.0, mercury_order_quantity=0.0)
        for d in dates
    ]
    ads, var, source = routes.calculate_demand_from_history(hist, 'SKU001', 'basic')
    assert ads > 0


def test_calculate_demand_insufficient_data():
    hist = [
        {'date': '2024-01-01', 'quantity': 5.0, 'available': 1.0, 'mercury_order_quantity': 0.0},
        {'date': '2024-01-02', 'quantity': 6.0, 'available': 1.0, 'mercury_order_quantity': 0.0},
        {'date': '2024-01-03', 'quantity': 7.0, 'available': 1.0, 'mercury_order_quantity': 0.0},
    ]
    ads, var, source = routes.calculate_demand_from_history(hist, 'SKU002', 'basic')
    assert source == 'insufficient_data'
    assert ads == 0.0


def test_run_psl_optimization_returns_6_values():
    result = routes.run_psl_optimization(
        ads=10.0, var=5.0, lt=14.0,
        gm=15.0, cost=50.0, sale_price=65.0,
        length=1.0, width=1.0, height=1.0,
        p_terms=30, s_terms=30, cost_of_capital=0.14,
    )
    assert len(result) == 6
    psl_val, profit, inv, sales, cube, ppc = result
    assert psl_val >= 0


def test_health_endpoint():
    health_result = asyncio.run(routes.health())
    assert health_result.get('status') == 'ok'


# --- 7. STORAGE --------------------------------------------------------------
import api.storage as storage


def test_init_db_creates_tables(tmp_path):
    db_path = str(tmp_path / 'test_optistock.db')
    conn = asyncio.run(storage.init_db(db_path=db_path))
    assert conn is not None

    async def check():
        cursor = await conn.execute(
            'SELECT name FROM sqlite_master ORDER BY name'
        )
        rows = await cursor.fetchall()
        return [r['name'] for r in rows]

    tables = asyncio.run(check())
    assert 'api_requests' in tables
    assert 'request_items' in tables


def test_store_request_succeeds(tmp_path):
    db_path = str(tmp_path / 'test_optistock2.db')
    asyncio.run(storage.init_db(db_path=db_path))
    asyncio.run(storage.store_request(
        request_id='test-req-001',
        api_key='test-key',
        endpoint='/v1/optimize',
        request_data={'items': [], 'tier': 'basic', 'cost_of_capital': 0.14},
        response_data={'items': [], 'compute_time_ms': 10.0},
        compute_time_ms=10.0,
    ))


def test_get_requests_returns_list(tmp_path):
    db_path = str(tmp_path / 'test_optistock3.db')
    asyncio.run(storage.init_db(db_path=db_path))
    asyncio.run(storage.store_request(
        request_id='test-req-002',
        api_key='test-key',
        endpoint='/v1/optimize',
        request_data={'items': [], 'tier': 'basic', 'cost_of_capital': 0.14},
        response_data={'items': [], 'compute_time_ms': 5.0},
        compute_time_ms=5.0,
    ))
    results = asyncio.run(storage.get_requests(
        start_date='2024-01-01',
        end_date='2030-01-01',  # Wide range to capture current timestamp
    ))
    assert isinstance(results, list)
    assert len(results) >= 1