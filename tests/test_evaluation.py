"""Model-quality gate on the labeled benchmark.

The benchmark is deterministic (seeded RNG + fixed model random_state), so these
thresholds are stable. They guard against silent quality regressions in the
detection logic.

The benchmark contains two anomaly kinds: point price-shocks (which the price
z-score catches on its own) and contextual volume/liquidity surges that leave the
price level untouched (which only the Isolation Forest can see). The contract
below is the project's central claim: the hybrid beats the z-score baseline
because it stays precise on price anomalies *and* adds the contextual ones.
"""
import pytest

from evaluate import run_evaluation


@pytest.fixture(scope="module")
def metrics():
    return run_evaluation()


def test_benchmark_is_well_formed(metrics):
    assert metrics["n_samples"] > 1000
    assert metrics["n_point"] > 0
    assert metrics["n_contextual"] > 0


def test_hybrid_catches_point_shocks(metrics):
    # Large injected price shocks must be caught — high recall is the contract.
    assert metrics["hybrid_recall_point"] >= 0.95


def test_hybrid_beats_zscore_on_f1(metrics):
    # The central regression guard: the hybrid must outperform the z-score-only
    # baseline overall. If a change makes the hybrid worse than its own simpler
    # component, this fails — which is exactly the contradiction we never want back.
    assert metrics["hybrid"]["f1"] > metrics["zscore_only"]["f1"]


def test_hybrid_is_more_precise_than_zscore(metrics):
    # Requiring the Isolation Forest to confirm a z-score trip discards the
    # z-score's standalone false positives, so the hybrid is the more precise one.
    assert metrics["hybrid"]["precision"] > metrics["zscore_only"]["precision"]


def test_hybrid_catches_contextual_anomalies_the_zscore_misses(metrics):
    # The Isolation Forest earns its place: it catches volume/liquidity anomalies
    # that the price z-score is structurally blind to.
    assert metrics["zscore_recall_contextual"] <= 0.15
    assert metrics["hybrid_recall_contextual"] >= 0.50
