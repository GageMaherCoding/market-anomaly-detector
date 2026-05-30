"""Model-quality gate on the labeled benchmark.

The benchmark is deterministic (seeded RNG + fixed model random_state), so these
thresholds are stable. They guard against silent quality regressions in the
detection logic.
"""
import pytest

from evaluate import run_evaluation


@pytest.fixture(scope="module")
def metrics():
    return run_evaluation()


def test_benchmark_is_well_formed(metrics):
    assert metrics["n_anomalies"] > 0
    assert metrics["n_samples"] > 1000


def test_hybrid_catches_injected_anomalies(metrics):
    # Large injected shocks must be caught — high recall is the contract.
    assert metrics["hybrid"]["recall"] >= 0.95


def test_zscore_is_more_precise_than_hybrid(metrics):
    # The statistical signal alone produces fewer false positives than the
    # hybrid; the hybrid trades precision for recall via the Isolation Forest.
    assert metrics["zscore_only"]["precision"] > metrics["hybrid"]["precision"]
    assert metrics["zscore_only"]["recall"] >= 0.85
