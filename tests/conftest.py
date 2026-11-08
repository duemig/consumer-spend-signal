import pytest

from spendsignal.simulate import SimulationConfig, simulate

SMALL_SECTORS = {
    25: ("Consumer Discretionary", 12, 1.0),
    30: ("Consumer Staples", 8, 0.8),
    40: ("Financials", 6, 0.0),
}


@pytest.fixture(scope="session")
def small_config() -> SimulationConfig:
    return SimulationConfig(sectors=SMALL_SECTORS, n_cohorts=6, start="2016-01-01", end="2019-12-31", seed=3)


@pytest.fixture(scope="session")
def small_data(small_config):
    return simulate(small_config)
