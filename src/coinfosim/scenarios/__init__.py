"""CoInfoSim scenario definitions and builders."""

from coinfosim.scenarios.occupancy import (
    GaussianAnchoredOccupancyScenario,
    OCCUPANCY_SCENARIO_NAME,
    OCCUPANCY_SCENARIO_QUESTION,
    build_gaussian_anchored_occupancy_model,
)
from coinfosim.scenarios.synthetic import (
    SCENARIO_1_NAME,
    SCENARIO_1_QUESTION,
    SyntheticScenario,
    make_synthetic_scenario_1,
)

__all__ = [
    "GaussianAnchoredOccupancyScenario",
    "OCCUPANCY_SCENARIO_NAME",
    "OCCUPANCY_SCENARIO_QUESTION",
    "SCENARIO_1_NAME",
    "SCENARIO_1_QUESTION",
    "SyntheticScenario",
    "build_gaussian_anchored_occupancy_model",
    "make_synthetic_scenario_1",
]
