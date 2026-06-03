"""
Cost guard — pre-flight budget check before tool node execution.

Approximate per-call costs (USD, as of 2026-06):
  Grounded-SAM segmentation: $0.0012
  MiDaS depth:               $0.0005
  SDXL ControlNet:            $0.0025
  Claude vision call:         $0.0060  (200K-token model, ~400 input tokens)

Default limit: $0.02 per job (configurable via MAX_COST_PER_JOB_USD env var).
Raises CostLimitExceeded if the estimated cost exceeds the budget.
"""

import os

_COST_MAP: dict[str, float] = {
    "segmentation":     0.0012,
    "depth_estimation": 0.0005,
    "compositing":      0.0025,
    "filter":           0.0000,   # local — no API cost
    "nlp_planner":      0.0060,   # Claude vision call
}

BUDGET_USD       = float(os.getenv("MAX_COST_PER_JOB_USD", "0.02"))
_SIZE_MULTIPLIER = 1.5   # surcharge for images > 5 MB


class CostLimitExceeded(Exception):
    """Raised when the projected job cost exceeds BUDGET_USD."""

    def __init__(self, estimated: float, budget: float):
        self.estimated = estimated
        self.budget    = budget
        super().__init__(
            f"Projected cost ${estimated:.4f} exceeds budget ${budget:.4f}. "
            "Remove nodes or use a smaller image."
        )


def check_cost(required_nodes: list[str], image_size_mb: float) -> float:
    """
    Estimate total cost for a set of nodes and image size.

    Returns the estimated cost in USD.
    Raises CostLimitExceeded if it exceeds BUDGET_USD.
    """
    base = sum(_COST_MAP.get(n, 0.0) for n in required_nodes)
    base += _COST_MAP["nlp_planner"]           # always charged
    if image_size_mb > 5.0:
        base *= _SIZE_MULTIPLIER

    if base > BUDGET_USD:
        raise CostLimitExceeded(base, BUDGET_USD)
    return base
