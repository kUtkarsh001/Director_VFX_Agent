# tests/test_routing.py
# Full routing test suite — verifies all five conditional routing functions
# in app/agent/graph.py using pure state dicts (no mocking required).

from langgraph.graph import END
from app.agent.graph import (
    route_after_planner,
    route_after_segmentation,
    route_after_depth,
    route_after_compositing,
    route_after_error_recovery,
)


# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------

def _state(**overrides) -> dict:
    base = {
        "status":           "running",
        "plan_confidence":  0.9,
        "extracted_intent": {"required_nodes": []},
        "errors":           [],
        "retry_count":      0,
        "retry_target":     None,
        "final_image":      None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# route_after_planner
# ---------------------------------------------------------------------------

def test_planner_routes_to_segmentation_when_present():
    s = _state(extracted_intent={"required_nodes": ["segmentation", "compositing"]})
    assert route_after_planner(s) == "segmentation"


def test_planner_routes_to_filter_when_filter_only():
    s = _state(extracted_intent={"required_nodes": ["filter"]})
    assert route_after_planner(s) == "filter"


def test_planner_routes_to_compositing_when_no_segmentation():
    s = _state(extracted_intent={"required_nodes": ["compositing"]})
    assert route_after_planner(s) == "compositing"


def test_planner_terminates_on_low_confidence():
    s = _state(
        plan_confidence=0.55,
        extracted_intent={"required_nodes": ["compositing"]},
    )
    assert route_after_planner(s) == END


def test_planner_terminates_on_clarification_required_status():
    s = _state(
        status="clarification_required",
        extracted_intent={"required_nodes": ["compositing"]},
    )
    assert route_after_planner(s) == END


def test_planner_routes_to_error_recovery_when_no_valid_nodes():
    s = _state(extracted_intent={"required_nodes": []})
    assert route_after_planner(s) == "error_recovery"


# ---------------------------------------------------------------------------
# route_after_segmentation
# ---------------------------------------------------------------------------

def test_segmentation_routes_to_depth_when_required():
    s = _state(extracted_intent={"required_nodes": ["depth_estimation", "compositing"]})
    assert route_after_segmentation(s) == "depth_estimation"


def test_segmentation_routes_to_compositing_when_no_depth():
    s = _state(extracted_intent={"required_nodes": ["compositing"]})
    assert route_after_segmentation(s) == "compositing"


def test_segmentation_routes_to_error_recovery_on_latest_node_error():
    s = _state(
        errors=[{"node": "segmentation", "error_code": "SEGMENTATION_FAILED",
                 "message": "timeout"}],
        retry_target=None,
        extracted_intent={"required_nodes": ["segmentation", "compositing"]},
    )
    assert route_after_segmentation(s) == "error_recovery"


def test_segmentation_does_not_loop_after_successful_retry():
    # Errors list has an old segmentation error but retry_target is cleared,
    # meaning the retry already ran and succeeded.
    s = _state(
        errors=[{"node": "segmentation", "error_code": "SEGMENTATION_FAILED",
                 "message": "prior transient error"}],
        retry_target="segmentation",   # still set from error_recovery dispatch
        retry_count=1,
        extracted_intent={"required_nodes": ["segmentation", "compositing"]},
    )
    # Should NOT loop back to error_recovery — should proceed to compositing
    result = route_after_segmentation(s)
    assert result != "error_recovery"


# ---------------------------------------------------------------------------
# route_after_depth
# ---------------------------------------------------------------------------

def test_depth_routes_to_compositing_even_with_error():
    # Depth failure is non-fatal by design
    s = _state(
        errors=[{"node": "depth_estimation", "error_code": "DEPTH_FAILED",
                 "message": "api timeout"}],
        extracted_intent={"required_nodes": ["depth_estimation", "compositing"]},
    )
    assert route_after_depth(s) == "compositing"


def test_depth_terminates_when_compositing_not_required():
    s = _state(extracted_intent={"required_nodes": ["depth_estimation"]})
    assert route_after_depth(s) == END


# ---------------------------------------------------------------------------
# route_after_compositing
# ---------------------------------------------------------------------------

def test_compositing_terminates_on_done_status():
    s = _state(status="done")
    assert route_after_compositing(s) == END


def test_compositing_routes_to_error_recovery_on_error():
    s = _state(
        errors=[{"node": "compositing", "error_code": "COMPOSITING_FAILED",
                 "message": "replicate timeout"}],
    )
    assert route_after_compositing(s) == "error_recovery"


# ---------------------------------------------------------------------------
# route_after_error_recovery
# ---------------------------------------------------------------------------

def test_error_recovery_retries_segmentation_on_first_attempt():
    s = _state(retry_count=1, retry_target="segmentation")
    assert route_after_error_recovery(s) == "segmentation"


def test_error_recovery_terminates_after_max_retries_exceeded():
    s = _state(retry_count=2, retry_target="segmentation")
    assert route_after_error_recovery(s) == END


def test_error_recovery_terminates_when_no_retry_target():
    s = _state(retry_count=0, retry_target=None)
    assert route_after_error_recovery(s) == END
