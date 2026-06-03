from langgraph.graph import StateGraph, END

from app.agent.state import VFXJobState
from app.agent.nodes.input_guard import input_guard_node
from app.agent.nodes.nlp_planner import nlp_planner_node
from app.agent.nodes.segmentation import segmentation_node
from app.agent.nodes.depth_estimation import depth_estimation_node
from app.agent.nodes.compositing import compositing_node
from app.agent.nodes.filter import filter_node
from app.agent.nodes.error_recovery import error_recovery_node


# ---------------------------------------------------------------------------
# Routing functions (architecture.md Section 5)
# ---------------------------------------------------------------------------

def route_after_planner(state: VFXJobState) -> str:
    if state.get("status") == "clarification_required":
        return END
    nodes = state.get("extracted_intent", {}).get("required_nodes", [])
    if state.get("plan_confidence", 1.0) < 0.7:
        return END                          # clarification_required terminal
    if "segmentation" in nodes:
        return "segmentation"
    if "depth_estimation" in nodes:
        return "depth_estimation"
    if "compositing" in nodes:
        return "compositing"
    if "filter" in nodes:
        return "filter"
    return "error_recovery"                 # no valid route found


def route_after_segmentation(state: VFXJobState) -> str:
    if state.get("errors"):
        return "error_recovery"
    nodes = state.get("extracted_intent", {}).get("required_nodes", [])
    if "depth_estimation" in nodes:
        return "depth_estimation"
    if "compositing" in nodes:
        return "compositing"
    return END


def route_after_depth(state: VFXJobState) -> str:
    nodes = state.get("extracted_intent", {}).get("required_nodes", [])
    # Depth failure is non-fatal if compositing is still required
    if "compositing" in nodes:
        return "compositing"
    if state.get("errors"):
        return "error_recovery"
    return END


def route_after_compositing(state: VFXJobState) -> str:
    if state.get("status") == "done":
        return END
    if state.get("errors"):
        return "error_recovery"
    return END


def route_after_error_recovery(state: VFXJobState) -> str:
    """Retry the failing node once, then give up."""
    target = state.get("retry_target")
    count  = state.get("retry_count", 0)
    if target and count <= 1 and target in (
        "segmentation", "depth_estimation", "compositing"
    ):
        return target
    return END


# ---------------------------------------------------------------------------
# Graph construction (architecture.md Section 6)
# ---------------------------------------------------------------------------

def build_vfx_graph():
    graph = StateGraph(VFXJobState)

    # Register nodes
    graph.add_node("input_guard",      input_guard_node)
    graph.add_node("nlp_planner",      nlp_planner_node)
    graph.add_node("segmentation",     segmentation_node)
    graph.add_node("depth_estimation", depth_estimation_node)
    graph.add_node("compositing",      compositing_node)
    graph.add_node("filter",           filter_node)
    graph.add_node("error_recovery",   error_recovery_node)

    # Entry point
    graph.set_entry_point("input_guard")
    graph.add_edge("input_guard", "nlp_planner")

    # Conditional routing from planner
    graph.add_conditional_edges("nlp_planner", route_after_planner)

    # Chained edges through tool nodes
    graph.add_conditional_edges("segmentation",     route_after_segmentation)
    graph.add_conditional_edges("depth_estimation", route_after_depth)
    graph.add_conditional_edges("compositing",      route_after_compositing)
    graph.add_edge("filter",         END)
    graph.add_conditional_edges("error_recovery", route_after_error_recovery)

    return graph.compile()
