# tests/conftest.py
#
# Global pytest configuration for the VFX Director Agent test suite.
#
# asyncio_mode = "auto" is required for pytest-asyncio >= 0.21 to automatically
# recognise async test functions decorated with @pytest.mark.asyncio.
# Without this, async tests are silently collected but never executed.

import pytest


# ---------------------------------------------------------------------------
# Shared state factory — used by both test_nodes.py and test_routing.py
# ---------------------------------------------------------------------------

def make_base_state(**overrides) -> dict:
    """Return a minimal VFXJobState-compatible dict for use in tests."""
    base = {
        "job_id":           "test-001",
        "original_image":   b"",
        "image_format":     "jpeg",
        "user_prompt":      "replace the sky",
        "extracted_intent": {"required_nodes": []},
        "execution_plan":   "",
        "plan_confidence":  0.9,
        "masks":            None,
        "mask_confidence":  None,
        "depth_map":        None,
        "final_image":      None,
        "nodes_executed":   [],
        "errors":           [],
        "quality_flags":    [],
        "status":           "queued",
        "retry_count":      0,
        "retry_target":     None,
        "node_timings":     {},
    }
    base.update(overrides)
    return base
