import io
import pytest
from PIL import Image

from app.agent.state import VFXJobState
from app.agent.nodes.input_guard import input_guard_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(image_bytes: bytes, prompt: str) -> VFXJobState:
    return VFXJobState(
        job_id="test-001",
        original_image=image_bytes,
        image_format="",
        user_prompt=prompt,
        extracted_intent={},
        execution_plan="",
        plan_confidence=0.0,
        masks=None,
        mask_confidence=None,
        depth_map=None,
        final_image=None,
        nodes_executed=[],
        errors=[],
        quality_flags=[],
        status="queued",
    )


def _make_jpeg(width: int = 100, height: int = 100) -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png(width: int = 100, height: int = 100) -> bytes:
    img = Image.new("RGB", (width, height), color=(0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests — success paths
# ---------------------------------------------------------------------------

def test_valid_jpeg_passes():
    result = input_guard_node(_make_state(_make_jpeg(), "make it black and white"))
    assert result["status"] == "planning"
    assert result["image_format"] == "jpeg"
    assert "input_guard" in result["nodes_executed"]
    assert result["errors"] == []


def test_valid_png_passes():
    result = input_guard_node(_make_state(_make_png(), "replace the sky"))
    assert result["status"] == "planning"
    assert result["image_format"] == "png"


def test_image_resized_when_over_2048px():
    big = _make_jpeg(width=4000, height=3000)
    result = input_guard_node(_make_state(big, "remove the background"))
    assert result["status"] == "planning"
    reopened = Image.open(io.BytesIO(result["original_image"]))
    assert max(reopened.width, reopened.height) <= 2048


# ---------------------------------------------------------------------------
# Tests — failure paths
# ---------------------------------------------------------------------------

def test_prompt_too_long_fails():
    result = input_guard_node(_make_state(_make_jpeg(), "x" * 600))
    assert result["status"] == "failed"
    assert result["errors"][0]["error_code"] == "PROMPT_INVALID"


def test_empty_prompt_fails():
    result = input_guard_node(_make_state(_make_jpeg(), ""))
    assert result["status"] == "failed"
    assert result["errors"][0]["error_code"] == "PROMPT_INVALID"


def test_whitespace_only_prompt_fails():
    result = input_guard_node(_make_state(_make_jpeg(), "   "))
    assert result["status"] == "failed"
    assert result["errors"][0]["error_code"] == "PROMPT_INVALID"


def test_oversized_image_fails():
    large = b"0" * (10 * 1024 * 1024 + 1)
    result = input_guard_node(_make_state(large, "valid prompt"))
    assert result["status"] == "failed"
    assert result["errors"][0]["error_code"] == "INPUT_TOO_LARGE"


def test_invalid_format_fails():
    result = input_guard_node(_make_state(b"not-an-image", "valid prompt"))
    assert result["status"] == "failed"
    assert result["errors"][0]["error_code"] == "INPUT_INVALID_FORMAT"
