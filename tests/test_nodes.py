import io
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from app.agent.state import VFXJobState
from app.agent.nodes.input_guard import input_guard_node
from app.agent.nodes.nlp_planner import nlp_planner_node
from app.services.anthropic_client import _strip_fences, call_planner


# ===========================================================================
# Helpers
# ===========================================================================

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


# ===========================================================================
# Milestone 1 — input_guard_node tests
# ===========================================================================

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


# ===========================================================================
# Milestone 2 — anthropic_client tests (all mocked, no live API calls)
# ===========================================================================

VALID_PLAN = {
    "target_objects":         ["person", "sky"],
    "actions":                ["remove_object", "replace_background"],
    "required_nodes":         ["segmentation", "depth_estimation", "compositing"],
    "plan_summary":           "Step 1: SAM masks. Step 2: MiDaS depth. Step 3: SDXL composite.",
    "confidence":             0.92,
    "clarification_needed":   False,
    "clarification_question": None,
}


# --- _strip_fences unit tests -----------------------------------------------

def test_strip_fences_removes_json_fence():
    raw = '```json\n{"key": "value"}\n```'
    assert _strip_fences(raw) == '{"key": "value"}'


def test_strip_fences_removes_plain_fence():
    raw = '```\n{"key": "value"}\n```'
    assert _strip_fences(raw) == '{"key": "value"}'


def test_strip_fences_passthrough_when_no_fence():
    raw = '{"key": "value"}'
    assert _strip_fences(raw) == '{"key": "value"}'


def test_strip_fences_handles_extra_whitespace():
    raw = '  ```json\n{ "a": 1 }\n```  '
    assert _strip_fences(raw) == '{ "a": 1 }'


# --- call_planner mock tests -------------------------------------------------

def _mock_client(response_text: str):
    """Return a patched AsyncAnthropic client that returns response_text."""
    mock_content         = MagicMock()
    mock_content.text    = response_text
    mock_message         = MagicMock()
    mock_message.content = [mock_content]
    mock_client          = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)
    return mock_client


@pytest.mark.asyncio
async def test_call_planner_returns_parsed_dict(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    with patch("app.services.anthropic_client.anthropic.AsyncAnthropic",
               return_value=_mock_client(json.dumps(VALID_PLAN))):
        result = await call_planner("base64img==", "replace the sky")
    assert result["confidence"] == 0.92
    assert "segmentation" in result["required_nodes"]
    assert result["clarification_needed"] is False


@pytest.mark.asyncio
async def test_call_planner_strips_fences_before_parsing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    fenced = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    with patch("app.services.anthropic_client.anthropic.AsyncAnthropic",
               return_value=_mock_client(fenced)):
        result = await call_planner("base64img==", "dramatic sky replacement")
    assert result["target_objects"] == ["person", "sky"]


@pytest.mark.asyncio
async def test_call_planner_raises_value_error_on_bad_json(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    with patch("app.services.anthropic_client.anthropic.AsyncAnthropic",
               return_value=_mock_client("Sure! I can help with that VFX task.")):
        with pytest.raises(ValueError, match="non-JSON"):
            await call_planner("base64img==", "do something")


@pytest.mark.asyncio
async def test_call_planner_passes_correct_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    mock_client = _mock_client(json.dumps(VALID_PLAN))
    with patch("app.services.anthropic_client.anthropic.AsyncAnthropic",
               return_value=mock_client):
        await call_planner("base64img==", "remove background")
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"
    assert call_kwargs["temperature"] == 0.1


# ===========================================================================
# Milestone 3 — nlp_planner_node tests (all mocked, no live API calls)
# ===========================================================================

def _patch_call_planner(plan: dict):
    """Patch call_planner in the nlp_planner module to return plan synchronously."""
    async def _fake(*args, **kwargs):
        return plan
    return patch("app.agent.nodes.nlp_planner.call_planner", side_effect=_fake)


def _planner_state(prompt: str = "remove the background and replace sky") -> VFXJobState:
    img = _make_jpeg()
    state = _make_state(img, prompt)
    state["image_format"] = "jpeg"
    state["status"]       = "planning"
    return state


def test_nlp_planner_happy_path():
    state = _planner_state()
    with _patch_call_planner(VALID_PLAN):
        result = nlp_planner_node(state)
    assert result["status"] == "running"
    assert result["plan_confidence"] == 0.92
    assert result["execution_plan"] == VALID_PLAN["plan_summary"]
    assert result["extracted_intent"]["required_nodes"] == ["segmentation", "depth_estimation", "compositing"]
    assert "nlp_planner" in result["nodes_executed"]


def test_nlp_planner_low_confidence_triggers_clarification():
    low_conf_plan = {**VALID_PLAN, "confidence": 0.55, "clarification_needed": False}
    state = _planner_state("do something cool")
    with _patch_call_planner(low_conf_plan):
        result = nlp_planner_node(state)
    assert result["status"] == "clarification_required"
    assert "nlp_planner" not in result["nodes_executed"]


def test_nlp_planner_clarification_flag_triggers_clarification():
    clarification_plan = {
        **VALID_PLAN,
        "confidence":             0.85,
        "clarification_needed":   True,
        "clarification_question": "Do you want to replace just the sky or the whole background?",
    }
    state = _planner_state("change the background maybe")
    with _patch_call_planner(clarification_plan):
        result = nlp_planner_node(state)
    assert result["status"] == "clarification_required"
    assert "nlp_planner" not in result["nodes_executed"]


def test_nlp_planner_stores_full_extracted_intent():
    state = _planner_state()
    with _patch_call_planner(VALID_PLAN):
        result = nlp_planner_node(state)
    intent = result["extracted_intent"]
    assert intent["target_objects"] == ["person", "sky"]
    assert intent["actions"]        == ["remove_object", "replace_background"]
    assert intent["clarification_needed"] is False
