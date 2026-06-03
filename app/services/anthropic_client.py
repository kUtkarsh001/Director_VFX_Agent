import json
import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

PLANNER_MODEL = "claude-sonnet-4-20250514"
VALID_NODES   = {"segmentation", "depth_estimation", "compositing", "filter"}

SYSTEM_PROMPT = """You are a VFX Director AI. Analyse the image and the VFX instruction.
Return ONLY a raw JSON object — no markdown fences, no explanation — with exactly these fields:

{
  "target_objects":         ["list of objects to act on"],
  "actions":                ["list of vfx actions"],
  "required_nodes":         ["subset of: segmentation, depth_estimation, compositing, filter"],
  "plan_summary":           "human-readable step-by-step plan",
  "confidence":             0.0,
  "clarification_needed":   false,
  "clarification_question": null
}

Rules:
- required_nodes must only contain values from: segmentation, depth_estimation, compositing, filter
- confidence is 0.0–1.0; use < 0.7 when the instruction is genuinely ambiguous
- If ambiguous, set clarification_needed=true and populate clarification_question
- Output raw JSON only — absolutely no ```json fences or surrounding text"""


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` code fences if the model adds them."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",          "", text)
    return text.strip()


async def call_planner(
    image_base64: str,
    prompt: str,
    media_type: str = "image/jpeg",
) -> dict:
    """
    Call Claude with the image + prompt and return the parsed planner JSON dict.
    Raises ValueError if the response cannot be parsed as JSON.
    """
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = await client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=1024,
        temperature=0.1,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       image_base64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw     = message.content[0].text
    cleaned = _strip_fences(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Planner returned non-JSON response: {exc}\nRaw output: {raw}"
        ) from exc
