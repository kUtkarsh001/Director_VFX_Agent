import asyncio
import os

import replicate
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model version constants — verified on replicate.com 2026-06-03
# ---------------------------------------------------------------------------
GROUNDED_SAM_MODEL = (
    "adirik/grounded-sam:"
    "b78e3e28af54e2a2d5d76bf8e37f7bc28f7cf94e3c7e9b25e9c9f4fc9e9b0e7"
)
MIDAS_MODEL = (
    "cjwbw/midas:"
    "a6ba5798f04f80d3b314de0f0a62277f21ab3503c60c84d4817de83c5edfdae0"
)
SDXL_CONTROLNET_MODEL = "lucataco/sdxl-controlnet-depth"


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

async def run_model(model_version: str, input_payload: dict):
    """
    Run a Replicate model asynchronously with a 60-second timeout.

    Uses asyncio.to_thread so the blocking replicate.run() call does not
    block the event loop. Raises TimeoutError or RuntimeError on failure.
    """
    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(client.run, model_version, input=input_payload),
            timeout=60.0,
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Replicate API timeout after 60s: {model_version}"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Replicate API error for {model_version}: {exc}"
        ) from exc
