import asyncio
import os

import replicate
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model identifiers — using owner/name without a pinned version hash so
# Replicate always resolves to the latest deployed version.
# Verify / pin hashes at replicate.com before moving to production.
# ---------------------------------------------------------------------------
GROUNDED_SAM_MODEL    = "adirik/grounded-sam"
MIDAS_MODEL           = "cjwbw/midas"
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
