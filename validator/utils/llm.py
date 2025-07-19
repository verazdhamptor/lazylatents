import json
import re
from typing import Any

from fiber import Keypair

from core.models.utility_models import Message
from validator.utils.call_endpoint import post_to_nineteen_chat
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def convert_to_nineteen_payload(
    messages: list[Message], model: str, temperature: float, max_tokens: int = 5000, stream: bool = False
) -> dict:
    return {
        "messages": [message.model_dump() for message in messages],
        "model": model,
        "temperature": temperature,
        "stream": stream,
        "max_tokens": max_tokens,
    }


def remove_reasoning_part(content: str, end_of_reasoning_tag: str) -> str:
    if end_of_reasoning_tag and end_of_reasoning_tag in content:
        content = content.split(end_of_reasoning_tag)[1].strip()
        return content
    else:
        logger.warning(f"No end of reasoning tag found in content: {content}")
        logger.warning("Returning empty string")
        return ""


def extract_json_from_response(response: str) -> dict:
    try:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError(f"No JSON found in response: {response}")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse JSON from response: {response}")
        raise e


async def post_to_nineteen_chat_with_reasoning(
    payload: dict[str, Any], keypair: Keypair, end_of_reasoning_tag: str
) -> str | None:
    response = await post_to_nineteen_chat(payload, keypair)
    return remove_reasoning_part(response, end_of_reasoning_tag)
