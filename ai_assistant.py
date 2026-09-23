from __future__ import annotations

import json
import os
from typing import Any

from urllib import request
from urllib.error import URLError, HTTPError


# ============================================================
# CONFIGURATION
# ============================================================

LM_STUDIO_URL = os.getenv(
    "LM_STUDIO_URL",
    "http://localhost:1234/v1/chat/completions",
)

LM_STUDIO_MODEL = os.getenv(
    "LM_STUDIO_MODEL",
    "",
)

LM_STUDIO_TIMEOUT = int(
    os.getenv(
        "LM_STUDIO_TIMEOUT",
        "120",
    )
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are RadioPathwayTool, a local amateur-radio propagation
assistant.

Your job is to answer questions about HF radio propagation
using the tools provided to you.

IMPORTANT RULES:

1. Use the available tools when the question depends on
   current propagation conditions.

2. Do not invent current propagation data.

3. HAP is the primary propagation prediction source.

4. Ionosphere observations and space weather are supporting
   information.

5. Clearly distinguish between:
   - measured/observed data
   - HAP predictions
   - your interpretation

6. If the available data does not answer the question,
   say so.

7. Do not pretend that HAP guarantees a contact.

8. Keep answers reasonably concise unless the user asks
   for detail.

9. The station is in Nelson, New Zealand.

10. The operator uses an Xiegu G90 at approximately 20 W.

11. The operator is primarily interested in HF propagation,
    especially 80 m currently and 20 m in the future.

12. You are an assistant for radio propagation analysis.
    Do not claim to have personally heard or worked stations.

When discussing a band, explain what the supplied propagation
data actually indicates rather than inventing an opening.
"""


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_propagation_report",
            "description": (
                "Get the complete current RadioPathwayTool "
                "propagation report. This includes HAP propagation "
                "predictions, regional HAP distribution, ionosphere "
                "observations, and space weather."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ============================================================
# HTTP HELPER
# ============================================================

def lmstudio_request(
    payload: dict[str, Any],
) -> dict[str, Any]:

    body = json.dumps(
        payload
    ).encode("utf-8")

    req = request.Request(
        LM_STUDIO_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:

        with request.urlopen(
            req,
            timeout=LM_STUDIO_TIMEOUT,
        ) as response:

            raw = response.read()

    except HTTPError as exc:

        error_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"LM Studio HTTP {exc.code}: "
            f"{error_body}"
        ) from exc

    except URLError as exc:

        raise RuntimeError(
            "Could not connect to LM Studio. "
            "Make sure the LM Studio server is running."
        ) from exc

    return json.loads(
        raw.decode("utf-8")
    )


# ============================================================
# REPORT SERIALISATION
# ============================================================

def _make_json_safe(value):
    """
    Convert RadioPathwayTool objects into JSON-safe data.

    This intentionally handles dataclasses, dictionaries,
    lists, tuples and ordinary objects.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, dict):

        return {
            str(key): _make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return [
            _make_json_safe(item)
            for item in value
        ]

    if hasattr(
        value,
        "__dataclass_fields__",
    ):

        return {
            key: _make_json_safe(
                getattr(value, key)
            )
            for key in value.__dataclass_fields__
        }

    if hasattr(
        value,
        "__dict__",
    ):

        return {
            key: _make_json_safe(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)


def serialise_report(report) -> str:
    """
    Convert a PropagationReport into JSON for the AI.
    """

    data = _make_json_safe(
        report
    )

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    name: str,
    arguments: dict[str, Any],
    report,
):
    """
    Execute an AI-requested tool.

    The AI never gets direct access to Python.
    Only explicitly registered tools can be executed.
    """

    if name == "get_propagation_report":

        return serialise_report(
            report
        )

    raise ValueError(
        f"Unknown AI tool: {name}"
    )


# ============================================================
# AI ASSISTANT
# ============================================================

def ask_radio_assistant(
    question: str,
    report,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Ask the local LM Studio model a radio question.

    The model can request one of the explicitly registered
    RadioPathwayTool functions.
    """

    if not LM_STUDIO_MODEL:

        raise RuntimeError(
            "LM_STUDIO_MODEL is not configured. "
            "Set it in your .env file."
        )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # --------------------------------------------------------
    # Optional short conversation history
    # --------------------------------------------------------

    if conversation_history:

        messages.extend(
            conversation_history[-10:]
        )

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # --------------------------------------------------------
    # First AI request
    # --------------------------------------------------------

    response = lmstudio_request(
        {
            "model": LM_STUDIO_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
    )

    choices = response.get(
        "choices",
        [],
    )

    if not choices:

        raise RuntimeError(
            "LM Studio returned no choices."
        )

    message = choices[0].get(
        "message",
        {},
    )

    tool_calls = message.get(
        "tool_calls",
        [],
    )

    # --------------------------------------------------------
    # No tool required
    # --------------------------------------------------------

    if not tool_calls:

        content = message.get(
            "content",
            "",
        )

        if content:

            return content.strip()

        return (
            "I couldn't generate a response."
        )

    # --------------------------------------------------------
    # Execute requested tools
    # --------------------------------------------------------

    messages.append(
        message
    )

    for tool_call in tool_calls:

        function = tool_call.get(
            "function",
            {},
        )

        name = function.get(
            "name"
        )

        raw_arguments = function.get(
            "arguments",
            "{}",
        )

        try:

            arguments = json.loads(
                raw_arguments
            )

        except json.JSONDecodeError:

            arguments = {}

        result = execute_tool(
            name,
            arguments,
            report,
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get(
                    "id",
                    "",
                ),
                "name": name,
                "content": result,
            }
        )

    # --------------------------------------------------------
    # Second AI request
    # --------------------------------------------------------

    final_response = lmstudio_request(
        {
            "model": LM_STUDIO_MODEL,
            "messages": messages,
            "temperature": 0.2,
        }
    )

    final_choices = final_response.get(
        "choices",
        [],
    )

    if not final_choices:

        raise RuntimeError(
            "LM Studio returned no final answer."
        )

    final_message = final_choices[0].get(
        "message",
        {},
    )

    content = final_message.get(
        "content",
        "",
    )

    if not content:

        return (
            "I collected the propagation data, "
            "but couldn't generate an answer."
        )

    return content.strip()