from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from main import get_propagation_report


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Small model is sufficient for routing/tool use.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

# Prevent the AI from getting stuck in a tool-call loop.
MAX_TOOL_ROUNDS = 5

# Internal propagation report cache.
#
# Discord already has a cache, but this second layer means
# multiple AI tool calls in the same request do not repeatedly
# collect HAP/ionosphere/space-weather data.
REPORT_CACHE_SECONDS = 300


# ============================================================
# GROQ CLIENT
# ============================================================

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file."
    )

client = Groq(
    api_key=GROQ_API_KEY,
)


# ============================================================
# REPORT CACHE
# ============================================================

_cached_report: Any | None = None
_cached_report_time: float = 0.0


def get_cached_propagation_report() -> Any:
    """
    Get the latest propagation report.

    A short cache prevents multiple AI tools from triggering
    multiple expensive HAP/ionosphere/space-weather collections.
    """

    global _cached_report
    global _cached_report_time

    now = time.monotonic()

    if (
        _cached_report is not None
        and now - _cached_report_time < REPORT_CACHE_SECONDS
    ):
        return _cached_report

    report = get_propagation_report()

    _cached_report = report
    _cached_report_time = now

    return report


# ============================================================
# SERIALISATION
# ============================================================

def serialize_value(value: Any) -> Any:
    """
    Convert dataclasses, dictionaries, lists and datetime
    objects into JSON-safe values.

    This keeps ai_assistant.py independent of the exact internal
    dataclass/dictionary structure used by main.py.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if is_dataclass(value):
        return serialize_value(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            serialize_value(item)
            for item in value
        ]

    if isinstance(value, (str, int, float, bool)):
        return value

    # Fallback for unexpected objects.
    return str(value)


def json_dumps(value: Any) -> str:
    """
    Convert a Python value to compact JSON for the model.
    """

    return json.dumps(
        serialize_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


# ============================================================
# REPORT HELPERS
# ============================================================

def get_report_value(
    report: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a field from either a dictionary or an object.

    This makes the AI layer tolerant of the exact PropagationReport
    implementation.
    """

    if isinstance(report, dict):
        return report.get(key, default)

    return getattr(
        report,
        key,
        default,
    )


# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def tool_get_propagation_report() -> str:
    """
    Return the complete current propagation report.
    """

    report = get_cached_propagation_report()

    return json_dumps(report)


def tool_get_hap_forecast() -> str:
    """
    Return HAP-specific propagation information.
    """

    report = get_cached_propagation_report()

    hap = get_report_value(
        report,
        "hap",
        None,
    )

    if hap is None:
        # Some implementations may store HAP information under
        # another structure. Returning the full report is safer
        # than inventing a result.
        return json_dumps({
            "error": "HAP data was not found in the report.",
        })

    return json_dumps(hap)


def tool_get_ionosphere() -> str:
    """
    Return current ionospheric observations.
    """

    report = get_cached_propagation_report()

    ionosphere = get_report_value(
        report,
        "ionosphere",
        None,
    )

    if ionosphere is None:
        return json_dumps({
            "error": (
                "Ionosphere data was not found "
                "in the report."
            ),
        })

    return json_dumps(ionosphere)


def tool_get_space_weather() -> str:
    """
    Return current solar and geomagnetic conditions.
    """

    report = get_cached_propagation_report()

    space_weather = get_report_value(
        report,
        "space_weather",
        None,
    )

    if space_weather is None:
        return json_dumps({
            "error": (
                "Space-weather data was not found "
                "in the report."
            ),
        })

    return json_dumps(space_weather)


# ============================================================
# TOOL SCHEMAS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_propagation_report",
            "description": (
                "Get the complete current RadioPathwayTool "
                "propagation report for Nelson, New Zealand. "
                "Use this when the user asks a broad question "
                "about current propagation, band conditions, "
                "or wants multiple aspects considered."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hap_forecast",
            "description": (
                "Get the current HF Availability Prediction "
                "(HAP) data. Use this when the user asks about "
                "which HF bands are currently predicted to work, "
                "band transitions, or regional HAP distribution."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ionosphere",
            "description": (
                "Get current Australian Space Weather Services "
                "ionospheric station observations. Use this when "
                "the user asks about ionospheric conditions, "
                "enhancement/depression at stations, or whether "
                "the ionosphere is behaving normally."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_space_weather",
            "description": (
                "Get current solar and geomagnetic conditions "
                "including F10.7, sunspots, K indices, Dst, "
                "and active space-weather alerts or warnings."
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
# TOOL MAP
# ============================================================

AVAILABLE_TOOLS = {
    "get_propagation_report": tool_get_propagation_report,
    "get_hap_forecast": tool_get_hap_forecast,
    "get_ionosphere": tool_get_ionosphere,
    "get_space_weather": tool_get_space_weather,
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the natural-language assistant for RadioPathwayTool,
an amateur-radio HF propagation assistant.

Your job is to help the user understand HF propagation using
the data provided by RadioPathwayTool.

IMPORTANT RULES:

1. Use the available tools whenever the question depends on
   current propagation, HAP, ionosphere, solar activity,
   geomagnetic conditions, or alerts.

2. NEVER invent current propagation data.

3. Clearly distinguish between:
   - HAP predictions
   - measured ionospheric observations
   - solar/geomagnetic measurements
   - your interpretation of those measurements

4. HAP is the primary propagation prediction in this system.
   Space weather and ionosphere observations provide supporting
   context.

5. Do not treat HAP percentages as guaranteed probability of
   making a contact. They represent the decoded HAP map
   distribution used by RadioPathwayTool.

6. When discussing amateur-radio propagation, explain things
   in practical terms where useful.

7. The user's main station location is Nelson, New Zealand.

8. The user is interested in HF amateur radio, particularly
   80m, 40m, 20m and other HF bands.

9. The user uses a Xiegu G90 at approximately 20 W.

10. Do not claim that a band is guaranteed to work.
    Propagation forecasts are predictions and actual contacts
    depend on many factors.

11. If the available data does not answer the question,
    say so rather than making up information.

12. You can answer normal conversational questions without
    using a tool when current data is not required.

13. Keep answers concise for Discord. Usually use a few
    paragraphs or short bullet points rather than a huge report.

14. If a user asks "why", explain the physical reasoning using
    the available data rather than simply repeating the data.

15. If the user asks about a future time and the available
    tool data does not actually contain a forecast for that
    specific time, say that clearly.

16. Never execute arbitrary Python, shell commands, or code.
    You may only use the explicitly provided tools.

17. Do not claim to have information from the internet unless
    that information was actually supplied by a tool.

18. If a tool returns an error or missing data, report that
    limitation honestly.

You are an assistant layered on top of a real propagation
engine. The propagation engine is the authority for current
data; you are responsible for interpreting it clearly.
"""


# ============================================================
# TOOL CALL SERIALISATION
# ============================================================

def assistant_message_to_dict(message: Any) -> dict[str, Any]:
    """
    Convert the Groq SDK assistant message into the dictionary
    format required for the next Chat Completion request.
    """

    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }

    if message.tool_calls:
        result["tool_calls"] = []

        for tool_call in message.tool_calls:
            result["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )

    return result


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool_call(
    tool_call: Any,
) -> str:
    """
    Safely execute one model-requested tool.

    The model can only execute functions present in
    AVAILABLE_TOOLS.
    """

    tool_name = tool_call.function.name

    tool_function = AVAILABLE_TOOLS.get(
        tool_name
    )

    if tool_function is None:
        return json_dumps({
            "error": (
                f"Unknown tool requested: {tool_name}"
            ),
        })

    try:
        # All current tools have no arguments.
        #
        # We still parse the JSON so malformed arguments can
        # be detected rather than silently ignored.
        raw_arguments = (
            tool_call.function.arguments or "{}"
        )

        try:
            arguments = json.loads(
                raw_arguments
            )
        except json.JSONDecodeError:
            return json_dumps({
                "error": (
                    f"Invalid arguments supplied for "
                    f"{tool_name}."
                ),
            })

        if not isinstance(arguments, dict):
            return json_dumps({
                "error": (
                    f"Invalid argument structure for "
                    f"{tool_name}."
                ),
            })

        # These tools currently take no arguments.
        return tool_function()

    except Exception as exc:
        return json_dumps({
            "error": (
                f"Tool {tool_name} failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        })


# ============================================================
# MAIN AI FUNCTION
# ============================================================

def ask_radio_assistant(
    user_message: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Send a user question to Groq and allow the model to call
    RadioPathwayTool functions.

    Returns:

        (
            final_answer,
            updated_conversation_history
        )

    The returned history contains the messages necessary to
    continue the conversation.
    """

    if not user_message.strip():
        return (
            "Please ask me a question.",
            conversation_history or [],
        )

    history = list(
        conversation_history or []
    )

    # Prevent the stored history from becoming enormous.
    #
    # We keep the most recent messages. The system prompt is
    # always added separately and is therefore never lost.
    history = history[-10:]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    # ========================================================
    # TOOL-CALLING LOOP
    # ========================================================

    for _ in range(MAX_TOOL_ROUNDS):

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_completion_tokens=1200,
        )

        message = response.choices[0].message

        # ----------------------------------------------------
        # No tool call = final answer
        # ----------------------------------------------------

        if not message.tool_calls:

            final_answer = (
                message.content
                or "I couldn't generate an answer."
            )

            # Store the assistant response in history.
            updated_history = history + [
                {
                    "role": "user",
                    "content": user_message,
                },
                {
                    "role": "assistant",
                    "content": final_answer,
                },
            ]

            # Keep history bounded.
            updated_history = updated_history[-10:]

            return (
                final_answer.strip(),
                updated_history,
            )

        # ----------------------------------------------------
        # Tool calls requested
        # ----------------------------------------------------

        assistant_dict = assistant_message_to_dict(
            message
        )

        messages.append(
            assistant_dict
        )

        for tool_call in message.tool_calls:

            result = execute_tool_call(
                tool_call
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result,
                }
            )

    # ========================================================
    # SAFETY LIMIT
    # ========================================================

    fallback = (
        "I reached the tool-call limit while trying to "
        "answer that question. Please try asking it again."
    )

    updated_history = history + [
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "assistant",
            "content": fallback,
        },
    ]

    return (
        fallback,
        updated_history[-10:],
    )