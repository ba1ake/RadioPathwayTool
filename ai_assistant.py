from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

from dotenv import load_dotenv
from mistralai.client import Mistral

from main import get_propagation_report


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Can be overridden in .env
MISTRAL_MODEL = os.getenv(
    "MISTRAL_MODEL",
    "ministral-14b-2512",
)

# Maximum number of model -> tool -> model cycles
MAX_TOOL_ROUNDS = 5

# Keep conversational history deliberately short.
# Tool data is NOT retained indefinitely.
MAX_HISTORY_MESSAGES = 6

# Cache propagation reports for a few minutes.
REPORT_CACHE_SECONDS = 300


# ============================================================
# CLIENT
# ============================================================

if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set. "
        "Add it to your .env file."
    )

client = Mistral(
    api_key=MISTRAL_API_KEY
)


# ============================================================
# PROPAGATION REPORT CACHE
# ============================================================

# Cache is keyed by TX/RX path.
#
# This is important because:
#
# Nelson -> Sydney
#
# and
#
# Nelson -> Tokyo
#
# must never accidentally return the same cached GRAFEX result.

_cached_reports: dict[
    tuple[str | None, str | None],
    tuple[Any, float],
] = {}


def get_cached_propagation_report(
    tx_location: str | None = None,
    rx_location: str | None = None,
) -> Any:
    """
    Return a cached propagation report for the requested path.

    The propagation engine remains the source of truth.

    The cache simply prevents repeated expensive HAP /
    ionosphere / space-weather / GRAFEX requests for the
    same path within the cache period.
    """

    global _cached_reports

    # Normalize locations for consistent cache keys.
    tx_key = (
        tx_location.strip()
        if isinstance(tx_location, str) and tx_location.strip()
        else None
    )

    rx_key = (
        rx_location.strip()
        if isinstance(rx_location, str) and rx_location.strip()
        else None
    )

    cache_key = (
        tx_key,
        rx_key,
    )

    now = time.time()

    cached = _cached_reports.get(
        cache_key
    )

    if cached is not None:
        cached_report, cached_time = cached

        if now - cached_time < REPORT_CACHE_SECONDS:
            return cached_report

        # Remove expired entry.
        _cached_reports.pop(
            cache_key,
            None,
        )

    # --------------------------------------------------------
    # Call the actual propagation engine.
    #
    # For path-specific requests this passes TX/RX through to
    # main.py, where GRAFEX should perform the point-to-point
    # calculation.
    # --------------------------------------------------------

    report = get_propagation_report(
        tx_location=tx_key,
        rx_location=rx_key,
    )

    _cached_reports[cache_key] = (
        report,
        now,
    )

    return report


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_value(value: Any) -> Any:
    """
    Convert dataclasses, dates, datetimes and arbitrary objects
    into JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if is_dataclass(value):
        return serialize_value(
            asdict(value)
        )

    if isinstance(value, dict):
        return {
            str(key): serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialize_value(item)
            for item in value
        ]

    if isinstance(value, set):
        return [
            serialize_value(item)
            for item in value
        ]

    if isinstance(value, (str, int, float, bool)):
        return value

    # Try common object serialization methods.
    if hasattr(value, "model_dump"):
        try:
            return serialize_value(
                value.model_dump()
            )
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return serialize_value(
                value.dict()
            )
        except Exception:
            pass

    return str(value)


def json_dumps(value: Any) -> str:
    """
    Convert a Python value into compact JSON suitable for a tool
    response.

    Compact JSON is intentional because tool output contributes
    to model context/token usage.
    """

    return json.dumps(
        serialize_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ============================================================
# REPORT ACCESS
# ============================================================

def get_report_value(
    report: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a field from either a dictionary or an object.
    """

    if report is None:
        return default

    if isinstance(report, dict):
        return report.get(
            key,
            default,
        )

    return getattr(
        report,
        key,
        default,
    )


def report_to_dict(
    report: Any,
) -> dict[str, Any]:
    """
    Convert the propagation report into a dictionary.

    Human-readable 'text' is deliberately removed because the AI
    should reason from structured data rather than from pre-written
    interpretations.

    The GRAFEX result is NOT removed.
    """

    if report is None:
        return {}

    if is_dataclass(report):
        data = asdict(report)

    elif isinstance(report, dict):
        data = dict(report)

    elif hasattr(report, "model_dump"):
        data = report.model_dump()

    elif hasattr(report, "dict"):
        data = report.dict()

    else:
        return {}

    # Never feed the human-readable report back into the AI.
    #
    # IMPORTANT:
    # Do NOT remove "grafex".
    # Do NOT remove "space_weather".
    # Do NOT remove "hap".
    # Do NOT remove "ionosphere".
    data.pop(
        "text",
        None,
    )

    return serialize_value(
        data
    )


# ============================================================
# AI TOOL: FULL STRUCTURED PROPAGATION REPORT
# ============================================================

def tool_get_propagation_report(
    tx_location: str | None = None,
    rx_location: str | None = None,
) -> dict[str, Any]:
    """
    Return structured propagation data for the AI.

    For path-specific questions, TX and RX locations are passed
    into the propagation engine so that GRAFEX can perform the
    actual point-to-point prediction.

    For general propagation questions, locations may be omitted.
    """

    try:
        # ----------------------------------------------------
        # Validate paired locations.
        # ----------------------------------------------------

        if (
            tx_location is not None
            and not isinstance(tx_location, str)
        ):
            return {
                "type": "propagation_report",
                "status": "error",
                "error": (
                    "tx_location must be a string."
                ),
            }

        if (
            rx_location is not None
            and not isinstance(rx_location, str)
        ):
            return {
                "type": "propagation_report",
                "status": "error",
                "error": (
                    "rx_location must be a string."
                ),
            }

        tx_location = (
            tx_location.strip()
            if tx_location
            else None
        )

        rx_location = (
            rx_location.strip()
            if rx_location
            else None
        )

        # A path-specific report requires both endpoints.
        if (
            (tx_location is None)
            != (rx_location is None)
        ):
            return {
                "type": "propagation_report",
                "status": "error",
                "error": (
                    "Path-specific propagation requires both "
                    "tx_location and rx_location."
                ),
            }

        # ----------------------------------------------------
        # Get the actual propagation report.
        # ----------------------------------------------------

        report = get_cached_propagation_report(
            tx_location=tx_location,
            rx_location=rx_location,
        )

        if report is None:
            return {
                "type": "propagation_report",
                "status": "unavailable",
                "error": (
                    "Propagation report is unavailable."
                ),
            }

        data = report_to_dict(
            report
        )

        if not data:
            return {
                "type": "propagation_report",
                "status": "error",
                "error": (
                    "Propagation report contained "
                    "no usable data."
                ),
            }

        return {
            "type": "propagation_report",
            "status": "ok",
            "tx_location": tx_location,
            "rx_location": rx_location,
            "data": data,
        }

    except Exception as exc:
        return {
            "type": "propagation_report",
            "status": "error",
            "error": str(exc),
        }


# ============================================================
# AI TOOL: HAP
# ============================================================

def tool_get_hap_forecast(
    tx_location: str | None = None,
    rx_location: str | None = None,
) -> dict[str, Any]:
    """
    Return only the HAP portion of the propagation report.

    If TX/RX locations are supplied, the underlying report is
    generated for that path.
    """
    try:
        report = get_cached_propagation_report(
            tx_location=tx_location,
            rx_location=rx_location,
        )

        # Use the independent HAP forecasts used by the dashboard.
        # These contain supported_grid_points/grid_points for regional
        # coverage rather than the raw HAP pixel support value.
        hap = getattr(
            report,
            "hap_band_forecast",
            None,
        )

        if hap is None:
            return {
                "type": "hap_forecast",
                "status": "unavailable",
                "error": (
                    "HAP data is unavailable."
                ),
            }

        # Add explicit regional coverage percentages for the AI.
        # The percentage is based on the same 49-point regional grid
        # used by the website: supported_grid_points / grid_points.
        hap_with_coverage = {}

        if isinstance(hap, dict):
            for band, hours in hap.items():

                if not isinstance(hours, dict):
                    hap_with_coverage[band] = hours
                    continue

                band_hours = {}

                for hour, prediction in hours.items():

                    if not isinstance(prediction, dict):
                        band_hours[hour] = prediction
                        continue

                    item = dict(prediction)

                    supported_points = item.get(
                        "supported_grid_points"
                    )
                    grid_points = item.get(
                        "grid_points"
                    )

                    if (
                        isinstance(supported_points, (int, float))
                        and isinstance(grid_points, (int, float))
                        and grid_points > 0
                    ):
                        item["coverage_percent"] = round(
                            supported_points / grid_points * 100,
                            1,
                        )

                    band_hours[hour] = item

                hap_with_coverage[band] = band_hours

        return {
            "type": "hap_forecast",
            "status": "ok",
            "tx_location": tx_location,
            "rx_location": rx_location,
            "data": serialize_value(hap_with_coverage),
        }

    except Exception as exc:
        return {
            "type": "hap_forecast",
            "status": "error",
            "error": str(exc),
        }


def tool_get_ionosphere(
    tx_location: str | None = None,
    rx_location: str | None = None,
) -> dict[str, Any]:
    """
    Return only ionospheric observations.

    If TX/RX locations are supplied, the underlying report is
    generated for that path.
    """

    try:
        report = get_cached_propagation_report(
            tx_location=tx_location,
            rx_location=rx_location,
        )

        ionosphere = get_report_value(
            report,
            "ionosphere",
        )

        if ionosphere is None:
            return {
                "type": "ionosphere",
                "status": "unavailable",
                "error": (
                    "Ionospheric data is unavailable."
                ),
            }

        return {
            "type": "ionosphere",
            "status": "ok",
            "tx_location": tx_location,
            "rx_location": rx_location,
            "data": serialize_value(
                ionosphere
            ),
        }

    except Exception as exc:
        return {
            "type": "ionosphere",
            "status": "error",
            "error": str(exc),
        }


# ============================================================
# AI TOOL: SPACE WEATHER
# ============================================================

def tool_get_space_weather(
    tx_location: str | None = None,
    rx_location: str | None = None,
) -> dict[str, Any]:
    """
    Return current space-weather measurements and alerts.

    If TX/RX locations are supplied, the underlying report is
    generated for that path so the associated T-index/GRAFEX
    information remains consistent with the requested report.
    """

    try:
        report = get_cached_propagation_report(
            tx_location=tx_location,
            rx_location=rx_location,
        )

        space_weather = get_report_value(
            report,
            "space_weather",
        )

        if space_weather is None:
            return {
                "type": "space_weather",
                "status": "unavailable",
                "error": (
                    "Space-weather data is unavailable."
                ),
            }

        return {
            "type": "space_weather",
            "status": "ok",
            "tx_location": tx_location,
            "rx_location": rx_location,
            "data": serialize_value(
                space_weather
            ),
        }

    except Exception as exc:
        return {
            "type": "space_weather",
            "status": "error",
            "error": str(exc),
        }


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_propagation_report",
            "description": (
                "Get the current structured HF propagation report. "
                "This includes HAP, ionosphere, space weather, "
                "and path-specific GRAFEX data when TX and RX "
                "locations are supplied. "
                "\n\n"
                "IMPORTANT: If the user asks about propagation "
                "between two locations, you MUST provide both "
                "tx_location and rx_location. "
                "The propagation engine will use those locations "
                "to run the actual point-to-point GRAFEX prediction. "
                "\n\n"
                "For an explicit GRAFEX request, such as "
                "'run GRAFEX for Nelson to Mumbai', this tool "
                "MUST be called with both endpoints. "
                "Do not use HAP as a substitute for GRAFEX. "
                "\n\n"
                "Examples of path-specific questions include: "
                "'How do I reach Sydney from Nelson?', "
                "'What bands could I use from Nelson to Sydney?', "
                "'What does GRAFEX predict for Nelson to Sydney?', "
                "or 'What is the propagation between New Zealand "
                "and Australia?'. "
                "\n\n"
                "For a general current propagation overview where "
                "no specific path is mentioned, locations may be "
                "omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tx_location": {
                        "type": "string",
                        "description": (
                            "Transmitter location. "
                            "Use a clear location such as "
                            "'Nelson, New Zealand'. "
                            "Required for a path-specific "
                            "propagation prediction."
                        ),
                    },
                    "rx_location": {
                        "type": "string",
                        "description": (
                            "Receiver location. "
                            "Use a clear location such as "
                            "'Mumbai, India'. "
                            "Required for a path-specific "
                            "propagation prediction."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hap_forecast",
            "description": (
                "Get the current HAP HF propagation forecast. "
                "Use this for questions specifically asking "
                "about HAP predictions, band predictions, "
                "upcoming HAP transitions, or regional HAP "
                "distribution. "
                "Do NOT use this as a substitute when the user "
                "explicitly requests GRAFEX."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tx_location": {
                        "type": "string",
                        "description": (
                            "Transmitter location if a specific "
                            "path is being requested."
                        ),
                    },
                    "rx_location": {
                        "type": "string",
                        "description": (
                            "Receiver location if a specific "
                            "path is being requested."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ionosphere",
            "description": (
                "Get current ionospheric station observations. "
                "Use this when the user asks about MUF observations, "
                "ionospheric enhancement/depression, or current "
                "station-level ionospheric conditions. "
                "Do NOT use this as a substitute when the user "
                "explicitly requests GRAFEX."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tx_location": {
                        "type": "string",
                        "description": (
                            "Transmitter location if relevant "
                            "to the requested propagation path."
                        ),
                    },
                    "rx_location": {
                        "type": "string",
                        "description": (
                            "Receiver location if relevant "
                            "to the requested propagation path."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_space_weather",
            "description": (
                "Get current solar and geomagnetic measurements "
                "and current space-weather alerts. "
                "Use this for questions specifically about solar "
                "or geomagnetic weather. "
                "Do NOT use this as a substitute when the user "
                "explicitly requests GRAFEX."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tx_location": {
                        "type": "string",
                        "description": (
                            "Transmitter location if relevant "
                            "to the requested propagation path."
                        ),
                    },
                    "rx_location": {
                        "type": "string",
                        "description": (
                            "Receiver location if relevant "
                            "to the requested propagation path."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]


AVAILABLE_TOOLS = {
    "get_propagation_report":
        tool_get_propagation_report,

    "get_hap_forecast":
        tool_get_hap_forecast,

    "get_ionosphere":
        tool_get_ionosphere,

    "get_space_weather":
        tool_get_space_weather,
}


# ============================================================
# GRAFEX REQUEST DETECTION
# ============================================================

def is_explicit_grafex_request(
    user_message: str,
) -> bool:
    """
    Detect whether the user explicitly requested GRAFEX.

    This is intentionally deterministic so the Python layer
    can enforce GRAFEX-specific behaviour even if the language
    model attempts to substitute another dataset.
    """

    if not isinstance(
        user_message,
        str,
    ):
        return False

    text = user_message.lower().strip()

    grafex_terms = (
        "grafex",
        "grafex prediction",
        "grafex forecast",
        "run grafex",
        "make a grafex",
        "do a grafex",
        "grafex for",
        "grafex on",
        "grafex between",
    )

    return any(
        term in text
        for term in grafex_terms
    )

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are RadioPathwayTool, an HF amateur-radio propagation assistant.

Your purpose is to answer questions about HF propagation using the actual data returned by the RadioPathwayTool Python tools.

The Python tools and propagation engine are the source of truth for current conditions and model predictions.

Never invent measurements, forecasts, path results, frequencies, times, alerts, propagation effects, or other data.

============================================================

1. CORE RULE
============================================================

Use this evidence chain:

DATA → DIRECT INTERPRETATION → ANSWER

Do not make unsupported additional inferences.

============================================================

GRAFEX RESPONSE STYLE
============================================================

When GRAFEX data is available, do NOT automatically dump the complete
hourly GRAFEX dataset into the answer.

The default response should be a concise, useful interpretation of the
GRAFEX prediction.

For a normal GRAFEX or path-specific propagation question:

- Give the TX → RX path.
- Give the distance when available.
- Give the T-index when available.
- Give the primary propagation mode when available.
- Summarize the important predicted propagation periods.
- Explain the useful frequency/propagation ranges when they can be
  directly supported by the returned GRAFEX data.
- Focus on information that helps the amateur-radio operator decide
  when or how to operate.
- Do NOT reproduce every hourly GRAFEX row by default.
- Do NOT reproduce the complete symbol table or legend by default.

The detailed GRAFEX data remains available to you and should be used
to produce the concise interpretation.

Only provide the full hourly GRAFEX data when the user explicitly asks
for detailed, full, complete, raw, hourly, technical, or otherwise
in-depth GRAFEX information.

Examples:

User:
"Can you make me a GRAFEX for Nelson to Mumbai?"

Preferred response:
Provide a concise GRAFEX path summary and interpretation. Do not dump
all 24 hourly rows.

User:
"When can I contact Mumbai from Nelson?"

Preferred response:
Use the returned GRAFEX data to identify the relevant predicted UTC
periods and explain the propagation conditions. Do not simply dump the
raw hourly table.

User:
"What band should I use to reach Mumbai from Nelson?"

Preferred response:
Use the returned GRAFEX data to explain the predicted useful frequency
range and relevant time periods. Do not dump the complete raw GRAFEX
dataset.

User:
"Give me the full GRAFEX data for Nelson to Mumbai."

Preferred response:
Provide the detailed hourly GRAFEX data, including OWF, EMUF, ALF,
frequency symbols, and the symbol legend when available.

User:
"Explain the GRAFEX for Nelson to Mumbai in detail."

Preferred response:
Give a detailed interpretation of the GRAFEX prediction and include
the relevant technical data needed to support that explanation.

IMPORTANT:

Do not invent amateur-radio band mappings from GRAFEX symbol positions
unless the returned data or verified application data explicitly
provides that mapping.

Do not treat GRAFEX model output as a guarantee that a contact will
succeed.

When discussing a predicted frequency range, distinguish between:
- the model's predicted propagation limits/ranges,
- amateur-radio operating bands,
- and an actual successful contact.

If the GRAFEX data does not support a precise conclusion, say so rather
than guessing.

The goal is:

NORMAL QUESTION → concise operational interpretation

EXPLICIT DETAILED QUESTION → detailed technical GRAFEX report

============================================================

For example:

GOOD:
"K-index is 0.67. That is consistent with relatively quiet
geomagnetic activity."

BAD:
"K-index is 0.67, therefore HF propagation is stable."

BAD:
"K-index is 0.67, therefore 80m will work."

A measurement may support a scientific interpretation, but it does
not automatically support a prediction about a particular band or
path.

When several datasets are available, do not combine them into a
stronger conclusion unless the supplied data explicitly supports
that conclusion.

============================================================
2. RESPONSE STYLE
============================================================

Be concise and directly answer the user's question.

Default response length:

* Simple question: 1–3 sentences.
* Simple recommendation: 2–5 sentences or a few bullets.
* Broader technical question: short headings and concise bullets.
* Detailed analysis: only when the user asks for detail.

Do not dump the entire propagation report.

Do not repeat raw tool output unnecessarily.

Only mention data that is relevant to the question.

Answer the question first, then provide the minimum explanation
needed to understand the answer.

If the user asks a simple question, give a simple answer.

============================================================
3. TOOL SELECTION
============================================================

Use the appropriate tool when current information is required.

Use get_space_weather for:

* F10.7
* sunspot number
* planetary K-index
* Australian K-index
* A-index
* Dst
* X-ray flux
* HF fadeout
* polar-cap absorption
* current alerts
* current warnings
* current watches

Use get_ionosphere for:

* current ionospheric observations
* station MUF observations
* station enhancement/depression
* station-level ionospheric conditions

Use get_hap_forecast for:

* HAP predictions
* current HAP band predictions
* upcoming HAP transitions
* HAP regional distributions

Use get_propagation_report for:

* broad current propagation questions
* questions involving multiple propagation datasets
* path-specific questions
* questions involving GRAFEX
* questions asking about propagation between two locations

For path-specific questions, get_propagation_report is the primary
tool.

============================================================
3A. EXPLICIT HAP REQUESTS
============================================================

If the user explicitly mentions HAP, HAP forecast, HAP predictions,
HAP bands, HAP transitions, or asks for HAP for a location, treat
that as a DIRECT HAP REQUEST.

Examples:

"hap for Nelson"

"HAP forecast for Nelson"

"what does HAP say for Nelson"

"show me the HAP prediction"

"what bands does HAP predict"

For a DIRECT HAP REQUEST:

1. MUST call get_hap_forecast.
2. Do NOT call get_propagation_report instead.
3. Do NOT use GRAFEX instead.
4. Do NOT use ionosphere instead.
5. Do NOT use space weather instead.
6. If one location is supplied, pass it as tx_location.
7. If two locations are supplied, pass them as tx_location and
   rx_location.
8. Use the returned HAP data directly in the answer.
9. If get_hap_forecast returns status="ok", do NOT say that HAP
   is unavailable.
10. Only say HAP is unavailable if the HAP tool actually returns
    status="unavailable" or status="error".

Examples:

User:
"hap for Nelson"

Call:

get_hap_forecast(
    tx_location="Nelson, New Zealand"
)

User:
"hap for Nelson to Sydney"

Call:

get_hap_forecast(
    tx_location="Nelson, New Zealand",
    rx_location="Sydney, Australia"
)

IMPORTANT:

An explicit HAP request takes priority over the general
get_propagation_report rule.

Do not interpret the phrase "hap for Nelson" as a broad propagation
question requiring get_propagation_report.

============================================================
4. PATH-SPECIFIC QUESTIONS
============================================================

When the user asks about propagation between two locations:

1. Identify the transmitter location (TX).
2. Identify the receiver location (RX).
3. Call get_propagation_report with BOTH locations.
4. Use the returned GRAFEX result when available.

Example:

"What is the best way to reach Sydney from Nelson?"

Call:

get_propagation_report(
    tx_location="Nelson, New Zealand",
    rx_location="Sydney, Australia"
)

Example:

"What does propagation look like from Blenheim to Brisbane?"

Call:

get_propagation_report(
    tx_location="Blenheim, New Zealand",
    rx_location="Brisbane, Australia"
)

Do NOT call the zero-location version for a path-specific question.

Do not substitute regional HAP information for a path-specific
GRAFEX result.

If GRAFEX is unavailable, say so rather than estimating the result.

============================================================
4A. GRAFEX LOCATION REQUIREMENTS
============================================================

GRAFEX requires a specific transmitter and receiver location.

A city, town, or specific station is a valid endpoint.

Examples:

"Nelson to Mumbai"

means:

TX = Nelson, New Zealand
RX = Mumbai, India

"Nelson to Sydney"

means:

TX = Nelson, New Zealand
RX = Sydney, Australia

Minor obvious spelling mistakes should be corrected when the
intended location is unambiguous.

For example:

"mubai" → Mumbai

"nelson" → Nelson, New Zealand

Do NOT silently convert a whole country, continent, or broad region
into an arbitrary point.

For example:

"Nelson to India"

is insufficient for a single path-specific GRAFEX prediction.

Ask which city or receiver location in India is intended.

Likewise:

"Nelson to Australia"

should ask which city or receiver location is intended.

If the user provides a city, town, or station, use it.

============================================================
4B. EXPLICIT GRAFEX REQUESTS
============================================================

If the user explicitly asks for GRAFEX, such as:

"grafex for Nelson to Mumbai"

"run GRAFEX"

"make a GRAFEX"

"what does GRAFEX predict"

"grafex prediction for Nelson to Sydney"

then this is a DIRECT GRAFEX REQUEST.

For a direct GRAFEX request:

1. Identify TX.
2. Identify RX.
3. If either endpoint is missing, ask for it.
4. If either endpoint is only a broad country or region, ask for
   a specific city or location.
5. Call get_propagation_report with BOTH endpoints.
6. Use the returned GRAFEX data.
7. If GRAFEX is not returned, state that GRAFEX was not returned.
8. Do NOT replace missing GRAFEX data with HAP.
9. Do NOT replace missing GRAFEX data with ionosphere data.
10. Do NOT replace missing GRAFEX data with space-weather data.
11. Do NOT provide a substitute band recommendation unless the
    user separately asked for one.

If the user explicitly asks for GRAFEX and the propagation report
does not contain GRAFEX data, the correct response is:

"The propagation report did not return a GRAFEX result for the
requested path."

Do not follow that statement with a HAP recommendation.

Do not say:

"GRAFEX is unavailable, however HAP suggests 160m."

Do not say:

"Since GRAFEX is unavailable, 160m is the best band."

Do not say:

"HAP can be used instead."

A missing GRAFEX result is simply missing GRAFEX data.

============================================================
5. GRAFEX
============================================================

GRAFEX is a path-specific HF propagation model.

When a GRAFEX result is returned, use its actual values.

Possible GRAFEX outputs include:

* TX/RX locations
* distance
* bearing
* T-index
* OWF
* EMUF
* ALF
* propagation modes
* frequency predictions
* other values explicitly returned by the engine

Treat all GRAFEX results as MODEL PREDICTIONS, not guarantees.

OWF:
The Optimum Working Frequency predicted by GRAFEX.

EMUF:
The Estimated Maximum Usable Frequency predicted by GRAFEX.

ALF:
The Absorption-Limited Frequency predicted by GRAFEX.

Do not claim that:

* OWF is guaranteed to work.
* EMUF is a guaranteed maximum.
* ALF means every lower frequency will fail.
* A GRAFEX prediction guarantees a contact.
* A GRAFEX prediction guarantees a particular signal strength.

Do not estimate GRAFEX values from other datasets.

Never derive GRAFEX results from:

* F10.7
* sunspot number
* K-index
* Dst
* station observations
* generic HF knowledge

If GRAFEX is absent from a path report, state:

"The propagation report did not return a GRAFEX result for the
requested path."

Do not invent a MUF or OWF.

============================================================
6. FREQUENCY RECOMMENDATIONS
============================================================

When the user asks which frequency or band to try:

Use the available model data.

For path-specific questions, prefer GRAFEX path-specific results.

Respect the user's actual amateur-radio band privileges.

Never recommend transmitting on a frequency outside the user's
permitted allocation.

GRAFEX frequencies are model frequencies, not automatically amateur
frequencies.

When GRAFEX provides a frequency near an amateur band, map it to the
relevant permitted amateur band rather than telling the user to
transmit exactly on the model frequency.

For example:

GRAFEX OWF = 17.9 MHz

Possible interpretation:

"17m is near the predicted OWF."

Do NOT automatically say:

"Transmit on 17.9 MHz."

If the supplied data does not establish that a band is suitable,
say that the model does not establish it.

============================================================
7. HAP
============================================================

HAP provides MODEL PREDICTIONS.

If HAP predicts a band:

GOOD:
"HAP currently predicts 80m at the configured base point."

BAD:
"80m is definitely open."

BAD:
"80m will work."

BAD:
"80m is guaranteed."

A HAP prediction is not proof that a real-world contact will occur.

---

## HAP REGIONAL PERCENTAGES

HAP regional percentages describe the proportion of decoded HAP grid
points supporting a band.

Example:

80m: 17%

Correct:

"17% of the decoded HAP grid points support 80m."

Do NOT describe this as:

* probability of making a contact
* chance of success
* signal-strength probability
* reliability
* probability that the band is open at the user's station

---

## HAP ZERO PERCENT

If HAP reports 0% for a band:

GOOD:
"HAP does not currently predict that band at the decoded grid
points."

Do NOT say:

* "The band is closed."
* "The band will not work."
* "Avoid that band."

A HAP model prediction does not guarantee real-world failure.

---

## HAP TRANSITIONS

A HAP transition is a model prediction for when its predicted
conditions change.

Do not describe it as an exact physical opening or closing time.

Use wording such as:

"HAP predicts a transition at 09:00 UTC."

Not:

"80m will open at 09:00."

============================================================
8. SPACE WEATHER
============================================================

Always distinguish the measurement from what it directly indicates.

---

## F10.7

F10.7 measures solar radio flux.

GOOD:
"F10.7 is 106 sfu."

GOOD:
"F10.7 is the measured solar radio flux."

Do not use F10.7 alone to claim that a particular HF band is open,
closed, good, bad, or reliable.

---

## SUNSPOT NUMBER

Sunspot number describes observed sunspot activity.

Do not use sunspot number alone to determine current band
conditions.

---

## K-INDEX

K-index describes geomagnetic activity.

GOOD:
"K-index is 0.67, consistent with relatively quiet geomagnetic
activity."

Do not use K-index alone to predict the performance of a specific
HF path or band.

Do not substitute K-index for GRAFEX's T-index.

---

## DST

Dst indicates geomagnetic disturbance associated with the
magnetospheric ring current.

Report the value and its direct scientific meaning without turning
Dst alone into a prediction of HF performance.

---

## X-RAY FLUX

X-ray flux can provide information about solar X-ray activity.

If unavailable, say:

"Current X-ray activity cannot be assessed from the supplied data."

Missing X-ray data does NOT mean there are no flares.

---

## HF FADEOUT

If HF fadeout is False:

"No HF fadeout event is currently reported by the supplied data."

Do not conclude that:

* HF propagation is unaffected.
* the ionosphere is stable.
* signals will not be affected.

---

## POLAR-CAP ABSORPTION

If PCA is False:

"No polar-cap absorption event is currently reported."

Do not conclude that there is no ionospheric absorption or that HF
propagation is unaffected.

============================================================
9. IONOSPHERIC OBSERVATIONS
============================================================

Ionospheric observations are station-specific.

If Perth reports +23%:

GOOD:
"Perth is reporting an ionospheric value 23% above its normal
reference."

Do not automatically infer:

* the user's local ionosphere has the same condition
* an entire country has the same condition
* an entire region has the same condition
* a particular path is enhanced
* stronger signals
* better long-distance propagation

Only make a path-specific claim when a path-specific result supports
it.

============================================================
10. SPACE-WEATHER ALERTS
============================================================

Distinguish:

WATCH:
A potential future event is being monitored or predicted.

WARNING:
The source is warning of an event.

ALERT:
The source reports an observed/current event.

A geomagnetic storm category describes geomagnetic activity.

Do not automatically translate:

G1 → bad HF
G2 → bad HF
G3 → bad HF

etc.

Do not invent radio effects from an alert.

============================================================
11. MISSING DATA
============================================================

Missing data means unavailable information.

It does NOT mean the opposite condition.

Examples:

BAD:
"X-ray data is unavailable, therefore there are no flares."

GOOD:
"X-ray data is unavailable, so current X-ray activity cannot be
assessed."

BAD:
"Ionospheric data is unavailable, therefore the ionosphere is
normal."

GOOD:
"Current ionospheric observations are unavailable."

For an explicit GRAFEX request, missing GRAFEX data must not be
replaced by another propagation dataset.

============================================================
12. CURRENT VS FUTURE
============================================================

Clearly distinguish observations from forecasts.

For "right now", use current data.

For "tonight", "tomorrow", or another future time, use relevant
forecast/model data when available.

Never present current conditions as future conditions.

Clearly label model predictions and forecasts as predictions.

============================================================
13. TIME
============================================================

Never invent or guess a timezone.

Use UTC when UTC is supplied.

Use local time only when the tool supplies the corresponding local
time or the timezone is explicitly known.

Always label UTC and local times clearly.

A predicted transition time is a model prediction, not a guaranteed
propagation event.

============================================================
14. GENERAL HF SCIENCE
============================================================

You may explain general scientific concepts such as:

* MUF
* LUF
* critical frequency
* NVIS
* skywave propagation
* ionospheric reflection
* absorption
* sporadic-E
* grayline propagation
* skip distance
* solar radiation
* geomagnetic storms
* frequency selection

Clearly separate general scientific explanation from current
measurements and model predictions.

Do not present general HF theory as evidence that a current path or
band is working.

============================================================
15. UNSUPPORTED SIGNAL CLAIMS
============================================================

Do not claim:

* stronger signals
* weaker signals
* louder signals
* better readability
* longer range
* shorter range
* greater reliability
* successful contacts

unless the supplied data explicitly supports that claim.

Propagation predictions are not signal-strength measurements.

============================================================
16. PATH INFERENCE
============================================================

Never infer a path condition from an unrelated station.

For example:

"Perth is enhanced, so Nelson → Perth should be good."

is NOT valid unless a path-specific result supports it.

Regional or station observations may provide context, but they do not
replace path-specific analysis.

============================================================
17. DATA PRIORITY
============================================================

When multiple datasets are available, use the dataset appropriate to
the question.

For a specific path:

GRAFEX path prediction
↓
HAP/regional information as supporting context
↓
station observations and space weather as additional context

For an explicit GRAFEX request, only the GRAFEX result answers the
GRAFEX request.

Do not use HAP, ionosphere, or space weather as a replacement for a
missing GRAFEX result.

Do not combine multiple weak indicators into a strong conclusion
unless the data explicitly supports that conclusion.

============================================================
18. CONCISENESS
============================================================

The user normally wants an operational answer, not a complete
scientific report.

Prioritize:

1. Direct answer.
2. Most relevant model result.
3. One short explanation if useful.
4. Important uncertainty or limitation.

Do not include unrelated fields from the propagation report.

Example:

User:
"What should I try to reach the Gold Coast?"

Preferred:

"GRAFEX currently puts the path's OWF at 17.9 MHz, so 17m is the
closest amateur band to that modelled optimum. GRAFEX is a model
prediction, so it does not guarantee a contact."

Do not respond with a long explanation of F10.7, K-index, Dst,
Perth observations, every HAP grid percentage, and every GRAFEX
field unless the user asks for a detailed analysis.

============================================================
19. FINAL VALIDATION
============================================================

Before answering, check:

1. Did I use the appropriate tool for current information?
2. If this is a path question, did I provide both TX and RX?
3. Did I use GRAFEX when it was available?
4. Did I distinguish measurements from interpretations?
5. Did I make an unsupported second-order inference?
6. Did I treat missing data as evidence of the opposite condition?
7. Did I treat HAP percentages as probabilities?
8. Did I describe a HAP prediction as guaranteed?
9. Did I describe a HAP transition as an exact opening/closing?
10. Did I infer a path condition from an unrelated station?
11. Did I infer signal strength from propagation data?
12. Did I invent an effect from a geomagnetic alert?
13. Did I confuse current conditions with forecasts?
14. Did I invent a GRAFEX value?
15. Did I present GRAFEX as a guarantee?
16. Did I recommend a frequency outside the user's permitted band?
17. Did I answer more broadly than the user's question requires?

If any answer is YES, revise the response before sending it.

The goal is:

ACCURATE + EVIDENCE-BASED + CONCISE + OPERATIONALLY USEFUL.

Answer the question that was asked, not every question that the
available data could answer.
"""


# ============================================================
# MISTRAL MESSAGE SERIALIZATION
# ============================================================

def message_to_dict(
    message: Any,
) -> dict[str, Any]:
    """
    Convert a Mistral SDK message object into a plain dictionary.

    Mistral's SDK returns typed objects for assistant responses,
    while our conversation history is easier to maintain as plain
    dictionaries.
    """

    if isinstance(
        message,
        dict,
    ):
        return message

    if hasattr(
        message,
        "model_dump",
    ):
        try:
            return message.model_dump(
                exclude_none=True
            )
        except Exception:
            pass

    if hasattr(
        message,
        "dict",
    ):
        try:
            return message.dict(
                exclude_none=True
            )
        except Exception:
            pass

    result: dict[str, Any] = {}

    for field in (
        "role",
        "content",
        "tool_calls",
        "name",
        "tool_call_id",
    ):
        value = getattr(
            message,
            field,
            None,
        )

        if value is not None:
            result[field] = serialize_value(
                value
            )

    return result


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool_call(
    tool_name: str,
    arguments: str | dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Execute a model-requested tool safely.
    """

    tool = AVAILABLE_TOOLS.get(
        tool_name
    )

    if tool is None:
        return {
            "status": "error",
            "error": f"Unknown tool: {tool_name}",
        }

    try:
        if arguments is None:
            parsed_arguments = {}

        elif isinstance(
            arguments,
            dict,
        ):
            parsed_arguments = arguments

        else:
            parsed_arguments = json.loads(
                arguments
            )

        if not isinstance(
            parsed_arguments,
            dict,
        ):
            return {
                "status": "error",
                "error": (
                    "Tool arguments must be a JSON object."
                ),
            }

        result = tool(
            **parsed_arguments
        )

        return serialize_value(
            result
        )

    except json.JSONDecodeError as exc:
        return {
            "status": "error",
            "error": (
                f"Invalid tool arguments: {exc}"
            ),
        }

    except TypeError as exc:
        return {
            "status": "error",
            "error": (
                f"Invalid tool arguments: {exc}"
            ),
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


# ============================================================
# AI ASSISTANT
# ============================================================

def ask_radio_assistant(
    user_message: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    """
    Ask the Mistral-powered RadioPathwayTool assistant a question.

    Returns:

        (
            final_answer,
            updated_conversation_history,
        )
    """

    if conversation_history is None:
        conversation_history = []

    # --------------------------------------------------------
    # Keep user/assistant conversation history short.
    #
    # IMPORTANT:
    # Tool outputs from previous turns are not intentionally
    # retained here. They can become very large and can consume
    # model context unnecessarily.
    # --------------------------------------------------------

    history = conversation_history[
        -MAX_HISTORY_MESSAGES:
    ]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(
        history
    )

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    # --------------------------------------------------------
    # Deterministic GRAFEX request tracking.
    #
    # This prevents the model from replacing a failed GRAFEX
    # request with HAP or another dataset.
    # --------------------------------------------------------

    explicit_grafex_request = (
        is_explicit_grafex_request(
            user_message
        )
    )

    grafex_was_returned = False

    # --------------------------------------------------------
    # Tool loop
    # --------------------------------------------------------

    for _ in range(
        MAX_TOOL_ROUNDS
    ):

        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.1,
            max_tokens=1200,
        )

        choice = response.choices[0]

        assistant_message = choice.message

        assistant_dict = message_to_dict(
            assistant_message
        )

        # Keep the assistant's response in the message history
        # so Mistral can associate subsequent tool responses
        # with the tool call.
        messages.append(
            assistant_dict
        )

        tool_calls = getattr(
            assistant_message,
            "tool_calls",
            None,
        )

        # ----------------------------------------------------
        # Normal final response
        # ----------------------------------------------------

        if not tool_calls:

            final_content = getattr(
                assistant_message,
                "content",
                None,
            )

            if final_content is None:
                final_content = ""

            # Mistral can return content as structured chunks.
            if isinstance(
                final_content,
                list,
            ):

                text_parts: list[str] = []

                for chunk in final_content:

                    if isinstance(
                        chunk,
                        dict,
                    ):

                        if chunk.get(
                            "type"
                        ) == "text":

                            text_parts.append(
                                chunk.get(
                                    "text",
                                    "",
                                )
                            )

                    elif hasattr(
                        chunk,
                        "text",
                    ):

                        text_parts.append(
                            getattr(
                                chunk,
                                "text",
                                "",
                            )
                        )

                final_content = "".join(
                    text_parts
                )

            final_answer = str(
                final_content
            ).strip()

            if not final_answer:
                final_answer = (
                    "I couldn't generate a response "
                    "from the available propagation data."
                )

            # ------------------------------------------------
            # HARD GRAFEX ENFORCEMENT
            #
            # If the user explicitly requested GRAFEX and the
            # actual propagation report did not contain GRAFEX,
            # NEVER allow the model to substitute HAP, ionosphere,
            # space weather, or a generic recommendation.
            # ------------------------------------------------

            if (
                explicit_grafex_request
                and not grafex_was_returned
            ):
                final_answer = (
                    "The propagation report did not return a "
                    "GRAFEX result for the requested path."
                )

            # ------------------------------------------------
            # Only retain normal user/assistant conversation.
            #
            # Do not preserve tool payloads from this turn.
            # ------------------------------------------------

            updated_history = [
                *history,
                {
                    "role": "user",
                    "content": user_message,
                },
                {
                    "role": "assistant",
                    "content": final_answer,
                },
            ]

            updated_history = updated_history[
                -MAX_HISTORY_MESSAGES:
            ]

            return (
                final_answer,
                updated_history,
            )

        # ----------------------------------------------------
        # Execute tool calls
        # ----------------------------------------------------

        for tool_call in tool_calls:

            function = getattr(
                tool_call,
                "function",
                None,
            )

            if function is None:
                continue

            tool_name = getattr(
                function,
                "name",
                "",
            )

            arguments = getattr(
                function,
                "arguments",
                "{}",
            )

            tool_call_id = getattr(
                tool_call,
                "id",
                "",
            )

            # ------------------------------------------------
            # For an explicit GRAFEX request, do not allow
            # secondary datasets to become substitutes.
            # ------------------------------------------------

            if (
                explicit_grafex_request
                and tool_name != "get_propagation_report"
            ):

                result = {
                    "status": "error",
                    "error": (
                        "The user explicitly requested GRAFEX. "
                        "Do not use this tool as a substitute. "
                        "Call get_propagation_report with both "
                        "the transmitter and receiver locations."
                    ),
                }

            else:

                result = execute_tool_call(
                    tool_name,
                    arguments,
                )

            # ------------------------------------------------
            # Track whether an explicit GRAFEX request actually
            # received GRAFEX data.
            # ------------------------------------------------

            if (
                explicit_grafex_request
                and tool_name == "get_propagation_report"
            ):

                print(
                    "\n===== GRAFEX TOOL RESULT =====",
                    flush=True,
                )

                print(
                    json.dumps(
                        result,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                    flush=True,
                )

                print(
                    "===== END GRAFEX TOOL RESULT =====\n",
                    flush=True,
                )

                data = (
                    result.get("data")
                    if isinstance(result, dict)
                    else None
                )

                data_status = (
                    data.get("data_status")
                    if isinstance(data, dict)
                    else None
                )

                grafex_was_returned = (
                    isinstance(data_status, dict)
                    and data_status.get("grafex_available") is True
                )

                # ------------------------------------------------
                # HARD STOP
                #
                # If GRAFEX was explicitly requested and the
                # propagation report does not contain it, stop
                # immediately.
                #
                # This prevents Mistral from seeing HAP and then
                # trying to construct a replacement answer.
                # ------------------------------------------------

                if grafex_was_returned:

                    grafex = data.get("grafex")

                    if isinstance(grafex, dict):

                        hours = grafex.get("hours", [])

                        # ------------------------------------------------
                        # Detect an optional UTC hour range in the user's
                        # original request.
                        #
                        # Examples:
                        #   "0utc to 12utc"
                        #   "00 utc to 12 utc"
                        #   "from 3utc to 8utc"
                        #
                        # End hour is inclusive.
                        # ------------------------------------------------

                        import re

                        range_match = re.search(
                            r"(?:from\\s*)?(\\d{1,2})\\s*utc\\s*(?:to|-|through)\\s*(\\d{1,2})\\s*utc",
                            user_message,
                            re.IGNORECASE,
                        )

                        if range_match:
                            start_hour = int(range_match.group(1))
                            end_hour = int(range_match.group(2))

                            if (
                                0 <= start_hour <= 23
                                and 0 <= end_hour <= 23
                            ):
                                if start_hour <= end_hour:
                                    selected_hours = [
                                        h
                                        for h in hours
                                        if isinstance(h, dict)
                                        and start_hour <= h.get("utc_hour", -1) <= end_hour
                                    ]
                                else:
                                    selected_hours = [
                                        h
                                        for h in hours
                                        if isinstance(h, dict)
                                        and (
                                            h.get("utc_hour", -1) >= start_hour
                                            or h.get("utc_hour", -1) <= end_hour
                                        )
                                    ]
                            else:
                                selected_hours = hours
                        else:
                            selected_hours = hours

                        summary_lines = [
                            "GRAFEX propagation prediction",
                            "",
                            f"TX: {grafex.get('tx_name', 'unknown')}",
                            f"RX: {grafex.get('rx_name', 'unknown')}",
                            f"Date: {grafex.get('prediction_date', 'unknown')}",
                            f"Distance: {grafex.get('distance_km', 'unknown')} km",
                            (
                                "TX → RX bearing: "
                                f"{grafex.get('bearing_tx_to_rx', 'unknown')}°"
                            ),
                            (
                                "RX → TX bearing: "
                                f"{grafex.get('bearing_rx_to_tx', 'unknown')}°"
                            ),
                            f"T-index: {grafex.get('t_index', 'unknown')}",
                            (
                                "Primary mode: "
                                f"{grafex.get('first_mode', 'unknown')}"
                            ),
                            (
                                "Secondary mode: "
                                f"{grafex.get('second_mode', 'unknown')}"
                            ),
                            "",
                            "UTC HOURLY GRAFEX DATA",
                            "-" * 72,
                        ]

                        for hour in selected_hours:

                            if not isinstance(hour, dict):
                                continue

                            utc_hour = hour.get("utc_hour")

                            def fmt(value):
                                if value is None:
                                    return "N/A"
                                try:
                                    return f"{float(value):.1f}"
                                except (TypeError, ValueError):
                                    return str(value)

                            summary_lines.append(
                                f"{int(utc_hour):02d}:00 UTC"
                            )

                            summary_lines.append(
                                "  First mode: "
                                f"OWF {fmt(hour.get('first_owf_mhz'))} MHz | "
                                f"EMUF {fmt(hour.get('first_emuf_mhz'))} MHz | "
                                f"ALF {fmt(hour.get('first_alf_mhz'))} MHz"
                            )

                            summary_lines.append(
                                "  Second mode: "
                                f"OWF {fmt(hour.get('second_owf_mhz'))} MHz | "
                                f"EMUF {fmt(hour.get('second_emuf_mhz'))} MHz | "
                                f"ALF {fmt(hour.get('second_alf_mhz'))} MHz"
                            )

                            frequencies = hour.get("frequencies")

                            if isinstance(frequencies, dict):
                                symbols = []

                                for frequency, info in sorted(
                                    frequencies.items(),
                                    key=lambda item: float(item[0]),
                                ):
                                    if not isinstance(info, dict):
                                        continue

                                    symbol = info.get("symbol", "?")

                                    try:
                                        frequency_text = f"{float(frequency):g}"
                                    except (TypeError, ValueError):
                                        frequency_text = str(frequency)

                                    symbols.append(
                                        f"{frequency_text}={symbol}"
                                    )

                                if symbols:
                                    summary_lines.append(
                                        "  Frequency symbols: "
                                        + " ".join(symbols)
                                    )

                            summary_lines.append("")

                        summary_lines.extend([
                            "GRAFEX SYMBOL LEGEND",
                            ". = usable less than 50% of days",
                            "% = usable 50% to 90% of days",
                            "B = both E and F modes 90% of days",
                            "M = mixed first and second F modes",
                            "F = first F mode only",
                            "E = E-layer propagation",
                            "P = 90% E and 50-90% F",
                            "S = second modes only",
                            "A = high absorption",
                            "X = complex modes",
                        ])

                        final_answer = "\n".join(summary_lines)

                    else:

                        final_answer = (
                            "GRAFEX was successfully returned for "
                            "the requested path, but the structured "
                            "GRAFEX data could not be formatted."
                        )

                    updated_history = [
                        *history,
                        {
                            "role": "user",
                            "content": user_message,
                        },
                        {
                            "role": "assistant",
                            "content": final_answer,
                        },
                    ]

                    updated_history = updated_history[
                        -MAX_HISTORY_MESSAGES:
                    ]

                    return (
                        final_answer,
                        updated_history,
                    )

                if not grafex_was_returned:

                    final_answer = (
                        "The propagation report did not return "
                        "a GRAFEX result for the requested path."
                    )

                    updated_history = [
                        *history,
                        {
                            "role": "user",
                            "content": user_message,
                        },
                        {
                            "role": "assistant",
                            "content": final_answer,
                        },
                    ]

                    updated_history = updated_history[
                        -MAX_HISTORY_MESSAGES:
                    ]

                    return (
                        final_answer,
                        updated_history,
                    )

            # ------------------------------------------------
            # Mistral expects tool results as role=tool
            # messages associated with the originating call.
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json_dumps(
                        result
                    ),
                    "tool_call_id": tool_call_id,
                }
            )

    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    fallback_answer = (
        "I wasn't able to complete the propagation analysis "
        "within the tool-call limit."
    )

    # If this was an explicit GRAFEX request, do not give a
    # generic fallback that could be interpreted as a successful
    # propagation analysis.
    if (
        explicit_grafex_request
        and not grafex_was_returned
    ):
        fallback_answer = (
            "The propagation report did not return a "
            "GRAFEX result for the requested path."
        )

    return (
        fallback_answer,
        [
            *history,
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": fallback_answer,
            },
        ][
            -MAX_HISTORY_MESSAGES:
        ],
    )


# ============================================================
# SIMPLE COMMAND-LINE TEST
# ============================================================

def main() -> None:
    """
    Simple local test.

    Run:

        python ai_assistant.py
    """

    print(
        "RadioPathwayTool Mistral AI assistant"
    )

    print(
        f"Model: {MISTRAL_MODEL}"
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    history: list[
        dict[str, Any]
    ] = []

    while True:

        try:
            user_input = input(
                "You: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            break

        if not user_input:
            continue

        if user_input.lower() in {
            "exit",
            "quit",
        }:
            break

        try:

            answer, history = ask_radio_assistant(
                user_input,
                history,
            )

            print()
            print(
                f"Assistant: {answer}"
            )
            print()

        except Exception as exc:

            print()
            print(
                "ERROR:"
            )

            print(
                str(exc)
            )

            print()


if __name__ == "__main__":
    main()
