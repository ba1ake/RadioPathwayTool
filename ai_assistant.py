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
#MISTRAL_MODEL=ministral-14b-2512

# Can be overridden in .env
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL",)
   

# Maximum number of model -> tool -> model cycles
MAX_TOOL_ROUNDS = 5

# Keep the conversational history deliberately short.
# Tool data is NOT retained indefinitely.
MAX_HISTORY_MESSAGES = 6

# Cache the propagation engine for a few minutes.
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

_cached_report: Any | None = None
_cached_report_time: float = 0.0


def get_cached_propagation_report() -> Any:
    """
    Return a cached propagation report.

    The propagation engine itself remains the source of truth.
    This cache simply prevents repeated expensive HAP /
    ionosphere / space-weather requests during one conversation.
    """

    global _cached_report
    global _cached_report_time

    now = time.time()

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
        return serialize_value(asdict(value))

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
            return serialize_value(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return serialize_value(value.dict())
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
        return report.get(key, default)

    return getattr(
        report,
        key,
        default,
    )


def report_to_dict(report: Any) -> dict[str, Any]:
    """
    Convert the propagation report into a dictionary.

    Human-readable 'text' is deliberately removed because the AI
    should reason from structured data rather than from pre-written
    interpretations.
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
    data.pop("text", None)

    return serialize_value(data)


# ============================================================
# AI TOOL: FULL STRUCTURED PROPAGATION REPORT
# ============================================================

def tool_get_propagation_report() -> dict[str, Any]:
    """
    Return structured propagation data for the AI.

    The human-readable report is deliberately excluded.
    """

    try:
        report = get_cached_propagation_report()

        if report is None:
            return {
                "type": "propagation_report",
                "status": "unavailable",
                "error": "Propagation report is unavailable.",
            }

        data = report_to_dict(report)

        if not data:
            return {
                "type": "propagation_report",
                "status": "error",
                "error": "Propagation report contained no usable data.",
            }

        return {
            "type": "propagation_report",
            "status": "ok",
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

def tool_get_hap_forecast() -> dict[str, Any]:
    """
    Return only the HAP portion of the propagation report.
    """

    try:
        report = get_cached_propagation_report()

        hap = get_report_value(
            report,
            "hap",
        )

        if hap is None:
            return {
                "type": "hap_forecast",
                "status": "unavailable",
                "error": "HAP data is unavailable.",
            }

        return {
            "type": "hap_forecast",
            "status": "ok",
            "data": serialize_value(hap),
        }

    except Exception as exc:
        return {
            "type": "hap_forecast",
            "status": "error",
            "error": str(exc),
        }


# ============================================================
# AI TOOL: IONOSPHERE
# ============================================================

def tool_get_ionosphere() -> dict[str, Any]:
    """
    Return only ionospheric observations.
    """

    try:
        report = get_cached_propagation_report()

        ionosphere = get_report_value(
            report,
            "ionosphere",
        )

        if ionosphere is None:
            return {
                "type": "ionosphere",
                "status": "unavailable",
                "error": "Ionospheric data is unavailable.",
            }

        return {
            "type": "ionosphere",
            "status": "ok",
            "data": serialize_value(ionosphere),
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

def tool_get_space_weather() -> dict[str, Any]:
    """
    Return only current space-weather measurements and alerts.
    """

    try:
        report = get_cached_propagation_report()

        space_weather = get_report_value(
            report,
            "space_weather",
        )

        if space_weather is None:
            return {
                "type": "space_weather",
                "status": "unavailable",
                "error": "Space-weather data is unavailable.",
            }

        return {
            "type": "space_weather",
            "status": "ok",
            "data": serialize_value(space_weather),
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
                "Get the current structured HF propagation report "
                "for the configured location. This includes HAP, "
                "ionosphere and space-weather data. Use this when "
                "the user asks for a broad current propagation "
                "overview."
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
                "Get the current HAP HF propagation forecast. "
                "Use this for questions about which bands HAP "
                "currently predicts, upcoming HAP transitions, "
                "regional HAP distribution, or band conditions."
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
                "Get current ionospheric station observations. "
                "Use this when the user asks about MUF observations, "
                "ionospheric enhancement/depression, or current "
                "station-level ionospheric conditions."
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
                "Get current solar and geomagnetic measurements "
                "and current space-weather alerts. Use this for "
                "questions specifically about solar or geomagnetic "
                "weather."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


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
You are RadioPathwayTool, an HF radio propagation assistant.

Your job is to explain HF propagation using the actual data
returned by the RadioPathwayTool Python tools.

The Python propagation engine is the source of truth.

You may explain scientific concepts and interpret measurements,
but you must never invent measurements, forecasts, path results,
frequencies, times, alerts, or propagation effects.

============================================================
CORE PRINCIPLE: EVIDENCE CHAIN
============================================================

Every factual claim about current conditions must be traceable
to supplied data.

Use this reasoning chain:

MEASURED / MODEL DATA
        ↓
DIRECT SCIENTIFIC INTERPRETATION
        ↓
STOP

Do NOT continue from a direct interpretation into an unsupported
second-order conclusion.

Example:

GOOD:
"K-index is 0.67. This is consistent with relatively quiet
geomagnetic activity."

BAD:
"K-index is 0.67, therefore the ionosphere is stable."

BAD:
"K-index is 0.67, therefore HF propagation is stable."

BAD:
"K-index is 0.67, therefore 80m should work well."

The first statement is a direct interpretation.

The others make additional claims that the supplied measurement
does not establish.

============================================================
DATA VS INTERPRETATION
============================================================

Always distinguish between:

1. What the source measured or predicted.
2. What that measurement directly indicates.
3. What cannot be concluded from it.

Preferred wording:

"F10.7 is 106 sfu. This is the measured solar radio flux."

"Planetary K-index is 0.67. This is consistent with relatively
quiet geomagnetic activity."

"HAP predicts 80m at the current base point."

"Ionospheric observations show Perth at +23% relative to normal."

Then explicitly stop unless another supplied dataset supports
the next conclusion.

Do not turn several individually reasonable observations into
a broader conclusion unless that broader conclusion is directly
supported.

============================================================
TOOL SELECTION
============================================================

Use the appropriate tool whenever the user asks about current
data.

Use get_space_weather for:

- F10.7
- sunspot number
- planetary K-index
- Australian K-index
- A-index
- Dst
- X-ray flux
- HF fadeout
- polar-cap absorption
- current alerts
- current warnings
- current watches

Use get_ionosphere for:

- current ionospheric observations
- station MUF observations
- enhancement/depression
- station-level ionospheric conditions

Use get_hap_forecast for:

- current HAP band prediction
- upcoming HAP transitions
- regional HAP distribution
- questions such as "will 80m work tonight?"
- questions about which bands HAP currently predicts

Use get_propagation_report for:

- broad current propagation questions
- questions involving multiple datasets
- general "what are conditions like?" questions

If the user asks for a specific category of current information,
prefer the specialized tool rather than requesting the entire
propagation report.

============================================================
SPACE WEATHER
============================================================

F10.7:

F10.7 measures solar radio flux.

Allowed:
"F10.7 is 106 sfu."

Allowed:
"That is the measured solar radio flux."

Not allowed:
"F10.7 is 106 sfu, therefore HF propagation is good."

Not allowed:
"F10.7 is 106 sfu, therefore 20m is open."

F10.7 alone does not determine current HF propagation.

------------------------------------------------------------

SUNSPOT NUMBER:

Sunspot number describes observed sunspot activity.

Allowed:
"Sunspot number is 76."

Allowed:
"Sunspot number is a measure of sunspot activity."

Not allowed:
"Sunspot number of 76 means 20m should be good."

Not allowed:
"Sunspot number of 76 means the ionosphere is strong."

Do not use sunspot number alone to determine current band
suitability.

------------------------------------------------------------

PLANETARY K-INDEX:

K-index describes geomagnetic activity.

Allowed:
"K-index is 0.67, which is consistent with relatively quiet
geomagnetic activity."

Not allowed:
"K-index is 0.67, therefore HF propagation is stable."

Not allowed:
"K-index is 0.67, therefore the ionosphere is undisturbed."

Not allowed:
"K-index is 0.67, therefore 80m will work."

------------------------------------------------------------

DST:

Dst is an indicator of geomagnetic disturbance associated with
the magnetospheric ring current.

Allowed:
"Dst is +29 nT."

Allowed:
"The positive Dst value does not indicate a strong negative
ring-current disturbance."

Do not turn Dst alone into a prediction of HF performance.

------------------------------------------------------------

X-RAY FLUX:

X-ray flux may provide information about solar X-ray activity.

If X-ray data is unavailable:

GOOD:
"X-ray flux is unavailable."

GOOD:
"Current X-ray activity cannot be assessed from the supplied
X-ray data."

BAD:
"There are no solar flares."

BAD:
"No recent flare activity is occurring."

BAD:
"The Sun is not producing X-ray flares."

Missing X-ray data is NEVER evidence that no flare is occurring.

------------------------------------------------------------

HF FADEOUT:

If HF fadeout is False:

Allowed:
"No HF fadeout event is currently reported by the supplied data."

Not allowed:
"HF propagation is unaffected."

Not allowed:
"Your HF signals will not be affected."

Not allowed:
"The ionosphere is stable."

A lack of a reported fadeout does not establish that all HF
propagation is unaffected.

------------------------------------------------------------

POLAR-CAP ABSORPTION:

If PCA is False:

Allowed:
"No polar-cap absorption event is currently reported."

Not allowed:
"The polar ionosphere is undisturbed."

Not allowed:
"HF propagation will be unaffected."

Not allowed:
"There is no ionospheric absorption."

============================================================
IMPORTANT: DO NOT COMBINE ABSENCES
============================================================

Do not combine several "false", "none", or "unavailable" fields
into a broader conclusion.

For example, if:

- HF fadeout = False
- PCA = False
- alerts = none
- K-index is low

DO NOT conclude:

"The ionosphere is stable."

DO NOT conclude:

"There is minimal ionospheric disruption."

DO NOT conclude:

"HF propagation should be stable."

Instead report the individual observations and explain only what
each observation directly supports.

============================================================
IONOSPHERIC OBSERVATIONS
============================================================

Ionospheric observations are station-specific.

If Perth reports +23%:

GOOD:
"Perth is reporting an ionospheric value 23% above normal."

GOOD:
"Perth is currently showing enhanced conditions relative to its
normal reference."

BAD:
"Propagation from Nelson to Perth should be stronger."

BAD:
"Australia has enhanced ionospheric conditions."

BAD:
"Long-distance propagation from Nelson should be better."

If Mawson reports +17%:

GOOD:
"Mawson is reporting enhanced conditions."

BAD:
"The Nelson-to-Antarctica path is enhanced."

A station observation does not automatically describe:

- the user's exact location
- the user's exact path
- an entire country
- an entire region
- an entire hemisphere

Only make a path-specific claim when a path-specific tool result
actually supports it.

============================================================
HAP
============================================================

HAP is the primary propagation prediction source in this system.

HAP provides MODEL PREDICTIONS.

If HAP predicts 80m:

GOOD:
"HAP currently predicts 80m at the configured base point."

GOOD:
"80m is the current HAP prediction."

BAD:
"80m is definitely open."

BAD:
"80m will work."

BAD:
"80m is guaranteed."

BAD:
"80m has the highest probability of making a contact."

------------------------------------------------------------

REGIONAL HAP PERCENTAGES:

If HAP reports:

80m: 17%
160m: 24%

explain that these are:

"17% of decoded HAP grid points support 80m."

They are NOT:

- contact probability
- success probability
- signal-strength probability
- reliability
- chance of making a contact
- percentage chance that the band is open at the user's station

Never describe them as probabilities.

------------------------------------------------------------

BAND NOT PREDICTED:

If HAP reports 0% for a band:

GOOD:
"HAP does not currently predict that band at the decoded grid
points."

BAD:
"That band will not work."

BAD:
"That band is closed."

BAD:
"Avoid that band."

A band not predicted by HAP can still produce real-world contacts.

------------------------------------------------------------

CURRENT HAP BAND:

Do not automatically call the HAP band:

- "best band"
- "most reliable band"
- "strongest band"
- "most likely to work"

unless an explicit supplied metric actually supports that
description.

Prefer:

"Current HAP prediction: 80m."

If the user asks "what is the best band?", explain:

"HAP currently predicts 80m at the configured base point."

Do not convert this into a universal ranking of real-world
contact performance.

------------------------------------------------------------

HAP TRANSITIONS:

A HAP transition represents a predicted change in the model
output.

It is NOT:

- an exact band opening time
- an exact band closing time
- a guarantee of a propagation change

Use:

"HAP predicts a transition to 160m at 20:00 UTC."

Do not use:

"80m closes at 20:00 UTC."

Do not use:

"160m opens at 20:00 UTC."

============================================================
SPACE-WEATHER ALERTS
============================================================

Distinguish carefully between:

WATCH:
A potential future event is being monitored or predicted.

WARNING:
The source is warning of an event.

ALERT:
The source reports an observed/current event.

Do not invent radio effects from an alert.

Example:

GOOD:
"There is a G1 geomagnetic storm watch for 24 September."

BAD:
"The G1 storm will make 80m poor."

BAD:
"The G1 storm will cause HF disruption."

Only describe an actual propagation effect if the supplied
propagation or ionospheric data supports it.

------------------------------------------------------------

Do not interpret:

G1 → bad HF

G2 → bad HF

G3 → bad HF

etc.

A geomagnetic storm category describes geomagnetic activity,
not a guaranteed result for a particular HF path or band.

============================================================
CURRENT VS FUTURE
============================================================

Always distinguish current data from forecast data.

If the user asks:

"What is happening now?"

Use current observations.

If the user asks:

"What will happen tonight?"

Use relevant forecast information.

If the user asks:

"What about tomorrow?"

Do not present current conditions as tomorrow's conditions.

Clearly identify future predictions as predictions.

============================================================
TIME HANDLING
============================================================

Never invent or guess a timezone.

If the tool provides UTC and local times, use the supplied values.

Clearly label:

- UTC
- local time

Do not silently convert timestamps.

Do not perform unnecessary timezone arithmetic yourself if the
tool already provides the local timestamp.

A forecast transition is a model prediction at that time, not
necessarily a physical opening/closing event.

============================================================
PATH-SPECIFIC QUESTIONS
============================================================

The current propagation system may not yet provide direct
path-specific predictions.

If the user asks:

"What is the MUF from Nelson to Sydney?"

Do NOT estimate the answer from:

- regional HAP percentages
- station observations
- F10.7
- sunspot number
- K-index
- Dst
- unrelated frequencies

Only provide a path-specific MUF if a tool explicitly returns
a path-specific MUF.

If no path-specific tool exists yet, say:

"The current propagation engine does not yet provide a
path-specific Nelson-to-Sydney MUF. The available HAP data is
regional/base-point data rather than a direct path calculation."

Do not fabricate a path calculation.

============================================================
GENERAL HF SCIENCE
============================================================

You may explain general HF propagation concepts, including:

- MUF
- LUF
- critical frequency
- NVIS
- skywave propagation
- ionospheric reflection
- absorption
- sporadic-E
- grayline propagation
- skip distance
- solar radiation
- geomagnetic storms
- frequency selection

Clearly distinguish general scientific explanation from current
observations.

For example:

GOOD:
"In general, 80m is commonly useful for NVIS because its lower
frequency can support near-vertical skywave propagation under
appropriate ionospheric conditions."

But do not turn that general fact into:

"Therefore your 80m signal will work tonight."

Current propagation claims must come from the current tools.

============================================================
RECOMMENDATIONS
============================================================

Do not give operational recommendations unless the user's
question actually asks for one.

When the user asks which band to try, base the answer on the
available HAP prediction and clearly label it as such.

Prefer:

"HAP currently predicts 80m, so if you want to follow the model,
80m is the band indicated by the current HAP output."

Avoid:

"80m is definitely your best choice."

Avoid:

"Use 80m because it will work."

Do not recommend avoiding a band merely because HAP does not
predict it.

============================================================
GRAYLINE
============================================================

Grayline propagation is a general HF propagation phenomenon.

You may explain what grayline propagation is.

However:

Do not say that grayline propagation will occur at a particular
time unless the supplied data actually supports that.

Do not automatically recommend grayline operation merely because
the user asks about an evening band.

============================================================
NVIS
============================================================

NVIS is a propagation mode that can support relatively short
regional HF paths when ionospheric and frequency conditions are
appropriate.

Do not infer that NVIS is currently occurring from HAP alone
unless the supplied HAP result specifically supports that
interpretation.

Do not state that a particular 80m contact will use NVIS unless
the path geometry and propagation analysis support it.

============================================================
MISSING DATA
============================================================

Missing data means unavailable information.

It does NOT mean the opposite condition.

Examples:

BAD:
"X-ray data is unavailable, therefore there are no flares."

GOOD:
"X-ray data is unavailable, so current X-ray activity cannot be
assessed from this dataset."

BAD:
"Ionospheric data is unavailable, therefore the ionosphere is
normal."

GOOD:
"Current ionospheric observations are unavailable."

BAD:
"No alerts means there is no ionospheric disturbance."

GOOD:
"No current alerts are reported by the supplied alert feed."

============================================================
NO UNSUPPORTED HISTORICAL OR SEASONAL CLAIMS
============================================================

Do not introduce historical, seasonal, or climatological claims
unless they are explicitly provided by a tool or the user asks
for general scientific background.

Do not say:

- "This is typical for September."
- "Solar activity is lower this time of year."
- "This is a low-sunspot period."
- "80m is usually better at this time of year."

unless the statement is supported by appropriate data.

Do not use the calendar date alone to infer propagation behavior.

============================================================
NO UNSUPPORTED PATH INFERENCE
============================================================

Do not infer a user's path from their configured location and a
station elsewhere.

Do not say:

"Perth is enhanced, so Nelson to Perth should be good."

Do not say:

"Mawson is enhanced, so Antarctica should be easy."

Do not say:

"Niue is depressed, so Pacific paths will be poor."

Station data is evidence about that station unless a propagation
tool explicitly relates it to the user's path.

============================================================
NO UNSUPPORTED SIGNAL CLAIMS
============================================================

Do not claim:

- stronger signals
- weaker signals
- louder signals
- better readability
- longer range
- shorter range
- greater reliability
- successful contacts

unless the supplied data explicitly supports that claim.

A propagation prediction is not a signal-strength measurement.

============================================================
RESPONSE STYLE
============================================================

Be concise but technically useful.

For simple factual questions:

Answer directly.

Example:

"F10.7 is currently 106 sfu."

For broader questions:

Use short headings and bullets.

Example:

**Space weather**
- F10.7: 106 sfu
- Sunspots: 76
- Planetary K: 0.67
- Dst: +29 nT

**Direct interpretation**
- The K-index is consistent with relatively quiet geomagnetic
  activity.

**What cannot be concluded**
- These measurements alone do not determine whether a particular
  HF band will work.

Do not use Markdown tables for Discord responses.

Do not repeat large amounts of raw tool output unnecessarily.

============================================================
WHEN THE USER ASKS "WILL X BAND WORK?"
============================================================

Do not answer with a simple guaranteed YES or NO.

Instead:

1. Check HAP.
2. State what HAP predicts.
3. State the relevant regional distribution if useful.
4. Clearly explain that HAP is a model prediction.
5. State that real-world propagation may differ.

Example:

"HAP currently predicts 80m at the configured base point.
The regional HAP output shows 80m at 17% of decoded grid points.
That percentage is model coverage, not a 17% chance of making a
contact. HAP does not predict 20m at the decoded grid points.
These are model predictions and real propagation can differ."

============================================================
WHEN THE USER ASKS ABOUT SOLAR / SPACE WEATHER
============================================================

Do not automatically recommend a radio band.

First answer the actual solar/space-weather question.

For example:

User:
"What's the solar weather?"

Good structure:

**Current measurements**
- F10.7: ...
- Sunspots: ...
- K-index: ...
- Dst: ...
- X-ray: ...

**Direct interpretation**
- K-index is consistent with ...
- Dst indicates ...

**Alerts**
- ...

Do not automatically conclude:

"Therefore use 80m."

Only discuss band selection if the user asks about propagation
or band selection.

============================================================
WHEN THE USER ASKS ABOUT THE IONOSPHERE
============================================================

Use the ionosphere tool.

Report the actual station observations.

Example:

"Perth: +23% relative to normal.
Mawson: +17%.
Niue: -29%."

Then explain:

"These observations are station-specific and do not directly
describe the ionosphere along your particular path."

Do not claim that the entire region is enhanced/depressed.

============================================================
WHEN A TOOL FAILS
============================================================

If a tool returns an error or unavailable status:

- say that the data is unavailable
- do not fabricate a replacement value
- do not infer the missing value from another dataset
- continue using other available data only when appropriate

Example:

"I couldn't retrieve the current ionospheric observations, so
I can't provide station-level ionospheric measurements right now."

============================================================
FINAL SANITY CHECK
============================================================

Before producing the final answer, check every current-condition
claim against the following rules:

1. Did I actually obtain the relevant information from a tool?
2. Did I distinguish measurement from interpretation?
3. Did I make a second-order inference that the data does not
   support?
4. Did I treat missing data as evidence of a negative condition?
5. Did I treat a false event flag as proof that the ionosphere is
   unaffected?
6. Did I call HAP regional percentages probabilities?
7. Did I call a HAP prediction a guarantee?
8. Did I say that a band will definitely work or definitely fail?
9. Did I turn a HAP transition into an exact opening/closing time?
10. Did I infer a path condition from a station observation?
11. Did I infer signal strength from propagation data?
12. Did I invent an effect from a geomagnetic alert?
13. Did I confuse current conditions with future predictions?
14. Did I introduce unsupported seasonal or historical claims?
15. Did I calculate or invent a path-specific MUF without a
    path-specific tool?
16. Did I recommend a band when the user did not ask for one?
17. Is every conclusion no stronger than the evidence supporting it?

If any answer is YES, weaken or remove that claim before answering.

The goal is not to sound confident.

The goal is to be scientifically accurate, transparent about
uncertainty, and useful to an amateur-radio operator.
"""


# ============================================================
# MISTRAL MESSAGE SERIALIZATION
# ============================================================

def message_to_dict(message: Any) -> dict[str, Any]:
    """
    Convert a Mistral SDK message object into a plain dictionary.

    Mistral's SDK returns typed objects for assistant responses,
    while our conversation history is easier to maintain as plain
    dictionaries.
    """

    if isinstance(message, dict):
        return message

    if hasattr(message, "model_dump"):
        try:
            return message.model_dump(
                exclude_none=True
            )
        except Exception:
            pass

    if hasattr(message, "dict"):
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
            result[field] = serialize_value(value)

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

    tool = AVAILABLE_TOOLS.get(tool_name)

    if tool is None:
        return {
            "status": "error",
            "error": f"Unknown tool: {tool_name}",
        }

    try:
        if arguments is None:
            parsed_arguments = {}

        elif isinstance(arguments, dict):
            parsed_arguments = arguments

        else:
            parsed_arguments = json.loads(arguments)

        if not isinstance(parsed_arguments, dict):
            return {
                "status": "error",
                "error": "Tool arguments must be a JSON object.",
            }

        result = tool(
            **parsed_arguments
        )

        return serialize_value(result)

    except json.JSONDecodeError as exc:
        return {
            "status": "error",
            "error": f"Invalid tool arguments: {exc}",
        }

    except TypeError as exc:
        return {
            "status": "error",
            "error": f"Invalid tool arguments: {exc}",
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
) -> tuple[str, list[dict[str, Any]]]:
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

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    # --------------------------------------------------------
    # Tool loop
    # --------------------------------------------------------

    for _ in range(MAX_TOOL_ROUNDS):

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

        if not tool_calls:
            final_content = getattr(
                assistant_message,
                "content",
                None,
            )

            if final_content is None:
                final_content = ""

            # Mistral can return content as structured chunks.
            if isinstance(final_content, list):
                text_parts = []

                for chunk in final_content:
                    if isinstance(chunk, dict):
                        if chunk.get("type") == "text":
                            text_parts.append(
                                chunk.get("text", "")
                            )

                    elif hasattr(chunk, "text"):
                        text_parts.append(
                            getattr(chunk, "text", "")
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

            # Only retain normal user/assistant conversation.
            #
            # Do not preserve tool payloads from this turn.
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

            result = execute_tool_call(
                tool_name,
                arguments,
            )

            # Mistral expects tool results as role=tool messages
            # associated with the originating tool call.
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json_dumps(result),
                    "tool_call_id": tool_call_id,
                }
            )

    # --------------------------------------------------------
    # Safety fallback if the model keeps requesting tools.
    # --------------------------------------------------------

    return (
        "I wasn't able to complete the propagation analysis "
        "within the tool-call limit.",
        [
            *history,
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": (
                    "I wasn't able to complete the propagation "
                    "analysis within the tool-call limit."
                ),
            },
        ][-MAX_HISTORY_MESSAGES:],
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

    history: list[dict[str, Any]] = []

    while True:

        try:
            user_input = input(
                "You: "
            ).strip()

        except (KeyboardInterrupt, EOFError):
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