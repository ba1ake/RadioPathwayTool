"""
Space Weather Data Collector
============================

V0.3

Purpose:
    Collect the space-weather information needed by the HF propagation
    analysis engine.

Primary geographic focus:
    New Zealand / Australia / South Pacific

Data sources:

    NOAA SWPC
        - F10.7 solar flux
        - planetary K index
        - Dst
        - space-weather alerts

    Australian Space Weather Services (SWS)
        - Australian-region K index
        - Australian-region A index
        - Australian-region Dst
        - magnetic alerts/warnings
        - aurora alerts/watches/outlooks
        - real-time T indices from the public SWS website

SWS API functions require an API key.

Set the environment variable:

    SWS_API_KEY

The public SWS real-time T-index webpage does NOT require
the SWS API key.

NOAA's public JSON products do not currently require an API key.

This module deliberately separates DATA COLLECTION from the propagation
analysis engine.

The output can later be passed into band_favorability.py.
"""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import (
    dataclass,
    asdict,
    field,
)
from datetime import (
    datetime,
    timezone,
    timedelta,
)
from typing import (
    Any,
    Optional,
)

import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

REQUEST_TIMEOUT = 15

SWS_API_BASE = (
    "https://sws-data.sws.bom.gov.au/api/v1"
)

# Public SWS real-time T-index webpage.
#
# This does NOT require an SWS API key.
SWS_REALTIME_T_INDEX_URL = (
    "https://www.sws.bom.gov.au/HF_Systems/6/4/2"
)

NOAA_BASE = (
    "https://services.swpc.noaa.gov"
)

# Alerts which do not contain an explicit expiry time are retained for this
# long after their issue time.
#
# This prevents old historical alerts from appearing indefinitely while still
# allowing recent watches/alerts to remain visible.
ALERT_FALLBACK_MAX_AGE_HOURS = 48


# ============================================================================
# NOAA ENDPOINTS
# ============================================================================

NOAA_ENDPOINTS = {

    # Solar radio flux / F10.7
    "solar_radio_flux":
        f"{NOAA_BASE}/products/solar-radio-flux.json",

    # Planetary K index
    "planetary_k":
        f"{NOAA_BASE}/products/noaa-planetary-k-index.json",

    # Dst
    "dst":
        f"{NOAA_BASE}/products/kyoto-dst.json",

    # General alerts
    "alerts":
        f"{NOAA_BASE}/products/alerts.json",

    # F10.7 historical data
    "f107":
        f"{NOAA_BASE}/json/f107_cm_flux.json",

    # Sunspot / solar-cycle observations
    "sunspots":
        f"{NOAA_BASE}/json/solar-cycle/observed-solar-cycle-indices.json",

    # Predicted F10.7
    "predicted_f107":
        f"{NOAA_BASE}/json/predicted_f107cm_flux.json",

    # Predicted sunspot number
    "predicted_sunspots":
        f"{NOAA_BASE}/json/predicted_monthly_sunspot_number.json",

    # GloTEC
    "glotec":
        f"{NOAA_BASE}/products/glotec/geojson_2d_urt.json",
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SpaceWeatherData:
    """
    Normalised space-weather dataset.

    This is the object that should eventually be passed into the propagation
    analysis engine.

    None means the value was unavailable.

    t_index:
        Default real-time T-index selected for general use.

    t_index_source:
        Human-readable description of where the default T-index came from.

    t_indices:
        Dictionary containing all real-time T-index values successfully
        retrieved from the SWS public webpage.

    t_index_retrieved_utc:
        UTC time at which the SWS T-index webpage was retrieved.

    t_index_updated_utc:
        UTC time reported by SWS for the actual T-index data update.
    """

    # ------------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------------

    retrieved_utc: Optional[str] = None

    # ------------------------------------------------------------------------
    # Solar activity
    # ------------------------------------------------------------------------

    solar_flux_10_7: Optional[float] = None
    sunspot_number: Optional[float] = None

    # ------------------------------------------------------------------------
    # Geomagnetic activity
    # ------------------------------------------------------------------------

    planetary_k_index: Optional[float] = None
    australian_k_index: Optional[float] = None
    a_index: Optional[float] = None
    dst_index: Optional[float] = None

    # ------------------------------------------------------------------------
    # Solar / HF disturbance
    # ------------------------------------------------------------------------

    xray_flux: Optional[float] = None
    hf_fadeout: bool = False
    polar_cap_absorption: bool = False

    # ------------------------------------------------------------------------
    # Real-time SWS T indices
    # ------------------------------------------------------------------------

    # Default T-index selected for general propagation use.
    t_index: Optional[float] = None

    # Human-readable description of where the selected value came from.
    t_index_source: Optional[str] = None

    # All successfully retrieved regional T indices.
    t_indices: dict[str, float] = field(
        default_factory=dict
    )

    # Time at which our program retrieved the SWS page.
    t_index_retrieved_utc: Optional[str] = None

    # Time at which SWS says the T-index data was last updated.
    t_index_updated_utc: Optional[str] = None

    # ------------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------------

    active_alerts: list[dict[str, Any]] | None = None
    active_warnings: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------------
    # SWS / Australian information
    # ------------------------------------------------------------------------

    sws_available: bool = False
    sws_error: Optional[str] = None

    # ------------------------------------------------------------------------
    # Source status
    # ------------------------------------------------------------------------

    source_status: dict[str, str] | None = None

    # ------------------------------------------------------------------------
    # Raw/extended information
    #
    # Kept available so we don't throw away useful data before deciding how
    # it should be incorporated into the model.
    # ------------------------------------------------------------------------

    raw: dict[str, Any] | None = None


# ============================================================================
# HTTP HELPERS
# ============================================================================

def get_json(
    url: str,
    timeout: int = REQUEST_TIMEOUT,
) -> Any:
    """
    Retrieve JSON from a public endpoint.
    """

    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent":
                "HF-Propagation-Assistant/0.1"
        },
    )

    response.raise_for_status()

    return response.json()


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int = REQUEST_TIMEOUT,
) -> Any:
    """
    POST JSON to an API.
    """

    response = requests.post(
        url,
        json=payload,
        timeout=timeout,
        headers={
            "Content-Type": "application/json",
            "User-Agent":
                "HF-Propagation-Assistant/0.1",
        },
    )

    response.raise_for_status()

    return response.json()


# ============================================================================
# SWS PUBLIC WEBSITE: REAL-TIME T INDICES
# ============================================================================

def parse_sws_t_index_page(
    page_html: str,
) -> tuple[dict[str, float], Optional[str]]:
    """
    Parse real-time T-index values from the public SWS webpage.

    The SWS page may publish:

        Northern hemisphere T index
        Southern hemisphere T index
        Northern Equatorial Australian Region T index
        Northern Australian Region T index
        Southern Australian Region T index
        Australian Region T index
        Antarctic Region T index

    SWS uses 999 to indicate that no autoscaled data is available.

    Returns:

        (
            t_indices,
            updated_utc,
        )
    """

    # ------------------------------------------------------------------------
    # Convert HTML into searchable text.
    # ------------------------------------------------------------------------

    page_text = html.unescape(page_html)

    # Remove scripts.
    page_text = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        page_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove styles.
    page_text = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        page_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove remaining HTML tags.
    page_text = re.sub(
        r"<[^>]+>",
        " ",
        page_text,
    )

    # Normalize whitespace.
    page_text = re.sub(
        r"\s+",
        " ",
        page_text,
    ).strip()

    # ------------------------------------------------------------------------
    # Parse T-index values.
    # ------------------------------------------------------------------------

    number_pattern = r"(-?\d+(?:\.\d+)?)"

    patterns = {

        "northern_hemisphere": (
            r"Northern\s+hemisphere\s+T\s+index\s*:\s*"
            + number_pattern
        ),

        "southern_hemisphere": (
            r"Southern\s+hemisphere\s+T\s+index\s*:\s*"
            + number_pattern
        ),

        "northern_equatorial_australia": (
            r"Northern\s+Equatorial\s+Australian\s+Region"
            r"\s+T\s+index\s*:\s*"
            + number_pattern
        ),

        "northern_australia": (
            r"Northern\s+Australian\s+Region"
            r"\s+T\s+index\s*:\s*"
            + number_pattern
        ),

        "southern_australia": (
            r"Southern\s+Australian\s+Region"
            r"\s+T\s+index\s*:\s*"
            + number_pattern
        ),

        "australian_region": (
            r"Australian\s+Region\s+T\s+index\s*:\s*"
            + number_pattern
        ),

        "antarctic_region": (
            r"Antarctic\s+Region\s+T\s+index\s*:\s*"
            + number_pattern
        ),
    }

    indices: dict[str, float] = {}

    for name, pattern in patterns.items():

        match = re.search(
            pattern,
            page_text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = safe_float(
            match.group(1)
        )

        if value is None:
            continue

        # SWS uses 999 to indicate that no autoscaled data is available.
        if value == 999:
            continue

        indices[name] = value

    # ------------------------------------------------------------------------
    # Parse SWS's own "last updated" time.
    #
    # Current page format:
    #
    #     last updated 24 Sep 2026 04:40 UT
    #
    # ------------------------------------------------------------------------

    updated_utc: Optional[str] = None

    updated_match = re.search(
        r"last\s+updated\s+"
        r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\s+"
        r"\d{1,2}:\d{2})\s+UT",
        page_text,
        flags=re.IGNORECASE,
    )

    if updated_match:

        parsed = parse_datetime(
            updated_match.group(1)
            + " UTC"
        )

        if parsed is not None:
            updated_utc = parsed.isoformat()

    # ------------------------------------------------------------------------
    # Make sure we actually got something useful.
    # ------------------------------------------------------------------------

    if not indices:
        raise RuntimeError(
            "SWS real-time T-index page returned no usable T indices."
        )

    return (
        indices,
        updated_utc,
    )


def get_sws_realtime_t_indices() -> dict[str, Any]:
    """
    Retrieve real-time T indices from the public SWS webpage.

    Source:
        SWS Global HF - Real Time T Indices

    The webpage provides real-time values derived from several hours
    of autoscaled ionosonde data.

    A value of 999 means that no autoscaled data is available and is
    therefore ignored.

    Returns:

        {
            "indices": {...},
            "updated_utc": "...",
            "retrieved_utc": "...",
        }

    The function does not require SWS_API_KEY.
    """

    retrieved_utc = datetime.now(
        timezone.utc
    ).isoformat()

    response = requests.get(
        SWS_REALTIME_T_INDEX_URL,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent":
                "HF-Propagation-Assistant/0.1"
        },
    )

    response.raise_for_status()

    (
        indices,
        updated_utc,
    ) = parse_sws_t_index_page(
        response.text
    )

    return {
        "indices": indices,
        "updated_utc": updated_utc,
        "retrieved_utc": retrieved_utc,
    }


def select_default_t_index(
    t_indices: dict[str, float],
) -> tuple[Optional[float], Optional[str]]:
    """
    Select the default T-index for the propagation system.

    Primary choice:
        Southern Hemisphere

    This is appropriate for the project's primary NZ/South Pacific
    operating region.

    Fallback:
        Australian Region
        Southern Australia
        Northern Australia
        Northern Hemisphere
        Antarctic Region

    This function only chooses a default value.

    Path-specific selection should be performed by main.py using the
    complete t_indices dictionary.
    """

    preferred = (

        (
            "southern_hemisphere",
            "SWS real-time Southern Hemisphere T index",
        ),

        (
            "australian_region",
            "SWS real-time Australian Region T index",
        ),

        (
            "southern_australia",
            "SWS real-time Southern Australian Region T index",
        ),

        (
            "northern_australia",
            "SWS real-time Northern Australian Region T index",
        ),

        (
            "northern_hemisphere",
            "SWS real-time Northern Hemisphere T index",
        ),

        (
            "antarctic_region",
            "SWS real-time Antarctic Region T index",
        ),
    )

    for key, source in preferred:

        value = t_indices.get(key)

        if value is not None:
            return (
                float(value),
                source,
            )

    return (
        None,
        None,
    )


# ============================================================================
# GENERIC VALUE HELPERS
# ============================================================================

def safe_float(
    value: Any,
) -> Optional[float]:
    """
    Convert a value to float when possible.
    """

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def latest_record(
    data: Any,
) -> Optional[dict[str, Any]]:
    """
    Try to find the most recent dictionary record in a JSON response.

    NOAA products aren't all formatted identically, so this intentionally
    handles several common structures.
    """

    if isinstance(data, list):

        records = [
            item
            for item in data
            if isinstance(item, dict)
        ]

        if not records:
            return None

        return records[-1]

    if isinstance(data, dict):

        # Common nested data field.
        if isinstance(
            data.get("data"),
            list,
        ):

            records = [
                item
                for item in data["data"]
                if isinstance(item, dict)
            ]

            if records:
                return records[-1]

        return data

    return None


# ============================================================================
# ALERT DATE/TIME HELPERS
# ============================================================================

def parse_datetime(
    value: Any,
) -> Optional[datetime]:
    """
    Try to convert a variety of timestamp formats into a timezone-aware
    UTC datetime.

    Handles:
        ISO 8601
        timestamps ending in Z
        timestamps with offsets
        common NOAA/SWS textual timestamps
    """

    if value is None:
        return None

    if isinstance(value, datetime):

        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )

    if not isinstance(
        value,
        str,
    ):
        return None

    text = value.strip()

    if not text:
        return None

    # Normalise common UTC notation.
    text = text.replace(
        " UTC",
        "+00:00",
    )

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    # First try Python's ISO parser.
    try:

        parsed = datetime.fromisoformat(
            text
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        )

    except ValueError:
        pass

    # Common NOAA/SWS formats.
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y %b %d %H:%M",
        "%Y %b %d %H:%M:%S",
        "%Y %B %d %H:%M",
        "%Y %B %d %H:%M:%S",
        "%d %b %Y %H:%M",
        "%d %b %Y %H:%M:%S",
    )

    for fmt in formats:

        try:

            parsed = datetime.strptime(
                text,
                fmt,
            )

            return parsed.replace(
                tzinfo=timezone.utc
            )

        except ValueError:
            continue

    return None


def extract_alert_issue_time(
    alert: dict[str, Any],
) -> Optional[datetime]:
    """
    Find the issue/publication time in an alert.

    Different NOAA/SWS products use different field names.
    """

    possible_fields = (
        "issue_datetime",
        "issue_time",
        "issued",
        "issued_datetime",
        "issueDateTime",
        "issueDate",
        "publication_time",
        "published",
        "created",
        "created_at",
        "timestamp",
        "datetime",
        "date_time",
    )

    for field_name in possible_fields:

        if field_name not in alert:
            continue

        parsed = parse_datetime(
            alert.get(field_name)
        )

        if parsed is not None:
            return parsed

    return None


def extract_alert_expiry_time(
    alert: dict[str, Any],
) -> Optional[datetime]:
    """
    Find an explicit expiry/valid-until time.

    Checks structured fields first, then searches the alert message for
    phrases such as:

        Valid Until: 2026 Sep 16 1800 UTC
        Valid until 2026 Sep 16 1800 UTC
        Valid To: ...
        Expires: ...
    """

    # ------------------------------------------------------------------------
    # Structured API fields
    # ------------------------------------------------------------------------

    possible_fields = (
        "valid_until",
        "valid_until_datetime",
        "valid_to",
        "valid_to_datetime",
        "validTo",
        "validUntil",
        "expiry",
        "expiry_time",
        "expiry_datetime",
        "expires",
        "expires_at",
        "expiration",
        "expiration_time",
        "end_time",
        "end_datetime",
        "end",
    )

    for field_name in possible_fields:

        if field_name not in alert:
            continue

        parsed = parse_datetime(
            alert.get(field_name)
        )

        if parsed is not None:
            return parsed

    # ------------------------------------------------------------------------
    # Search message/text fields
    # ------------------------------------------------------------------------

    text_fields = (
        "message",
        "text",
        "description",
        "body",
        "content",
        "summary",
    )

    combined_text = " ".join(
        str(
            alert.get(
                field_name,
                "",
            )
        )
        for field_name in text_fields
    )

    if not combined_text:
        return None

    patterns = (

        r"valid\s+until\s*:?\s*"
        r"(\d{4}\s+[A-Za-z]{3,9}\s+\d{1,2}\s+\d{3,4}\s*UTC)",

        r"valid\s+to\s*:?\s*"
        r"(\d{4}\s+[A-Za-z]{3,9}\s+\d{1,2}\s+\d{3,4}\s*UTC)",

        r"expires?\s*:?\s*"
        r"(\d{4}\s+[A-Za-z]{3,9}\s+\d{1,2}\s+\d{3,4}\s*UTC)",
    )

    for pattern in patterns:

        match = re.search(
            pattern,
            combined_text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        date_text = match.group(1)

        date_match = re.match(
            r"(\d{4})\s+([A-Za-z]{3,9})\s+"
            r"(\d{1,2})\s+(\d{3,4})\s*UTC",
            date_text,
            flags=re.IGNORECASE,
        )

        if not date_match:
            continue

        year = date_match.group(1)
        month = date_match.group(2)
        day = date_match.group(3)
        time_part = date_match.group(4)

        # Convert 1800 -> 18:00 and 900 -> 09:00.
        if len(time_part) == 3:
            time_part = (
                "0"
                + time_part
            )

        normalised = (
            f"{year} {month} {day} "
            f"{time_part[:2]}:{time_part[2:]} UTC"
        )

        parsed = parse_datetime(
            normalised
        )

        if parsed is not None:
            return parsed

    return None


def is_alert_current(
    alert: dict[str, Any],
    now_utc: Optional[datetime] = None,
    max_age_hours: int = ALERT_FALLBACK_MAX_AGE_HOURS,
) -> bool:
    """
    Determine whether an alert should be exposed as current.

    Rules:

    1. If an explicit expiry time exists:
         keep only if it has not expired.

    2. If there is no expiry time but an issue time exists:
         keep it for max_age_hours.

    3. If neither expiry nor issue time can be determined:
         keep it rather than silently discarding potentially important
         information.
    """

    if now_utc is None:
        now_utc = datetime.now(
            timezone.utc
        )

    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(
            tzinfo=timezone.utc
        )
    else:
        now_utc = now_utc.astimezone(
            timezone.utc
        )

    # ------------------------------------------------------------------------
    # Explicit expiry
    # ------------------------------------------------------------------------

    expiry = extract_alert_expiry_time(
        alert
    )

    if expiry is not None:
        return expiry >= now_utc

    # ------------------------------------------------------------------------
    # No explicit expiry: use issue age
    # ------------------------------------------------------------------------

    issue_time = extract_alert_issue_time(
        alert
    )

    if issue_time is not None:

        age = (
            now_utc
            - issue_time
        )

        # Future-dated records are retained.
        if age.total_seconds() < 0:
            return True

        return age <= timedelta(
            hours=max_age_hours
        )

    # ------------------------------------------------------------------------
    # Unknown format
    # ------------------------------------------------------------------------

    return True


def filter_current_alerts(
    alerts: Any,
    now_utc: Optional[datetime] = None,
    max_age_hours: int = ALERT_FALLBACK_MAX_AGE_HOURS,
) -> list[dict[str, Any]]:
    """
    Filter a collection of NOAA/SWS alerts down to currently relevant
    records.
    """

    if not isinstance(
        alerts,
        list,
    ):
        return []

    filtered = []

    for alert in alerts:

        if not isinstance(
            alert,
            dict,
        ):
            continue

        if is_alert_current(
            alert,
            now_utc=now_utc,
            max_age_hours=max_age_hours,
        ):
            filtered.append(
                alert
            )

    return filtered


# ============================================================================
# NOAA: SOLAR RADIO FLUX
# ============================================================================

def get_noaa_solar_flux() -> Optional[float]:
    """
    Retrieve the latest available F10.7 solar radio flux.

    Returns:
        Solar flux in solar flux units (sfu)
    """

    try:

        data = get_json(
            NOAA_ENDPOINTS[
                "solar_radio_flux"
            ]
        )

        record = latest_record(
            data
        )

        if not record:
            return None

        for key in (
            "flux",
            "f10.7",
            "f107",
            "flux_10_7",
        ):

            if key in record:

                value = safe_float(
                    record[key]
                )

                if value is not None:
                    return value

        return None

    except Exception:
        return None


# ============================================================================
# NOAA: F10.7 FALLBACK
# ============================================================================

def get_noaa_f107_fallback() -> Optional[float]:
    """
    Fallback F10.7 source.
    """

    try:

        data = get_json(
            NOAA_ENDPOINTS["f107"]
        )

        record = latest_record(
            data
        )

        if not record:
            return None

        for key in (
            "flux",
            "f10.7",
            "f107",
            "f107_adj",
            "observed",
        ):

            if key in record:

                value = safe_float(
                    record[key]
                )

                if value is not None:
                    return value

        return None

    except Exception:
        return None


# ============================================================================
# NOAA: PLANETARY K
# ============================================================================

def get_noaa_planetary_k() -> Optional[float]:
    """
    Retrieve the latest NOAA planetary K index.
    """

    try:

        data = get_json(
            NOAA_ENDPOINTS["planetary_k"]
        )

        record = latest_record(
            data
        )

        if not record:
            return None

        for key in (
            "kp",
            "Kp",
            "k",
            "index",
        ):

            if key in record:

                value = safe_float(
                    record[key]
                )

                if value is not None:
                    return value

        return None

    except Exception:
        return None


# ============================================================================
# NOAA: DST
# ============================================================================

def get_noaa_dst() -> Optional[float]:
    """
    Retrieve the latest available Dst value.
    """

    try:

        data = get_json(
            NOAA_ENDPOINTS["dst"]
        )

        record = latest_record(
            data
        )

        if not record:
            return None

        for key in (
            "dst",
            "Dst",
            "value",
            "index",
        ):

            if key in record:

                value = safe_float(
                    record[key]
                )

                if value is not None:
                    return value

        return None

    except Exception:
        return None


# ============================================================================
# NOAA: SUNSPOTS
# ============================================================================

def get_noaa_sunspot_number() -> Optional[float]:
    """
    Retrieve the latest available sunspot number.
    """

    try:

        data = get_json(
            NOAA_ENDPOINTS["sunspots"]
        )

        record = latest_record(
            data
        )

        if not record:
            return None

        for key in (
            "ssn",
            "sunspot_number",
            "sunspots",
            "sunspot",
        ):

            if key in record:

                value = safe_float(
                    record[key]
                )

                if value is not None:
                    return value

        return None

    except Exception:
        return None


# ============================================================================
# NOAA: ALERTS
# ============================================================================

def get_noaa_alerts() -> list[dict[str, Any]]:
    """
    Retrieve NOAA space-weather alerts and filter them to currently relevant
    records.
    """

    try:

        data = get_json(
            NOAA_ENDPOINTS["alerts"]
        )

        if isinstance(
            data,
            list,
        ):

            return filter_current_alerts(
                data
            )

        return []

    except Exception:
        return []


# ============================================================================
# SWS API
# ============================================================================

def get_sws_api_key() -> Optional[str]:
    """
    Read the SWS API key from an environment variable.
    """

    return os.getenv(
        "SWS_API_KEY"
    )


def sws_request(
    endpoint: str,
    location: Optional[str] = None,
) -> Any:
    """
    Make a request to the Australian Space Weather Services API.
    """

    api_key = get_sws_api_key()

    if not api_key:
        raise RuntimeError(
            "SWS_API_KEY environment variable is not configured."
        )

    payload = {
        "api_key": api_key,
    }

    if location is not None:
        payload["options"] = {
            "location": location,
        }

    return post_json(
        f"{SWS_API_BASE}/{endpoint}",
        payload,
    )


# ============================================================================
# SWS: K INDEX
# ============================================================================

def get_sws_k_index(
    location: str = "Australian region",
) -> Optional[float]:
    """
    Retrieve the latest SWS K index.
    """

    try:

        response = sws_request(
            "get-k-index",
            location,
        )

        record = latest_record(
            response
        )

        if not record:
            return None

        return safe_float(
            record.get("index")
        )

    except Exception:
        return None


# ============================================================================
# SWS: A INDEX
# ============================================================================

def get_sws_a_index() -> Optional[float]:
    """
    Retrieve the Australian-region A index.
    """

    try:

        response = sws_request(
            "get-a-index",
            "Australian region",
        )

        record = latest_record(
            response
        )

        if not record:
            return None

        return safe_float(
            record.get("index")
        )

    except Exception:
        return None


# ============================================================================
# SWS: DST
# ============================================================================

def get_sws_dst() -> Optional[float]:
    """
    Retrieve the Australian-region Dst index.
    """

    try:

        response = sws_request(
            "get-dst-index",
            "Australian region",
        )

        record = latest_record(
            response
        )

        if not record:
            return None

        return safe_float(
            record.get("index")
        )

    except Exception:
        return None


# ============================================================================
# SWS: MAGNETIC ALERT
# ============================================================================

def get_sws_magnetic_alert() -> list[dict[str, Any]]:
    """
    Retrieve current Australian-region magnetic alerts.
    """

    try:

        response = sws_request(
            "get-mag-alert"
        )

        data = response.get(
            "data",
            [],
        )

        if isinstance(
            data,
            list,
        ):

            return filter_current_alerts(
                data
            )

        return []

    except Exception:
        return []


# ============================================================================
# SWS: MAGNETIC WARNING
# ============================================================================

def get_sws_magnetic_warning() -> list[dict[str, Any]]:
    """
    Retrieve current Australian-region magnetic warnings.
    """

    try:

        response = sws_request(
            "get-mag-warning"
        )

        data = response.get(
            "data",
            [],
        )

        if isinstance(
            data,
            list,
        ):

            return filter_current_alerts(
                data
            )

        return []

    except Exception:
        return []


# ============================================================================
# SWS: AURORA WATCH
# ============================================================================

def get_sws_aurora_watch() -> list[dict[str, Any]]:
    """
    Retrieve any current Australian-region aurora watch.
    """

    try:

        response = sws_request(
            "get-aurora-watch"
        )

        data = response.get(
            "data",
            [],
        )

        if isinstance(
            data,
            list,
        ):

            return filter_current_alerts(
                data
            )

        return []

    except Exception:
        return []


# ============================================================================
# SWS: AURORA OUTLOOK
# ============================================================================

def get_sws_aurora_outlook() -> list[dict[str, Any]]:
    """
    Retrieve the current Australian-region aurora outlook.
    """

    try:

        response = sws_request(
            "get-aurora-outlook"
        )

        data = response.get(
            "data",
            [],
        )

        if isinstance(
            data,
            list,
        ):

            return filter_current_alerts(
                data
            )

        return []

    except Exception:
        return []


# ============================================================================
# MAIN COLLECTION FUNCTION
# ============================================================================

def collect_space_weather() -> SpaceWeatherData:
    """
    Collect all currently supported space-weather information.

    The collector is deliberately fault tolerant.

    If one service fails, the remaining services are still used.

    The public SWS T-index webpage is queried independently of the SWS API
    key. This means the T-index can still be available even if the SWS API
    credentials are not configured.
    """

    retrieved_dt = datetime.now(
        timezone.utc
    )

    retrieved = retrieved_dt.isoformat()

    status: dict[str, str] = {}
    raw: dict[str, Any] = {}

    # ------------------------------------------------------------------------
    # SWS REAL-TIME T INDICES
    #
    # This is deliberately performed independently of SWS_API_KEY.
    # The public SWS webpage does not require the authenticated API.
    # ------------------------------------------------------------------------

    try:

        t_index_result = (
            get_sws_realtime_t_indices()
        )

        t_indices = t_index_result.get(
            "indices",
            {},
        )

        t_index_updated_utc = (
            t_index_result.get(
                "updated_utc"
            )
        )

        t_index_retrieved_utc = (
            t_index_result.get(
                "retrieved_utc"
            )
        )

        (
            t_index,
            t_index_source,
        ) = select_default_t_index(
            t_indices
        )

        if t_index is not None:
            status["SWS T index"] = "OK"
        else:
            status["SWS T index"] = "FAILED"

    except Exception as exc:

        t_indices = {}
        t_index = None
        t_index_source = None
        t_index_updated_utc = None

        t_index_retrieved_utc = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        status["SWS T index"] = "FAILED"

        raw["SWS T index error"] = str(
            exc
        )

    raw["SWS real-time T indices"] = (
        t_indices
    )

    raw["SWS T index updated UTC"] = (
        t_index_updated_utc
    )

    # ------------------------------------------------------------------------
    # NOAA F10.7
    # ------------------------------------------------------------------------

    solar_flux = get_noaa_solar_flux()

    if solar_flux is None:
        solar_flux = (
            get_noaa_f107_fallback()
        )

    if solar_flux is not None:
        status["NOAA F10.7"] = "OK"
    else:
        status["NOAA F10.7"] = "FAILED"

    # ------------------------------------------------------------------------
    # NOAA K
    # ------------------------------------------------------------------------

    planetary_k = (
        get_noaa_planetary_k()
    )

    if planetary_k is not None:
        status["NOAA K"] = "OK"
    else:
        status["NOAA K"] = "FAILED"

    # ------------------------------------------------------------------------
    # NOAA Dst
    # ------------------------------------------------------------------------

    dst = get_noaa_dst()

    if dst is not None:
        status["NOAA Dst"] = "OK"
    else:
        status["NOAA Dst"] = "FAILED"

    # ------------------------------------------------------------------------
    # NOAA sunspots
    # ------------------------------------------------------------------------

    sunspots = (
        get_noaa_sunspot_number()
    )

    if sunspots is not None:
        status["NOAA sunspots"] = "OK"
    else:
        status["NOAA sunspots"] = "FAILED"

    # ------------------------------------------------------------------------
    # NOAA alerts
    #
    # Retrieve the complete API response first.
    #
    # This is important because raw data should remain available for
    # debugging, historical analysis and future improvements to the filter.
    # ------------------------------------------------------------------------

    try:

        noaa_alert_response = get_json(
            NOAA_ENDPOINTS["alerts"]
        )

        if isinstance(
            noaa_alert_response,
            list,
        ):

            noaa_alerts_raw = [
                item
                for item in noaa_alert_response
                if isinstance(
                    item,
                    dict,
                )
            ]

        else:
            noaa_alerts_raw = []

    except Exception:

        noaa_alerts_raw = []

    noaa_alerts = filter_current_alerts(
        noaa_alerts_raw,
        now_utc=retrieved_dt,
    )

    status["NOAA alerts"] = "OK"

    raw["NOAA alerts"] = (
        noaa_alerts_raw
    )

    # ------------------------------------------------------------------------
    # SWS AUTHENTICATED API
    # ------------------------------------------------------------------------

    sws_key_available = bool(
        get_sws_api_key()
    )

    if sws_key_available:

        australian_k = get_sws_k_index(
            "Australian region"
        )

        a_index = get_sws_a_index()

        sws_dst = get_sws_dst()

        magnetic_alerts = (
            get_sws_magnetic_alert()
        )

        magnetic_warnings = (
            get_sws_magnetic_warning()
        )

        aurora_watch = (
            get_sws_aurora_watch()
        )

        aurora_outlook = (
            get_sws_aurora_outlook()
        )

        sws_available = True
        sws_error = None

        status["SWS K"] = (
            "OK"
            if australian_k is not None
            else "FAILED"
        )

        status["SWS A"] = (
            "OK"
            if a_index is not None
            else "FAILED"
        )

        status["SWS Dst"] = (
            "OK"
            if sws_dst is not None
            else "FAILED"
        )

        raw["SWS magnetic alerts"] = (
            magnetic_alerts
        )

        raw["SWS magnetic warnings"] = (
            magnetic_warnings
        )

        raw["SWS aurora watch"] = (
            aurora_watch
        )

        raw["SWS aurora outlook"] = (
            aurora_outlook
        )

    else:

        australian_k = None
        a_index = None
        sws_dst = None

        magnetic_alerts = []
        magnetic_warnings = []
        aurora_watch = []
        aurora_outlook = []

        sws_available = False

        sws_error = (
            "SWS_API_KEY is not configured."
        )

        status["SWS"] = (
            "NOT CONFIGURED"
        )

    # ------------------------------------------------------------------------
    # Create normalised result
    # ------------------------------------------------------------------------

    result = SpaceWeatherData(

        retrieved_utc=retrieved,

        solar_flux_10_7=solar_flux,

        sunspot_number=sunspots,

        planetary_k_index=planetary_k,

        australian_k_index=australian_k,

        a_index=a_index,

        # Prefer the Australian value when available for this project.
        dst_index=(
            sws_dst
            if sws_dst is not None
            else dst
        ),

        xray_flux=None,

        hf_fadeout=False,

        polar_cap_absorption=False,

        # --------------------------------------------------------------------
        # Real-time SWS T-index
        # --------------------------------------------------------------------

        t_index=t_index,

        t_index_source=t_index_source,

        t_indices=t_indices,

        t_index_retrieved_utc=(
            t_index_retrieved_utc
        ),

        t_index_updated_utc=(
            t_index_updated_utc
        ),

        active_alerts=noaa_alerts,

        active_warnings=(
            magnetic_warnings
        ),

        sws_available=sws_available,

        sws_error=sws_error,

        source_status=status,

        raw=raw,
    )

    return result


# ============================================================================
# CONVERSION TO PROPAGATION INPUTS
# ============================================================================

def to_propagation_inputs(
    weather: SpaceWeatherData,
    latitude: float,
    longitude: float,
    timestamp_utc: Optional[datetime] = None,
):
    """
    Convert the collected data into the PropagationInputs structure used by
    band_favorability.py.

    This function intentionally imports the scoring module only when called,
    avoiding a hard dependency during simple data collection/testing.
    """

    import importlib

    propagation_inputs_type = getattr(
        importlib.import_module(
            "band_favorability"
        ),
        "PropagationInputs",
    )

    if timestamp_utc is None:
        timestamp_utc = datetime.now(
            timezone.utc
        )

    return propagation_inputs_type(

        latitude=latitude,

        longitude=longitude,

        timestamp_utc=timestamp_utc,

        solar_flux_10_7=(
            weather.solar_flux_10_7
        ),

        sunspot_number=(
            weather.sunspot_number
        ),

        k_index=(
            weather.australian_k_index
            if weather.australian_k_index
            is not None
            else weather.planetary_k_index
        ),

        a_index=weather.a_index,

        dst_index=weather.dst_index,

        xray_flux=weather.xray_flux,

        hf_fadeout=weather.hf_fadeout,

        polar_cap_absorption=(
            weather.polar_cap_absorption
        ),
    )


# ============================================================================
# PRINTABLE SUMMARY
# ============================================================================

def print_summary(
    weather: SpaceWeatherData,
) -> None:
    """
    Print a human-readable diagnostic summary.
    """

    print()
    print("=" * 65)
    print("HF PROPAGATION SPACE-WEATHER DATA")
    print("=" * 65)
    print()

    print("Retrieved:")
    print(
        f"  {weather.retrieved_utc}"
    )

    # ------------------------------------------------------------------------
    # SOLAR
    # ------------------------------------------------------------------------

    print()
    print("SOLAR")
    print("-" * 65)

    print(
        f"  F10.7:        "
        f"{weather.solar_flux_10_7}"
    )

    print(
        f"  Sunspots:     "
        f"{weather.sunspot_number}"
    )

    # ------------------------------------------------------------------------
    # GEOMAGNETIC
    # ------------------------------------------------------------------------

    print()
    print("GEOMAGNETIC")
    print("-" * 65)

    print(
        f"  Planetary K:  "
        f"{weather.planetary_k_index}"
    )

    print(
        f"  Australian K: "
        f"{weather.australian_k_index}"
    )

    print(
        f"  A index:      "
        f"{weather.a_index}"
    )

    print(
        f"  Dst:          "
        f"{weather.dst_index}"
    )

    # ------------------------------------------------------------------------
    # T INDEX
    # ------------------------------------------------------------------------

    print()
    print("REAL-TIME T INDICES")
    print("-" * 65)

    print(
        f"  Selected T:   "
        f"{weather.t_index}"
    )

    print(
        f"  Source:       "
        f"{weather.t_index_source}"
    )

    print(
        f"  SWS updated:  "
        f"{weather.t_index_updated_utc}"
    )

    print(
        f"  Retrieved:    "
        f"{weather.t_index_retrieved_utc}"
    )

    if weather.t_indices:

        for name, value in (
            weather.t_indices.items()
        ):

            display_name = (
                name.replace(
                    "_",
                    " ",
                )
                .title()
            )

            print(
                f"  {display_name + ':':<36}"
                f"{value}"
            )

    else:

        print(
            "  No real-time T-index data available."
        )

    # ------------------------------------------------------------------------
    # ALERTS
    # ------------------------------------------------------------------------

    print()
    print("ALERTS")
    print("-" * 65)

    print(
        f"  Current NOAA alerts: "
        f"{len(weather.active_alerts or [])}"
    )

    print(
        f"  Current SWS warnings:"
        f" {len(weather.active_warnings or [])}"
    )

    print(
        f"  SWS API available:"
        f" {weather.sws_available}"
    )

    # ------------------------------------------------------------------------
    # SOURCE STATUS
    # ------------------------------------------------------------------------

    print()
    print("SOURCE STATUS")
    print("-" * 65)

    for source, status in (
        weather.source_status or {}
    ).items():

        print(
            f"  {source:<28} {status}"
        )

    if weather.sws_error:

        print()
        print(
            f"SWS API: {weather.sws_error}"
        )

    print()
    print("=" * 65)


# ============================================================================
# JSON EXPORT
# ============================================================================

def save_json(
    weather: SpaceWeatherData,
    filename: str = "space_weather.json",
) -> None:
    """
    Save collected data to JSON.

    Useful during development and debugging.
    """

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            asdict(weather),
            file,
            indent=4,
            default=str,
        )


# ============================================================================
# TEST / COMMAND-LINE MODE
# ============================================================================

if __name__ == "__main__":

    weather = collect_space_weather()

    print_summary(
        weather
    )

    save_json(
        weather
    )

    print()

    print(
        "Saved: space_weather.json"
    )