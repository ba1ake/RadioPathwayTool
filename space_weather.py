"""
Space Weather Data Collector
============================

V0.2

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
        - Public real-time T-index pages
        - Australian-region K index
        - Australian-region A index
        - Australian-region Dst
        - magnetic alerts/warnings
        - aurora alerts/watches/outlooks

IMPORTANT:
    SWS real-time T-index values are collected from the public SWS
    Real Time T Index web pages and do NOT require an API key.

    The SWS API remains available for the other SWS products when
    SWS_API_KEY is configured.

    We deliberately do NOT substitute K-index for T-index.

This module deliberately separates DATA COLLECTION from the propagation
scoring engine.

The output can later be passed into band_favorability.py.
"""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Optional

import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

REQUEST_TIMEOUT = 15

SWS_API_BASE = (
    "https://sws-data.sws.bom.gov.au/api/v1"
)

# Public SWS pages.
#
# These pages contain the current real-time T-index values in their HTML.
# No API key is required.
SWS_PUBLIC_T_INDEX_AUSTRALASIA_URL = (
    "https://www.sws.bom.gov.au/HF_Systems/1/6/3"
)

SWS_PUBLIC_T_INDEX_GLOBAL_URL = (
    "https://www.sws.bom.gov.au/HF_Systems/6/4/2"
)

NOAA_BASE = (
    "https://services.swpc.noaa.gov"
)

HTTP_HEADERS = {
    "User-Agent":
        "HF-Propagation-Assistant/0.2 "
        "(amateur-radio propagation analysis)"
}


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
    # Direct SWS real-time T-index values
    #
    # These are collected from the public SWS HTML pages.
    #
    # They are deliberately kept separate from K-index and other
    # geomagnetic measurements.
    # ------------------------------------------------------------------------

    t_index_nz: Optional[float] = None

    t_index_australia: Optional[float] = None
    t_index_australian_region: Optional[float] = None

    t_index_southern_australia: Optional[float] = None
    t_index_northern_australia: Optional[float] = None
    t_index_northern_equatorial_australia: Optional[float] = None

    t_index_southern_hemisphere: Optional[float] = None
    t_index_northern_hemisphere: Optional[float] = None

    t_index_antarctic: Optional[float] = None

    # Generic/default T-index.
    t_index: Optional[float] = None

    t_index_source: Optional[str] = None
    t_index_retrieved_utc: Optional[str] = None

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
        headers=HTTP_HEADERS,
    )

    response.raise_for_status()

    return response.json()


def get_text(
    url: str,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """
    Retrieve text/HTML from a public endpoint.
    """

    response = requests.get(
        url,
        timeout=timeout,
        headers=HTTP_HEADERS,
    )

    response.raise_for_status()

    return response.text


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
            **HTTP_HEADERS,
            "Content-Type": "application/json",
        },
    )

    response.raise_for_status()

    return response.json()


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

    except (TypeError, ValueError):
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

        # Common nested data field
        if isinstance(data.get("data"), list):

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
# SWS PUBLIC REAL-TIME T INDEX
# ============================================================================

def _clean_html_text(
    page: str,
) -> str:
    """
    Convert an SWS HTML page into searchable plain text.

    This intentionally avoids requiring BeautifulSoup. The SWS pages expose
    the T-index values as ordinary page text, so a small HTML stripper is
    sufficient and keeps this module dependency-light.
    """

    # Remove scripts/styles first.
    page = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )

    page = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove remaining tags.
    page = re.sub(
        r"<[^>]+>",
        " ",
        page,
        flags=re.DOTALL,
    )

    # Decode HTML entities.
    page = html.unescape(page)

    # Normalise whitespace.
    page = re.sub(
        r"\s+",
        " ",
        page,
    )

    return page.strip()


def _extract_sws_t_index(
    text: str,
    label: str,
) -> Optional[float]:
    """
    Extract an SWS T-index value from plain page text.

    Example:

        Australian Region T index: 52

    Returns None when the value is unavailable.

    SWS documents 999 as meaning that no autoscaled data is available.
    We therefore treat 999 as unavailable rather than a real T-index.
    """

    pattern = (
        re.escape(label)
        + r"\s*:\s*"
        + r"(-?\d+(?:\.\d+)?)"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    value = safe_float(
        match.group(1)
    )

    if value is None:
        return None

    # SWS uses 999 when autoscaled data is unavailable.
    if value == 999:
        return None

    return value


def get_sws_public_t_indices() -> dict[str, Any]:
    """
    Retrieve real-time T-index values from the public SWS webpages.

    This does NOT require SWS_API_KEY.

    Returns a dictionary containing any successfully parsed indices plus
    source/error metadata.

    Public SWS Australasia page provides:

        Northern Equatorial Australian Region
        Northern Australian Region
        Southern Australian Region
        Australian Region
        Antarctic Region

    Public SWS Global page provides:

        Northern Hemisphere
        Southern Hemisphere
        Australian Region
        Southern/Northern Australian regions
        Antarctic Region

    The public pages do not expose the Tnz value, so t_index_nz remains
    unavailable unless another public SWS source is added later.
    """

    result: dict[str, Any] = {
        "available": False,
        "error": None,
        "source": None,
        "retrieved_utc": None,

        "t_index_nz": None,

        "t_index_australia": None,
        "t_index_australian_region": None,

        "t_index_southern_australia": None,
        "t_index_northern_australia": None,
        "t_index_northern_equatorial_australia": None,

        "t_index_southern_hemisphere": None,
        "t_index_northern_hemisphere": None,

        "t_index_antarctic": None,

        "raw": {},
    }

    retrieved = datetime.now(
        timezone.utc
    ).isoformat()

    result["retrieved_utc"] = retrieved

    # ------------------------------------------------------------------------
    # Australasia public page
    # ------------------------------------------------------------------------

    try:

        page = get_text(
            SWS_PUBLIC_T_INDEX_AUSTRALASIA_URL
        )

        text = _clean_html_text(
            page
        )

        result["raw"]["australasia_text"] = text

        result[
            "t_index_northern_equatorial_australia"
        ] = _extract_sws_t_index(
            text,
            "Northern Equatorial Australian Region T index",
        )

        result[
            "t_index_northern_australia"
        ] = _extract_sws_t_index(
            text,
            "Northern Australian Region T index",
        )

        result[
            "t_index_southern_australia"
        ] = _extract_sws_t_index(
            text,
            "Southern Australian Region T index",
        )

        result[
            "t_index_australian_region"
        ] = _extract_sws_t_index(
            text,
            "Australian Region T index",
        )

        result[
            "t_index_antarctic"
        ] = _extract_sws_t_index(
            text,
            "Antarctic Region T index",
        )

    except Exception as exc:

        result["raw"]["australasia_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # ------------------------------------------------------------------------
    # Global public page
    # ------------------------------------------------------------------------

    try:

        page = get_text(
            SWS_PUBLIC_T_INDEX_GLOBAL_URL
        )

        text = _clean_html_text(
            page
        )

        result["raw"]["global_text"] = text

        result[
            "t_index_northern_hemisphere"
        ] = _extract_sws_t_index(
            text,
            "Northern hemisphere T index",
        )

        result[
            "t_index_southern_hemisphere"
        ] = _extract_sws_t_index(
            text,
            "Southern hemisphere T index",
        )

        # The global page also exposes these values. Use them only when
        # the Australasia page did not already provide them.
        if (
            result["t_index_australian_region"]
            is None
        ):
            result[
                "t_index_australian_region"
            ] = _extract_sws_t_index(
                text,
                "Australian Region T index",
            )

        if (
            result["t_index_antarctic"]
            is None
        ):
            result[
                "t_index_antarctic"
            ] = _extract_sws_t_index(
                text,
                "Antarctic Region T index",
            )

        if (
            result["t_index_southern_australia"]
            is None
        ):
            result[
                "t_index_southern_australia"
            ] = _extract_sws_t_index(
                text,
                "Southern Australian Region T index",
            )

        if (
            result["t_index_northern_australia"]
            is None
        ):
            result[
                "t_index_northern_australia"
            ] = _extract_sws_t_index(
                text,
                "Northern Australian Region T index",
            )

        if (
            result[
                "t_index_northern_equatorial_australia"
            ]
            is None
        ):
            result[
                "t_index_northern_equatorial_australia"
            ] = _extract_sws_t_index(
                text,
                "Northern Equatorial Australian Region T index",
            )

    except Exception as exc:

        result["raw"]["global_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # ------------------------------------------------------------------------
    # Determine whether anything useful was obtained.
    # ------------------------------------------------------------------------

    usable_fields = (
        "t_index_australian_region",
        "t_index_southern_australia",
        "t_index_northern_australia",
        "t_index_northern_equatorial_australia",
        "t_index_southern_hemisphere",
        "t_index_northern_hemisphere",
        "t_index_antarctic",
    )

    if any(
        result.get(field) is not None
        for field in usable_fields
    ):

        result["available"] = True

        result["source"] = (
            "Australian Space Weather Services "
            "public Real Time T Index pages"
        )

        return result

    # No usable values.
    result["error"] = (
        "No usable real-time SWS T-index values "
        "could be extracted from the public pages."
    )

    return result


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
            NOAA_ENDPOINTS["solar_radio_flux"]
        )

        record = latest_record(data)

        if not record:
            return None

        # NOAA products may use different field names depending on product.
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

        record = latest_record(data)

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

        record = latest_record(data)

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

        record = latest_record(data)

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

        record = latest_record(data)

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
    Retrieve active NOAA space-weather alerts.
    """

    try:

        data = get_json(
            NOAA_ENDPOINTS["alerts"]
        )

        if isinstance(data, list):

            return [
                item
                for item in data
                if isinstance(item, dict)
            ]

        return []

    except Exception:

        return []


# ============================================================================
# SWS API
# ============================================================================

def get_sws_api_key() -> Optional[str]:
    """
    Read the SWS API key from an environment variable.

    This is only required for the authenticated SWS API products.

    The public real-time T-index collector does NOT require this key.
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

    Possible locations include:

        Australian region
        Hobart
        Melbourne
        Sydney
        Canberra
        Darwin
        Perth
        Learmonth
        Norfolk Island
        etc.
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

        if isinstance(data, list):
            return data

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

        if isinstance(data, list):
            return data

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

        if isinstance(data, list):
            return data

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

        if isinstance(data, list):
            return data

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

    The public SWS T-index pages are collected independently of the
    authenticated SWS API.
    """

    retrieved = datetime.now(
        timezone.utc
    ).isoformat()

    status: dict[str, str] = {}
    raw: dict[str, Any] = {}

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
    # ------------------------------------------------------------------------

    noaa_alerts = get_noaa_alerts()

    status["NOAA alerts"] = "OK"

    raw["NOAA alerts"] = noaa_alerts

    # ------------------------------------------------------------------------
    # PUBLIC SWS REAL-TIME T INDEX
    # ------------------------------------------------------------------------

    public_t = (
        get_sws_public_t_indices()
    )

    raw["SWS public T index"] = (
        public_t.get("raw", {})
    )

    t_index_nz = (
        public_t.get("t_index_nz")
    )

    t_index_australia = (
        public_t.get("t_index_australia")
    )

    t_index_australian_region = (
        public_t.get(
            "t_index_australian_region"
        )
    )

    t_index_southern_australia = (
        public_t.get(
            "t_index_southern_australia"
        )
    )

    t_index_northern_australia = (
        public_t.get(
            "t_index_northern_australia"
        )
    )

    t_index_northern_equatorial_australia = (
        public_t.get(
            "t_index_northern_equatorial_australia"
        )
    )

    t_index_southern = (
        public_t.get(
            "t_index_southern_hemisphere"
        )
    )

    t_index_northern = (
        public_t.get(
            "t_index_northern_hemisphere"
        )
    )

    t_index_antarctic = (
        public_t.get(
            "t_index_antarctic"
        )
    )

    # ------------------------------------------------------------------------
    # Select a sensible generic direct T-index.
    #
    # This is NOT derived from K.
    #
    # For this project, the Southern Hemisphere value is preferred because
    # the primary operating region is New Zealand / Australia.
    # ------------------------------------------------------------------------

    generic_t_index = (
        t_index_southern
        if t_index_southern is not None
        else (
            t_index_australian_region
            if t_index_australian_region is not None
            else t_index_southern_australia
        )
    )

    if public_t.get("available"):

        status["SWS public T-index"] = "OK"

        raw["SWS public T-index source"] = (
            public_t.get("source")
        )

        raw["SWS public T-index retrieved UTC"] = (
            public_t.get("retrieved_utc")
        )

    else:

        status["SWS public T-index"] = "FAILED"

    # ------------------------------------------------------------------------
    # Authenticated SWS API
    #
    # This remains optional.
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

        # API itself is not configured, but this does NOT mean that
        # public SWS T-index data is unavailable.
        status["SWS API"] = "NOT CONFIGURED"

    # ------------------------------------------------------------------------
    # Overall SWS availability
    #
    # Public T-index data counts as SWS availability because it is an
    # official SWS data product and is enough for GRAFEX T-index selection.
    # ------------------------------------------------------------------------

    sws_available = bool(
        public_t.get("available")
        or sws_key_available
    )

    if sws_available:

        sws_error = None

    else:

        sws_error = (
            "No public SWS T-index data was available "
            "and SWS_API_KEY is not configured."
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

        active_alerts=noaa_alerts,

        active_warnings=(
            raw.get(
                "SWS magnetic warnings",
                [],
            )
        ),

        sws_available=sws_available,
        sws_error=sws_error,

        # --------------------------------------------------------------------
        # Direct SWS T-index values
        # --------------------------------------------------------------------

        t_index_nz=t_index_nz,

        t_index_australia=t_index_australia,

        t_index_australian_region=(
            t_index_australian_region
        ),

        t_index_southern_australia=(
            t_index_southern_australia
        ),

        t_index_northern_australia=(
            t_index_northern_australia
        ),

        t_index_northern_equatorial_australia=(
            t_index_northern_equatorial_australia
        ),

        t_index_southern_hemisphere=(
            t_index_southern
        ),

        t_index_northern_hemisphere=(
            t_index_northern
        ),

        t_index_antarctic=(
            t_index_antarctic
        ),

        t_index=generic_t_index,

        t_index_source=(
            public_t.get("source")
        ),

        t_index_retrieved_utc=(
            public_t.get("retrieved_utc")
        ),

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

    from band_favorability import (
        PropagationInputs,
    )

    if timestamp_utc is None:

        timestamp_utc = datetime.now(
            timezone.utc
        )

    return PropagationInputs(

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

    print()
    print("GEOMAGNETIC")
    print("-" * 65)

    print(
        f"  Planetary K:  "
        f"{weather.planetary_k_index}"
    )

    print(
        f"  Australian K:"
        f" {weather.australian_k_index}"
    )

    print(
        f"  A index:      "
        f"{weather.a_index}"
    )

    print(
        f"  Dst:          "
        f"{weather.dst_index}"
    )

    print()
    print("REAL-TIME SWS T INDICES")
    print("-" * 65)

    print(
        f"  NZ T-index:                    "
        f"{weather.t_index_nz}"
    )

    print(
        f"  Australian Region T-index:     "
        f"{weather.t_index_australian_region}"
    )

    print(
        f"  Southern Australia T-index:    "
        f"{weather.t_index_southern_australia}"
    )

    print(
        f"  Northern Australia T-index:    "
        f"{weather.t_index_northern_australia}"
    )

    print(
        f"  Northern Equatorial Australia:"
        f" {weather.t_index_northern_equatorial_australia}"
    )

    print(
        f"  Southern Hemisphere T-index:   "
        f"{weather.t_index_southern_hemisphere}"
    )

    print(
        f"  Northern Hemisphere T-index:   "
        f"{weather.t_index_northern_hemisphere}"
    )

    print(
        f"  Antarctic T-index:             "
        f"{weather.t_index_antarctic}"
    )

    print(
        f"  Generic T-index:               "
        f"{weather.t_index}"
    )

    print(
        f"  T-index source:                "
        f"{weather.t_index_source}"
    )

    print()
    print("ALERTS")
    print("-" * 65)

    print(
        f"  NOAA alerts:  "
        f"{len(weather.active_alerts or [])}"
    )

    print(
        f"  SWS available:"
        f" {weather.sws_available}"
    )

    print()
    print("SOURCE STATUS")
    print("-" * 65)

    for source, status in (
        weather.source_status or {}
    ).items():

        print(
            f"  {source:<30} {status}"
        )

    if weather.sws_error:

        print()
        print(
            f"SWS: {weather.sws_error}"
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
