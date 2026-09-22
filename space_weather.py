
"""
Space Weather Data Collector
============================

V0.1

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

    SWS requires an API key.
    Set the environment variable:

        SWS_API_KEY

    NOAA's public JSON products do not currently require an API key.

This module deliberately separates DATA COLLECTION from the propagation
scoring engine.

The output can later be passed into band_favorability.py.
"""

from __future__ import annotations

import json
import os
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

NOAA_BASE = (
    "https://services.swpc.noaa.gov"
)


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
    """

    retrieved = datetime.now(
        timezone.utc
    ).isoformat()

    status = {}
    raw = {}

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
    # SWS
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

        sws_available = False

        sws_error = (
            "SWS_API_KEY is not configured."
        )

        status["SWS"] = "NOT CONFIGURED"

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
            f"  {source:<20} {status}"
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
