"""
RadioPathwayTool - Main Propagation Engine

This module is the orchestration layer between the individual data sources
and the Discord bot / AI assistant.

Architecture:

    space_weather.py
            |
    ionosphere.py
            |
    band_favorability.py
            |
    propagation_engine.py
            |
            v
         main.py
            |
            +--> structured propagation report
            |
            +--> human-readable text
            |
            v
       discord_bot.py
       ai_assistant.py

Important design principles:

1. HAP is the primary HF propagation prediction for the transmitter/base
   location.

2. GRAFEX is used when a specific TX -> RX path is supplied.

3. Space weather is supporting context, not an arbitrary score.

4. Live ionospheric observations are supporting evidence.

5. No 0-100 band score is generated here.

6. main.py does not contain Discord-specific code.

7. Importing main.py does not automatically perform network requests.

8. Individual data sources can fail without necessarily killing the
   entire propagation report.

9. GRAFEX is optional. A normal propagation report does not require
   TX/RX path information.

10. K-index values are NOT automatically treated as GRAFEX T-index values.
    GRAFEX is only requested when a usable T-index is available.

11. T-index values come directly from the SWS real-time T-index data
    collected by space_weather.py.

12. For cross-region paths, the lower applicable SWS T-index is used where
    both regional values are available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional

import re


# ---------------------------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------------------------

from space_weather import collect_space_weather

from ionosphere import get_all_ionosphere_observations

from band_favorability import (
    HAPCollector,
    HAPDecoder,
    HAPConfig,
)

from band_hap import (
    IndependentHAPCollector,
)

from propagation_engine import (
    LocationResolver,
    PropagationEngine,
)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

LOCATION_NAME = "Nelson"
LOCATION_LATITUDE = -41.27
LOCATION_LONGITUDE = 173.28

# The HAP system currently uses the nine frequencies supplied to the
# SWS prediction tool.
HAP_FREQUENCIES_KHZ = [
    1838,
    3650,
    7150,
    10125,
    14175,
    18118,
    21225,
    24940,
    28850,
]

# HAP grid centred around the requested transmitter location.
HAP_GRID_ROWS = 7
HAP_GRID_COLS = 7
HAP_GRID_STEP_LAT = 5.0
HAP_GRID_STEP_LON = 5.0


# ---------------------------------------------------------------------------
# LONG-LIVED PROPAGATION SERVICES
# ---------------------------------------------------------------------------

# Keep these alive at module level so their internal caches can persist
# between report requests.

LOCATION_RESOLVER = LocationResolver()
PROPAGATION_ENGINE = PropagationEngine()


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class DataStatus:
    """
    Describes whether each major data source was successfully obtained.
    """

    hap_available: bool = False
    ionosphere_available: bool = False
    space_weather_available: bool = False
    grafex_available: bool = False

    hap_error: Optional[str] = None
    ionosphere_error: Optional[str] = None
    space_weather_error: Optional[str] = None
    grafex_error: Optional[str] = None


@dataclass
class PropagationReport:
    """
    Complete result returned by get_propagation_report().

    GRAFEX is optional.

    If rx_location is None, this is a normal local/base-point propagation
    report.

    If rx_location is supplied, grafex contains the path-specific GRAFEX
    prediction when it can be obtained.
    """

    # -----------------------------------------------------------------------
    # Primary transmitter/base location
    # -----------------------------------------------------------------------

    location_name: str
    latitude: float
    longitude: float

    # -----------------------------------------------------------------------
    # Optional path information
    # -----------------------------------------------------------------------

    tx_location: str
    rx_location: Optional[str]

    # -----------------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------------

    generated_utc: str
    generated_local: str
    current_utc_hour: int

    # -----------------------------------------------------------------------
    # HAP
    # -----------------------------------------------------------------------

    current_band: Optional[str]
    current_frequency_mhz: Optional[float]
    current_support: Optional[int]

    next_transition_utc: Optional[int]
    next_transition_band: Optional[str]
    next_transition_frequency_mhz: Optional[float]

    regional_distribution: dict[str, int]
    hap_forecast: dict[int, dict[str, Any]]

    # Independent one-frequency HAP predictions.
    #
    # Structure:
    #
    #   band -> UTC hour -> prediction data
    #
    # These are deliberately separate from the universal multi-frequency
    # HAP recommendation.
    hap_band_forecast: dict[str, dict[int, dict[str, Any]]]

    # -----------------------------------------------------------------------
    # Supporting data
    # -----------------------------------------------------------------------

    ionosphere_observations: list[dict[str, Any]]
    space_weather: dict[str, Any]

    # -----------------------------------------------------------------------
    # Path-specific GRAFEX
    # -----------------------------------------------------------------------

    grafex: Optional[dict[str, Any]]

    # -----------------------------------------------------------------------
    # Status + text
    # -----------------------------------------------------------------------

    data_status: DataStatus
    text: str


# ---------------------------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------------------------

def get_current_utc() -> datetime:
    """
    Return the current timezone-aware UTC datetime.
    """
    return datetime.now(timezone.utc)


def format_local_time(dt: datetime) -> str:
    """
    Format the supplied datetime for the local NZ user-facing report.

    New Zealand daylight saving is handled automatically by ZoneInfo.
    """
    try:
        from zoneinfo import ZoneInfo

        nz_time = dt.astimezone(
            ZoneInfo("Pacific/Auckland")
        )

        return nz_time.strftime(
            "%d %b %Y %H:%M NZ"
        )

    except Exception:
        return dt.strftime(
            "%d %b %Y %H:%M UTC"
        )


# ---------------------------------------------------------------------------
# LOCATION HELPERS
# ---------------------------------------------------------------------------

def resolve_location(
    location_name: str,
) -> tuple[str, float, float]:
    """
    Resolve a human-readable location into:

        (display_name, latitude, longitude)

    Uses the LocationResolver from propagation_engine.py.
    """

    result = LOCATION_RESOLVER.resolve(
        location_name
    )

    # Object-style result.
    if hasattr(result, "latitude") and hasattr(
        result,
        "longitude",
    ):
        latitude = float(result.latitude)
        longitude = float(result.longitude)

        display_name = getattr(
            result,
            "display_name",
            location_name,
        )

        return (
            str(display_name),
            latitude,
            longitude,
        )

    # Dictionary-style result.
    if isinstance(result, dict):
        latitude = float(
            result["latitude"]
        )

        longitude = float(
            result["longitude"]
        )

        display_name = result.get(
            "display_name",
            location_name,
        )

        return (
            str(display_name),
            latitude,
            longitude,
        )

    # Tuple/list-style result.
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        latitude = float(result[0])
        longitude = float(result[1])

        return (
            location_name,
            latitude,
            longitude,
        )

    raise ValueError(
        f"Unsupported location resolver result for "
        f"{location_name!r}: {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# HAP HELPERS
# ---------------------------------------------------------------------------

def create_hap_config(
    current_time: datetime,
    location_name: str = LOCATION_NAME,
    latitude: float = LOCATION_LATITUDE,
    longitude: float = LOCATION_LONGITUDE,
    t_index: Optional[float] = None,
) -> HAPConfig:
    """
    Create the HAP configuration centred on the requested transmitter
    location.

    For a normal report, this defaults to Nelson.

    For a path-specific report, HAP is centred on the TX location while
    GRAFEX handles the complete TX -> RX path.
    """

    return HAPCollector.create_centred_config(
        base_name=location_name,
        base_lat=latitude,
        base_lon=longitude,
        nrows=HAP_GRID_ROWS,
        ncols=HAP_GRID_COLS,
        step_lat=HAP_GRID_STEP_LAT,
        step_lon=HAP_GRID_STEP_LON,
        tindex=(
            int(round(t_index))
            if t_index is not None
            else 5
        ),
    )


def collect_hap(
    current_time: datetime,
    location_name: str = LOCATION_NAME,
    latitude: float = LOCATION_LATITUDE,
    longitude: float = LOCATION_LONGITUDE,
    t_index: Optional[float] = None,
) -> tuple[Any, Any]:
    """
    Collect and decode the complete HAP forecast.

    Returns:
        (config, decoded_hap)
    """

    config = create_hap_config(
        current_time=current_time,
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        t_index=t_index,
    )

    collector = HAPCollector()

    hap_data = collector.collect(
        config
    )

    decoder = HAPDecoder(
        config
    )

    decoded_hap = decoder.decode_all(
        hap_data
    )

    return config, decoded_hap


def collect_independent_hap(
    config: HAPConfig,
    current_time: datetime,
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Collect independent HAP predictions for every configured band.

    The collector returns:

        band -> {
            "frequency_mhz": float,
            "hours": {
                utc_hour -> prediction
            }
        }

    The report expects:

        band -> {
            utc_hour -> prediction
        }

    Therefore this function unwraps the "hours" container.
    """

    collector = IndependentHAPCollector()

    raw_results = collector.collect_all(
        config=config,
        frequencies_khz=HAP_FREQUENCIES_KHZ,
        timestamp_utc=current_time,
    )

    results: dict[str, dict[int, dict[str, Any]]] = {}

    for band, band_data in raw_results.items():
        results[band] = band_data.get("hours", {})

    return results


def get_hap_hour(
    decoder: HAPDecoder,
    decoded_hap: Any,
    hour_utc: int,
) -> Any:
    """
    Return the decoded HAP data for a particular UTC hour.
    """

    return decoded_hap[hour_utc]


def extract_hap_base_recommendation(
    decoder: HAPDecoder,
    decoded_hap: Any,
    hour_utc: int,
) -> tuple[
    Optional[str],
    Optional[float],
    Optional[int],
]:
    """
    Extract the HAP recommendation for the transmitter/base point.

    Returns:
        band
        frequency MHz
        sample support

    If no usable recommendation exists, all three values are None.

    NOTE:
        sample_support is decoder/sample support. It represents how many
        local pixels contributed to the decoded recommendation. It is not
        a propagation probability, signal-strength estimate, or contact
        reliability score.
    """

    try:
        recommendation = decoder.get_base_recommendation(
            decoded=decoded_hap,
            hour_utc=hour_utc,
        )

        if recommendation is None:
            return None, None, None

        band = getattr(
            recommendation,
            "band",
            None,
        )

        frequency_khz = getattr(
            recommendation,
            "frequency_khz",
            None,
        )

        frequency_mhz = (
            frequency_khz / 1000.0
            if frequency_khz is not None
            else None
        )

        # HAPRecommendation uses sample_support.
        #
        # This is decoder/sample support, NOT a propagation confidence
        # percentage or probability of successful communication.
        support = getattr(
            recommendation,
            "sample_support",
            None,
        )

        return (
            band,
            frequency_mhz,
            support,
        )

    except Exception:
        return None, None, None


def extract_regional_distribution(
    decoder: HAPDecoder,
    decoded_hap: Any,
    hour_utc: int,
) -> dict[str, int]:
    """
    Extract the number of HAP grid points assigned to each band.

    The values are counts, not percentages and not probability estimates.
    """

    try:
        distribution = decoder.get_regional_distribution(
            decoded=decoded_hap,
            hour_utc=hour_utc,
        )

        if distribution is None:
            return {}

        return {
            str(band): int(count)
            for band, count in distribution.items()
        }

    except Exception:
        return {}


def find_next_hap_transition(
    decoder: HAPDecoder,
    decoded_hap: Any,
    current_hour: int,
) -> tuple[
    Optional[int],
    Optional[str],
    Optional[float],
]:
    """
    Find the next UTC hour at which the HAP recommendation at the
    transmitter/base point changes.

    Searches forward through the next 23 hours.
    """

    try:
        current_band, _, _ = (
            extract_hap_base_recommendation(
                decoder,
                decoded_hap,
                current_hour,
            )
        )

        if current_band is None:
            return None, None, None

        for offset in range(1, 24):
            hour = (
                current_hour + offset
            ) % 24

            (
                band,
                frequency_mhz,
                _,
            ) = extract_hap_base_recommendation(
                decoder,
                decoded_hap,
                hour,
            )

            if band is None:
                continue

            if band != current_band:
                return (
                    hour,
                    band,
                    frequency_mhz,
                )

    except Exception:
        pass

    return None, None, None


# ---------------------------------------------------------------------------
# SPACE WEATHER HELPERS
# ---------------------------------------------------------------------------

def _numeric(value: Any) -> Optional[float]:
    """
    Safely convert a value to float.
    """

    try:
        if value is None:
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def _get_t_indices_from_weather(
    weather: Any,
) -> dict[str, float]:
    """
    Extract the direct SWS T-index dictionary from SpaceWeatherData.

    space_weather.py now stores the live SWS values in:

        weather.t_indices

    Expected keys:

        northern_hemisphere
        southern_hemisphere
        northern_equatorial_australia
        northern_australia
        southern_australia
        australian_region
        antarctic_region

    No K-index conversion is performed here.
    """

    raw_indices = getattr(
        weather,
        "t_indices",
        None,
    )

    if not isinstance(
        raw_indices,
        dict,
    ):
        return {}

    result: dict[str, float] = {}

    for key, value in raw_indices.items():
        numeric_value = _numeric(value)

        if numeric_value is None:
            continue

        # 999 is the SWS missing/unusable sentinel.
        if numeric_value == 999:
            continue

        result[str(key)] = numeric_value

    return result


def build_space_weather_report(
    weather: Any,
) -> dict[str, Any]:
    """
    Convert SpaceWeatherData into a simple serialisable dictionary.

    This intentionally exposes the measurements rather than converting them
    into a propagation score.

    T-index values are read directly from space_weather.py.

    We do NOT derive T-index values from K-index.
    """

    t_indices = _get_t_indices_from_weather(
        weather
    )

    # -----------------------------------------------------------------------
    # Direct regional T-index values
    # -----------------------------------------------------------------------

    t_index_northern = t_indices.get(
        "northern_hemisphere"
    )

    t_index_southern = t_indices.get(
        "southern_hemisphere"
    )

    t_index_northern_equatorial_australia = (
        t_indices.get(
            "northern_equatorial_australia"
        )
    )

    t_index_northern_australia = (
        t_indices.get(
            "northern_australia"
        )
    )

    t_index_southern_australia = (
        t_indices.get(
            "southern_australia"
        )
    )

    t_index_australian_region = (
        t_indices.get(
            "australian_region"
        )
    )

    t_index_antarctic = t_indices.get(
        "antarctic_region"
    )

    # -----------------------------------------------------------------------
    # NZ compatibility
    #
    # The public SWS page does not necessarily expose a dedicated NZ T-index.
    # If space_weather.py provides one in the future, use it.
    #
    # Otherwise use Southern Hemisphere as the NZ regional fallback.
    # -----------------------------------------------------------------------

    t_index_nz = (
        t_indices.get("new_zealand")
        or t_indices.get("nz")
        or t_indices.get("tnz")
    )

    if t_index_nz is None:
        t_index_nz = t_index_southern

    # -----------------------------------------------------------------------
    # Generic/default T-index
    #
    # space_weather.py already selects a sensible default T-index.
    # Preserve that value when available.
    # -----------------------------------------------------------------------

    t_index = _numeric(
        getattr(
            weather,
            "t_index",
            None,
        )
    )

    if t_index is None:
        t_index = (
            t_index_southern
            or t_index_australian_region
            or t_index_southern_australia
            or t_index_northern_australia
            or t_index_northern
            or t_index_antarctic
        )

    return {
        # -------------------------------------------------------------------
        # Standard space-weather measurements
        # -------------------------------------------------------------------

        "retrieved_utc": getattr(
            weather,
            "retrieved_utc",
            None,
        ),

        "solar_flux_10_7": getattr(
            weather,
            "solar_flux_10_7",
            None,
        ),

        "sunspot_number": getattr(
            weather,
            "sunspot_number",
            None,
        ),

        "planetary_k_index": getattr(
            weather,
            "planetary_k_index",
            None,
        ),

        "australian_k_index": getattr(
            weather,
            "australian_k_index",
            None,
        ),

        "a_index": getattr(
            weather,
            "a_index",
            None,
        ),

        "dst_index": getattr(
            weather,
            "dst_index",
            None,
        ),

        "xray_flux": getattr(
            weather,
            "xray_flux",
            None,
        ),

        "hf_fadeout": getattr(
            weather,
            "hf_fadeout",
            None,
        ),

        "polar_cap_absorption": getattr(
            weather,
            "polar_cap_absorption",
            None,
        ),

        # -------------------------------------------------------------------
        # Alerts
        # -------------------------------------------------------------------

        "active_alerts": getattr(
            weather,
            "active_alerts",
            [],
        ),

        "active_warnings": getattr(
            weather,
            "active_warnings",
            [],
        ),

        # -------------------------------------------------------------------
        # SWS status
        # -------------------------------------------------------------------

        "sws_available": getattr(
            weather,
            "sws_available",
            False,
        ),

        "sws_error": getattr(
            weather,
            "sws_error",
            None,
        ),

        "source_status": getattr(
            weather,
            "source_status",
            {},
        ),

        # -------------------------------------------------------------------
        # T-index data
        # -------------------------------------------------------------------

        "t_index": t_index,

        "t_indices": t_indices,

        "t_index_source": getattr(
            weather,
            "t_index_source",
            None,
        ),

        "t_index_retrieved_utc": getattr(
            weather,
            "t_index_retrieved_utc",
            None,
        ),

        "t_index_updated_utc": getattr(
            weather,
            "t_index_updated_utc",
            None,
        ),

        # Compatibility / convenient named fields.
        "t_index_nz": t_index_nz,

        "t_index_australia": (
            t_index_australian_region
            or t_index_southern_australia
            or t_index_northern_australia
        ),

        "t_index_australian_region": (
            t_index_australian_region
        ),

        "t_index_southern_australia": (
            t_index_southern_australia
        ),

        "t_index_northern_australia": (
            t_index_northern_australia
        ),

        "t_index_northern_equatorial_australia": (
            t_index_northern_equatorial_australia
        ),

        "t_index_southern_hemisphere": (
            t_index_southern
        ),

        "t_index_northern_hemisphere": (
            t_index_northern
        ),

        "t_index_antarctic": (
            t_index_antarctic
        ),
    }


def extract_grafex_t_index(
    tx_location: str,
    rx_location: str,
    weather: Any,
    weather_report: dict[str, Any],
) -> Optional[float]:
    """
    Select the most appropriate direct SWS T-index for a GRAFEX
    TX -> RX path.

    IMPORTANT:
        This function NEVER converts K-index into T-index.

    Selection:

    NZ <-> Australia:
        Lower of the NZ/Southern Hemisphere and Australian regional
        T-indices when both are available.

    NZ <-> NZ:
        NZ-specific T-index if available, otherwise Southern Hemisphere.

    Australia <-> Australia:
        Australian regional T-index.

    Other Southern Hemisphere:
        Southern Hemisphere T-index.

    Other Northern Hemisphere:
        Northern Hemisphere T-index.

    Antarctic:
        Antarctic T-index.

    Generic:
        SpaceWeatherData's selected default T-index.

    The lower-value selection for cross-region paths follows SWS guidance
    for communications between different regions during disturbed
    conditions.
    """

    # -----------------------------------------------------------------------
    # Get direct SWS T-index values.
    # -----------------------------------------------------------------------

    t_indices = weather_report.get(
        "t_indices",
        {},
    )

    if not isinstance(
        t_indices,
        dict,
    ):
        t_indices = {}

    def get_index(
        *keys: str,
    ) -> Optional[float]:

        for key in keys:
            value = _numeric(
                t_indices.get(key)
            )

            if value is not None:
                return value

        return None

    # NZ:
    #
    # The current public SWS page may not provide a dedicated NZ value.
    # Southern Hemisphere is therefore the valid fallback.
    t_index_nz = get_index(
        "new_zealand",
        "nz",
        "tnz",
    )

    if t_index_nz is None:
        t_index_nz = _numeric(
            weather_report.get(
                "t_index_nz"
            )
        )

    if t_index_nz is None:
        t_index_nz = get_index(
            "southern_hemisphere"
        )

    # Australia.
    t_index_australia = get_index(
        "australian_region",
        "southern_australia",
        "northern_australia",
        "northern_equatorial_australia",
    )

    if t_index_australia is None:
        t_index_australia = _numeric(
            weather_report.get(
                "t_index_australia"
            )
        )

    # Southern Hemisphere.
    t_index_southern = get_index(
        "southern_hemisphere"
    )

    if t_index_southern is None:
        t_index_southern = _numeric(
            weather_report.get(
                "t_index_southern_hemisphere"
            )
        )

    # Northern Hemisphere.
    t_index_northern = get_index(
        "northern_hemisphere"
    )

    if t_index_northern is None:
        t_index_northern = _numeric(
            weather_report.get(
                "t_index_northern_hemisphere"
            )
        )

    # Antarctic.
    t_index_antarctic = get_index(
        "antarctic_region"
    )

    if t_index_antarctic is None:
        t_index_antarctic = _numeric(
            weather_report.get(
                "t_index_antarctic"
            )
        )

    # Generic direct T-index.
    generic_t_index = _numeric(
        weather_report.get(
            "t_index"
        )
    )

    # -----------------------------------------------------------------------
    # Normalise location names.
    # -----------------------------------------------------------------------

    tx = (
        tx_location.strip().casefold()
    )

    rx = (
        rx_location.strip().casefold()
    )

    def is_new_zealand(
        location: str,
    ) -> bool:
        return (
            "new zealand" in location
            or location.endswith(", nz")
            or location == "nz"
        )

    def is_australia(
        location: str,
    ) -> bool:
        return (
            "australia" in location
            or location.endswith(", au")
            or location == "au"
        )

    tx_nz = is_new_zealand(tx)
    rx_nz = is_new_zealand(rx)

    tx_aus = is_australia(tx)
    rx_aus = is_australia(rx)

    # -----------------------------------------------------------------------
    # NZ <-> Australia
    # -----------------------------------------------------------------------

    nz_to_australia = (
        (tx_nz and rx_aus)
        or
        (tx_aus and rx_nz)
    )

    if nz_to_australia:
        available = [
            value
            for value in (
                t_index_nz,
                t_index_australia,
            )
            if value is not None
        ]

        if available:
            return min(
                available
            )

        if t_index_southern is not None:
            return t_index_southern

        return generic_t_index

    # -----------------------------------------------------------------------
    # New Zealand -> New Zealand
    # -----------------------------------------------------------------------

    if tx_nz and rx_nz:

        if t_index_nz is not None:
            return t_index_nz

        if t_index_southern is not None:
            return t_index_southern

        return generic_t_index

    # -----------------------------------------------------------------------
    # Australia -> Australia
    # -----------------------------------------------------------------------

    if tx_aus and rx_aus:

        if t_index_australia is not None:
            return t_index_australia

        if t_index_southern is not None:
            return t_index_southern

        return generic_t_index

    # -----------------------------------------------------------------------
    # Generic Southern Hemisphere
    # -----------------------------------------------------------------------

    if t_index_southern is not None:
        return t_index_southern

    # -----------------------------------------------------------------------
    # Generic Northern Hemisphere
    # -----------------------------------------------------------------------

    if t_index_northern is not None:
        return t_index_northern

    # -----------------------------------------------------------------------
    # Antarctic
    # -----------------------------------------------------------------------

    if t_index_antarctic is not None:
        return t_index_antarctic

    # -----------------------------------------------------------------------
    # Generic direct T-index
    # -----------------------------------------------------------------------

    return generic_t_index


# ---------------------------------------------------------------------------
# IONOSPHERE HELPERS
# ---------------------------------------------------------------------------

def build_ionosphere_report(
    observations: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert ionospheric observations into simple dictionaries suitable for
    Discord/JSON.
    """

    result = []

    for observation in observations.values():

        percent = getattr(
            observation,
            "percent_difference",
            None,
        )

        if percent is not None:

            if percent > 0:
                condition_text = (
                    f"{getattr(observation, 'condition', 'unknown')} "
                    f"(+{percent:.0f}%)"
                )

            else:
                condition_text = (
                    f"{getattr(observation, 'condition', 'unknown')} "
                    f"({percent:.0f}%)"
                )

        else:

            condition_text = getattr(
                observation,
                "condition",
                "unknown",
            )

        timestamp = getattr(
            observation,
            "timestamp_utc",
            None,
        )

        if timestamp is not None:
            timestamp_text = timestamp.isoformat()
        else:
            timestamp_text = None

        result.append(
            {
                "station": getattr(
                    observation,
                    "station",
                    None,
                ),

                "station_name": getattr(
                    observation,
                    "station_name",
                    None,
                ),

                "timestamp_utc": timestamp_text,

                "condition": getattr(
                    observation,
                    "condition",
                    None,
                ),

                "condition_display": condition_text,

                "percent_difference": percent,

                "data_available": getattr(
                    observation,
                    "data_available",
                    False,
                ),

                "source": getattr(
                    observation,
                    "source",
                    None,
                ),
            }
        )

    return result


# ---------------------------------------------------------------------------
# GRAFEX HELPERS
# ---------------------------------------------------------------------------

def run_grafex_path_prediction(
    tx_location: str,
    rx_location: str,
    prediction_date: date,
    t_index: Optional[float],
) -> tuple[
    Optional[dict[str, Any]],
    Optional[str],
]:
    """
    Run a path-specific GRAFEX prediction.

    Returns:
        (prediction_dict, error)

    GRAFEX is deliberately isolated from the rest of the propagation
    pipeline so a path prediction failure does not kill HAP, ionosphere,
    or space-weather data.
    """

    if not tx_location.strip():
        return (
            None,
            "TX location was empty.",
        )

    if not rx_location.strip():
        return (
            None,
            "RX location was empty.",
        )

    if t_index is None:
        return (
            None,
            "GRAFEX requires a direct SWS T-index, but the current "
            "space-weather data source did not provide one. "
            "K-index is not substituted for T-index.",
        )

    try:

        prediction = PROPAGATION_ENGINE.calculate_path(
            tx_location=tx_location,
            rx_location=rx_location,
            prediction_date=prediction_date,
            t_index=t_index,
        )

        if prediction is None:
            return (
                None,
                "GRAFEX returned no prediction.",
            )

        # Dataclass result.
        try:
            prediction_dict = asdict(
                prediction
            )

        except Exception:

            # Dictionary result.
            if isinstance(
                prediction,
                dict,
            ):
                prediction_dict = prediction

            # Object result.
            else:
                prediction_dict = {
                    key: value
                    for key, value in vars(
                        prediction
                    ).items()
                    if not key.startswith("_")
                }

        return (
            prediction_dict,
            None,
        )

    except Exception as exc:
        return (
            None,
            f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# TEXT FORMATTING
# ---------------------------------------------------------------------------

def format_frequency(
    frequency_mhz: Optional[float],
) -> str:

    if frequency_mhz is None:
        return "unknown"

    return f"{frequency_mhz:.3f} MHz"


def format_regional_distribution(
    distribution: dict[str, int],
) -> list[str]:
    """
    Convert regional counts to human-readable lines.

    HAP currently represents a 7 x 7 = 49 point grid.
    """

    if not distribution:
        return [
            "No regional HAP distribution available."
        ]

    total = sum(
        distribution.values()
    )

    if total <= 0:
        return [
            "No regional HAP distribution available."
        ]

    ordered = sorted(
        distribution.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    lines = []

    for band, count in ordered:

        percentage = (
            count / total
        ) * 100.0

        lines.append(
            f"{band}: {count}/{total} "
            f"({percentage:.0f}%)"
        )

    return lines


def format_ionosphere(
    observations: list[dict[str, Any]],
) -> list[str]:
    """
    Format the most useful ionospheric observations.

    Stations with actual data are shown first.
    """

    if not observations:
        return [
            "No ionospheric observations available."
        ]

    usable = [
        observation
        for observation in observations
        if observation.get("data_available")
    ]

    unusable = [
        observation
        for observation in observations
        if not observation.get("data_available")
    ]

    lines = []

    for observation in usable:

        name = observation.get(
            "station_name",
            observation.get(
                "station",
                "Unknown",
            ),
        )

        condition = observation.get(
            "condition_display",
            "unknown",
        )

        lines.append(
            f"{name}: {condition}"
        )

    if not usable:

        for observation in unusable:

            name = observation.get(
                "station_name",
                observation.get(
                    "station",
                    "Unknown",
                ),
            )

            condition = observation.get(
                "condition_display",
                "no data",
            )

            lines.append(
                f"{name}: {condition}"
            )

    return lines


def format_space_weather(
    weather: dict[str, Any],
) -> list[str]:
    """
    Format the main space-weather measurements.
    """

    lines = []

    solar_flux = weather.get(
        "solar_flux_10_7"
    )

    if solar_flux is not None:
        lines.append(
            f"F10.7: {solar_flux}"
        )

    sunspots = weather.get(
        "sunspot_number"
    )

    if sunspots is not None:
        lines.append(
            f"Sunspots: {sunspots}"
        )

    australian_k = weather.get(
        "australian_k_index"
    )

    planetary_k = weather.get(
        "planetary_k_index"
    )

    if australian_k is not None:
        lines.append(
            f"Australian K: {australian_k}"
        )

    elif planetary_k is not None:
        lines.append(
            f"Planetary K: {planetary_k}"
        )

    a_index = weather.get(
        "a_index"
    )

    if a_index is not None:
        lines.append(
            f"A-index: {a_index}"
        )

    dst = weather.get(
        "dst_index"
    )

    if dst is not None:
        lines.append(
            f"Dst: {dst}"
        )

    # -----------------------------------------------------------------------
    # T-index
    # -----------------------------------------------------------------------

    t_indices = weather.get(
        "t_indices",
        {},
    )

    if not isinstance(
        t_indices,
        dict,
    ):
        t_indices = {}

    t_index_nz = weather.get(
        "t_index_nz"
    )

    t_index_australia = weather.get(
        "t_index_australia"
    )

    t_index_australian_region = weather.get(
        "t_index_australian_region"
    )

    t_index_southern_australia = weather.get(
        "t_index_southern_australia"
    )

    t_index_northern_australia = weather.get(
        "t_index_northern_australia"
    )

    t_index_southern = weather.get(
        "t_index_southern_hemisphere"
    )

    t_index_northern = weather.get(
        "t_index_northern_hemisphere"
    )

    t_index_antarctic = weather.get(
        "t_index_antarctic"
    )

    t_index = weather.get(
        "t_index"
    )

    if t_index_nz is not None:
        lines.append(
            f"NZ T-index: {t_index_nz}"
        )

    if t_index_australian_region is not None:
        lines.append(
            f"Australian Region T-index: "
            f"{t_index_australian_region}"
        )

    elif t_index_australia is not None:
        lines.append(
            f"Australian T-index: "
            f"{t_index_australia}"
        )

    if t_index_southern_australia is not None:
        lines.append(
            f"Southern Australia T-index: "
            f"{t_index_southern_australia}"
        )

    if t_index_northern_australia is not None:
        lines.append(
            f"Northern Australia T-index: "
            f"{t_index_northern_australia}"
        )

    if t_index_southern is not None:
        lines.append(
            f"Southern Hemisphere T-index: "
            f"{t_index_southern}"
        )

    if t_index_northern is not None:
        lines.append(
            f"Northern Hemisphere T-index: "
            f"{t_index_northern}"
        )

    if t_index_antarctic is not None:
        lines.append(
            f"Antarctic T-index: "
            f"{t_index_antarctic}"
        )

    # Generic fallback for compatibility.
    if (
        t_index is not None
        and not any(
            value is not None
            for value in (
                t_index_nz,
                t_index_australia,
                t_index_australian_region,
                t_index_southern_australia,
                t_index_northern_australia,
                t_index_southern,
                t_index_northern,
                t_index_antarctic,
            )
        )
    ):
        lines.append(
            f"T-index: {t_index}"
        )

    return lines


# ---------------------------------------------------------------------------
# GRAFEX TEXT FORMATTING
# ---------------------------------------------------------------------------

def format_grafex_summary(
    grafex: dict[str, Any],
) -> list[str]:
    """
    Produce a concise human-readable summary of the GRAFEX prediction.

    The complete structured GRAFEX data remains available in report.grafex.
    """

    lines = []

    tx_name = grafex.get(
        "tx_name"
    )

    rx_name = grafex.get(
        "rx_name"
    )

    distance_km = grafex.get(
        "distance_km"
    )

    prediction_date = grafex.get(
        "prediction_date"
    )

    t_index = grafex.get(
        "t_index"
    )

    if tx_name and rx_name:
        lines.append(
            f"Path: {tx_name} -> {rx_name}"
        )

    if distance_km is not None:
        try:
            lines.append(
                f"Distance: {float(distance_km):.0f} km"
            )
        except (TypeError, ValueError):
            lines.append(
                f"Distance: {distance_km} km"
            )

    if prediction_date:
        lines.append(
            f"Prediction date: {prediction_date}"
        )

    if t_index is not None:
        lines.append(
            f"GRAFEX T-index: {t_index}"
        )

    bearing = grafex.get(
        "bearing_tx_to_rx"
    )

    if bearing is not None:
        try:
            lines.append(
                f"TX bearing: {float(bearing):.1f}°"
            )
        except (TypeError, ValueError):
            pass

    first_mode = grafex.get(
        "first_mode"
    )

    second_mode = grafex.get(
        "second_mode"
    )

    if first_mode or second_mode:

        modes = [
            mode
            for mode in (
                first_mode,
                second_mode,
            )
            if mode
        ]

        lines.append(
            "Propagation modes: "
            + " / ".join(
                str(mode)
                for mode in modes
            )
        )

    # -----------------------------------------------------------------------
    # Hourly predictions
    # -----------------------------------------------------------------------

    hours = grafex.get(
        "hours"
    )

    if isinstance(
        hours,
        list,
    ) and hours:

        lines.append("")
        lines.append(
            "GRAFEX hourly path prediction:"
        )

        shown = 0

        for hour in hours:

            if not isinstance(
                hour,
                dict,
            ):
                continue

            utc_hour = hour.get(
                "utc_hour"
            )

            frequencies = hour.get(
                "frequencies",
                {},
            )

            if not isinstance(
                frequencies,
                dict,
            ):
                continue

            supported_bands = []

            for band, prediction in frequencies.items():

                if not isinstance(
                    prediction,
                    dict,
                ):
                    continue

                if prediction.get(
                    "supported"
                ):
                    supported_bands.append(
                        str(band)
                    )

            if supported_bands:

                try:
                    hour_text = f"{int(utc_hour):02d}"
                except (TypeError, ValueError):
                    hour_text = str(utc_hour)

                lines.append(
                    f"  {hour_text} UTC: "
                    + ", ".join(
                        supported_bands
                    )
                )

                shown += 1

            if shown >= 24:
                break

    return lines


# ---------------------------------------------------------------------------
# SPACE WEATHER ALERT FORMATTING
# ---------------------------------------------------------------------------

def extract_alert_message(
    alert: dict[str, Any],
) -> str:
    """
    Extract the human-readable message from a space-weather alert.
    """

    for key in (
        "message",
        "text",
        "description",
        "body",
        "content",
        "summary",
    ):

        value = alert.get(
            key
        )

        if isinstance(
            value,
            str,
        ) and value.strip():

            return value.strip()

    return ""


def format_space_weather_alert(
    alert: dict[str, Any],
) -> str:
    """
    Convert a raw space-weather alert dictionary into a concise
    human-readable summary.
    """

    message = extract_alert_message(
        alert
    )

    if not message:
        return "Space-weather alert"

    # -----------------------------------------------------------------------
    # Geomagnetic storm watch
    # -----------------------------------------------------------------------

    storm_match = re.search(
        r"WATCH:\s*\*+\s*Geomagnetic Storm Category\s+"
        r"(G\d+)\s+Predicted",
        message,
        flags=re.IGNORECASE,
    )

    if storm_match:

        level = storm_match.group(
            1
        ).upper()

        date_match = re.search(
            r"([A-Z][a-z]{2}\s+\d{1,2})\s*:\s*\*?"
            rf"{re.escape(level)}",
            message,
            flags=re.IGNORECASE,
        )

        if date_match:
            return (
                f"{level} geomagnetic storm watch "
                f"for {date_match.group(1)}"
            )

        return (
            f"{level} geomagnetic storm watch"
        )

    # -----------------------------------------------------------------------
    # Geomagnetic storm warning
    # -----------------------------------------------------------------------

    warning_match = re.search(
        r"WARNING:\s*\*+\s*Geomagnetic Storm Category\s+"
        r"(G\d+)",
        message,
        flags=re.IGNORECASE,
    )

    if warning_match:

        level = warning_match.group(
            1
        ).upper()

        return (
            f"{level} geomagnetic storm warning"
        )

    # -----------------------------------------------------------------------
    # Generic WATCH
    # -----------------------------------------------------------------------

    generic_watch = re.search(
        r"WATCH:\s*\*+\s*(.+?)(?:\r?\n|$)",
        message,
        flags=re.IGNORECASE,
    )

    if generic_watch:

        text = generic_watch.group(
            1
        ).strip(" *")

        if text:
            return text

    # -----------------------------------------------------------------------
    # Generic WARNING
    # -----------------------------------------------------------------------

    generic_warning = re.search(
        r"WARNING:\s*\*+\s*(.+?)(?:\r?\n|$)",
        message,
        flags=re.IGNORECASE,
    )

    if generic_warning:

        text = generic_warning.group(
            1
        ).strip(" *")

        if text:
            return text

    # -----------------------------------------------------------------------
    # Generic ALERT
    # -----------------------------------------------------------------------

    generic_alert = re.search(
        r"ALERT:\s*\*+\s*(.+?)(?:\r?\n|$)",
        message,
        flags=re.IGNORECASE,
    )

    if generic_alert:

        text = generic_alert.group(
            1
        ).strip(" *")

        if text:
            return text

    # -----------------------------------------------------------------------
    # Fallback
    # -----------------------------------------------------------------------

    first_line = (
        message.splitlines()[0].strip()
    )

    if first_line:
        return first_line

    return "Space-weather alert"


def format_space_weather_alerts(
    alerts: list[dict[str, Any]] | None,
) -> list[str]:
    """
    Format all currently active space-weather alerts.
    """

    if not alerts:
        return []

    formatted = []

    for alert in alerts:

        if not isinstance(
            alert,
            dict,
        ):
            continue

        text = format_space_weather_alert(
            alert
        )

        if (
            text
            and text not in formatted
        ):
            formatted.append(
                text
            )

    return formatted


# ---------------------------------------------------------------------------
# REPORT TEXT
# ---------------------------------------------------------------------------

def build_report_text(
    report: PropagationReport,
) -> str:
    """
    Create a compact human-readable report.

    This is deliberately plain text rather than Discord-specific Markdown.

    GRAFEX is only displayed when a path was requested.
    """

    lines = []

    lines.append(
        "=" * 60
    )

    lines.append(
        "RADIOPATHWAYTOOL"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        f"Location: {report.location_name}"
    )

    lines.append(
        f"Coordinates: "
        f"{report.latitude:.2f}, "
        f"{report.longitude:.2f}"
    )

    # -----------------------------------------------------------------------
    # Path
    # -----------------------------------------------------------------------

    if report.rx_location:

        lines.append(
            f"Path: "
            f"{report.tx_location} -> "
            f"{report.rx_location}"
        )

    lines.append(
        f"Generated: {report.generated_local}"
    )

    lines.append("")

    # -----------------------------------------------------------------------
    # HAP
    # -----------------------------------------------------------------------

    lines.append(
        "HAP"
    )

    lines.append(
        "-" * 60
    )

    if report.data_status.hap_available:

        if report.current_band:

            lines.append(
                "Current recommendation: "
                f"{report.current_band} "
                f"({format_frequency(report.current_frequency_mhz)})"
            )

            if report.current_support is not None:
                lines.append(
                    f"Grid support: "
                    f"{report.current_support}/49"
                )

        else:

            lines.append(
                "Current recommendation: "
                "No usable HAP recommendation"
            )

        if report.next_transition_utc is not None:

            transition_frequency = (
                format_frequency(
                    report.next_transition_frequency_mhz
                )
            )

            lines.append(
                f"Next transition: "
                f"{report.next_transition_utc:02d} UTC -> "
                f"{report.next_transition_band} "
                f"({transition_frequency})"
            )

        lines.append("")

        lines.append(
            "Regional HAP:"
        )

        for regional_line in format_regional_distribution(
            report.regional_distribution
        ):

            lines.append(
                f"  {regional_line}"
            )

    else:

        lines.append(
            "HAP unavailable."
        )

        if report.data_status.hap_error:

            lines.append(
                f"Error: "
                f"{report.data_status.hap_error}"
            )

    lines.append("")

    # -----------------------------------------------------------------------
    # GRAFEX
    # -----------------------------------------------------------------------

    if report.rx_location:

        lines.append(
            "GRAFEX PATH PREDICTION"
        )

        lines.append(
            "-" * 60
        )

        if report.data_status.grafex_available:

            if report.grafex:

                for line in format_grafex_summary(
                    report.grafex
                ):

                    lines.append(
                        f"  {line}"
                    )

        else:

            lines.append(
                "  GRAFEX unavailable."
            )

            if report.data_status.grafex_error:

                lines.append(
                    f"  Reason: "
                    f"{report.data_status.grafex_error}"
                )

        lines.append("")

    # -----------------------------------------------------------------------
    # IONOSPHERE
    # -----------------------------------------------------------------------

    lines.append(
        "IONOSPHERE"
    )

    lines.append(
        "-" * 60
    )

    if report.data_status.ionosphere_available:

        for line in format_ionosphere(
            report.ionosphere_observations
        ):

            lines.append(
                f"  {line}"
            )

    else:

        lines.append(
            "  Ionospheric observations unavailable."
        )

        if report.data_status.ionosphere_error:

            lines.append(
                f"  Error: "
                f"{report.data_status.ionosphere_error}"
            )

    lines.append("")

    # -----------------------------------------------------------------------
    # SPACE WEATHER
    # -----------------------------------------------------------------------

    lines.append(
        "SPACE WEATHER"
    )

    lines.append(
        "-" * 60
    )

    if report.data_status.space_weather_available:

        for line in format_space_weather(
            report.space_weather
        ):

            lines.append(
                f"  {line}"
            )

        alerts = report.space_weather.get(
            "active_alerts",
            [],
        )

        warnings = report.space_weather.get(
            "active_warnings",
            [],
        )

        formatted_alerts = (
            format_space_weather_alerts(
                alerts
            )
        )

        if formatted_alerts:

            lines.append("")

            lines.append(
                "  Current watches/alerts:"
            )

            for alert_text in formatted_alerts:

                lines.append(
                    f"    • {alert_text}"
                )

        formatted_warnings = (
            format_space_weather_alerts(
                warnings
            )
        )

        if formatted_warnings:

            lines.append("")

            lines.append(
                "  Current warnings:"
            )

            for warning_text in formatted_warnings:

                lines.append(
                    f"    • {warning_text}"
                )

    else:

        lines.append(
            "  Space weather unavailable."
        )

        if report.data_status.space_weather_error:

            lines.append(
                f"  Error: "
                f"{report.data_status.space_weather_error}"
            )

    lines.append("")

    # -----------------------------------------------------------------------
    # DATA STATUS
    # -----------------------------------------------------------------------

    lines.append(
        "DATA STATUS"
    )

    lines.append(
        "-" * 60
    )

    lines.append(
        f"  HAP: "
        f"{'OK' if report.data_status.hap_available else 'FAILED'}"
    )

    lines.append(
        f"  Ionosphere: "
        f"{'OK' if report.data_status.ionosphere_available else 'FAILED'}"
    )

    lines.append(
        f"  Space weather: "
        f"{'OK' if report.data_status.space_weather_available else 'FAILED'}"
    )

    if report.rx_location:

        lines.append(
            f"  GRAFEX: "
            f"{'OK' if report.data_status.grafex_available else 'FAILED'}"
        )

    if report.space_weather.get(
        "retrieved_utc"
    ):

        lines.append(
            f"  Space weather retrieved: "
            f"{report.space_weather['retrieved_utc']}"
        )

    if report.space_weather.get(
        "t_index_updated_utc"
    ):

        lines.append(
            f"  SWS T-index updated: "
            f"{report.space_weather['t_index_updated_utc']}"
        )

    lines.append("")

    # -----------------------------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------------------------

    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "-" * 60
    )

    if report.current_band:

        lines.append(
            f"  HAP currently predicts "
            f"{report.current_band} for the "
            f"{report.location_name} base point."
        )

        if report.regional_distribution:

            dominant_band = max(
                report.regional_distribution,
                key=lambda band:
                    report.regional_distribution[band],
            )

            dominant_count = (
                report.regional_distribution[
                    dominant_band
                ]
            )

            total = sum(
                report.regional_distribution.values()
            )

            lines.append(
                f"  The surrounding HAP region is "
                f"predominantly {dominant_band} "
                f"({dominant_count}/{total} grid points)."
            )

        lines.append(
            "  HAP is the primary local/base-point "
            "propagation prediction; space weather and "
            "ionospheric observations provide supporting context."
        )

    else:

        lines.append(
            "  There is not enough HAP data to provide "
            "a current band recommendation."
        )

    # -----------------------------------------------------------------------
    # Path-specific interpretation
    # -----------------------------------------------------------------------

    if report.rx_location:

        lines.append("")

        if report.data_status.grafex_available:

            lines.append(
                "  GRAFEX provides the path-specific "
                "TX -> RX prediction for the requested route."
            )

            lines.append(
                "  GRAFEX predictions should be treated as "
                "model output rather than a guarantee that a "
                "contact will succeed."
            )

        else:

            lines.append(
                "  A path-specific GRAFEX prediction was requested "
                "but was not available."
            )

            lines.append(
                "  The HAP, ionosphere, and space-weather data above "
                "remain available as supporting information."
            )

    lines.append("")

    lines.append(
        "=" * 60
    )

    return "\n".join(
        lines
    )


# ---------------------------------------------------------------------------
# MAIN REPORT GENERATOR
# ---------------------------------------------------------------------------

def get_propagation_report(
    tx_location: str = LOCATION_NAME,
    rx_location: Optional[str] = None,
) -> PropagationReport:
    """
    Collect all propagation information and return a complete report.

    NORMAL REPORT
    -------------

        report = get_propagation_report()

    This produces the normal Nelson/base-point report.

    PATH-SPECIFIC REPORT
    --------------------

        report = get_propagation_report(
            tx_location="Nelson, New Zealand",
            rx_location="Sydney, Australia",
        )

    This produces:

        - HAP centred on the TX location
        - ionospheric observations
        - space weather
        - SWS T-index selection
        - GRAFEX TX -> RX prediction

    GRAFEX is optional.

    If rx_location is omitted, no GRAFEX request is made.

    If rx_location is supplied but GRAFEX fails, the rest of the report
    remains available.
    """

    # -----------------------------------------------------------------------
    # NORMALISE INPUT
    # -----------------------------------------------------------------------

    if isinstance(
        tx_location,
        str,
    ):
        tx_location = tx_location.strip()
    else:
        tx_location = LOCATION_NAME

    if not tx_location:
        tx_location = LOCATION_NAME

    if isinstance(
        rx_location,
        str,
    ):

        rx_location = rx_location.strip()

        if not rx_location:
            rx_location = None

    else:
        rx_location = None

    # -----------------------------------------------------------------------
    # TIME
    # -----------------------------------------------------------------------

    current_time = get_current_utc()

    current_hour = current_time.hour

    generated_utc = (
        current_time.isoformat()
    )

    generated_local = (
        format_local_time(
            current_time
        )
    )

    # -----------------------------------------------------------------------
    # RESOLVE TRANSMITTER LOCATION
    # -----------------------------------------------------------------------

    resolved_tx_name = tx_location

    tx_latitude = LOCATION_LATITUDE
    tx_longitude = LOCATION_LONGITUDE

    try:

        (
            resolved_tx_name,
            tx_latitude,
            tx_longitude,
        ) = resolve_location(
            tx_location
        )

    except Exception:

        # Preserve the configured Nelson fallback if Nelson was requested.
        #
        # For arbitrary locations, GRAFEX independently performs its own
        # path/location handling. HAP falls back to Nelson if the TX
        # location cannot be resolved.

        if (
            tx_location.casefold()
            not in {
                LOCATION_NAME.casefold(),
                "nelson, new zealand",
            }
        ):

            resolved_tx_name = tx_location

            tx_latitude = LOCATION_LATITUDE
            tx_longitude = LOCATION_LONGITUDE

    # -----------------------------------------------------------------------
    # STATUS
    # -----------------------------------------------------------------------

    status = DataStatus()

    # -----------------------------------------------------------------------
    # SPACE WEATHER
    # -----------------------------------------------------------------------

    weather = None

    try:

        weather = collect_space_weather()

        status.space_weather_available = True

        space_weather = (
            build_space_weather_report(
                weather
            )
        )

    except Exception as exc:

        space_weather = {}

        status.space_weather_available = False

        status.space_weather_error = (
            f"{type(exc).__name__}: {exc}"
        )

    # -----------------------------------------------------------------------
    # SHARED T-INDEX SELECTION
    # -----------------------------------------------------------------------

    # Select the direct SWS T-index once for the requested path.
    #
    # This exact value is shared by both HAP and GRAFEX so that the two
    # propagation models are operating from the same ionospheric input.
    t_index = None

    if rx_location is not None:
        t_index = extract_grafex_t_index(
            tx_location=tx_location,
            rx_location=rx_location,
            weather=weather,
            weather_report=space_weather,
        )

    # -----------------------------------------------------------------------
    # IONOSPHERE
    # -----------------------------------------------------------------------

    try:

        ionosphere_raw = (
            get_all_ionosphere_observations()
        )

        status.ionosphere_available = True

        ionosphere_observations = (
            build_ionosphere_report(
                ionosphere_raw
            )
        )

    except Exception as exc:

        ionosphere_observations = []

        status.ionosphere_available = False

        status.ionosphere_error = (
            f"{type(exc).__name__}: {exc}"
        )

    # -----------------------------------------------------------------------
    # HAP
    # -----------------------------------------------------------------------

    current_band = None
    current_frequency_mhz = None
    current_support = None

    next_transition_utc = None
    next_transition_band = None
    next_transition_frequency_mhz = None

    regional_distribution = {}
    hap_forecast = {}
    hap_band_forecast = {}

    try:

        (
            config,
            decoded_hap,
        ) = collect_hap(
            current_time=current_time,
            location_name=resolved_tx_name,
            latitude=tx_latitude,
            longitude=tx_longitude,
            t_index=t_index,
        )

        status.hap_available = True

        decoder = HAPDecoder(
            config
        )

        # Current UTC hour.
        get_hap_hour(
            decoder,
            decoded_hap,
            current_hour,
        )

        (
            current_band,
            current_frequency_mhz,
            current_support,
        ) = extract_hap_base_recommendation(
            decoder,
            decoded_hap,
            current_hour,
        )

        regional_distribution = (
            extract_regional_distribution(
                decoder,
                decoded_hap,
                current_hour,
            )
        )

        (
            next_transition_utc,
            next_transition_band,
            next_transition_frequency_mhz,
        ) = find_next_hap_transition(
            decoder,
            decoded_hap,
            current_hour,
        )
        for hour in range(24):

            recommendation = (
                decoder.get_base_recommendation(
                    decoded=decoded_hap,
                    hour_utc=hour,
                )
            )

            if recommendation is None:
                hap_forecast[hour] = {
                    "band": None,
                    "frequency_mhz": None,
                    "support": None,
                }

                continue

            hap_forecast[hour] = {
                "band": recommendation.band,
                "frequency_mhz": (
                    recommendation.frequency_khz / 1000.0
                ),
                "support": getattr(
                    recommendation,
                    "sample_support",
                    None,
                ),
            }

        # ---------------------------------------------------------------
        # Independent one-frequency HAP predictions
        # ---------------------------------------------------------------
        #
        # The universal HAP above answers:
        #
        #     "Which supplied frequency does SWS recommend?"
        #
        # These requests independently test each band:
        #
        #     "Does SWS HAP support this frequency?"
        #
        # They are intentionally kept separate.
        try:

            hap_band_forecast = collect_independent_hap(
                config=config,
                current_time=current_time,
            )

        except Exception as exc:

            hap_band_forecast = {
                band: {
                    24: {
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        )
                    }
                }
                for band in (
                    "160m",
                    "80m",
                    "40m",
                    "30m",
                    "20m",
                    "17m",
                    "15m",
                    "12m",
                    "10m",
                )
            }

    except Exception as exc:

        status.hap_available = False

        status.hap_error = (
            f"{type(exc).__name__}: {exc}"
        )

    # -----------------------------------------------------------------------
    # GRAFEX
    # -----------------------------------------------------------------------

    grafex = None

    if rx_location is not None:

        prediction_date = (
            current_time.date()
        )

        (
            grafex,
            grafex_error,
        ) = run_grafex_path_prediction(
            tx_location=tx_location,
            rx_location=rx_location,
            prediction_date=prediction_date,
            t_index=t_index,
        )

        if grafex is not None:

            status.grafex_available = True

        else:

            status.grafex_available = False

            status.grafex_error = (
                grafex_error
            )

    # -----------------------------------------------------------------------
    # BUILD REPORT
    # -----------------------------------------------------------------------

    report = PropagationReport(
        # TX/base point
        location_name=resolved_tx_name,
        latitude=tx_latitude,
        longitude=tx_longitude,

        # Path
        tx_location=tx_location,
        rx_location=rx_location,

        # Timing
        generated_utc=generated_utc,
        generated_local=generated_local,
        current_utc_hour=current_hour,

        # HAP
        current_band=current_band,
        current_frequency_mhz=current_frequency_mhz,
        current_support=current_support,

        next_transition_utc=next_transition_utc,
        next_transition_band=next_transition_band,
        next_transition_frequency_mhz=(
            next_transition_frequency_mhz
        ),

        regional_distribution=(
            regional_distribution
        ),

        hap_forecast=(
            hap_forecast
        ),

        hap_band_forecast=(
            hap_band_forecast
        ),

        # Supporting data
        ionosphere_observations=(
            ionosphere_observations
        ),

        space_weather=space_weather,

        # GRAFEX
        grafex=grafex,

        # Status
        data_status=status,

        # Filled immediately below
        text="",
    )

    # Generate the human-readable representation after the complete
    # report exists.
    report.text = build_report_text(
        report
    )

    return report


# ---------------------------------------------------------------------------
# TERMINAL DISPLAY
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point.

    By default this tests the normal non-GRAFEX report.

    To test a path-specific report, change the call to:

        get_propagation_report(
            tx_location="Nelson, New Zealand",
            rx_location="Sydney, Australia",
        )
    """

    report = get_propagation_report()

    print(
        "\n" + report.text
    )


# ---------------------------------------------------------------------------
# DIRECT EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main() 
