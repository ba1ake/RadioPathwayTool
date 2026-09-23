"""
RadioPathwayTool - Main Propagation Engine

This module is the orchestration layer between the individual data sources
and the Discord bot.

Architecture:

    space_weather.py
            |
    ionosphere.py
            |
    band_favorability.py
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


Important design principles:

1. HAP is the primary HF propagation prediction.
2. Space weather is supporting context, not an arbitrary score.
3. Live ionospheric observations are supporting evidence.
4. No 0-100 band score is generated here.
5. main.py does not contain Discord-specific code.
6. Importing main.py does not automatically perform network requests.
7. Individual data sources can fail without necessarily killing the
   entire propagation report.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Optional


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


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

LOCATION_NAME = "Nelson"

LOCATION_LATITUDE = -41.27
LOCATION_LONGITUDE = 173.28

# The HAP system currently uses the nine frequencies supplied to the SWS
# prediction tool.
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

# How many degrees around the requested location should the HAP grid cover.
HAP_GRID_ROWS = 7
HAP_GRID_COLS = 7
HAP_GRID_STEP_LAT = 5.0
HAP_GRID_STEP_LON = 5.0


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

    hap_error: Optional[str] = None
    ionosphere_error: Optional[str] = None
    space_weather_error: Optional[str] = None


@dataclass
class PropagationReport:
    """
    Complete result returned by get_propagation_report().

    The Discord bot should consume this object rather than having to know
    anything about HAP decoding, SWS URLs, or the individual data modules.
    """

    location_name: str
    latitude: float
    longitude: float

    generated_utc: str
    generated_local: str

    current_utc_hour: int

    current_band: Optional[str]
    current_frequency_mhz: Optional[float]
    current_support: Optional[int]

    next_transition_utc: Optional[int]
    next_transition_band: Optional[str]
    next_transition_frequency_mhz: Optional[float]

    regional_distribution: dict[str, int]

    ionosphere_observations: list[dict[str, Any]]

    space_weather: dict[str, Any]

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

    We deliberately do not hard-code NZST/NZDT because the date determines
    whether New Zealand is currently observing daylight saving.
    """

    try:
        from zoneinfo import ZoneInfo

        nz_time = dt.astimezone(ZoneInfo("Pacific/Auckland"))

        return nz_time.strftime(
            "%d %b %Y %H:%M NZ"
        )

    except Exception:
        # Fallback if zoneinfo is unavailable.
        return dt.strftime("%d %b %Y %H:%M UTC")


# ---------------------------------------------------------------------------
# HAP HELPERS
# ---------------------------------------------------------------------------


def create_hap_config(current_time: datetime) -> HAPConfig:
    """
    Create the HAP configuration centred on Nelson.

    The SWS HAP system returns all 24 UTC hours, so the exact current hour
    does not need to be requested separately here.
    """

    return HAPCollector.create_centred_config(
        base_name=LOCATION_NAME,
        base_lat=LOCATION_LATITUDE,
        base_lon=LOCATION_LONGITUDE,
        nrows=HAP_GRID_ROWS,
        ncols=HAP_GRID_COLS,
        step_lat=HAP_GRID_STEP_LAT,
        step_lon=HAP_GRID_STEP_LON,
    )


def collect_hap(current_time: datetime) -> tuple[Any, Any]:
    """
    Collect and decode the complete HAP forecast.

    Returns:

        (config, decoded_hap)

    Raises an exception if the HAP system cannot be collected.
    """

    config = create_hap_config(current_time)

    collector = HAPCollector()

    hap_data = collector.collect(config)

    decoder = HAPDecoder(config)

    decoded_hap = decoder.decode_all(hap_data)

    return config, decoded_hap


def get_hap_hour(
    decoder: HAPDecoder,
    decoded_hap: Any,
    hour_utc: int,
) -> Any:
    """
    Obtain the HAP result for a particular UTC hour.

    The exact object returned by HAPDecoder is deliberately kept internal
    to this module.
    """

    # HAPDecoder.decode_all() produces the hourly results indexed by UTC
    # hour in the current implementation.
    return decoded_hap[hour_utc]


def extract_hap_base_recommendation(
    decoder: HAPDecoder,
    hour_result: Any,
) -> tuple[Optional[str], Optional[float], Optional[int]]:
    """
    Extract the HAP recommendation for the base/grid point.

    Returns:

        band
        frequency MHz
        support

    If the decoder has no usable result, all three values are None.
    """

    try:
        recommendation = decoder.get_base_recommendation(
            hour_result
        )

        if recommendation is None:
            return None, None, None

        band = getattr(recommendation, "band", None)

        frequency_khz = getattr(
            recommendation,
            "frequency_khz",
            None,
        )

        if frequency_khz is None:
            frequency_mhz = None
        else:
            frequency_mhz = frequency_khz / 1000.0

        support = getattr(
            recommendation,
            "support",
            None,
        )

        return band, frequency_mhz, support

    except Exception:
        return None, None, None


def extract_regional_distribution(
    decoder: HAPDecoder,
    hour_result: Any,
) -> dict[str, int]:
    """
    Extract the number of HAP grid points assigned to each band.

    The values are counts, not percentages and not probability estimates.
    """

    try:
        distribution = decoder.get_regional_distribution(
            hour_result
        )

        if distribution is None:
            return {}

        # Convert to a plain dictionary so the returned report is easy for
        # Discord, JSON, logging, etc. to consume.
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
) -> tuple[Optional[int], Optional[str], Optional[float]]:
    """
    Find the next UTC hour at which the HAP recommendation at the Nelson
    base point changes.

    Searches forward through the next 23 hours.

    Returns:

        transition_hour
        band
        frequency MHz
    """

    try:
        current_result = get_hap_hour(
            decoder,
            decoded_hap,
            current_hour,
        )

        current_band, _, _ = extract_hap_base_recommendation(
            decoder,
            current_result,
        )

        if current_band is None:
            return None, None, None

        for offset in range(1, 24):

            hour = (current_hour + offset) % 24

            result = get_hap_hour(
                decoder,
                decoded_hap,
                hour,
            )

            band, frequency_mhz, _ = (
                extract_hap_base_recommendation(
                    decoder,
                    result,
                )
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


def build_space_weather_report(weather: Any) -> dict[str, Any]:
    """
    Convert SpaceWeatherData into a simple serialisable dictionary.

    This intentionally exposes the measurements rather than converting them
    into a propagation score.
    """

    return {
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
    }


# ---------------------------------------------------------------------------
# IONOSPHERE HELPERS
# ---------------------------------------------------------------------------


def build_ionosphere_report(
    observations: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert ionospheric observations into simple dictionaries suitable for
    Discord/JSON.

    Stations with no useful data are retained so the user can see that the
    station was checked but did not provide a usable observation.
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
        return ["No regional HAP distribution available."]

    total = sum(distribution.values())

    if total <= 0:
        return ["No regional HAP distribution available."]

    # Sort by number of grid points, largest first.
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
        return ["No ionospheric observations available."]

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
            observation.get("station", "Unknown"),
        )

        condition = observation.get(
            "condition_display",
            "unknown",
        )

        lines.append(
            f"{name}: {condition}"
        )

    # Only mention missing stations if there are no usable observations.
    # Otherwise the Discord message can become unnecessarily large.
    if not usable:
        for observation in unusable:

            name = observation.get(
                "station_name",
                observation.get("station", "Unknown"),
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

    return lines


def build_report_text(
    report: PropagationReport,
) -> str:
    """
    Create a compact human-readable report.

    This is deliberately plain text rather than Discord-specific Markdown.
    Discord can use it directly now and format it further later.
    """

    lines = []

    lines.append("=" * 60)
    lines.append("RADIOPATHWAYTOOL")
    lines.append("=" * 60)

    lines.append(
        f"Location: {report.location_name}"
    )

    lines.append(
        f"Coordinates: "
        f"{report.latitude:.2f}, "
        f"{report.longitude:.2f}"
    )

    lines.append(
        f"Generated: {report.generated_local}"
    )

    lines.append("")

    # -----------------------------------------------------------------------
    # HAP
    # -----------------------------------------------------------------------

    lines.append("HAP")
    lines.append("-" * 60)

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
                f"{report.next_transition_utc:02d} UTC → "
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
                f"Error: {report.data_status.hap_error}"
            )

    lines.append("")

    # -----------------------------------------------------------------------
    # IONOSPHERE
    # -----------------------------------------------------------------------

    lines.append("IONOSPHERE")
    lines.append("-" * 60)

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

    lines.append("SPACE WEATHER")
    lines.append("-" * 60)

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

        if alerts:
            lines.append("")
            lines.append("  Active alerts:")

            for alert in alerts:
                lines.append(
                    f"    - {alert}"
                )

        if warnings:
            lines.append("")
            lines.append("  Active warnings:")

            for warning in warnings:
                lines.append(
                    f"    - {warning}"
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

    lines.append("DATA STATUS")
    lines.append("-" * 60)

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

    if report.space_weather.get("retrieved_utc"):
        lines.append(
            f"  Space weather retrieved: "
            f"{report.space_weather['retrieved_utc']}"
        )

    lines.append("")

    # -----------------------------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------------------------

    lines.append("INTERPRETATION")
    lines.append("-" * 60)

    if report.current_band:

        lines.append(
            f"  HAP currently predicts "
            f"{report.current_band} for the "
            f"{report.location_name} base point."
        )

        if report.regional_distribution:

            dominant_band = max(
                report.regional_distribution,
                key=report.regional_distribution.get,
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
            "  HAP is the primary propagation prediction; "
            "space weather and ionospheric observations "
            "provide supporting context."
        )

    else:

        lines.append(
            "  There is not enough HAP data to provide "
            "a current band recommendation."
        )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN REPORT GENERATOR
# ---------------------------------------------------------------------------


def get_propagation_report() -> PropagationReport:
    """
    Collect all propagation information and return a complete report.

    This is the main function that discord_bot.py should call.

    Example:

        from main import get_propagation_report

        report = get_propagation_report()

        print(report.text)

    The function attempts to keep working even if one data source fails.
    """

    # -----------------------------------------------------------------------
    # TIME
    # -----------------------------------------------------------------------

    current_time = get_current_utc()

    current_hour = current_time.hour

    generated_utc = current_time.isoformat()

    generated_local = format_local_time(
        current_time
    )

    # -----------------------------------------------------------------------
    # STATUS
    # -----------------------------------------------------------------------

    status = DataStatus()

    # -----------------------------------------------------------------------
    # SPACE WEATHER
    # -----------------------------------------------------------------------

    try:

        weather = collect_space_weather()

        status.space_weather_available = True

        space_weather = build_space_weather_report(
            weather
        )

    except Exception as exc:

        weather = None

        space_weather = {}

        status.space_weather_available = False

        status.space_weather_error = (
            f"{type(exc).__name__}: {exc}"
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

        ionosphere_raw = {}

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

    try:

        config, decoded_hap = collect_hap(
            current_time
        )

        status.hap_available = True

        decoder = HAPDecoder(config)

        # Current UTC hour.
        current_hour_result = get_hap_hour(
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
            current_hour_result,
        )

        regional_distribution = (
            extract_regional_distribution(
                decoder,
                current_hour_result,
            )
        )

        # Look for the next change in the base-point recommendation.
        (
            next_transition_utc,
            next_transition_band,
            next_transition_frequency_mhz,
        ) = find_next_hap_transition(
            decoder,
            decoded_hap,
            current_hour,
        )

    except Exception as exc:

        status.hap_available = False

        status.hap_error = (
            f"{type(exc).__name__}: {exc}"
        )

    # -----------------------------------------------------------------------
    # BUILD REPORT
    # -----------------------------------------------------------------------

    report = PropagationReport(
        location_name=LOCATION_NAME,

        latitude=LOCATION_LATITUDE,

        longitude=LOCATION_LONGITUDE,

        generated_utc=generated_utc,

        generated_local=generated_local,

        current_utc_hour=current_hour,

        current_band=current_band,

        current_frequency_mhz=current_frequency_mhz,

        current_support=current_support,

        next_transition_utc=next_transition_utc,

        next_transition_band=next_transition_band,

        next_transition_frequency_mhz=(
            next_transition_frequency_mhz
        ),

        regional_distribution=regional_distribution,

        ionosphere_observations=(
            ionosphere_observations
        ),

        space_weather=space_weather,

        data_status=status,

        text="",
    )

    # Generate the human-readable representation after the complete report
    # exists. This means Discord can use either report.text or the individual
    # structured fields.
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

    This is only for testing/development. The Discord bot should normally
    call get_propagation_report() directly.
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