"""
Real-time ionospheric conditions from the Australian Space Weather
Services MUF Report.

Source:
https://www.sws.bom.gov.au/HF_Systems/1/2/4

The SWS MUF Report is generated as HTML and contains current
station-level MUF conditions based on autoscaled ionosonde data.

Important:
The report does NOT provide the absolute MUF value for each station.
It provides a condition relative to the monthly predicted MUF, e.g.:

    near normal
    enhanced by 20%
    depressed by -17%
    no vertical MUF data
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests


SWS_MUF_URL = "https://www.sws.bom.gov.au/HF_Systems/1/2/4"

REQUEST_TIMEOUT = 20


@dataclass
class IonosphereObservation:
    station: str
    station_name: str

    # Time that SWS says the report was generated/updated.
    timestamp_utc: Optional[datetime]

    # Condition reported by SWS.
    condition: str

    # Percentage departure from predicted MUF, if SWS gives one.
    # Example:
    #   "enhanced by 30%" -> +30
    #   "depressed by -17%" -> -17
    percent_difference: Optional[float]

    # Whether SWS currently has usable vertical MUF data.
    data_available: bool

    source: str = SWS_MUF_URL


STATIONS = {
    "brisbane": "Brisbane",
    "canberra": "Canberra",
    "cocos": "Cocos Is",
    "darwin": "Darwin",
    "hobart": "Hobart",
    "learmonth": "Learmonth",
    "niue": "Niue Is",
    "norfolk": "Norfolk Is",
    "perth": "Perth",
    "sydney": "Sydney",
    "townsville": "Townsville",

    # Antarctic stations
    "casey": "Casey",
    "davis": "Davis",
    "mawson": "Mawson",
}


def _parse_report_timestamp(text: str) -> Optional[datetime]:
    """
    Parse:

        (last updated 10 Sep 2026 19:01 UT)

    into a timezone-aware UTC datetime.
    """

    pattern = (
        r"last updated\s+"
        r"(\d{1,2})\s+"
        r"([A-Za-z]{3})\s+"
        r"(\d{4})\s+"
        r"(\d{2}):(\d{2})\s+UT"
    )

    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        return None

    day = int(match.group(1))
    month_text = match.group(2)
    year = int(match.group(3))
    hour = int(match.group(4))
    minute = int(match.group(5))

    try:
        timestamp = datetime.strptime(
            f"{day} {month_text} {year} {hour}:{minute}",
            "%d %b %Y %H:%M",
        )

        return timestamp.replace(tzinfo=timezone.utc)

    except ValueError:
        return None


def _parse_station_condition(
    text: str,
    station_name: str,
) -> tuple[str, Optional[float], bool]:
    """
    Extract the MUF condition for one station.

    Examples:

        Norfolk Is : near normal

        Townsville : depressed by -17%

        Casey : enhanced by 30%

        Hobart : no vertical MUF data.
    """

    # Escape the station name because some contain spaces.
    pattern = (
        rf"{re.escape(station_name)}\s*:\s*"
        r"([^\n\r]+)"
    )

    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        return (
            "not reported",
            None,
            False,
        )

    condition = match.group(1).strip()

    # Remove trailing punctuation.
    condition = condition.rstrip(". ")

    # Check whether this station has usable MUF data.
    data_available = (
        "no vertical muf data" not in condition.lower()
    )

    # Extract percentage if present.
    percent_match = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*%",
        condition,
    )

    percent_difference = None

    if percent_match:
        percent_difference = float(percent_match.group(1))

    return (
        condition,
        percent_difference,
        data_available,
    )


def fetch_muf_report() -> tuple[str, datetime]:
    """
    Download the current SWS MUF Report.

    Returns:
        (report_text, retrieval_timestamp_utc)
    """

    response = requests.get(
        SWS_MUF_URL,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": (
                "RadioPathwayTool/1.0 "
                "(HF propagation research)"
            )
        },
    )

    response.raise_for_status()

    retrieval_time = datetime.now(timezone.utc)

    return response.text, retrieval_time


def get_all_ionosphere_observations() -> dict[str, IonosphereObservation]:
    """
    Fetch and parse the current SWS MUF report.
    """

    html, retrieval_time = fetch_muf_report()

    # Strip HTML tags so we can parse the actual report text.
    text = re.sub(r"<[^>]+>", " ", html)

    # Decode the most common HTML entities.
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
    )

    # Collapse excessive whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Convert HTML line breaks into useful separators.
    text = re.sub(r"\s*\n\s*", "\n", text)

    report_timestamp = _parse_report_timestamp(text)

    results = {}

    for station_id, station_name in STATIONS.items():

        condition, percent_difference, data_available = (
            _parse_station_condition(
                text,
                station_name,
            )
        )

        results[station_id] = IonosphereObservation(
            station=station_id,
            station_name=station_name,
            timestamp_utc=report_timestamp,
            condition=condition,
            percent_difference=percent_difference,
            data_available=data_available,
        )

    return results


def print_report(
    observations: dict[str, IonosphereObservation],
) -> None:

    print("=" * 70)
    print("HF PROPAGATION IONOSPHERIC OBSERVATIONS")
    print("=" * 70)

    retrieval_time = datetime.now(timezone.utc)

    print()
    print(f"Retrieved: {retrieval_time}")

    print()
    print("SWS MUF REPORT")
    print("-" * 70)

    for observation in observations.values():

        timestamp = observation.timestamp_utc

        if timestamp:
            timestamp_text = timestamp.strftime(
                "%Y-%m-%d %H:%M UTC"
            )
        else:
            timestamp_text = "unknown"

        print(
            f"{observation.station_name:<15} "
            f"{observation.condition:<25} "
            f"report: {timestamp_text}"
        )


if __name__ == "__main__":

    try:

        observations = get_all_ionosphere_observations()

        print_report(observations)

    except requests.RequestException as exc:

        print("=" * 70)
        print("HF PROPAGATION IONOSPHERIC OBSERVATIONS")
        print("=" * 70)

        print()
        print("FAILED TO RETRIEVE SWS MUF REPORT")
        print("-" * 70)
        print(exc)
