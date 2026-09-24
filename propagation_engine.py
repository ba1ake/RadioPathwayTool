from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import json
import math
import re
import time


# ============================================================================
# CONFIGURATION
# ============================================================================

GRAFEX_URL = (
    "https://www.sws.bom.gov.au/"
    "local-cgi-bin/grafexj-cgi.tcl"
)

# GRAFEX provides predictions from 1 MHz to 40 MHz.
MIN_FREQUENCY_MHZ = 1
MAX_FREQUENCY_MHZ = 40


# ============================================================================
# LOCATION / GEOCODING CONFIGURATION
# ============================================================================

NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/search"
)

NOMINATIM_USER_AGENT = (
    "RadioPathwayTool/1.0 "
    "(amateur radio propagation analysis)"
)

NOMINATIM_MIN_REQUEST_INTERVAL = 1.1


# ============================================================================
# GRAFEX SYMBOLS
# ============================================================================

GRAFEX_SYMBOL_MEANINGS = {
    ".": "usable less than 50% of days",
    "%": "usable 50% to 90% of days",
    "B": "both E and F modes 90% of days",
    "M": "mixed first and second F modes",
    "F": "first F mode only",
    "E": "E-layer propagation",
    "P": "90% E and 50-90% F",
    "S": "second modes only",
    "A": "high absorption",
    "X": "complex modes",
}


# ============================================================================
# COMMON AMATEUR RADIO FREQUENCIES
# ============================================================================

# Lookup frequencies only.
# GRAFEX itself works in 1 MHz steps.
BAND_FREQUENCIES_MHZ = {
    "160m": 1.838,
    "80m": 3.650,
    "40m": 7.150,
    "30m": 10.125,
    "20m": 14.175,
    "17m": 18.118,
    "15m": 21.225,
    "12m": 24.940,
    "10m": 28.850,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ResolvedLocation:
    """
    A human-readable location resolved into geographic coordinates.
    """

    query: str
    display_name: str
    latitude: float
    longitude: float


@dataclass
class GrafexFrequencyPrediction:
    """
    Prediction for one frequency during one UTC hour.
    """

    frequency_mhz: float
    symbol: str
    meaning: str
    supported: bool | None


@dataclass
class GrafexHour:
    """
    One hour of GRAFEX prediction.
    """

    utc_hour: int

    first_owf_mhz: float | None
    first_emuf_mhz: float | None
    first_alf_mhz: float | None

    second_owf_mhz: float | None
    second_emuf_mhz: float | None
    second_alf_mhz: float | None

    frequencies: dict[
        float,
        GrafexFrequencyPrediction
    ]


@dataclass
class GrafexPrediction:
    """
    Complete GRAFEX prediction for a transmitter/receiver path.
    """

    tx_name: str
    tx_latitude: float
    tx_longitude: float

    rx_name: str
    rx_latitude: float
    rx_longitude: float

    prediction_date: str
    t_index: float

    distance_km: float | None
    bearing_tx_to_rx: float | None
    bearing_rx_to_tx: float | None

    first_mode: str | None
    second_mode: str | None

    hours: list[GrafexHour]

    source: str
    retrieved_utc: str

    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# GEOMETRY
# ============================================================================

def calculate_distance_and_bearing(
    tx_lat: float,
    tx_lon: float,
    rx_lat: float,
    rx_lon: float,
) -> tuple[float, float, float]:

    earth_radius_km = 6371.0088

    lat1 = math.radians(tx_lat)
    lat2 = math.radians(rx_lat)

    delta_lat = math.radians(rx_lat - tx_lat)
    delta_lon = math.radians(rx_lon - tx_lon)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    distance_km = earth_radius_km * c

    # TX -> RX
    y = (
        math.sin(delta_lon)
        * math.cos(lat2)
    )

    x = (
        math.cos(lat1)
        * math.sin(lat2)
        - math.sin(lat1)
        * math.cos(lat2)
        * math.cos(delta_lon)
    )

    bearing_tx_to_rx = math.degrees(
        math.atan2(y, x)
    )

    bearing_tx_to_rx = (
        bearing_tx_to_rx + 360
    ) % 360

    # RX -> TX
    reverse_delta_lon = math.radians(
        tx_lon - rx_lon
    )

    y_reverse = (
        math.sin(reverse_delta_lon)
        * math.cos(lat1)
    )

    x_reverse = (
        math.cos(lat2)
        * math.sin(lat1)
        - math.sin(lat2)
        * math.cos(lat1)
        * math.cos(reverse_delta_lon)
    )

    bearing_rx_to_tx = math.degrees(
        math.atan2(
            y_reverse,
            x_reverse,
        )
    )

    bearing_rx_to_tx = (
        bearing_rx_to_tx + 360
    ) % 360

    return (
        distance_km,
        bearing_tx_to_rx,
        bearing_rx_to_tx,
    )


# ============================================================================
# LOCATION RESOLVER
# ============================================================================

class LocationResolver:
    """
    Resolve human-readable location names into latitude/longitude.
    """

    def __init__(
        self,
        timeout_seconds: int = 15,
    ) -> None:

        self.timeout_seconds = timeout_seconds

        self._cache: dict[
            str,
            ResolvedLocation
        ] = {}

        self._last_request_time = 0.0

    def _wait_for_rate_limit(self) -> None:

        elapsed = (
            time.monotonic()
            - self._last_request_time
        )

        if elapsed < NOMINATIM_MIN_REQUEST_INTERVAL:
            time.sleep(
                NOMINATIM_MIN_REQUEST_INTERVAL
                - elapsed
            )

    def resolve(
        self,
        location_name: str,
    ) -> ResolvedLocation:

        location_name = location_name.strip()

        if not location_name:
            raise ValueError(
                "Location name cannot be empty."
            )

        cache_key = location_name.casefold()

        if cache_key in self._cache:
            return self._cache[cache_key]

        self._wait_for_rate_limit()

        params = {
            "q": location_name,
            "format": "jsonv2",
            "limit": 1,
        }

        url = (
            f"{NOMINATIM_URL}"
            f"?{urlencode(params)}"
        )

        request = Request(
            url,
            headers={
                "User-Agent": NOMINATIM_USER_AGENT,
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                data = response.read()

        except HTTPError as exc:
            raise RuntimeError(
                f"Location lookup HTTP error "
                f"{exc.code}: {exc.reason}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                "Location lookup connection error: "
                f"{exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise RuntimeError(
                "Location lookup timed out."
            ) from exc

        finally:
            self._last_request_time = (
                time.monotonic()
            )

        try:
            results = json.loads(
                data.decode(
                    "utf-8",
                    errors="replace",
                )
            )

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Location lookup returned invalid JSON."
            ) from exc

        if not results:
            raise ValueError(
                f"Could not find location: "
                f"'{location_name}'"
            )

        result = results[0]

        try:
            latitude = float(result["lat"])
            longitude = float(result["lon"])

            display_name = str(
                result.get(
                    "display_name",
                    location_name,
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                "Location lookup returned "
                f"an invalid result for "
                f"'{location_name}'."
            ) from exc

        resolved = ResolvedLocation(
            query=location_name,
            display_name=display_name,
            latitude=latitude,
            longitude=longitude,
        )

        self._cache[cache_key] = resolved

        return resolved


# ============================================================================
# GRAFEX HTTP CLIENT
# ============================================================================

class GrafexClient:
    """
    Client for the SWS/BOM GRAFEX CGI endpoint.
    """

    def __init__(
        self,
        timeout_seconds: int = 20,
    ) -> None:

        self.timeout_seconds = timeout_seconds

    def build_url(
        self,
        tx_name: str,
        tx_lat: float,
        tx_lon: float,
        rx_name: str,
        rx_lat: float,
        rx_lon: float,
        prediction_date: date,
        t_index: float,
    ) -> str:

        params = {
            "txname": tx_name,
            "txlat": f"{tx_lat:.4f}",
            "txlng": f"{tx_lon:.4f}",
            "rxname": rx_name,
            "rxlat": f"{rx_lat:.4f}",
            "rxlng": f"{rx_lon:.4f}",
            "year": prediction_date.year,
            "month": prediction_date.month,
            "day": prediction_date.day,
            "tindex": t_index,
        }

        return (
            f"{GRAFEX_URL}"
            f"?{urlencode(params)}"
        )

    def fetch(
        self,
        tx_name: str,
        tx_lat: float,
        tx_lon: float,
        rx_name: str,
        rx_lat: float,
        rx_lon: float,
        prediction_date: date,
        t_index: float,
    ) -> str:

        url = self.build_url(
            tx_name=tx_name,
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            rx_name=rx_name,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            prediction_date=prediction_date,
            t_index=t_index,
        )

        request = Request(
            url,
            headers={
                "User-Agent":
                    "RadioPathwayTool/1.0",
                "Accept":
                    "text/html,text/plain,*/*",
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:

                data = response.read()

        except HTTPError as exc:

            raise RuntimeError(
                f"GRAFEX HTTP error "
                f"{exc.code}: {exc.reason}"
            ) from exc

        except URLError as exc:

            raise RuntimeError(
                "GRAFEX connection error: "
                f"{exc.reason}"
            ) from exc

        except TimeoutError as exc:

            raise RuntimeError(
                "GRAFEX request timed out."
            ) from exc

        return data.decode(
            "utf-8",
            errors="replace",
        )


# ============================================================================
# PARSER
# ============================================================================

class GrafexParser:
    """
    Parser for SWS/BOM grafexj-cgi.tcl output.

    GRAFEX output has historically appeared in slightly different
    whitespace/HTML representations, so the parser deliberately
    normalises the response before parsing.
    """

    @classmethod
    def clean_html(
        cls,
        raw_text: str,
    ) -> str:

        text = raw_text

        # Convert common HTML line-break/table delimiters.
        text = re.sub(
            r"<\s*(?:br|/tr|/p|/pre|/div|/td|/th)\s*>",
            "\n",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"<\s*(?:tr|p|pre|div|td|th)[^>]*>",
            "\n",
            text,
            flags=re.IGNORECASE,
        )

        # Remove remaining tags.
        text = re.sub(
            r"<[^>]+>",
            " ",
            text,
        )

        # Decode common HTML entities manually.
        replacements = {
            "&nbsp;": " ",
            "&gt;": ">",
            "&lt;": "<",
            "&amp;": "&",
            "&#39;": "'",
            "&quot;": '"',
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    @classmethod
    def normalise_text(
        cls,
        raw_text: str,
    ) -> str:

        text = cls.clean_html(raw_text)

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Replace tabs with spaces.
        text = text.replace("\t", " ")

        # Collapse repeated spaces but preserve line boundaries.
        text = re.sub(
            r"[ ]{2,}",
            " ",
            text,
        )

        return text

    @classmethod
    def parse_float(
        cls,
        value: str | None,
    ) -> float | None:

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def normalise_symbol_block(
        cls,
        symbols: str,
    ) -> str:

        # Remove spaces and any accidental HTML artefacts.
        return re.sub(
            r"[^.%BFMEPSAX]",
            "",
            symbols.upper(),
        )

    @classmethod
    def parse_frequency_predictions(
        cls,
        symbol_block: str,
    ) -> dict[
        float,
        GrafexFrequencyPrediction
    ]:

        symbols = cls.normalise_symbol_block(
            symbol_block
        )

        predictions: dict[
            float,
            GrafexFrequencyPrediction
        ] = {}

        for index, symbol in enumerate(
            symbols[:40]
        ):

            frequency = float(index + 1)

            meaning = (
                GRAFEX_SYMBOL_MEANINGS.get(
                    symbol,
                    "unknown GRAFEX symbol",
                )
            )

            if symbol in {".", "A"}:
                supported: bool | None = False

            elif symbol in {
                "F",
                "B",
                "P",
                "E",
                "M",
                "S",
                "X",
                "%",
            }:
                supported = True

            else:
                supported = None

            predictions[frequency] = (
                GrafexFrequencyPrediction(
                    frequency_mhz=frequency,
                    symbol=symbol,
                    meaning=meaning,
                    supported=supported,
                )
            )

        return predictions

    @classmethod
    def _looks_like_symbol_block(
        cls,
        value: str,
    ) -> bool:

        cleaned = re.sub(
            r"\s+",
            "",
            value.upper(),
        )

        if not cleaned:
            return False

        if len(cleaned) < 10:
            return False

        allowed = set(
            ".%BFMEPSAX"
        )

        return (
            all(
                char in allowed
                for char in cleaned
            )
            and len(cleaned) >= 20
        )

    @classmethod
    def _parse_hour_line(
        cls,
        line: str,
    ) -> GrafexHour | None:

        """
        Parse a GRAFEX hourly line.

        Expected logical structure:

            hour
            first OWF
            first EMUF
            first ALF
            symbol block
            second OWF
            second EMUF
            second ALF
            ending hour

        The exact spacing is intentionally not strict.
        """

        stripped = line.strip()

        if not stripped:
            return None

        # Remove leading '*' characters used by some GRAFEX
        # representations.
        stripped = stripped.strip("* ")

        fields = stripped.split()

        if len(fields) < 9:
            return None

        # Search for a two-digit hour.
        hour_index = None

        for index, field in enumerate(fields):

            if re.fullmatch(
                r"\d{2}",
                field,
            ):

                candidate = int(field)

                if 0 <= candidate <= 23:
                    hour_index = index
                    break

        if hour_index is None:
            return None

        remaining = fields[hour_index:]

        if len(remaining) < 9:
            return None

        hour_text = remaining[0]

        try:
            hour = int(hour_text)
        except ValueError:
            return None

        # Try the canonical layout first.
        #
        # hour, OWF, EMUF, ALF, symbols,
        # OWF, EMUF, ALF, end_hour

        if len(remaining) >= 9:

            first_owf = cls.parse_float(
                remaining[1]
            )

            first_emuf = cls.parse_float(
                remaining[2]
            )

            first_alf = cls.parse_float(
                remaining[3]
            )

            # The symbol block can itself contain spaces.
            # Find the first numeric field after it.
            symbol_end = None

            for index in range(4, len(remaining)):

                if re.fullmatch(
                    r"-?\d+(?:\.\d+)?",
                    remaining[index],
                ):

                    if index >= 5:
                        symbol_end = index
                        break

            if symbol_end is not None:

                symbols = " ".join(
                    remaining[4:symbol_end]
                )

                if cls._looks_like_symbol_block(
                    symbols
                ):

                    numeric_after = remaining[
                        symbol_end:
                    ]

                    if len(numeric_after) >= 4:

                        second_owf = cls.parse_float(
                            numeric_after[0]
                        )

                        second_emuf = cls.parse_float(
                            numeric_after[1]
                        )

                        second_alf = cls.parse_float(
                            numeric_after[2]
                        )

                        end_hour_text = numeric_after[3]

                        if re.fullmatch(
                            r"\d{2}",
                            end_hour_text,
                        ):

                            end_hour = int(
                                end_hour_text
                            )

                            if end_hour == hour:

                                frequency_predictions = (
                                    cls.parse_frequency_predictions(
                                        symbols
                                    )
                                )

                                if frequency_predictions:

                                    return GrafexHour(
                                        utc_hour=hour,

                                        first_owf_mhz=(
                                            first_owf
                                        ),

                                        first_emuf_mhz=(
                                            first_emuf
                                        ),

                                        first_alf_mhz=(
                                            first_alf
                                        ),

                                        second_owf_mhz=(
                                            second_owf
                                        ),

                                        second_emuf_mhz=(
                                            second_emuf
                                        ),

                                        second_alf_mhz=(
                                            second_alf
                                        ),

                                        frequencies=(
                                            frequency_predictions
                                        ),
                                    )

        return None

    @classmethod
    def _extract_hour_rows(
        cls,
        lines: list[str],
    ) -> list[GrafexHour]:

        hours: list[GrafexHour] = []

        seen_hours: set[int] = set()

        for line in lines:

            hour = cls._parse_hour_line(
                line
            )

            if hour is None:
                continue

            if hour.utc_hour in seen_hours:
                continue

            seen_hours.add(
                hour.utc_hour
            )

            hours.append(hour)

        hours.sort(
            key=lambda item: item.utc_hour
        )

        return hours

    @classmethod
    def _extract_distance(
        cls,
        text: str,
    ) -> float | None:

        patterns = [
            r"Distance\s*:\s*(-?\d+(?:\.\d+)?)\s*km",
            r"Distance\s+(-?\d+(?:\.\d+)?)\s*km",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                return float(
                    match.group(1)
                )

        return None

    @classmethod
    def _extract_bearings(
        cls,
        text: str,
    ) -> tuple[float | None, float | None]:

        patterns = [
            (
                r"Bearings?\s*:\s*"
                r"(-?\d+(?:\.\d+)?)\s+"
                r"(-?\d+(?:\.\d+)?)"
            ),
            (
                r"Bearings?\s+"
                r"(-?\d+(?:\.\d+)?)\s+"
                r"(-?\d+(?:\.\d+)?)"
            ),
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:

                return (
                    float(match.group(1)),
                    float(match.group(2)),
                )

        return None, None

    @classmethod
    def _extract_t_index(
        cls,
        text: str,
    ) -> float | None:

        patterns = [
            r"T[- ]*index\s*:\s*(-?\d+(?:\.\d+)?)",
            r"T\s*index\s*=\s*(-?\d+(?:\.\d+)?)",
            r"\bT\s*=\s*(-?\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                return float(
                    match.group(1)
                )

        return None

    @classmethod
    def _extract_modes(
        cls,
        lines: list[str],
    ) -> tuple[str | None, str | None]:

        first_mode = None
        second_mode = None

        for index, line in enumerate(lines):

            if "first mode" not in line.lower():
                continue

            # Search nearby lines rather than assuming the
            # data is immediately adjacent.
            candidates = lines[
                index + 1:index + 5
            ]

            for candidate in candidates:

                fields = candidate.strip().split()

                if len(fields) < 2:
                    continue

                # Remove decorative punctuation.
                cleaned = [
                    field.strip("*")
                    for field in fields
                ]

                if len(cleaned) >= 2:

                    first_mode = cleaned[0]
                    second_mode = cleaned[-1]

                    return (
                        first_mode,
                        second_mode,
                    )

        return (
            first_mode,
            second_mode,
        )

    @classmethod
    def parse(
        cls,
        raw_text: str,
        tx_name: str,
        tx_lat: float,
        tx_lon: float,
        rx_name: str,
        rx_lat: float,
        rx_lon: float,
        prediction_date: date,
        t_index: float,
        source: str = GRAFEX_URL,
    ) -> GrafexPrediction:

        if not raw_text or not raw_text.strip():

            raise ValueError(
                "GRAFEX returned an empty response."
            )

        normalised = cls.normalise_text(
            raw_text
        )

        lines = normalised.splitlines()

        hours = cls._extract_hour_rows(
            lines
        )

        if not hours:

            # Provide useful diagnostic information.
            preview = normalised.strip()

            if len(preview) > 6000:
                preview = preview[:6000] + (
                    "\n...[truncated]"
                )

            raise ValueError(
                "GRAFEX response contained no "
                "recognisable hourly prediction rows.\n\n"
                "GRAFEX response preview:\n"
                f"{preview}"
            )

        (
            distance_km,
            bearing_tx_to_rx,
            bearing_rx_to_tx,
        ) = calculate_distance_and_bearing(
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
        )

        extracted_distance = (
            cls._extract_distance(
                normalised
            )
        )

        if extracted_distance is not None:
            distance_km = extracted_distance

        (
            extracted_bearing_tx,
            extracted_bearing_rx,
        ) = cls._extract_bearings(
            normalised
        )

        if extracted_bearing_tx is not None:
            bearing_tx_to_rx = (
                extracted_bearing_tx
            )

        if extracted_bearing_rx is not None:
            bearing_rx_to_tx = (
                extracted_bearing_rx
            )

        extracted_t_index = (
            cls._extract_t_index(
                normalised
            )
        )

        if extracted_t_index is not None:
            t_index = extracted_t_index

        (
            first_mode,
            second_mode,
        ) = cls._extract_modes(
            lines
        )

        retrieved_utc = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        return GrafexPrediction(
            tx_name=tx_name,
            tx_latitude=tx_lat,
            tx_longitude=tx_lon,

            rx_name=rx_name,
            rx_latitude=rx_lat,
            rx_longitude=rx_lon,

            prediction_date=(
                prediction_date.isoformat()
            ),

            t_index=t_index,

            distance_km=distance_km,

            bearing_tx_to_rx=(
                bearing_tx_to_rx
            ),

            bearing_rx_to_tx=(
                bearing_rx_to_tx
            ),

            first_mode=first_mode,
            second_mode=second_mode,

            hours=hours,

            source=source,

            retrieved_utc=retrieved_utc,

            raw_text=raw_text,
        )


# ============================================================================
# HIGH-LEVEL PROPAGATION ENGINE
# ============================================================================

class PropagationEngine:
    """
    High-level interface for RadioPathwayTool.
    """

    def __init__(
        self,
        grafex_client: GrafexClient | None = None,
        location_resolver:
            LocationResolver | None = None,
    ) -> None:

        self.grafex = (
            grafex_client
            or GrafexClient()
        )

        self.location_resolver = (
            location_resolver
            or LocationResolver()
        )

    def calculate_path(
        self,
        tx_location: str,
        rx_location: str,
        prediction_date: date,
        t_index: float,
    ) -> GrafexPrediction:

        tx = self.location_resolver.resolve(
            tx_location
        )

        rx = self.location_resolver.resolve(
            rx_location
        )

        raw_text = self.grafex.fetch(
            tx_name=tx_location,
            tx_lat=tx.latitude,
            tx_lon=tx.longitude,

            rx_name=rx_location,
            rx_lat=rx.latitude,
            rx_lon=rx.longitude,

            prediction_date=prediction_date,

            t_index=t_index,
        )

        return GrafexParser.parse(
            raw_text=raw_text,

            tx_name=tx_location,
            tx_lat=tx.latitude,
            tx_lon=tx.longitude,

            rx_name=rx_location,
            rx_lat=rx.latitude,
            rx_lon=rx.longitude,

            prediction_date=prediction_date,

            t_index=t_index,
        )


# ============================================================================
# BAND LOOKUP HELPERS
# ============================================================================

def get_hour(
    prediction: GrafexPrediction,
    utc_hour: int,
) -> GrafexHour:

    if not 0 <= utc_hour <= 23:
        raise ValueError(
            "utc_hour must be between 0 and 23."
        )

    for hour in prediction.hours:

        if hour.utc_hour == utc_hour:
            return hour

    raise KeyError(
        f"GRAFEX prediction does not contain "
        f"UTC hour {utc_hour:02d}."
    )


def get_frequency_prediction(
    prediction: GrafexPrediction,
    utc_hour: int,
    frequency_mhz: float,
) -> GrafexFrequencyPrediction:

    hour = get_hour(
        prediction,
        utc_hour,
    )

    grafex_frequency = int(
        round(frequency_mhz)
    )

    if not (
        MIN_FREQUENCY_MHZ
        <= grafex_frequency
        <= MAX_FREQUENCY_MHZ
    ):

        raise ValueError(
            f"Frequency must be between "
            f"{MIN_FREQUENCY_MHZ} and "
            f"{MAX_FREQUENCY_MHZ} MHz."
        )

    return hour.frequencies[
        float(grafex_frequency)
    ]


def get_band_prediction(
    prediction: GrafexPrediction,
    utc_hour: int,
    band: str,
) -> GrafexFrequencyPrediction:

    band_key = band.lower().strip()

    if band_key not in BAND_FREQUENCIES_MHZ:

        raise KeyError(
            f"Unknown band '{band}'. "
            f"Available bands: "
            f"{', '.join(BAND_FREQUENCIES_MHZ)}"
        )

    return get_frequency_prediction(
        prediction=prediction,
        utc_hour=utc_hour,
        frequency_mhz=(
            BAND_FREQUENCIES_MHZ[
                band_key
            ]
        ),
    )


# ============================================================================
# SUMMARY HELPERS
# ============================================================================

def summarise_band(
    prediction: GrafexPrediction,
    utc_hour: int,
    band: str,
) -> dict[str, Any]:

    band_key = band.lower().strip()

    if band_key not in BAND_FREQUENCIES_MHZ:

        raise KeyError(
            f"Unknown band '{band}'. "
            f"Available bands: "
            f"{', '.join(BAND_FREQUENCIES_MHZ)}"
        )

    requested_frequency = (
        BAND_FREQUENCIES_MHZ[
            band_key
        ]
    )

    result = get_band_prediction(
        prediction=prediction,
        utc_hour=utc_hour,
        band=band_key,
    )

    hour = get_hour(
        prediction=prediction,
        utc_hour=utc_hour,
    )

    return {
        "band": band_key,

        "requested_frequency_mhz":
            requested_frequency,

        "grafex_frequency_mhz":
            result.frequency_mhz,

        "symbol":
            result.symbol,

        "meaning":
            result.meaning,

        "supported":
            result.supported,

        "first_mode_owf_mhz":
            hour.first_owf_mhz,

        "first_mode_emuf_mhz":
            hour.first_emuf_mhz,

        "first_mode_alf_mhz":
            hour.first_alf_mhz,

        "second_mode_owf_mhz":
            hour.second_owf_mhz,

        "second_mode_emuf_mhz":
            hour.second_emuf_mhz,

        "second_mode_alf_mhz":
            hour.second_alf_mhz,
    }


def summarise_all_bands(
    prediction: GrafexPrediction,
    utc_hour: int,
) -> list[dict[str, Any]]:

    results: list[dict[str, Any]] = []

    for band in BAND_FREQUENCIES_MHZ:

        results.append(
            summarise_band(
                prediction=prediction,
                utc_hour=utc_hour,
                band=band,
            )
        )

    return results


# ============================================================================
# SERIALISATION
# ============================================================================

def prediction_to_json(
    prediction: GrafexPrediction,
    indent: int = 2,
) -> str:

    return json.dumps(
        prediction.to_dict(),
        indent=indent,
        ensure_ascii=False,
    )


# ============================================================================
# SIMPLE TEST / CLI
# ============================================================================

def main() -> None:

    """
    Basic standalone test.

    Nelson -> Sydney
    24 September 2026
    T-index 62
    """

    engine = PropagationEngine()

    print("=" * 70)
    print("RADIOPATHWAYTOOL - GRAFEX TEST")
    print("=" * 70)
    print()

    print("Resolving locations...")

    prediction = engine.calculate_path(
        tx_location="Nelson, New Zealand",
        rx_location="Sydney, Australia",
        prediction_date=date(
            2026,
            9,
            24,
        ),
        t_index=62,
    )

    print()

    print(
        f"Path: "
        f"{prediction.tx_name}"
        f" -> "
        f"{prediction.rx_name}"
    )

    print(
        f"TX coordinates: "
        f"{prediction.tx_latitude:.4f}, "
        f"{prediction.tx_longitude:.4f}"
    )

    print(
        f"RX coordinates: "
        f"{prediction.rx_latitude:.4f}, "
        f"{prediction.rx_longitude:.4f}"
    )

    print()

    if prediction.distance_km is not None:

        print(
            f"Distance: "
            f"{prediction.distance_km:.0f} km"
        )

    else:

        print(
            "Distance: unavailable"
        )

    if (
        prediction.bearing_tx_to_rx is not None
        and prediction.bearing_rx_to_tx is not None
    ):

        print(
            f"Bearings: "
            f"{prediction.bearing_tx_to_rx:.0f}° / "
            f"{prediction.bearing_rx_to_tx:.0f}°"
        )

    else:

        print(
            "Bearings: unavailable"
        )

    print(
        f"T-index: "
        f"{prediction.t_index}"
    )

    print(
        f"Date: "
        f"{prediction.prediction_date}"
    )

    print()

    print(
        f"First mode: "
        f"{prediction.first_mode or 'unknown'}"
    )

    print(
        f"Second mode: "
        f"{prediction.second_mode or 'unknown'}"
    )

    print()

    print(
        f"Hourly predictions parsed: "
        f"{len(prediction.hours)}"
    )

    print()

    print("BAND CONDITIONS")
    print("-" * 70)

    for utc_hour in [0, 6, 12, 18]:

        try:
            hour = get_hour(
                prediction,
                utc_hour,
            )
        except KeyError:
            continue

        print(
            f"\nUTC {hour.utc_hour:02d}:00"
        )

        for band in BAND_FREQUENCIES_MHZ:

            result = summarise_band(
                prediction=prediction,
                utc_hour=hour.utc_hour,
                band=band,
            )

            print(
                f"  {band:>4}: "
                f"{result['symbol']}  "
                f"{result['meaning']}"
            )

    print()
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()