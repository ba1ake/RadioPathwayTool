"""
RadioPathwayTool
SWS HAP propagation engine

This module retrieves and decodes the Australian Bureau of Meteorology
Space Weather Services (SWS) HF Availability Prediction (HAP) charts.

The HAP data is treated as the primary propagation prediction.

This module deliberately does NOT produce an arbitrary 0-100 score.

Instead it exposes structured propagation information which can later
be combined with:

    - SWS ionosphere observations
    - space weather
    - data age
    - PSK Reporter
    - WSPR
    - user radio / antenna information

HAP frequencies:

    160m = 1.838 MHz
     80m = 3.650 MHz
     40m = 7.150 MHz
     30m = 10.125 MHz
     20m = 14.175 MHz
     17m = 18.118 MHz
     15m = 21.225 MHz
     12m = 24.940 MHz
     10m = 28.850 MHz

SWS returns four GIF pages:

    Page 1 -> 00-05 UTC
    Page 2 -> 06-11 UTC
    Page 3 -> 12-17 UTC
    Page 4 -> 18-23 UTC
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import hashlib
import re

import requests
from PIL import Image


# ============================================================================
# CONFIGURATION
# ============================================================================

SWS_BASE_URL = "https://www.sws.bom.gov.au"

SWS_HAP_CGI = (
    f"{SWS_BASE_URL}/local-cgi-bin/phapj-cgi.tcl"
)

CACHE_DIR = Path(".hap_cache")

REQUEST_TIMEOUT = 30


# ============================================================================
# FREQUENCIES
# ============================================================================

BANDS = {
    "160m": 1838,
    "80m": 3650,
    "40m": 7150,
    "30m": 10125,
    "20m": 14175,
    "17m": 18118,
    "15m": 21225,
    "12m": 24940,
    "10m": 28850,
}

FREQUENCY_TO_BAND = {
    frequency: band
    for band, frequency in BANDS.items()
}


# ============================================================================
# HAP COLOURS
# ============================================================================

HAP_COLOURS = {
    (255, 255, 0): 1838,      # Yellow
    (255, 0, 0): 3650,        # Red
    (128, 128, 0): 7150,      # Olive
    (0, 128, 0): 10125,       # Green
    (0, 255, 255): 14175,     # Cyan
    (0, 255, 0): 18118,       # Lime
    (0, 128, 128): 21225,     # Teal
    (0, 0, 255): 24940,       # Blue
    (0, 0, 128): 28850,       # Navy
}


# ============================================================================
# DISCOVERED MAP GEOMETRY
# ============================================================================

"""
The actual geographic map area inside each hourly HAP panel was found
experimentally from the SWS GIF.

Panel:

    approximately 290 x 179 pixels

Effective map:

    X = 12 .. 289
    Y = 2  .. 178

The extreme edges can contain borders. Therefore boundary grid points
are sampled slightly inward.
"""

HAP_MAP_X_MIN = 12
HAP_MAP_X_MAX = 289

HAP_MAP_Y_MIN = 2
HAP_MAP_Y_MAX = 178

EDGE_INSET = 2


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HAPConfig:
    """
    Configuration used to request an HAP grid.
    """

    base_name: str

    base_lat: float
    base_lon: float

    nw_lat: float
    nw_lon: float

    step_lat: float = 5.0
    step_lon: float = 5.0

    nrows: int = 7
    ncols: int = 7

    tindex: int = 5


@dataclass
class HAPGridPoint:
    """
    Geographic point and its corresponding location in the rendered HAP map.
    """

    row: int
    col: int

    latitude: float
    longitude: float

    pixel_x: float
    pixel_y: float


@dataclass
class HAPRecommendation:
    """
    HAP recommendation at one geographic grid point and hour.
    """

    hour_utc: int

    row: int
    col: int

    latitude: float
    longitude: float

    band: Optional[str]
    frequency_khz: Optional[int]

    pixel_x: float
    pixel_y: float

    sample_support: int


@dataclass
class HAPHourResult:
    """
    Raw hourly HAP image panel.
    """

    hour_utc: int
    panel: Image.Image


# ============================================================================
# HAP COLLECTOR
# ============================================================================

class HAPCollector:

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
    ) -> None:

        self.cache_dir = cache_dir

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "RadioPathwayTool/1.0 "
                    "(HF propagation research)"
                )
            }
        )

    # ------------------------------------------------------------------------
    # Create centred geographic grid
    # ------------------------------------------------------------------------

    @staticmethod
    def create_centred_config(
        base_name: str,
        base_lat: float,
        base_lon: float,
        nrows: int = 7,
        ncols: int = 7,
        step_lat: float = 5.0,
        step_lon: float = 5.0,
    ) -> HAPConfig:

        centre_row = nrows // 2
        centre_col = ncols // 2

        nw_lat = (
            base_lat
            + centre_row * step_lat
        )

        nw_lon = (
            base_lon
            - centre_col * step_lon
        )

        return HAPConfig(
            base_name=base_name,
            base_lat=base_lat,
            base_lon=base_lon,
            nw_lat=nw_lat,
            nw_lon=nw_lon,
            step_lat=step_lat,
            step_lon=step_lon,
            nrows=nrows,
            ncols=ncols,
        )

    # ------------------------------------------------------------------------
    # Build geographic grid
    # ------------------------------------------------------------------------

    @staticmethod
    def build_grid(
        config: HAPConfig,
    ) -> list[HAPGridPoint]:

        points = []

        for row in range(config.nrows):

            latitude = (
                config.nw_lat
                - row * config.step_lat
            )

            for col in range(config.ncols):

                longitude = (
                    config.nw_lon
                    + col * config.step_lon
                )

                # ------------------------------------------------------------
                # Geographic -> rendered map coordinates
                # ------------------------------------------------------------

                if config.ncols > 1:

                    x_fraction = (
                        col
                        / (config.ncols - 1)
                    )

                else:

                    x_fraction = 0.5

                if config.nrows > 1:

                    y_fraction = (
                        row
                        / (config.nrows - 1)
                    )

                else:

                    y_fraction = 0.5

                pixel_x = (
                    HAP_MAP_X_MIN
                    + x_fraction
                    * (
                        HAP_MAP_X_MAX
                        - HAP_MAP_X_MIN
                    )
                )

                pixel_y = (
                    HAP_MAP_Y_MIN
                    + y_fraction
                    * (
                        HAP_MAP_Y_MAX
                        - HAP_MAP_Y_MIN
                    )
                )

                # ------------------------------------------------------------
                # Avoid sampling directly on rendered map borders.
                # ------------------------------------------------------------

                if col == 0:

                    pixel_x += EDGE_INSET

                elif col == config.ncols - 1:

                    pixel_x -= EDGE_INSET

                if row == 0:

                    pixel_y += EDGE_INSET

                elif row == config.nrows - 1:

                    pixel_y -= EDGE_INSET

                points.append(
                    HAPGridPoint(
                        row=row,
                        col=col,
                        latitude=latitude,
                        longitude=longitude,
                        pixel_x=pixel_x,
                        pixel_y=pixel_y,
                    )
                )

        return points

    # ------------------------------------------------------------------------
    # Build CGI request
    # ------------------------------------------------------------------------

    def build_request_url(
        self,
        config: HAPConfig,
        timestamp_utc: Optional[datetime] = None,
    ) -> str:

        if timestamp_utc is None:

            timestamp_utc = datetime.now(
                timezone.utc
            )

        frequencies = list(
            BANDS.values()
        )

        params = {
            "baslat": f"{config.base_lat:.4f}",
            "baslng": f"{config.base_lon:.4f}",
            "basename": config.base_name,

            "numfreqs": len(frequencies),

            "year": timestamp_utc.year,
            "month": timestamp_utc.month,
            "day": timestamp_utc.day,

            "tindex": config.tindex,

            "nwlat": f"{config.nw_lat:.4f}",
            "nwlng": f"{config.nw_lon:.4f}",

            "steplat": f"{config.step_lat:.4f}",
            "steplng": f"{config.step_lon:.4f}",

            "nrows": config.nrows,
            "ncols": config.ncols,
        }

        for index, frequency in enumerate(
            frequencies,
            start=1,
        ):

            params[
                f"freq{index}"
            ] = frequency

        params["freq10"] = ""

        return (
            f"{SWS_HAP_CGI}?"
            + urlencode(params)
        )

    # ------------------------------------------------------------------------
    # Fetch CGI
    # ------------------------------------------------------------------------

    def fetch_hap_page(
        self,
        url: str,
    ) -> str:

        response = self.session.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.text

    # ------------------------------------------------------------------------
    # Extract GIF URLs
    # ------------------------------------------------------------------------

    @staticmethod
    def extract_image_urls(
        html: str,
    ) -> list[str]:

        matches = re.findall(
            r'(?:https?://[^"\']+)?'
            r'/olts/hapimgs/[^"\']+\.gif',
            html,
            flags=re.IGNORECASE,
        )

        urls = []

        for match in matches:

            if match.startswith("http"):

                url = match

            else:

                url = (
                    SWS_BASE_URL
                    + match
                )

            if url not in urls:

                urls.append(url)

        return urls

    # ------------------------------------------------------------------------
    # Cache filename
    # ------------------------------------------------------------------------

    def cache_filename(
        self,
        url: str,
    ) -> Path:

        digest = hashlib.md5(
            url.encode("utf-8")
        ).hexdigest()

        return (
            self.cache_dir
            / f"{digest}.gif"
        )

    # ------------------------------------------------------------------------
    # Download image
    # ------------------------------------------------------------------------

    def download_image(
        self,
        url: str,
    ) -> Path:

        path = self.cache_filename(
            url
        )

        if path.exists():

            return path

        response = self.session.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        path.write_bytes(
            response.content
        )

        return path

    # ------------------------------------------------------------------------
    # Split page into hourly panels
    # ------------------------------------------------------------------------

    @staticmethod
    def extract_hour_panels(
        image: Image.Image,
    ) -> dict[int, Image.Image]:

        image = image.convert("RGB")

        boxes = [
            (50, 134, 340, 313),
            (360, 134, 650, 313),

            (50, 381, 340, 560),
            (360, 381, 650, 560),

            (50, 631, 340, 808),
            (360, 631, 650, 808),
        ]

        panels = {}

        for index, box in enumerate(boxes):

            panels[index] = image.crop(
                box
            )

        return panels

    # ------------------------------------------------------------------------
    # Collect all 24 hours
    # ------------------------------------------------------------------------

    def collect(
        self,
        config: HAPConfig,
        timestamp_utc: Optional[datetime] = None,
    ) -> dict[int, HAPHourResult]:

        if timestamp_utc is None:

            timestamp_utc = datetime.now(
                timezone.utc
            )

        url = self.build_request_url(
            config,
            timestamp_utc,
        )

        print()
        print(
            "HAP request:"
        )

        print(url)

        html = self.fetch_hap_page(
            url
        )

        image_urls = (
            self.extract_image_urls(
                html
            )
        )

        if not image_urls:

            raise RuntimeError(
                "SWS HAP returned no image URLs."
            )

        print()
        print(
            f"Found {len(image_urls)} HAP pages."
        )

        results = {}

        for page_index, image_url in enumerate(
            image_urls,
            start=1,
        ):

            print(
                f"Downloading HAP page "
                f"{page_index}/{len(image_urls)}..."
            )

            image_path = (
                self.download_image(
                    image_url
                )
            )

            image = Image.open(
                image_path
            )

            panels = (
                self.extract_hour_panels(
                    image
                )
            )

            first_hour = (
                (page_index - 1)
                * 6
            )

            for panel_index, panel in panels.items():

                hour = (
                    first_hour
                    + panel_index
                )

                results[hour] = (
                    HAPHourResult(
                        hour_utc=hour,
                        panel=panel,
                    )
                )

        return results


# ============================================================================
# HAP DECODER
# ============================================================================

class HAPDecoder:

    def __init__(
        self,
        config: HAPConfig,
    ) -> None:

        self.config = config

        self.grid = (
            HAPCollector.build_grid(
                config
            )
        )

    # ------------------------------------------------------------------------
    # RGB -> frequency
    # ------------------------------------------------------------------------

    @staticmethod
    def colour_to_frequency(
        rgb: tuple[int, int, int],
    ) -> Optional[int]:

        return HAP_COLOURS.get(
            rgb
        )

    # ------------------------------------------------------------------------
    # Frequency -> band
    # ------------------------------------------------------------------------

    @staticmethod
    def frequency_to_band(
        frequency_khz: Optional[int],
    ) -> Optional[str]:

        if frequency_khz is None:

            return None

        return FREQUENCY_TO_BAND.get(
            frequency_khz
        )

    # ------------------------------------------------------------------------
    # Sample point
    # ------------------------------------------------------------------------

    @staticmethod
    def sample_point(
        panel: Image.Image,
        point: HAPGridPoint,
        radius: int = 2,
    ) -> tuple[
        Optional[int],
        int,
    ]:

        """
        Sample a small neighbourhood around a geographic point.

        The dominant recognised HAP colour is returned.

        sample_support is the number of pixels supporting that colour.

        This is a decoder quality metric only. It is NOT propagation
        confidence.
        """

        image = panel.convert(
            "RGB"
        )

        centre_x = round(
            point.pixel_x
        )

        centre_y = round(
            point.pixel_y
        )

        counts = {}

        for y in range(
            centre_y - radius,
            centre_y + radius + 1,
        ):

            if (
                y < 0
                or y >= image.height
            ):

                continue

            for x in range(
                centre_x - radius,
                centre_x + radius + 1,
            ):

                if (
                    x < 0
                    or x >= image.width
                ):

                    continue

                rgb = image.getpixel(
                    (x, y)
                )

                frequency = (
                    HAP_COLOURS.get(
                        rgb
                    )
                )

                if frequency is None:

                    continue

                counts[rgb] = (
                    counts.get(
                        rgb,
                        0,
                    )
                    + 1
                )

        if not counts:

            return None, 0

        dominant_colour = max(
            counts,
            key=counts.get,
        )

        frequency = (
            HAP_COLOURS[
                dominant_colour
            ]
        )

        return (
            frequency,
            counts[
                dominant_colour
            ],
        )

    # ------------------------------------------------------------------------
    # Decode one hour
    # ------------------------------------------------------------------------

    def decode_hour(
        self,
        hour_result: HAPHourResult,
    ) -> list[HAPRecommendation]:

        results = []

        for point in self.grid:

            frequency, support = (
                self.sample_point(
                    hour_result.panel,
                    point,
                )
            )

            band = (
                self.frequency_to_band(
                    frequency
                )
            )

            results.append(
                HAPRecommendation(
                    hour_utc=hour_result.hour_utc,

                    row=point.row,
                    col=point.col,

                    latitude=point.latitude,
                    longitude=point.longitude,

                    band=band,
                    frequency_khz=frequency,

                    pixel_x=point.pixel_x,
                    pixel_y=point.pixel_y,

                    sample_support=support,
                )
            )

        return results

    # ------------------------------------------------------------------------
    # Decode all hours
    # ------------------------------------------------------------------------

    def decode_all(
        self,
        hap_hours: dict[int, HAPHourResult],
    ) -> dict[
        int,
        list[HAPRecommendation],
    ]:

        decoded = {}

        for hour in sorted(
            hap_hours
        ):

            decoded[hour] = (
                self.decode_hour(
                    hap_hours[hour]
                )
            )

        return decoded

    # ------------------------------------------------------------------------
    # Find nearest grid point
    # ------------------------------------------------------------------------

    def nearest_grid_point(
        self,
        latitude: float,
        longitude: float,
    ) -> HAPGridPoint:

        return min(
            self.grid,
            key=lambda point: (
                (point.latitude - latitude) ** 2
                +
                (point.longitude - longitude) ** 2
            ),
        )

    # ------------------------------------------------------------------------
    # Get recommendation for arbitrary coordinate
    # ------------------------------------------------------------------------

    def get_at_location(
        self,
        decoded: dict[
            int,
            list[HAPRecommendation],
        ],
        latitude: float,
        longitude: float,
        hour_utc: int,
    ) -> Optional[HAPRecommendation]:

        if hour_utc not in decoded:

            return None

        point = (
            self.nearest_grid_point(
                latitude,
                longitude,
            )
        )

        for result in decoded[
            hour_utc
        ]:

            if (
                result.row == point.row
                and
                result.col == point.col
            ):

                return result

        return None

    # ------------------------------------------------------------------------
    # Get base recommendation
    # ------------------------------------------------------------------------

    def get_base_recommendation(
        self,
        decoded: dict[
            int,
            list[HAPRecommendation],
        ],
        hour_utc: int,
    ) -> Optional[HAPRecommendation]:

        return self.get_at_location(
            decoded=decoded,
            latitude=self.config.base_lat,
            longitude=self.config.base_lon,
            hour_utc=hour_utc,
        )

    # ------------------------------------------------------------------------
    # Get complete base forecast
    # ------------------------------------------------------------------------

    def get_base_forecast(
        self,
        decoded: dict[
            int,
            list[HAPRecommendation],
        ],
    ) -> dict[
        int,
        Optional[HAPRecommendation],
    ]:

        forecast = {}

        for hour in sorted(
            decoded
        ):

            forecast[hour] = (
                self.get_base_recommendation(
                    decoded,
                    hour,
                )
            )

        return forecast

    # ------------------------------------------------------------------------
    # Regional band distribution
    # ------------------------------------------------------------------------

    def get_regional_distribution(
        self,
        decoded: dict[
            int,
            list[HAPRecommendation],
        ],
        hour_utc: int,
    ) -> dict[str, int]:

        counts = {
            band: 0
            for band in BANDS
        }

        if hour_utc not in decoded:

            return counts

        for result in decoded[
            hour_utc
        ]:

            if result.band in counts:

                counts[
                    result.band
                ] += 1

        return counts


# ============================================================================
# PRINTING
# ============================================================================

def print_geometry(
    decoder: HAPDecoder,
) -> None:

    print()
    print(
        "=" * 78
    )

    print(
        "HAP GEOGRAPHIC → PIXEL GEOMETRY"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"Map X range: "
        f"{HAP_MAP_X_MIN} → "
        f"{HAP_MAP_X_MAX}"
    )

    print(
        f"Map Y range: "
        f"{HAP_MAP_Y_MIN} → "
        f"{HAP_MAP_Y_MAX}"
    )

    print(
        f"Boundary inset: "
        f"{EDGE_INSET}px"
    )

    print()

    centre = (
        decoder.nearest_grid_point(
            decoder.config.base_lat,
            decoder.config.base_lon,
        )
    )

    print(
        "Base grid point:"
    )

    print(
        f"  Geographic: "
        f"{centre.latitude:.4f}, "
        f"{centre.longitude:.4f}"
    )

    print(
        f"  Pixel: "
        f"({centre.pixel_x:.1f}, "
        f"{centre.pixel_y:.1f})"
    )


def print_grid(
    decoder: HAPDecoder,
    recommendations: list[HAPRecommendation],
) -> None:

    if not recommendations:

        return

    hour = recommendations[0].hour_utc

    print()
    print(
        "=" * 78
    )

    print(
        f"HAP GRID — {hour:02d} UTC"
    )

    print(
        "=" * 78
    )

    print()

    print(
        "Latitude \\ Longitude",
        end="   ",
    )

    for col in range(
        decoder.config.ncols
    ):

        longitude = (
            decoder.config.nw_lon
            + col
            * decoder.config.step_lon
        )

        print(
            f"{longitude:8.2f}",
            end="",
        )

    print()

    print(
        "-" * 78
    )

    for row in range(
        decoder.config.nrows
    ):

        latitude = (
            decoder.config.nw_lat
            - row
            * decoder.config.step_lat
        )

        print(
            f"{latitude:8.2f}",
            end="",
        )

        row_results = [
            result
            for result in recommendations
            if result.row == row
        ]

        row_results.sort(
            key=lambda result: result.col
        )

        for result in row_results:

            value = (
                result.band
                if result.band is not None
                else "--"
            )

            print(
                f"{value:>10}",
                end="",
            )

        print()

    base = (
        decoder.get_base_recommendation(
            {
                hour: recommendations
            },
            hour,
        )
    )

    print()

    if base is None:

        print(
            "Base location: no decoded HAP result"
        )

    elif base.band is None:

        print(
            "Base location: no recognised HAP frequency"
        )

    else:

        print(
            "Base location:"
        )

        print(
            f"  {decoder.config.base_name}"
        )

        print(
            f"  HAP recommendation: "
            f"{base.band} "
            f"({base.frequency_khz / 1000:.3f} MHz)"
        )

        print(
            f"  Sample support: "
            f"{base.sample_support}"
        )


def print_base_forecast(
    decoder: HAPDecoder,
    decoded: dict[
        int,
        list[HAPRecommendation],
    ],
) -> None:

    print()
    print(
        "=" * 78
    )

    print(
        "HAP BASE LOCATION — 24 HOUR FORECAST"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"Base: {decoder.config.base_name}"
    )

    print(
        f"Latitude: "
        f"{decoder.config.base_lat:.4f}"
    )

    print(
        f"Longitude: "
        f"{decoder.config.base_lon:.4f}"
    )

    print()

    print(
        f"{'UTC':>5}  "
        f"{'Band':>6}  "
        f"{'Frequency':>12}  "
        f"{'Support':>8}"
    )

    print(
        "-" * 48
    )

    forecast = (
        decoder.get_base_forecast(
            decoded
        )
    )

    for hour, result in forecast.items():

        if result is None:

            print(
                f"{hour:02d}     "
                f"{'--':>6}  "
                f"{'--':>12}  "
                f"{'--':>8}"
            )

            continue

        if result.band is None:

            print(
                f"{hour:02d}     "
                f"{'--':>6}  "
                f"{'--':>12}  "
                f"{result.sample_support:>8}"
            )

            continue

        print(
            f"{hour:02d}     "
            f"{result.band:>6}  "
            f"{result.frequency_khz / 1000:>9.3f} MHz  "
            f"{result.sample_support:>8}"
        )


def print_regional_summary(
    decoder: HAPDecoder,
    decoded: dict[
        int,
        list[HAPRecommendation],
    ],
) -> None:

    print()
    print(
        "=" * 78
    )

    print(
        "REGIONAL HAP DISTRIBUTION"
    )

    print(
        "=" * 78
    )

    print()

    total = (
        decoder.config.nrows
        * decoder.config.ncols
    )

    for hour in sorted(
        decoded
    ):

        distribution = (
            decoder.get_regional_distribution(
                decoded,
                hour,
            )
        )

        active = [
            (
                band,
                count,
            )
            for band, count
            in distribution.items()
            if count > 0
        ]

        active.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        print(
            f"{hour:02d} UTC:",
            end=" ",
        )

        parts = []

        for band, count in active:

            percentage = (
                count
                / total
                * 100
            )

            parts.append(
                f"{band} "
                f"{count}/{total} "
                f"({percentage:.0f}%)"
            )

        print(
            " | ".join(parts)
        )


# ============================================================================
# MAIN TEST
# ============================================================================

def main() -> None:

    print()
    print(
        "=" * 78
    )

    print(
        "RADIOPATHWAYTOOL — SWS HAP ENGINE"
    )

    print(
        "=" * 78
    )

    # ------------------------------------------------------------------------
    # Nelson
    # ------------------------------------------------------------------------

    base_name = "Nelson"

    base_lat = -41.27

    base_lon = 173.28

    collector = HAPCollector()

    config = (
        collector.create_centred_config(
            base_name=base_name,
            base_lat=base_lat,
            base_lon=base_lon,
            nrows=7,
            ncols=7,
            step_lat=5.0,
            step_lon=5.0,
        )
    )

    print()

    print(
        "Location:"
    )

    print(
        f"  {base_name}"
    )

    print(
        f"  Latitude : "
        f"{base_lat:.4f}"
    )

    print(
        f"  Longitude: "
        f"{base_lon:.4f}"
    )

    print()

    print(
        "Calculated geographic grid:"
    )

    print(
        f"  NW latitude : "
        f"{config.nw_lat:.4f}"
    )

    print(
        f"  NW longitude: "
        f"{config.nw_lon:.4f}"
    )

    print(
        f"  Step        : "
        f"{config.step_lat}° / "
        f"{config.step_lon}°"
    )

    print(
        f"  Grid        : "
        f"{config.nrows} × "
        f"{config.ncols}"
    )

    print(
        f"  Centre      : row "
        f"{config.nrows // 2}, col "
        f"{config.ncols // 2}"
    )

    # ------------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------------

    decoder = HAPDecoder(
        config
    )

    print_geometry(
        decoder
    )

    # ------------------------------------------------------------------------
    # Collect
    # ------------------------------------------------------------------------

    print()
    print(
        "Collecting HAP data..."
    )

    try:

        hap_hours = collector.collect(
            config
        )

    except Exception as exc:

        print()
        print(
            "HAP collection failed:"
        )

        print(
            f"  {exc}"
        )

        raise SystemExit(1)

    print()
    print(
        "HAP collection successful."
    )

    print(
        f"Decoded hours: "
        f"{len(hap_hours)}"
    )

    # ------------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------------

    decoded = (
        decoder.decode_all(
            hap_hours
        )
    )

    # ------------------------------------------------------------------------
    # Show key daylight examples
    # ------------------------------------------------------------------------

    for hour in (
        12,
        15,
    ):

        if hour in decoded:

            print_grid(
                decoder,
                decoded[hour],
            )

    # ------------------------------------------------------------------------
    # 24-hour base forecast
    # ------------------------------------------------------------------------

    print_base_forecast(
        decoder,
        decoded,
    )

    # ------------------------------------------------------------------------
    # Regional distribution
    # ------------------------------------------------------------------------

    print_regional_summary(
        decoder,
        decoded,
    )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    print(
        "HAP ENGINE COMPLETE"
    )

    print(
        "=" * 78
    )

    print()

    print(
        "HAP is now exposed as structured propagation data."
    )

    print()

    print(
        "Next integration:"
    )

    print(
        "  HAP"
        " + "
        "SWS ionosphere"
        " + "
        "space weather"
        " + "
        "data age"
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()