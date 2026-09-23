
"""
band_favorability.py

HF propagation analysis using the Australian Bureau of Meteorology
Space Weather Services (SWS) Hourly HF Availability Prediction (HAP).

Current focus:
    - Generate an SWS HAP request centred on the requested base location.
    - Download the four HAP image pages covering 00-23 UTC.
    - Decode the HAP colour at each geographic grid point.
    - Translate HAP colours into the requested frequencies/bands.
    - Provide raw HAP results for later integration into the propagation model.

IMPORTANT:
    HAP is a propagation prediction product, not a probability-of-contact
    calculator.

    The HAP chart indicates which requested HF frequency is predicted to
    be suitable for the base-to-location circuit.

    The decoder therefore reports:
        "HAP recommends 7.150 MHz here"

    rather than:
        "There is an 80% chance of contact."

This module deliberately keeps the HAP decoding separate from the final
band-favorability scoring system until the decoder has been validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
import hashlib
import re

import requests
from PIL import Image


# ============================================================================
# SWS CONFIGURATION
# ============================================================================

SWS_BASE_URL = "https://www.sws.bom.gov.au"

SWS_HAP_CGI = (
    "https://www.sws.bom.gov.au/"
    "local-cgi-bin/phapj-cgi.tcl"
)

SWS_HAP_IMAGE_BASE = (
    "https://www.sws.bom.gov.au"
)

CACHE_DIR = Path(".hap_cache")


# ============================================================================
# BAND / FREQUENCY DEFINITIONS
# ============================================================================

# Frequencies are the same frequencies used by the SWS HAP request.
#
# The HAP legend associates each colour with one requested frequency.
#
# Keep these in ascending order.

BANDS = {
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
# HAP COLOUR DEFINITIONS
# ============================================================================

# These colours were determined from the SWS HAP legend used by the
# nine-frequency request.
#
# RGB:
#
#   160m  yellow
#   80m   red
#   40m   olive
#   30m   green
#   20m   cyan
#   17m   lime
#   15m   teal
#   12m   blue
#   10m   navy
#
# Exact colours are important because the GIF uses a fixed palette.

HAP_COLOURS = {
    (255, 255, 0): ("160m", 1.838),
    (255, 0, 0): ("80m", 3.650),
    (128, 128, 0): ("40m", 7.150),
    (0, 128, 0): ("30m", 10.125),
    (0, 255, 255): ("20m", 14.175),
    (0, 255, 0): ("17m", 18.118),
    (0, 128, 128): ("15m", 21.225),
    (0, 0, 255): ("12m", 24.940),
    (0, 0, 128): ("10m", 28.850),
}


# Reverse lookup.

BAND_COLOURS = {
    band: colour
    for colour, (band, _frequency) in HAP_COLOURS.items()
}


# ============================================================================
# HAP DATA STRUCTURES
# ============================================================================

@dataclass
class HAPConfig:
    """
    Configuration used to generate an SWS HAP request.

    The geographic grid is deliberately centred around the base location.

    For a 7x7 grid with 5-degree spacing:

        155  160  165  170  175  180  185
         |    |    |    |    |    |    |
        ...                  |
                             ↓
                         BASE LOCATION
    """

    latitude: float
    longitude: float
    base_name: str = "Base"

    grid_step_lat: float = 5.0
    grid_step_lon: float = 5.0

    grid_rows: int = 7
    grid_cols: int = 7

    # HAP frequencies.
    frequencies_khz: tuple[int, ...] = (
        1838,
        3650,
        7150,
        10125,
        14175,
        18118,
        21225,
        24940,
        28850,
    )

    def __post_init__(self):
        if self.grid_rows % 2 == 0:
            raise ValueError("grid_rows must be odd so the base can be centred")

        if self.grid_cols % 2 == 0:
            raise ValueError("grid_cols must be odd so the base can be centred")

    @property
    def centre_row(self) -> int:
        return self.grid_rows // 2

    @property
    def centre_col(self) -> int:
        return self.grid_cols // 2

    @property
    def nw_latitude(self) -> float:
        """
        Calculate the northernmost grid latitude.

        Example:
            base = -41.27
            7 rows
            5-degree spacing

            NW = -26.27
        """

        half_height = self.centre_row * self.grid_step_lat

        return self.latitude + half_height

    @property
    def nw_longitude(self) -> float:
        """
        Calculate the westernmost grid longitude.
        """

        half_width = self.centre_col * self.grid_step_lon

        return self.longitude - half_width

    def grid_latitude(self, row: int) -> float:
        return self.nw_latitude - (row * self.grid_step_lat)

    def grid_longitude(self, col: int) -> float:
        return self.nw_longitude + (col * self.grid_step_lon)

    def grid_coordinates(self) -> list[tuple[int, int, float, float]]:
        """
        Return every geographic grid point.

        Each tuple is:

            row,
            column,
            latitude,
            longitude
        """

        points = []

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):

                lat = self.grid_latitude(row)
                lon = self.grid_longitude(col)

                points.append(
                    (
                        row,
                        col,
                        lat,
                        lon,
                    )
                )

        return points


@dataclass
class HAPGridPoint:
    """
    One decoded HAP geographic grid point.
    """

    row: int
    col: int

    latitude: float
    longitude: float

    band: Optional[str] = None
    frequency_mhz: Optional[float] = None

    rgb: Optional[tuple[int, int, int]] = None

    supported: bool = False

    confidence: float = 0.0

    notes: list[str] = field(default_factory=list)


@dataclass
class HAPHourResult:
    """
    Decoded result for one UTC hour.
    """

    hour_utc: int

    image_url: Optional[str] = None
    image_path: Optional[Path] = None

    grid_points: list[HAPGridPoint] = field(default_factory=list)

    decoding_success: bool = False

    notes: list[str] = field(default_factory=list)

    @property
    def supported_bands(self) -> set[str]:
        return {
            point.band
            for point in self.grid_points
            if point.supported and point.band is not None
        }

    def band_counts(self) -> dict[str, int]:
        counts = {
            band: 0
            for band in BANDS
        }

        for point in self.grid_points:
            if point.band in counts:
                counts[point.band] += 1

        return counts


@dataclass
class HAPFrequencyResult:
    """
    Full HAP result for one base location.
    """

    config: HAPConfig

    hourly: dict[int, HAPHourResult] = field(default_factory=dict)

    source_url: Optional[str] = None

    retrieved_utc: Optional[datetime] = None

    success: bool = False

    error: Optional[str] = None


# ============================================================================
# HAP COLLECTOR
# ============================================================================

class HAPCollector:
    """
    Downloads and decodes SWS HAP charts.
    """

    def __init__(
        self,
        cache_dir: Path | str = CACHE_DIR,
        timeout: int = 30,
    ):
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

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
    # REQUEST GENERATION
    # ------------------------------------------------------------------------

    def build_hap_url(
        self,
        config: HAPConfig,
        timestamp_utc: datetime,
    ) -> str:
        """
        Build the SWS HAP CGI URL.

        The geographic grid is automatically centred around the base.
        """

        timestamp_utc = timestamp_utc.astimezone(timezone.utc)

        params = {
            "baslat": f"{config.latitude:.4f}",
            "baslng": f"{config.longitude:.4f}",
            "basename": config.base_name,

            "numfreqs": str(len(config.frequencies_khz)),

            "year": str(timestamp_utc.year),
            "month": str(timestamp_utc.month),
            "day": str(timestamp_utc.day),

            # We request the whole UTC day.
            #
            # The returned result consists of four pages:
            #   00-05
            #   06-11
            #   12-17
            #   18-23
            "tindex": str(timestamp_utc.hour),

            "nwlat": f"{config.nw_latitude:.4f}",
            "nwlng": f"{config.nw_longitude:.4f}",

            "steplat": f"{config.grid_step_lat:.4f}",
            "steplng": f"{config.grid_step_lon:.4f}",

            "nrows": str(config.grid_rows),
            "ncols": str(config.grid_cols),
        }

        for index, frequency in enumerate(
            config.frequencies_khz,
            start=1,
        ):
            params[f"freq{index}"] = str(frequency)

        # The CGI expects the remaining frequency fields to exist in some
        # versions, so leave freq10 empty for our nine-frequency request.
        if len(config.frequencies_khz) < 10:
            params["freq10"] = ""

        return (
            f"{SWS_HAP_CGI}?"
            f"{urlencode(params)}"
        )

    # ------------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------------

    def download_hap_html(
        self,
        config: HAPConfig,
        timestamp_utc: datetime,
    ) -> str:

        url = self.build_hap_url(
            config,
            timestamp_utc,
        )

        response = self.session.get(
            url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.text

    # ------------------------------------------------------------------------
    # IMAGE URL EXTRACTION
    # ------------------------------------------------------------------------

    def extract_image_urls(
        self,
        html: str,
    ) -> list[str]:
        """
        Extract HAP GIF image URLs from the CGI output.

        The SWS response normally contains four GIF links:

            hap1....gif  -> 00-05 UTC
            hap2....gif  -> 06-11 UTC
            hap3....gif  -> 12-17 UTC
            hap4....gif  -> 18-23 UTC
        """

        matches = re.findall(
            r'(?:src|href)=["\']([^"\']*hap[1-4][^"\']*\.gif)["\']',
            html,
            flags=re.IGNORECASE,
        )

        urls = []

        for match in matches:

            if match.startswith("http://"):
                url = match

            elif match.startswith("https://"):
                url = match

            elif match.startswith("/"):
                url = (
                    f"{SWS_BASE_URL}"
                    f"{match}"
                )

            else:
                url = (
                    f"{SWS_BASE_URL}/"
                    f"{match}"
                )

            if url not in urls:
                urls.append(url)

        return urls

    # ------------------------------------------------------------------------
    # CACHE
    # ------------------------------------------------------------------------

    def _cache_name(
        self,
        url: str,
    ) -> Path:

        digest = hashlib.sha256(
            url.encode("utf-8")
        ).hexdigest()[:20]

        return self.cache_dir / f"{digest}.gif"

    def download_image(
        self,
        url: str,
    ) -> Path:

        cache_path = self._cache_name(url)

        if cache_path.exists():
            return cache_path

        response = self.session.get(
            url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        cache_path.write_bytes(
            response.content
        )

        return cache_path

    # ------------------------------------------------------------------------
    # IMAGE PANEL EXTRACTION
    # ------------------------------------------------------------------------

    def extract_hour_panels(
        self,
        image: Image.Image,
    ) -> dict[int, Image.Image]:
        """
        Extract the six hourly map panels from a 700x900 SWS HAP image.

        Page layout:

            00   01
            02   03
            04   05

        The exact image currently produced by SWS has approximate map
        rectangles:

            x = 50..340 / 360..650
            y = 134..313
            y = 381..560
            y = 631..808

        We retain the coordinates here as an explicit decoder parameter
        rather than mixing them into the scoring logic.
        """

        width, height = image.size

        if width < 650 or height < 850:
            raise ValueError(
                f"Unexpected HAP image size: {width}x{height}"
            )

        boxes = [
            (0, (50, 134, 340, 313)),
            (1, (360, 134, 650, 313)),
            (2, (50, 381, 340, 560)),
            (3, (360, 381, 650, 560)),
            (4, (50, 631, 340, 808)),
            (5, (360, 631, 650, 808)),
        ]

        panels = {}

        for index, box in boxes:

            panels[index] = image.crop(box)

        return panels

    # ------------------------------------------------------------------------
    # COLOUR DECODING
    # ------------------------------------------------------------------------

    @staticmethod
    def nearest_hap_colour(
        rgb: tuple[int, int, int],
    ) -> tuple[Optional[tuple[int, int, int]], float]:
        """
        Find the closest known HAP colour.

        Returns:

            colour,
            similarity

        Similarity is a simple normalized RGB-distance metric.

        Exact HAP colours should normally produce similarity = 1.0.
        """

        best_colour = None
        best_distance = float("inf")

        for colour in HAP_COLOURS:

            distance = sum(
                (
                    rgb[index] - colour[index]
                ) ** 2
                for index in range(3)
            )

            if distance < best_distance:

                best_distance = distance
                best_colour = colour

        # Maximum RGB Euclidean distance.
        max_distance = (
            3 * (255 ** 2)
        ) ** 0.5

        distance = best_distance ** 0.5

        similarity = 1.0 - (
            distance / max_distance
        )

        return best_colour, similarity

    @staticmethod
    def dominant_colour(
        image: Image.Image,
        radius: int = 5,
    ) -> tuple[Optional[tuple[int, int, int]], float]:
        """
        Determine the dominant HAP colour near the centre of a map panel.

        This is intentionally only a diagnostic helper.

        The next decoder stage will replace this with geographic grid
        sampling once the map geometry is fully validated.
        """

        rgb_image = image.convert("RGB")

        width, height = rgb_image.size

        cx = width // 2
        cy = height // 2

        pixels = []

        for y in range(
            max(0, cy - radius),
            min(height, cy + radius + 1),
        ):

            for x in range(
                max(0, cx - radius),
                min(width, cx + radius + 1),
            ):

                rgb = rgb_image.getpixel(
                    (x, y)
                )

                if rgb in HAP_COLOURS:
                    pixels.append(rgb)

        if not pixels:
            return None, 0.0

        counts = {}

        for rgb in pixels:
            counts[rgb] = (
                counts.get(rgb, 0) + 1
            )

        colour = max(
            counts,
            key=counts.get,
        )

        confidence = (
            counts[colour] / len(pixels)
        )

        return colour, confidence

    # ------------------------------------------------------------------------
    # GEOGRAPHIC GRID DECODER
    # ------------------------------------------------------------------------

    def estimate_grid_pixel(
        self,
        panel: Image.Image,
        config: HAPConfig,
        row: int,
        col: int,
    ) -> tuple[int, int]:
        """
        Estimate the pixel position of a geographic grid point.

        IMPORTANT:
            This is currently based on proportional mapping.

            We are deliberately keeping this isolated so it can be replaced
            after validating the actual SWS map geometry.

        The map panel is treated as representing the requested geographic
        bounding box.
        """

        width, height = panel.size

        if config.grid_cols <= 1:
            x = width // 2
        else:
            x = round(
                col
                / (config.grid_cols - 1)
                * (width - 1)
            )

        if config.grid_rows <= 1:
            y = height // 2
        else:
            y = round(
                row
                / (config.grid_rows - 1)
                * (height - 1)
            )

        return x, y

    def sample_grid_point(
        self,
        panel: Image.Image,
        config: HAPConfig,
        row: int,
        col: int,
        radius: int = 4,
    ) -> tuple[
        Optional[tuple[int, int, int]],
        float,
    ]:
        """
        Sample a small area around an estimated geographic grid point.

        Only known HAP colours are considered.

        This reduces the chance of sampling text, borders, coastlines,
        or other map graphics.
        """

        rgb_image = panel.convert("RGB")

        width, height = rgb_image.size

        cx, cy = self.estimate_grid_pixel(
            panel,
            config,
            row,
            col,
        )

        counts: dict[
            tuple[int, int, int],
            int
        ] = {}

        for y in range(
            max(0, cy - radius),
            min(height, cy + radius + 1),
        ):

            for x in range(
                max(0, cx - radius),
                min(width, cx + radius + 1),
            ):

                rgb = rgb_image.getpixel(
                    (x, y)
                )

                if rgb in HAP_COLOURS:

                    counts[rgb] = (
                        counts.get(rgb, 0) + 1
                    )

        if not counts:
            return None, 0.0

        colour = max(
            counts,
            key=counts.get,
        )

        total = sum(counts.values())

        confidence = (
            counts[colour] / total
        )

        return colour, confidence

    def decode_panel(
        self,
        panel: Image.Image,
        config: HAPConfig,
        hour_utc: int,
    ) -> HAPHourResult:
        """
        Decode one hourly map.

        Produces one HAPGridPoint per geographic grid coordinate.
        """

        result = HAPHourResult(
            hour_utc=hour_utc,
        )

        for row, col, lat, lon in config.grid_coordinates():

            colour, confidence = (
                self.sample_grid_point(
                    panel,
                    config,
                    row,
                    col,
                )
            )

            point = HAPGridPoint(
                row=row,
                col=col,
                latitude=lat,
                longitude=lon,
                rgb=colour,
                confidence=confidence,
            )

            if colour in HAP_COLOURS:

                band, frequency = HAP_COLOURS[
                    colour
                ]

                point.band = band
                point.frequency_mhz = frequency
                point.supported = True

            else:

                point.notes.append(
                    "No HAP frequency colour detected."
                )

            result.grid_points.append(
                point
            )

        result.decoding_success = True

        return result

    # ------------------------------------------------------------------------
    # PAGE → HOURS
    # ------------------------------------------------------------------------

    def decode_page(
        self,
        image_path: Path,
        page_index: int,
        config: HAPConfig,
    ) -> dict[int, HAPHourResult]:
        """
        Decode one HAP page.

        Page numbering:

            1 = 00-05 UTC
            2 = 06-11 UTC
            3 = 12-17 UTC
            4 = 18-23 UTC
        """

        image = Image.open(
            image_path
        )

        panels = self.extract_hour_panels(
            image
        )

        starting_hour = (
            page_index - 1
        ) * 6

        results = {}

        for panel_index, panel in panels.items():

            hour = (
                starting_hour
                + panel_index
            )

            results[hour] = (
                self.decode_panel(
                    panel,
                    config,
                    hour,
                )
            )

            results[hour].image_path = (
                image_path
            )

        return results

    # ------------------------------------------------------------------------
    # FULL COLLECTION
    # ------------------------------------------------------------------------

    def collect(
        self,
        config: HAPConfig,
        timestamp_utc: Optional[datetime] = None,
    ) -> HAPFrequencyResult:
        """
        Generate, download and decode the HAP for a location.
        """

        if timestamp_utc is None:

            timestamp_utc = datetime.now(
                timezone.utc
            )

        timestamp_utc = (
            timestamp_utc.astimezone(
                timezone.utc
            )
        )

        result = HAPFrequencyResult(
            config=config,
            source_url=None,
            retrieved_utc=datetime.now(
                timezone.utc
            ),
        )

        try:

            url = self.build_hap_url(
                config,
                timestamp_utc,
            )

            result.source_url = url

            print(
                "HAP request:"
            )
            print(
                url
            )

            html = self.download_hap_html(
                config,
                timestamp_utc,
            )

            image_urls = (
                self.extract_image_urls(
                    html
                )
            )

            if len(image_urls) < 4:

                raise RuntimeError(
                    "Expected four HAP GIF pages "
                    f"but found {len(image_urls)}."
                )

            image_urls = image_urls[:4]

            for page_number, image_url in enumerate(
                image_urls,
                start=1,
            ):

                print(
                    f"Downloading HAP page "
                    f"{page_number}/4..."
                )

                image_path = (
                    self.download_image(
                        image_url
                    )
                )

                hourly = (
                    self.decode_page(
                        image_path,
                        page_number,
                        config,
                    )
                )

                for hour, hour_result in hourly.items():

                    hour_result.image_url = (
                        image_url
                    )

                    result.hourly[hour] = (
                        hour_result
                    )

            result.success = True

        except Exception as exc:

            result.success = False
            result.error = str(exc)

        return result


# ============================================================================
# PUBLIC API
# ============================================================================

def collect_hap_data(
    latitude: float,
    longitude: float,
    base_name: str = "Base",
    timestamp_utc: Optional[datetime] = None,
) -> HAPFrequencyResult:
    """
    Public helper used by main.py and future Discord/API code.
    """

    config = HAPConfig(
        latitude=latitude,
        longitude=longitude,
        base_name=base_name,
    )

    collector = HAPCollector()

    return collector.collect(
        config,
        timestamp_utc,
    )


# ============================================================================
# DEBUG / VALIDATION OUTPUT
# ============================================================================

def print_hap_grid(
    result: HAPFrequencyResult,
    hour: int,
) -> None:
    """
    Print a decoded HAP grid for one UTC hour.
    """

    if hour not in result.hourly:

        print(
            f"No HAP data available for {hour:02d} UTC."
        )

        return

    hourly = result.hourly[hour]

    config = result.config

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

    # Longitude header.

    print(
        "Latitude \\ Longitude",
        end=" "
    )

    for col in range(
        config.grid_cols
    ):

        lon = config.grid_longitude(
            col
        )

        print(
            f"{lon:>8.2f}",
            end=""
        )

    print()

    print(
        "-" * 78
    )

    points = {
        (
            point.row,
            point.col,
        ): point
        for point in hourly.grid_points
    }

    for row in range(
        config.grid_rows
    ):

        lat = config.grid_latitude(
            row
        )

        print(
            f"{lat:>8.2f}",
            end=" "
        )

        for col in range(
            config.grid_cols
        ):

            point = points.get(
                (row, col)
            )

            if point is None:

                label = "?"

            elif point.band:

                label = point.band

            else:

                label = "--"

            print(
                f"{label:>8}",
                end=""
            )

        print()

    print()

    print(
        "Base location:"
    )

    print(
        f"  {config.base_name}"
    )

    print(
        f"  Latitude : {config.latitude:.4f}"
    )

    print(
        f"  Longitude: {config.longitude:.4f}"
    )

    print()

    print(
        "Centre grid point:"
    )

    print(
        f"  Row: {config.centre_row}"
    )

    print(
        f"  Col: {config.centre_col}"
    )

    print(
        f"  Lat: {config.grid_latitude(config.centre_row):.4f}"
    )

    print(
        f"  Lon: {config.grid_longitude(config.centre_col):.4f}"
    )


def print_hap_summary(
    result: HAPFrequencyResult,
) -> None:
    """
    Print a compact summary of all decoded HAP hours.
    """

    print()
    print(
        "=" * 78
    )

    print(
        "HAP FREQUENCY SUMMARY"
    )

    print(
        "=" * 78
    )

    for hour in sorted(
        result.hourly
    ):

        hourly = result.hourly[
            hour
        ]

        counts = hourly.band_counts()

        print()

        print(
            f"{hour:02d} UTC"
        )

        for band, frequency in BANDS.items():

            count = counts.get(
                band,
                0,
            )

            total = (
                result.config.grid_rows
                * result.config.grid_cols
            )

            percentage = (
                count
                / total
                * 100
            )

            print(
                f"  {band:>4} "
                f"{frequency:>7.3f} MHz : "
                f"{count:2d}/{total} "
                f"grid points "
                f"({percentage:5.1f}%)"
            )


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":

    print(
        "=" * 78
    )

    print(
        "SWS HAP CENTRED DECODER TEST"
    )

    print(
        "=" * 78
    )

    print()

    # ------------------------------------------------------------------------
    # TEST LOCATION
    # ------------------------------------------------------------------------
    #
    # Nelson, NZ.
    #
    # The HAP grid will automatically be calculated around this location.

    latitude = -41.27
    longitude = 173.28

    base_name = "Nelson"

    print(
        f"Location: {base_name}"
    )

    print(
        f"Latitude : {latitude}"
    )

    print(
        f"Longitude: {longitude}"
    )

    print()

    config = HAPConfig(
        latitude=latitude,
        longitude=longitude,
        base_name=base_name,
    )

    print(
        "Calculated HAP geographic grid:"
    )

    print(
        f"  NW latitude : "
        f"{config.nw_latitude:.4f}"
    )

    print(
        f"  NW longitude: "
        f"{config.nw_longitude:.4f}"
    )

    print(
        f"  Step        : "
        f"{config.grid_step_lat}° / "
        f"{config.grid_step_lon}°"
    )

    print(
        f"  Grid        : "
        f"{config.grid_rows} × "
        f"{config.grid_cols}"
    )

    print(
        f"  Centre      : "
        f"row {config.centre_row}, "
        f"col {config.centre_col}"
    )

    print()

    print(
        "Collecting HAP data..."
    )

    print()

    collector = HAPCollector()

    result = collector.collect(
        config
    )

    if not result.success:

        print()

        print(
            "HAP COLLECTION FAILED"
        )

        print(
            f"Error: {result.error}"
        )

        raise SystemExit(1)

    print()

    print(
        "HAP collection successful."
    )

    print(
        f"Decoded hours: "
        f"{len(result.hourly)}"
    )

    # ------------------------------------------------------------------------
    # Print the 12 UTC grid.
    # ------------------------------------------------------------------------

    if 12 in result.hourly:

        print_hap_grid(
            result,
            12,
        )

    # ------------------------------------------------------------------------
    # Print the 15 UTC grid.
    # ------------------------------------------------------------------------

    if 15 in result.hourly:

        print_hap_grid(
            result,
            15,
        )

    # ------------------------------------------------------------------------
    # Print all-hour summary.
    # ------------------------------------------------------------------------

    print_hap_summary(
        result
    )

    print()

    print(
        "=" * 78
    )

    print(
        "DONE"
    )

    print(
        "=" * 78
    )


# ============================================================================
# HAP PIXEL MAP DIAGNOSTIC
# ============================================================================

def print_hap_pixel_map(
    image_path: Path,
    page_index: int = 3,
    hour_in_page: int = 3,
    step: int = 4,
) -> None:
    """
    Print a coarse ASCII representation of an HAP map.

    page_index:
        1 = 00-05 UTC
        2 = 06-11 UTC
        3 = 12-17 UTC
        4 = 18-23 UTC

    hour_in_page:
        0-5

    step:
        Number of source pixels represented by one ASCII character.

    This is purely a diagnostic tool. It is NOT the final decoder.
    """

    image = Image.open(image_path)

    panels = HAPCollector().extract_hour_panels(image)

    if hour_in_page not in panels:
        print(
            f"Invalid hour_in_page: {hour_in_page}"
        )
        return

    panel = panels[hour_in_page].convert("RGB")

    colour_chars = {
        (255, 255, 0): "Y",      # 160m
        (255, 0, 0): "R",        # 80m
        (128, 128, 0): "O",      # 40m
        (0, 128, 0): "G",        # 30m
        (0, 255, 255): "C",      # 20m
        (0, 255, 0): "L",        # 17m
        (0, 128, 128): "T",      # 15m
        (0, 0, 255): "B",        # 12m
        (0, 0, 128): "N",        # 10m
        (255, 255, 255): ".",    # white/background
        (0, 0, 0): "#",          # black/border/text
    }

    width, height = panel.size

    print()
    print("=" * 100)
    print("HAP PIXEL MAP DIAGNOSTIC")
    print("=" * 100)

    print()
    print(
        f"Image: {image_path}"
    )

    print(
        f"Panel: {width} × {height}"
    )

    actual_hour = (
        (page_index - 1) * 6
        + hour_in_page
    )

    print(
        f"UTC hour: {actual_hour:02d}"
    )

    print(
        f"Pixel sampling step: {step}"
    )

    print()

    print(
        "Legend:"
    )

    print(
        "  Y = 160m   R = 80m   O = 40m   G = 30m   C = 20m"
    )

    print(
        "  L = 17m    T = 15m   B = 12m   N = 10m"
    )

    print(
        "  . = white/background"
    )

    print(
        "  # = black/border/text"
    )

    print()

    # ------------------------------------------------------------------------
    # Print column coordinate markers.
    # ------------------------------------------------------------------------

    print(
        "X coordinate:"
    )

    print(
        "    ",
        end=""
    )

    for x in range(
        0,
        width,
        step,
    ):

        print(
            str(x // 100 % 10),
            end=""
        )

    print()

    print(
        "    ",
        end=""
    )

    for x in range(
        0,
        width,
        step,
    ):

        print(
            str(x // 10 % 10),
            end=""
        )

    print()

    print(
        "    ",
        end=""
    )

    for x in range(
        0,
        width,
        step,
    ):

        print(
            str(x % 10),
            end=""
        )

    print()

    # ------------------------------------------------------------------------
    # Sample image.
    #
    # Instead of taking the first pixel in each block, count the colours
    # inside the block and use the dominant recognised HAP colour.
    # ------------------------------------------------------------------------

    for y in range(
        0,
        height,
        step,
    ):

        print(
            f"{y:03d} ",
            end=""
        )

        for x in range(
            0,
            width,
            step,
        ):

            counts = {}

            for yy in range(
                y,
                min(y + step, height),
            ):

                for xx in range(
                    x,
                    min(x + step, width),
                ):

                    rgb = panel.getpixel(
                        (xx, yy)
                    )

                    if rgb in colour_chars:

                        counts[rgb] = (
                            counts.get(rgb, 0)
                            + 1
                        )

            if not counts:

                char = " "

            else:

                dominant = max(
                    counts,
                    key=counts.get,
                )

                char = colour_chars[
                    dominant
                ]

            print(
                char,
                end=""
            )

        print()

    print()

    print("=" * 100)
    print("END PIXEL MAP")
    print("=" * 100)
    print()


# ============================================================================
# RUN PIXEL MAP TEST
# ============================================================================

if __name__ == "__main__":

    # Find the most recently cached GIF.
    #
    # This is only for diagnostics. We will replace this with a proper
    # reference to the page returned by the collector later.

    cache_files = sorted(
        CACHE_DIR.glob("*.gif"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not cache_files:

        print(
            "No cached HAP GIFs found."
        )

        print(
            "Run the HAP collection test first."
        )

        raise SystemExit(1)

    image_path = cache_files[0]

    print_hap_pixel_map(
        image_path=image_path,
        page_index=3,
        hour_in_page=3,
        step=10,
    )