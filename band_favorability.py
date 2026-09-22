
"""
HF Band Favorability Engine
===========================

V0.1

Purpose:
    Take solar, geomagnetic, ionospheric and time/location inputs and produce
    a simple 0-100 favorability score for several HF bands.

Supported bands:
    80m, 40m, 20m, 15m, 10m

IMPORTANT:
    This is an initial framework.

    The numerical scoring weights are placeholders. They are NOT intended to
    represent a scientifically validated probability of making a contact.

    The next stages of the project will replace the placeholder inputs with
    real data from online sources such as:

        - Australian Space Weather Services (SWS)
        - NOAA Space Weather Prediction Center (SWPC)
        - HF propagation / ionospheric products
        - Propagation observation services

The analysis engine is intentionally separated from data collection so that
the Discord bot, website, or future SDR monitoring system can all use the
same engine.
"""

from dataclasses import dataclass
from datetime import datetime
from math import asin, cos, radians, sin
from typing import Optional


# ============================================================================
# BAND DEFINITIONS
# ============================================================================

BANDS = {
    "80m": {
        "frequency_mhz": 3.65,
        "lower_mhz": 3.5,
        "upper_mhz": 4.0,
    },

    "40m": {
        "frequency_mhz": 7.1,
        "lower_mhz": 7.0,
        "upper_mhz": 7.3,
    },

    "20m": {
        "frequency_mhz": 14.1,
        "lower_mhz": 14.0,
        "upper_mhz": 14.35,
    },

    "15m": {
        "frequency_mhz": 21.2,
        "lower_mhz": 21.0,
        "upper_mhz": 21.45,
    },

    "10m": {
        "frequency_mhz": 28.4,
        "lower_mhz": 28.0,
        "upper_mhz": 29.7,
    },
}


# ============================================================================
# INPUT DATA
# ============================================================================

@dataclass
class PropagationInputs:
    """
    Inputs used by the propagation engine.

    Most of these fields will eventually be populated automatically from
    online sources.

    None means that the value is currently unavailable.

    It is important that unavailable data is represented by None rather than
    zero, because zero can itself be a legitimate measurement.
    """

    # ------------------------------------------------------------------------
    # Location / time
    # ------------------------------------------------------------------------

    latitude: float
    longitude: float
    timestamp_utc: datetime

    # ------------------------------------------------------------------------
    # Solar activity
    #
    # Potential future sources:
    #   SWS / NOAA
    # ------------------------------------------------------------------------

    solar_flux_10_7: Optional[float] = None
    sunspot_number: Optional[float] = None

    # ------------------------------------------------------------------------
    # Geomagnetic activity
    #
    # Potential future sources:
    #   SWS / NOAA
    # ------------------------------------------------------------------------

    k_index: Optional[float] = None
    a_index: Optional[float] = None
    dst_index: Optional[float] = None

    # ------------------------------------------------------------------------
    # Ionospheric / propagation information
    #
    # Potential future sources:
    #   SWS
    #   ionospheric monitoring services
    #   propagation prediction services
    # ------------------------------------------------------------------------

    muf_mhz: Optional[float] = None
    fof2_mhz: Optional[float] = None
    t_index: Optional[float] = None

    # ------------------------------------------------------------------------
    # Solar radiation / HF disturbance indicators
    # ------------------------------------------------------------------------

    xray_flux: Optional[float] = None

    hf_fadeout: bool = False
    polar_cap_absorption: bool = False

    # ------------------------------------------------------------------------
    # Future observation/prediction inputs
    #
    # These allow us to eventually integrate services such as PSK Reporter,
    # WSPR or other real-world observations.
    # ------------------------------------------------------------------------

    propagation_forecast_score: Optional[float] = None
    observed_activity_score: Optional[float] = None


# ============================================================================
# OUTPUT DATA
# ============================================================================

@dataclass
class BandFavorability:
    """
    Result produced for one band.
    """

    band: str
    frequency_mhz: float

    score: float
    rating: str
    confidence: str

    reasons: list[str]

    data_completeness: float


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    """Keep a value inside a defined range."""

    return max(minimum, min(maximum, value))


def rating_from_score(score: float) -> str:
    """
    Convert the numerical score into a simple human-readable rating.
    """

    if score >= 85:
        return "Excellent"

    if score >= 70:
        return "Good"

    if score >= 50:
        return "Fair"

    if score >= 30:
        return "Marginal"

    return "Poor"


def confidence_from_completeness(completeness: float) -> str:
    """
    Estimate confidence based on how much input data was available.

    This is deliberately simple for V0.1.
    """

    if completeness >= 80:
        return "High"

    if completeness >= 55:
        return "Moderate"

    return "Low"


# ============================================================================
# SOLAR POSITION
# ============================================================================

def solar_elevation(
    latitude: float,
    longitude: float,
    timestamp_utc: datetime,
) -> float:
    """
    Calculate a rough solar elevation.

    This is intentionally lightweight.

    It is NOT intended to replace a proper astronomical calculation. The
    purpose here is simply to give the first scoring engine a rough idea of
    whether the location is in daylight, twilight or darkness.

    We can replace this with a proper solar-position calculation later.
    """

    # Day of year
    day = timestamp_utc.timetuple().tm_yday

    # Approximate solar declination
    declination = 23.44 * sin(
        radians((360 / 365.0) * (day - 81))
    )

    # Approximate local solar time
    utc_hours = (
        timestamp_utc.hour
        + timestamp_utc.minute / 60
        + timestamp_utc.second / 3600
    )

    solar_time = (
        utc_hours + longitude / 15.0
    ) % 24

    # Solar hour angle
    hour_angle = 15 * (solar_time - 12)

    altitude = asin(
        sin(radians(latitude))
        * sin(radians(declination))
        +
        cos(radians(latitude))
        * cos(radians(declination))
        * cos(radians(hour_angle))
    )

    return altitude * 180 / 3.141592653589793


# ============================================================================
# TIME / DAYLIGHT SCORING
# ============================================================================

def solar_day_score(
    inputs: PropagationInputs,
    band: str,
) -> tuple[float, str]:
    """
    Estimate the effect of daylight/night on a band.

    This is intentionally broad.

    A future version should use proper propagation/path models rather than
    simply classifying the location as day/night.
    """

    elevation = solar_elevation(
        inputs.latitude,
        inputs.longitude,
        inputs.timestamp_utc,
    )

    # Higher HF bands
    if band in ("10m", "15m", "20m"):

        if elevation > 15:
            return (
                1.00,
                "Daylight currently favours the higher HF bands.",
            )

        if elevation > -6:
            return (
                0.65,
                "Near sunrise/sunset may provide useful transition conditions.",
            )

        return (
            0.30,
            "Night-time conditions generally reduce ionisation for the higher bands.",
        )

    # 40 metres
    if band == "40m":

        if elevation > 15:
            return (
                0.75,
                "40m can support useful daytime and regional propagation.",
            )

        return (
            0.95,
            "40m often remains useful after sunset.",
        )

    # 80 metres
    if elevation > 15:
        return (
            0.60,
            "80m can support regional propagation during daylight.",
        )

    return (
        0.95,
        "Night-time conditions are often favourable for 80m.",
    )


# ============================================================================
# SOLAR ACTIVITY
# ============================================================================

def solar_activity_score(
    inputs: PropagationInputs,
    band: str,
) -> tuple[float, str]:
    """
    Estimate the effect of solar activity.

    This currently uses Solar Flux Index / F10.7 as a placeholder.

    The actual relationship between solar flux and propagation is more
    complicated than this and should be calibrated against real propagation
    observations later.
    """

    if inputs.solar_flux_10_7 is None:
        return (
            0.50,
            "Solar flux data unavailable.",
        )

    sfi = inputs.solar_flux_10_7

    # Higher-frequency bands benefit more from strong ionisation.
    if band in ("10m", "15m", "20m"):

        score = clamp(
            (sfi - 60) / 140,
            0.0,
            1.0,
        )

        return (
            score,
            f"Solar flux: {sfi:.0f} sfu.",
        )

    if band == "40m":

        score = clamp(
            (sfi - 60) / 180,
            0.0,
            1.0,
        )

        return (
            score,
            f"Solar flux: {sfi:.0f} sfu.",
        )

    # 80m is much less dependent on strong solar activity.
    return (
        0.70,
        f"Solar flux: {sfi:.0f} sfu; 80m is less dependent on high solar flux.",
    )


# ============================================================================
# GEOMAGNETIC CONDITIONS
# ============================================================================

def geomagnetic_score(
    inputs: PropagationInputs,
) -> tuple[float, str]:
    """
    Estimate the effect of geomagnetic conditions.

    K-index is a 0-9 index.

    Lower values generally indicate quieter geomagnetic conditions.

    This is a simplified model and should eventually be calibrated against
    actual HF propagation data.
    """

    if inputs.k_index is None:
        return (
            0.50,
            "Geomagnetic K-index unavailable.",
        )

    k = inputs.k_index

    if k <= 1:

        return (
            1.00,
            f"Very quiet geomagnetic conditions. K={k:.1f}.",
        )

    if k <= 3:

        return (
            0.85,
            f"Quiet to unsettled geomagnetic conditions. K={k:.1f}.",
        )

    if k <= 5:

        return (
            0.60,
            f"Elevated geomagnetic activity may reduce HF reliability. K={k:.1f}.",
        )

    if k <= 6:

        return (
            0.35,
            f"Disturbed geomagnetic conditions. K={k:.1f}.",
        )

    return (
        0.15,
        f"Strong geomagnetic disturbance. K={k:.1f}.",
    )


# ============================================================================
# MUF
# ============================================================================

def muf_score(
    inputs: PropagationInputs,
    band: str,
) -> tuple[float, str]:
    """
    Compare the operating frequency against a supplied MUF.

    This is potentially one of the most important components of the final
    system.

    In later versions we want path-specific MUF rather than a single generic
    MUF value.
    """

    if inputs.muf_mhz is None:
        return (
            0.50,
            "MUF data unavailable.",
        )

    frequency = BANDS[band]["frequency_mhz"]
    muf = inputs.muf_mhz

    if frequency <= muf * 0.70:

        return (
            1.00,
            f"Operating frequency is comfortably below the MUF ({muf:.1f} MHz).",
        )

    if frequency <= muf * 0.90:

        return (
            0.90,
            f"Operating frequency is below the MUF ({muf:.1f} MHz).",
        )

    if frequency <= muf:

        return (
            0.65,
            f"Operating frequency is close to the MUF ({muf:.1f} MHz).",
        )

    return (
        0.20,
        f"Operating frequency is above the supplied MUF ({muf:.1f} MHz).",
    )


# ============================================================================
# HF DISTURBANCE PENALTIES
# ============================================================================

def disturbance_penalty(
    inputs: PropagationInputs,
) -> tuple[float, str]:
    """
    Apply penalties for major HF disturbance indicators.
    """

    penalty = 0.0
    reasons = []

    if inputs.hf_fadeout:

        penalty += 0.35

        reasons.append(
            "HF fadeout is currently indicated."
        )

    if inputs.polar_cap_absorption:

        penalty += 0.25

        reasons.append(
            "Polar-cap absorption is indicated."
        )

    if inputs.xray_flux is not None:

        # Conservative placeholder threshold.
        if inputs.xray_flux >= 1e-5:

            penalty += 0.20

            reasons.append(
                "Elevated X-ray flux may increase HF absorption."
            )

    if not reasons:

        return (
            0.0,
            "No major supplied HF disturbance indicators.",
        )

    return (
        min(penalty, 0.80),
        " ".join(reasons),
    )


# ============================================================================
# MAIN BAND CALCULATION
# ============================================================================

def calculate_band_favorability(
    band: str,
    inputs: PropagationInputs,
) -> BandFavorability:
    """
    Calculate the favorability of one HF band.

    Returns:
        BandFavorability

    Current weighting:

        Time/daylight       20%
        Solar activity      25%
        Geomagnetic         25%
        MUF                 30%

    These weights are deliberately provisional.
    """

    if band not in BANDS:

        raise ValueError(
            f"Unknown band '{band}'. "
            f"Available bands: {', '.join(BANDS)}"
        )

    components = []
    reasons = []

    # ------------------------------------------------------------------------
    # Time / daylight
    # ------------------------------------------------------------------------

    day_score, day_reason = solar_day_score(
        inputs,
        band,
    )

    components.append(
        ("time", day_score, 0.20)
    )

    reasons.append(day_reason)

    # ------------------------------------------------------------------------
    # Solar activity
    # ------------------------------------------------------------------------

    solar_score, solar_reason = solar_activity_score(
        inputs,
        band,
    )

    components.append(
        ("solar", solar_score, 0.25)
    )

    reasons.append(solar_reason)

    # ------------------------------------------------------------------------
    # Geomagnetic conditions
    # ------------------------------------------------------------------------

    geo_score, geo_reason = geomagnetic_score(
        inputs
    )

    components.append(
        ("geomagnetic", geo_score, 0.25)
    )

    reasons.append(geo_reason)

    # ------------------------------------------------------------------------
    # MUF
    # ------------------------------------------------------------------------

    muf_component, muf_reason = muf_score(
        inputs,
        band,
    )

    components.append(
        ("muf", muf_component, 0.30)
    )

    reasons.append(muf_reason)

    # ------------------------------------------------------------------------
    # Weighted score
    # ------------------------------------------------------------------------

    total_weight = sum(
        weight
        for _, _, weight in components
    )

    weighted_score = sum(
        score * weight
        for _, score, weight in components
    )

    score = (
        weighted_score / total_weight
    ) * 100

    # ------------------------------------------------------------------------
    # Disturbance penalties
    # ------------------------------------------------------------------------

    penalty, penalty_reason = disturbance_penalty(
        inputs
    )

    score *= (
        1.0 - penalty
    )

    reasons.append(penalty_reason)

    score = round(
        clamp(score),
        1,
    )

    # ------------------------------------------------------------------------
    # Data completeness
    # ------------------------------------------------------------------------

    tracked_fields = [
        inputs.solar_flux_10_7,
        inputs.sunspot_number,
        inputs.k_index,
        inputs.a_index,
        inputs.dst_index,
        inputs.muf_mhz,
        inputs.fof2_mhz,
        inputs.t_index,
        inputs.xray_flux,
    ]

    available = sum(
        value is not None
        for value in tracked_fields
    )

    completeness = (
        available / len(tracked_fields)
    )

    completeness_percent = round(
        completeness * 100,
        1,
    )

    return BandFavorability(
        band=band,
        frequency_mhz=BANDS[band]["frequency_mhz"],
        score=score,
        rating=rating_from_score(score),
        confidence=confidence_from_completeness(
            completeness_percent
        ),
        reasons=reasons,
        data_completeness=completeness_percent,
    )


# ============================================================================
# ALL BANDS
# ============================================================================

def calculate_all_bands(
    inputs: PropagationInputs,
) -> dict[str, BandFavorability]:
    """
    Calculate favorability for all supported bands.
    """

    return {
        band: calculate_band_favorability(
            band,
            inputs,
        )
        for band in BANDS
    }


# ============================================================================
# TEST / DEVELOPMENT MODE
# ============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------------
    # Example location:
    #
    # Nelson, New Zealand
    #
    # These values are ONLY example values.
    # They are NOT live conditions.
    # ------------------------------------------------------------------------

    example_inputs = PropagationInputs(

        latitude=-41.2706,
        longitude=173.2837,

        timestamp_utc=datetime(
            2026,
            9,
            23,
            0,
            30,
        ),

        # --------------------------------------------------------------------
        # Example online data
        # --------------------------------------------------------------------

        solar_flux_10_7=140,
        sunspot_number=100,

        k_index=2,
        a_index=8,
        dst_index=-10,

        muf_mhz=25,
        fof2_mhz=8,
        t_index=50,

        xray_flux=1e-7,
    )

    results = calculate_all_bands(
        example_inputs
    )

    print()
    print("HF BAND FAVORABILITY")
    print("=" * 60)

    for band, result in results.items():

        print(
            f"{band:>4} | "
            f"{result.score:5.1f}/100 | "
            f"{result.rating:<9} | "
            f"Confidence: {result.confidence}"
        )

    print()
    print("DETAILS")
    print("=" * 60)

    for band, result in results.items():

        print()
        print(f"{band} ({result.frequency_mhz:.2f} MHz)")
        print("-" * 40)

        print(
            f"Score: {result.score}/100"
        )

        print(
            f"Rating: {result.rating}"
        )

        print(
            f"Confidence: {result.confidence}"
        )

        print(
            f"Data completeness: "
            f"{result.data_completeness}%"
        )

        print("Reasons:")

        for reason in result.reasons:

            print(
                f"  - {reason}"
            )
