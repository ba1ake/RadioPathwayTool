"""
band_favorability.py

HF band favorability scoring engine.

Version: 0.2

This module combines:
    - time of day / solar elevation
    - solar activity
    - geomagnetic conditions
    - Australian Space Weather Services ionospheric observations
    - HF disturbance indicators

The output is a heuristic favorability score from 0-100.

IMPORTANT:
This is not a validated propagation probability model.
The ionospheric observations currently available from SWS are
regional indicators rather than direct path-specific MUF measurements.

Future versions can become more accurate by incorporating:
    - numerical MUF / foF2 data
    - path-specific propagation models
    - PSK Reporter observations
    - WSPR observations
    - station-to-station geometry
    - VOACAP-style calculations
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import asin, cos, degrees, pi, sin
from typing import Optional


# ---------------------------------------------------------------------------
# BAND DEFINITIONS
# ---------------------------------------------------------------------------

BANDS = {
    "80m": {
        "frequency_mhz": 3.65,
        "lower_mhz": 3.5,
        "upper_mhz": 4.0,
    },
    "40m": {
        "frequency_mhz": 7.10,
        "lower_mhz": 7.0,
        "upper_mhz": 7.3,
    },
    "20m": {
        "frequency_mhz": 14.10,
        "lower_mhz": 14.0,
        "upper_mhz": 14.35,
    },
    "15m": {
        "frequency_mhz": 21.20,
        "lower_mhz": 21.0,
        "upper_mhz": 21.45,
    },
    "10m": {
        "frequency_mhz": 28.40,
        "lower_mhz": 28.0,
        "upper_mhz": 29.7,
    },
}


# ---------------------------------------------------------------------------
# BAND-SPECIFIC WEIGHTS
# ---------------------------------------------------------------------------

# These weights describe how strongly each factor influences the current
# favorability estimate for each band.
#
# They are intentionally heuristic and can be tuned later against real
# propagation observations.

BAND_WEIGHTS = {
    "80m": {
        "time": 0.35,
        "solar": 0.10,
        "geomagnetic": 0.30,
        "ionosphere": 0.15,
    },
    "40m": {
        "time": 0.25,
        "solar": 0.15,
        "geomagnetic": 0.25,
        "ionosphere": 0.35,
    },
    "20m": {
        "time": 0.20,
        "solar": 0.20,
        "geomagnetic": 0.20,
        "ionosphere": 0.40,
    },
    "15m": {
        "time": 0.15,
        "solar": 0.20,
        "geomagnetic": 0.15,
        "ionosphere": 0.50,
    },
    "10m": {
        "time": 0.10,
        "solar": 0.20,
        "geomagnetic": 0.10,
        "ionosphere": 0.60,
    },
}


# ---------------------------------------------------------------------------
# REGIONAL STATION WEIGHTS
# ---------------------------------------------------------------------------

# These are broad regional relevance weights for a station operating from
# Nelson, NZ.
#
# They should NOT be interpreted as path-specific propagation weights.
# They simply prevent a station such as Davis Antarctica from having the
# same influence as a geographically more relevant southern-Pacific /
# Australia observation.

STATION_WEIGHTS = {
    "norfolk": 1.00,
    "hobart": 0.95,
    "canberra": 0.85,
    "niue": 0.85,
    "sydney": 0.75,
    "brisbane": 0.60,
    "townsville": 0.50,
    "darwin": 0.45,
    "learmonth": 0.35,
    "perth": 0.35,
    "cocos": 0.25,
    "casey": 0.20,
    "davis": 0.20,
    "mawson": 0.20,
}


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class PropagationInputs:
    """
    Inputs used by the propagation scoring engine.
    """

    latitude: float
    longitude: float
    timestamp_utc: datetime

    solar_flux_10_7: Optional[float] = None
    sunspot_number: Optional[float] = None

    k_index: Optional[float] = None
    a_index: Optional[float] = None
    dst_index: Optional[float] = None

    # Numerical ionospheric data.
    muf_mhz: Optional[float] = None
    fof2_mhz: Optional[float] = None
    t_index: Optional[float] = None

    # Disturbance indicators.
    xray_flux: Optional[float] = None
    hf_fadeout: bool = False
    polar_cap_absorption: bool = False

    # Future / external propagation observations.
    propagation_forecast_score: Optional[float] = None
    observed_activity_score: Optional[float] = None

    # SWS station observations.
    ionosphere_observations: Optional[dict] = None


@dataclass
class BandFavorability:
    """
    Result for a single HF band.
    """

    band: str
    frequency_mhz: float
    score: float
    rating: str
    confidence: str
    reasons: list[str]
    data_completeness: float


# ---------------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------------

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def rating_from_score(score: float) -> str:
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
    if completeness >= 80:
        return "High"
    if completeness >= 55:
        return "Moderate"

    return "Low"


# ---------------------------------------------------------------------------
# SOLAR POSITION
# ---------------------------------------------------------------------------

def solar_elevation(
    latitude: float,
    longitude: float,
    timestamp_utc: datetime,
) -> float:
    """
    Approximate solar elevation in degrees.

    This is deliberately lightweight rather than a high-precision
    astronomical calculation.
    """

    day_of_year = timestamp_utc.timetuple().tm_yday

    # Approximate solar declination.
    declination = 23.44 * sin(
        2 * pi * (284 + day_of_year) / 365
    )

    # UTC decimal hour.
    utc_hour = (
        timestamp_utc.hour
        + timestamp_utc.minute / 60
        + timestamp_utc.second / 3600
    )

    # Approximate solar time.
    solar_time = utc_hour + longitude / 15.0

    # Normalize to 0-24 hours.
    solar_time %= 24

    hour_angle = 15 * (solar_time - 12)

    lat_rad = pi * latitude / 180
    dec_rad = pi * declination / 180
    hour_rad = pi * hour_angle / 180

    elevation = asin(
        sin(lat_rad) * sin(dec_rad)
        + cos(lat_rad) * cos(dec_rad) * cos(hour_rad)
    )

    return degrees(elevation)


# ---------------------------------------------------------------------------
# TIME / DAYLIGHT SCORE
# ---------------------------------------------------------------------------

def solar_day_score(
    band: str,
    latitude: float,
    longitude: float,
    timestamp_utc: datetime,
) -> tuple[float, str]:
    """
    Estimate how favorable the current solar elevation is for the band.
    """

    elevation = solar_elevation(
        latitude,
        longitude,
        timestamp_utc,
    )

    if band in {"10m", "15m", "20m"}:

        if elevation > 30:
            return 1.00, f"Sun well above horizon ({elevation:.0f}°)"

        if elevation > 15:
            return 0.90, f"Daylight conditions ({elevation:.0f}°)"

        if elevation > 0:
            return 0.70, f"Low solar elevation ({elevation:.0f}°)"

        if elevation > -6:
            return 0.50, f"Twilight ({elevation:.0f}°)"

        return 0.25, f"Night-time conditions ({elevation:.0f}°)"

    if band == "40m":

        if elevation > 30:
            return 0.80, f"Daylight conditions ({elevation:.0f}°)"

        if elevation > 0:
            return 0.90, f"Transition conditions ({elevation:.0f}°)"

        return 1.00, f"Night-time conditions ({elevation:.0f}°)"

    # 80m
    if elevation > 30:
        return 0.60, f"Daylight conditions ({elevation:.0f}°)"

    if elevation > 0:
        return 0.75, f"Low solar elevation ({elevation:.0f}°)"

    return 1.00, f"Night-time conditions ({elevation:.0f}°)"


# ---------------------------------------------------------------------------
# SOLAR ACTIVITY
# ---------------------------------------------------------------------------

def solar_activity_score(
    band: str,
    solar_flux_10_7: Optional[float],
    sunspot_number: Optional[float],
) -> tuple[float, str]:

    if solar_flux_10_7 is not None:

        sfi = solar_flux_10_7

        # Normalized broad estimate.
        low = 65.0
        high = 220.0

        normalized = clamp(
            (sfi - low) / (high - low),
            0.0,
            1.0,
        )

        # Higher bands benefit more from increased solar activity.
        if band == "80m":
            score = 0.60 + 0.20 * normalized

        elif band == "40m":
            score = 0.60 + 0.25 * normalized

        elif band == "20m":
            score = 0.50 + 0.45 * normalized

        elif band == "15m":
            score = 0.35 + 0.60 * normalized

        else:  # 10m
            score = 0.25 + 0.70 * normalized

        return (
            clamp(score, 0.0, 1.0),
            f"SFI {sfi:.0f}",
        )

    if sunspot_number is not None:

        ssn = sunspot_number

        normalized = clamp(
            ssn / 180.0,
            0.0,
            1.0,
        )

        if band == "80m":
            score = 0.65 + 0.15 * normalized

        elif band == "40m":
            score = 0.60 + 0.25 * normalized

        elif band == "20m":
            score = 0.50 + 0.40 * normalized

        elif band == "15m":
            score = 0.35 + 0.55 * normalized

        else:
            score = 0.25 + 0.65 * normalized

        return (
            clamp(score, 0.0, 1.0),
            f"SSN {ssn:.0f}",
        )

    return 0.50, "Solar activity unavailable"


# ---------------------------------------------------------------------------
# GEOMAGNETIC CONDITIONS
# ---------------------------------------------------------------------------

def geomagnetic_score(
    k_index: Optional[float],
    band: str,
) -> tuple[float, str]:

    if k_index is None:
        return 0.50, "K-index unavailable"

    k = k_index

    if k <= 1:
        base = 1.00
        description = "Very quiet geomagnetic conditions"

    elif k <= 2:
        base = 0.95
        description = "Quiet geomagnetic conditions"

    elif k <= 3:
        base = 0.85
        description = "Generally quiet geomagnetic conditions"

    elif k <= 4:
        base = 0.70
        description = "Unsettled geomagnetic conditions"

    elif k <= 5:
        base = 0.55
        description = "Active geomagnetic conditions"

    elif k <= 6:
        base = 0.35
        description = "Strong geomagnetic disturbance"

    else:
        base = 0.15
        description = "Severe geomagnetic disturbance"

    # Lower bands are generally more resilient to geomagnetic disturbance.
    if band == "80m":
        score = 0.70 + 0.30 * base

    elif band == "40m":
        score = 0.60 + 0.40 * base

    elif band == "20m":
        score = 0.50 + 0.50 * base

    elif band == "15m":
        score = 0.40 + 0.60 * base

    else:
        score = 0.30 + 0.70 * base

    return clamp(score, 0.0, 1.0), f"{description} (K={k:.1f})"


# ---------------------------------------------------------------------------
# IONOSPHERIC OBSERVATIONS
# ---------------------------------------------------------------------------

def _condition_value(
    condition: str,
    percent_difference: Optional[float],
) -> float:
    """
    Convert an SWS textual condition into a neutralized score.

    0.50 = neutral
    >0.50 = enhanced
    <0.50 = depressed

    This deliberately avoids treating the SWS percentage as a direct
    path MUF percentage.
    """

    text = condition.lower()

    if "absorption" in text:
        return 0.20

    if "no vertical muf data" in text:
        return 0.50

    if "enhanced" in text:

        if percent_difference is not None:
            adjustment = clamp(
                percent_difference / 100.0,
                0.0,
                0.35,
            )

            return clamp(
                0.60 + adjustment,
                0.0,
                0.95,
            )

        return 0.65

    if "depressed" in text:

        if percent_difference is not None:
            adjustment = clamp(
                abs(percent_difference) / 100.0,
                0.0,
                0.35,
            )

            return clamp(
                0.45 - adjustment,
                0.05,
                0.50,
            )

        return 0.40

    if "near normal" in text:
        return 0.55

    return 0.50


def ionosphere_condition_score(
    observations: Optional[dict],
    band: str,
) -> tuple[float, str, int]:
    """
    Calculate a regional ionospheric score from SWS station observations.

    Returns:
        score
        explanation
        number of usable observations
    """

    if not observations:
        return (
            0.50,
            "No SWS ionospheric observations available",
            0,
        )

    weighted_total = 0.0
    total_weight = 0.0

    enhanced = 0
    depressed = 0
    normal = 0
    absorption = 0

    for station_key, observation in observations.items():

        weight = STATION_WEIGHTS.get(
            station_key.lower(),
            0.25,
        )

        if not getattr(observation, "data_available", False):
            continue

        condition = getattr(
            observation,
            "condition",
            "",
        )

        percent = getattr(
            observation,
            "percent_difference",
            None,
        )

        value = _condition_value(
            condition,
            percent,
        )

        weighted_total += value * weight
        total_weight += weight

        condition_lower = condition.lower()

        if "absorption" in condition_lower:
            absorption += 1

        elif "enhanced" in condition_lower:
            enhanced += 1

        elif "depressed" in condition_lower:
            depressed += 1

        elif "near normal" in condition_lower:
            normal += 1

    if total_weight == 0:
        return (
            0.50,
            "No usable SWS ionospheric observations",
            0,
        )

    base_score = weighted_total / total_weight

    # Higher bands should require stronger evidence of ionospheric support.
    #
    # We do NOT assume "near normal" means the 10m band is open.
    # Without numerical MUF or actual high-band observations, the result
    # remains conservative.

    if band == "10m":
        if enhanced == 0:
            base_score = min(base_score, 0.55)

    elif band == "15m":
        if enhanced == 0:
            base_score = min(base_score, 0.65)

    elif band == "20m":
        if enhanced == 0:
            base_score = min(base_score, 0.75)

    # 40m and 80m do not need this evidence gate.

    if absorption:
        explanation = (
            f"SWS: {enhanced} enhanced, "
            f"{normal} near normal, "
            f"{depressed} depressed, "
            f"{absorption} absorption"
        )

    elif enhanced:
        explanation = (
            f"SWS: {enhanced} enhanced, "
            f"{normal} near normal, "
            f"{depressed} depressed"
        )

    else:
        explanation = (
            f"SWS: {normal} near normal, "
            f"{depressed} depressed, "
            f"{absorption} absorption"
        )

    return (
        clamp(base_score, 0.0, 1.0),
        explanation,
        enhanced + depressed + normal + absorption,
    )


# ---------------------------------------------------------------------------
# NUMERICAL MUF
# ---------------------------------------------------------------------------

def muf_score(
    band_frequency_mhz: float,
    muf_mhz: Optional[float],
) -> tuple[float, str]:

    if muf_mhz is None:
        return 0.50, "Numerical MUF unavailable"

    frequency = band_frequency_mhz

    if frequency <= muf_mhz * 0.70:
        score = 1.00
        description = "Well below estimated MUF"

    elif frequency <= muf_mhz * 0.90:
        score = 0.90
        description = "Below estimated MUF"

    elif frequency <= muf_mhz:
        score = 0.70
        description = "Near estimated MUF"

    else:
        score = 0.20
        description = "Above estimated MUF"

    return score, description


# ---------------------------------------------------------------------------
# DISTURBANCE
# ---------------------------------------------------------------------------

def disturbance_penalty(
    inputs: PropagationInputs,
) -> tuple[float, list[str]]:

    penalty = 0.0
    reasons = []

    if inputs.hf_fadeout:
        penalty += 0.35
        reasons.append("HF fadeout reported")

    if inputs.polar_cap_absorption:
        penalty += 0.25
        reasons.append("Polar-cap absorption reported")

    if inputs.xray_flux is not None:

        if inputs.xray_flux >= 1e-4:
            penalty += 0.35
            reasons.append("Strong X-ray activity")

        elif inputs.xray_flux >= 1e-5:
            penalty += 0.20
            reasons.append("Elevated X-ray activity")

    return (
        clamp(penalty, 0.0, 0.80),
        reasons,
    )


# ---------------------------------------------------------------------------
# DATA COMPLETENESS
# ---------------------------------------------------------------------------

def calculate_data_completeness(
    inputs: PropagationInputs,
) -> float:

    fields = [
        inputs.solar_flux_10_7,
        inputs.sunspot_number,
        inputs.k_index,
        inputs.a_index,
        inputs.dst_index,
    ]

    available = sum(
        value is not None
        for value in fields
    )

    total = len(fields)

    # Ionosphere observations are a separate category.
    ionosphere_available = bool(
        inputs.ionosphere_observations
    )

    score = available / total

    if ionosphere_available:
        score += 0.20

    # Numerical MUF is particularly valuable.
    if inputs.muf_mhz is not None:
        score += 0.15

    if inputs.fof2_mhz is not None:
        score += 0.10

    return clamp(
        score * 100,
        0.0,
        100.0,
    )


# ---------------------------------------------------------------------------
# BAND CALCULATION
# ---------------------------------------------------------------------------

def calculate_band_favorability(
    band: str,
    inputs: PropagationInputs,
) -> BandFavorability:

    if band not in BANDS:
        raise ValueError(
            f"Unknown band: {band}"
        )

    band_info = BANDS[band]
    frequency = band_info["frequency_mhz"]

    weights = BAND_WEIGHTS[band]

    reasons = []

    # ---------------------------------------------------------------
    # Time / solar elevation
    # ---------------------------------------------------------------

    time_score, time_reason = solar_day_score(
        band,
        inputs.latitude,
        inputs.longitude,
        inputs.timestamp_utc,
    )

    reasons.append(time_reason)

    # ---------------------------------------------------------------
    # Solar activity
    # ---------------------------------------------------------------

    solar_score, solar_reason = solar_activity_score(
        band,
        inputs.solar_flux_10_7,
        inputs.sunspot_number,
    )

    reasons.append(solar_reason)

    # ---------------------------------------------------------------
    # Geomagnetic conditions
    # ---------------------------------------------------------------

    geo_score, geo_reason = geomagnetic_score(
        inputs.k_index,
        band,
    )

    reasons.append(geo_reason)

    # ---------------------------------------------------------------
    # Ionosphere
    # ---------------------------------------------------------------

    ionosphere_score, ionosphere_reason, observation_count = (
        ionosphere_condition_score(
            inputs.ionosphere_observations,
            band,
        )
    )

    reasons.append(ionosphere_reason)

    # ---------------------------------------------------------------
    # Numerical MUF
    # ---------------------------------------------------------------

    numerical_muf_score, muf_reason = muf_score(
        frequency,
        inputs.muf_mhz,
    )

    if inputs.muf_mhz is not None:
        # Numerical MUF is more trustworthy than the generic regional
        # condition indicator, so blend it into the ionosphere component.
        ionosphere_score = (
            ionosphere_score * 0.35
            + numerical_muf_score * 0.65
        )

        reasons.append(muf_reason)

    # ---------------------------------------------------------------
    # Weighted base score
    # ---------------------------------------------------------------

    base_score = (
        time_score * weights["time"]
        + solar_score * weights["solar"]
        + geo_score * weights["geomagnetic"]
        + ionosphere_score * weights["ionosphere"]
    )

    # ---------------------------------------------------------------
    # Disturbances
    # ---------------------------------------------------------------

    penalty, disturbance_reasons = disturbance_penalty(
        inputs
    )

    reasons.extend(disturbance_reasons)

    final_score = base_score * (1.0 - penalty)

    final_score = clamp(
        final_score * 100,
        0.0,
        100.0,
    )

    # ---------------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------------

    completeness = calculate_data_completeness(
        inputs
    )

    # More observations improve confidence in the ionospheric component.
    if observation_count >= 5:
        completeness += 5

    completeness = clamp(
        completeness,
        0.0,
        100.0,
    )

    confidence = confidence_from_completeness(
        completeness
    )

    # ---------------------------------------------------------------
    # Add explicit high-band uncertainty
    # ---------------------------------------------------------------

    if band in {"10m", "15m"}:
        if inputs.muf_mhz is None:
            reasons.append(
                "No numerical MUF: high-band assessment is conservative"
            )

    return BandFavorability(
        band=band,
        frequency_mhz=frequency,
        score=round(final_score, 1),
        rating=rating_from_score(final_score),
        confidence=confidence,
        reasons=reasons,
        data_completeness=round(
            completeness,
            1,
        ),
    )


# ---------------------------------------------------------------------------
# ALL BANDS
# ---------------------------------------------------------------------------

def calculate_all_bands(
    inputs: PropagationInputs,
) -> dict[str, BandFavorability]:

    return {
        band: calculate_band_favorability(
            band,
            inputs,
        )
        for band in BANDS
    }


# ---------------------------------------------------------------------------
# TEST / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    test_inputs = PropagationInputs(
        latitude=-41.27,
        longitude=173.28,
        timestamp_utc=datetime.now(),
        solar_flux_10_7=120,
        sunspot_number=80,
        k_index=2,
        a_index=8,
        dst_index=-10,
    )

    results = calculate_all_bands(
        test_inputs
    )

    print("=" * 70)
    print("HF BAND FAVORABILITY TEST")
    print("=" * 70)

    for result in results.values():

        print(
            f"{result.band:>4} "
            f"{result.frequency_mhz:>6.2f} MHz  "
            f"{result.score:>5.1f}/100  "
            f"{result.rating:<9} "
            f"confidence={result.confidence}"
        )

        for reason in result.reasons:
            print(f"      - {reason}")

        print()