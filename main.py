from datetime import datetime

from space_weather import collect_space_weather
from ionosphere import get_all_ionosphere_observations
from band_favorability import (
    calculate_all_bands,
    PropagationInputs,
)


# ---------------------------------------------------------------------------
# COLLECT LIVE DATA
# ---------------------------------------------------------------------------

weather = collect_space_weather()
ionosphere = get_all_ionosphere_observations()


# ---------------------------------------------------------------------------
# BUILD PROPAGATION INPUTS
# ---------------------------------------------------------------------------

inputs = PropagationInputs(
    latitude=-41.27,
    longitude=173.28,

    timestamp_utc=datetime.fromisoformat(
        weather.retrieved_utc
    ),

    # Solar conditions
    solar_flux_10_7=weather.solar_flux_10_7,
    sunspot_number=weather.sunspot_number,

    # Prefer Australian K-index for this project when available.
    k_index=(
        weather.australian_k_index
        if weather.australian_k_index is not None
        else weather.planetary_k_index
    ),

    a_index=weather.a_index,
    dst_index=weather.dst_index,

    # Disturbance indicators
    xray_flux=weather.xray_flux,
    hf_fadeout=weather.hf_fadeout,
    polar_cap_absorption=weather.polar_cap_absorption,

    # Live Australian SWS ionospheric observations
    ionosphere_observations=ionosphere,
)


# ---------------------------------------------------------------------------
# CALCULATE BAND FAVORABILITY
# ---------------------------------------------------------------------------

results = calculate_all_bands(inputs)


# ---------------------------------------------------------------------------
# DISPLAY BAND RESULTS
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("HF PROPAGATION ANALYSIS")
print("=" * 70)

for result in results.values():

    print(
        f"{result.band:>4} "
        f"{result.frequency_mhz:>6.2f} MHz  "
        f"{result.score:>5.1f}/100  "
        f"{result.rating:<9} "
        f"Confidence: {result.confidence}"
    )

    for reason in result.reasons:
        print(f"      - {reason}")

    print()


# ---------------------------------------------------------------------------
# DISPLAY IONOSPHERIC OBSERVATIONS
# ---------------------------------------------------------------------------

print("=" * 70)
print("IONOSPHERIC OBSERVATIONS")
print("=" * 70)

for observation in ionosphere.values():

    if observation.percent_difference is not None:

        condition = (
            f"{observation.condition} "
            f"({observation.percent_difference:+.0f}%)"
        )

    else:
        condition = observation.condition

    print(
        f"{observation.station_name:<15} "
        f"{condition}"
    )


# ---------------------------------------------------------------------------
# DISPLAY DATA SOURCES / STATUS
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA STATUS")
print("=" * 70)

print(
    f"Space Weather SWS available: "
    f"{weather.sws_available}"
)

if weather.sws_error:
    print(
        f"SWS error: {weather.sws_error}"
    )

print(
    f"Space weather retrieved: "
    f"{weather.retrieved_utc}"
)

print(
    f"Ionospheric observations: "
    f"{len(ionosphere)} stations"
)

print("=" * 70)