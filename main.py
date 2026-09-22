from datetime import datetime

from space_weather import collect_space_weather
from ionosphere import get_all_ionosphere_observations
from band_favorability import calculate_all_bands, PropagationInputs


# --------------------------------------------------
# Collect data
# --------------------------------------------------

weather = collect_space_weather()
ionosphere = get_all_ionosphere_observations()


# --------------------------------------------------
# Prepare propagation inputs
# --------------------------------------------------

inputs = PropagationInputs(
    latitude=-41.27,
    longitude=173.28,

    timestamp_utc=datetime.fromisoformat(
        weather.retrieved_utc
    ),

    solar_flux_10_7=weather.solar_flux_10_7,
    sunspot_number=weather.sunspot_number,

    k_index=(
        weather.australian_k_index
        if weather.australian_k_index is not None
        else weather.planetary_k_index
    ),

    a_index=weather.a_index,
    dst_index=weather.dst_index,

    xray_flux=weather.xray_flux,
    hf_fadeout=weather.hf_fadeout,
    polar_cap_absorption=weather.polar_cap_absorption,
)


# --------------------------------------------------
# Calculate band favourability
# --------------------------------------------------

results = calculate_all_bands(inputs)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("HF PROPAGATION ANALYSIS")
print("=" * 70)

for result in results.values():
    print(
        f"{result.band:>4} "
        f"{result.frequency_mhz:>6.2f} MHz  "
        f"{result.score:>3}/100  "
        f"{result.rating}"
    )

# --------------------------------------------------
# Display ionospheric observations
# --------------------------------------------------

print("\n" + "=" * 70)
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