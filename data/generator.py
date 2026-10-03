import numpy as np
import pandas as pd


def generate_microgrid_data(days=365, seed=42):
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=days * 24,
        freq="h"
    )

    hours = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()


    base_load = 1.5

    morning_peak = 1.5 * np.exp(-((hours - 8) / 2.5) ** 2)
    evening_peak = 2.5 * np.exp(-((hours - 19) / 3.0) ** 2)

    daily_variation = rng.normal(0, 0.15, len(timestamps))

    load = (
        base_load
        + morning_peak
        + evening_peak
        + daily_variation
    )

    load = np.clip(load, 0.5, None)


    solar = np.zeros(len(timestamps))

    daylight = (hours >= 6) & (hours <= 18)

    solar_angle = (
        (hours[daylight] - 6) / 12
    ) * np.pi

    solar[daylight] = (
        5.0 * np.sin(solar_angle)
    )


    seasonal_factor = (
        0.8
        + 0.2 * np.sin(
            2 * np.pi * (day_of_year - 80) / 365
        )
    )

    solar *= seasonal_factor

    cloud_factor = rng.uniform(
        0.7,
        1.0,
        len(timestamps)
    )

    solar *= cloud_factor

    solar = np.clip(solar, 0, None)


    price = np.full(
        len(timestamps),
        6.0
    )

    # Morning peak
    price[(hours >= 6) & (hours < 10)] = 8.0

    # Afternoon
    price[(hours >= 10) & (hours < 17)] = 6.0

    # Evening peak
    price[(hours >= 17) & (hours < 22)] = 12.0

    # Night
    price[(hours >= 22) | (hours < 6)] = 5.0

    # Small random price variation
    price += rng.normal(
        0,
        0.3,
        len(timestamps)
    )

    price = np.clip(price, 3.0, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "load_kw": load,
        "solar_kw": solar,
        "grid_price": price
    })

    return df


if __name__ == "__main__":
    df = generate_microgrid_data()

    df.to_csv(
        "data/raw/microgrid.csv",
        index=False
    )

    print(f"Generated {len(df)} rows")
    print(df.head())