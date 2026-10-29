
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(rows=1500, random_state=42):
    rng = np.random.default_rng(random_state)

    locations = [
        "City Center",
        "Market",
        "Near Railway station",
        "Near Metro",
        "IT Hub",
    ]

    location_premium = {
        "City Center": 30,
        "Market": 12,
        "Near Railway station": 0,
        "Near Metro": 24,
        "IT Hub": 20,
    }

    furnishing_premium = {
        "Unfurnished": 0,
        "Semi-Furnished": 5,
        "Furnished": 10,
    }

    area = rng.integers(500, 4001, rows)

    bedrooms = np.clip(
        np.round(area / 650 + rng.normal(0, 0.8, rows)),
        1,
        6,
    ).astype(int)

    bathrooms = np.clip(
        bedrooms + rng.integers(-1, 2, rows),
        1,
        5,
    )

    age = rng.integers(0, 41, rows)

    location = rng.choice(
        locations,
        rows,
        p=[0.22, 0.28, 0.18, 0.16, 0.16],
    )

    parking = rng.integers(0, 4, rows)

    furnishing = rng.choice(
        list(furnishing_premium.keys()),
        rows,
        p=[0.35, 0.40, 0.25],
    )

    price = (
        8
        + area * 0.055
        + bedrooms * 2.5
        + bathrooms * 1.5
        + np.array([location_premium[x] for x in location])
        + np.array([furnishing_premium[x] for x in furnishing])
        + parking * 1.2
        - age * 0.35
        + rng.normal(0, 8, rows)
    )

    price = np.maximum(price, 10)

    return pd.DataFrame({
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "location": location,
        "parking": parking,
        "furnishing": furnishing,
        "price_lakh": np.round(price, 2),
    })


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    output_path = output_dir / "house_prices.csv"

    df.to_csv(output_path, index=False)

    print("Dataset generated successfully!")
    print(f"Location: {output_path}")
    print(f"Total records: {len(df)}")
    print(df.head())
