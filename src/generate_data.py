"""
generate_data.py
----------------
Generates a realistic synthetic dataset for "Will My Plant Survive Me?"
Simulates plant health outcomes (Thrives / Survives / Doomed) based on
plant care habits, botanical preferences, and realistic environmental noise.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Set random seeds for reproducible synthetic generation
np.random.seed(42)

# Define project directories safely relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "plants.csv"

PLANT_TYPES = ["succulent", "fern", "cactus", "tropical", "flowering"]
HUMIDITY_LEVELS = ["low", "medium", "high"]
DRAINAGE_OPTIONS = ["yes", "no"]


def calculate_plant_health(
    plant_type: str,
    watering: int,
    sunlight: float,
    humidity: str,
    temp: float,
    forgetfulness: int,
    drainage: str,
) -> float:
    """
    Computes a deterministic botanical suitability score (0-100)
    based on plant requirements and user care habits.
    """
    score = 100.0

    # 1. Drainage penalty (improper drainage is deadly for arid/succulent plants)
    if drainage == "no":
        if plant_type in ["cactus", "succulent"]:
            score -= 35.0  # Fatal root rot risk
        else:
            score -= 20.0

    # 2. Watering & Sunlight rules per plant type
    if plant_type == "succulent":
        # Likes dry conditions, moderate-to-high light, tolerates neglect
        if watering > 2:
            score -= (watering - 2) * 15.0  # Overwatering penalty
        elif watering == 0:
            score -= 5.0

        if sunlight < 5:
            score -= (5 - sunlight) * 6.0
        elif sunlight > 10:
            score -= 5.0

        if humidity == "high":
            score -= 15.0
        elif humidity == "medium":
            score -= 5.0

        # Succulents forgive forgetfulness
        if forgetfulness > 7:
            score -= (forgetfulness - 7) * 3.0

        # Temperature preferences: 16 - 28 C
        if temp < 15:
            score -= (15 - temp) * 2.5
        elif temp > 32:
            score -= (temp - 32) * 2.0

    elif plant_type == "fern":
        # Loves moisture, high humidity, indirect light, hates neglect
        if watering < 3:
            score -= (3 - watering) * 18.0  # Underwatering penalty
        elif watering > 6:
            score -= (watering - 6) * 10.0

        # Direct sun scorches fronds
        if sunlight > 5:
            score -= (sunlight - 5) * 12.0
        elif sunlight < 1:
            score -= 10.0

        if humidity == "low":
            score -= 25.0  # Fronds turn brown and crispy
        elif humidity == "medium":
            score -= 8.0

        # Ferns hate forgetful owners
        score -= forgetfulness * 2.8

        # Temperature preferences: 15 - 24 C
        if temp < 13:
            score -= (13 - temp) * 3.0
        elif temp > 27:
            score -= (temp - 27) * 3.0

    elif plant_type == "cactus":
        # Desert plant: extremely low water, high sun, low humidity
        if watering > 1:
            score -= (watering - 1) * 22.0  # Water is poison for indoor cacti
        if sunlight < 6:
            score -= (6 - sunlight) * 7.0
        if humidity == "high":
            score -= 22.0
        elif humidity == "medium":
            score -= 10.0

        # Neglect is a virtue for cacti!
        if forgetfulness < 3:
            score -= 8.0

        # Temperature preferences: 18 - 35 C
        if temp < 14:
            score -= (14 - temp) * 3.5

    elif plant_type == "tropical":
        # Warm, humid, moderate-to-high indirect light, consistent watering
        if watering < 2:
            score -= (2 - watering) * 16.0
        elif watering > 5:
            score -= (watering - 5) * 12.0

        if sunlight < 3:
            score -= (3 - sunlight) * 8.0
        elif sunlight > 9:
            score -= (sunlight - 9) * 6.0

        if humidity == "low":
            score -= 20.0
        elif humidity == "medium":
            score -= 6.0

        if forgetfulness > 5:
            score -= (forgetfulness - 5) * 3.5

        # Tropical plants are very sensitive to cold drafts
        if temp < 17:
            score -= (17 - temp) * 3.5
        elif temp > 33:
            score -= (temp - 33) * 2.0

    elif plant_type == "flowering":
        # Balanced watering, generous sun to trigger blooms, stable conditions
        if watering < 2:
            score -= (2 - watering) * 14.0
        elif watering > 4:
            score -= (watering - 4) * 12.0

        if sunlight < 5:
            score -= (5 - sunlight) * 8.0
        elif sunlight > 10:
            score -= (sunlight - 10) * 5.0

        if humidity == "low":
            score -= 12.0
        elif humidity == "high":
            score -= 8.0

        if forgetfulness > 4:
            score -= (forgetfulness - 4) * 3.2

        # Temperature preferences: 16 - 26 C
        if temp < 15:
            score -= (15 - temp) * 2.5
        elif temp > 28:
            score -= (temp - 28) * 2.5

    return max(0.0, min(100.0, score))


def generate_plants_dataset(n_samples: int = 2200) -> pd.DataFrame:
    """
    Generates synthetic dataset combining rule-based botanical scores
    with calibrated Gaussian noise to simulate real-world unpredictable variance.
    """
    plant_type = np.random.choice(PLANT_TYPES, size=n_samples)
    watering_per_week = np.random.randint(0, 8, size=n_samples)  # 0 to 7
    sunlight_hours = np.round(np.random.uniform(0.0, 12.0, size=n_samples), 1)  # 0 to 12
    humidity = np.random.choice(HUMIDITY_LEVELS, size=n_samples, p=[0.35, 0.40, 0.25])
    temperature_c = np.round(np.random.uniform(10.0, 35.0, size=n_samples), 1)  # 10 to 35
    forgetfulness_score = np.random.randint(1, 11, size=n_samples)  # 1 to 10
    has_drainage = np.random.choice(DRAINAGE_OPTIONS, size=n_samples, p=[0.65, 0.35])

    records = []
    # Calibrated Gaussian noise (std dev = 5.0) ensures realistic overlap
    # achieving target 80-90% accuracy while preventing artificial 100% test accuracy.
    gaussian_noise = np.random.normal(loc=0.0, scale=5.0, size=n_samples)

    for i in range(n_samples):
        base_score = calculate_plant_health(
            plant_type=plant_type[i],
            watering=watering_per_week[i],
            sunlight=sunlight_hours[i],
            humidity=humidity[i],
            temp=temperature_c[i],
            forgetfulness=forgetfulness_score[i],
            drainage=has_drainage[i],
        )

        noisy_score = base_score + gaussian_noise[i]

        # Map to 3 classes
        if noisy_score >= 68.0:
            outcome = "Thrives"
        elif noisy_score >= 42.0:
            outcome = "Survives"
        else:
            outcome = "Doomed"

        records.append(
            {
                "plant_type": plant_type[i],
                "watering_per_week": int(watering_per_week[i]),
                "sunlight_hours": float(sunlight_hours[i]),
                "humidity": humidity[i],
                "temperature_c": float(temperature_c[i]),
                "forgetfulness_score": int(forgetfulness_score[i]),
                "has_drainage": has_drainage[i],
                "outcome": outcome,
            }
        )

    df = pd.DataFrame(records)
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic plant care dataset...")
    df = generate_plants_dataset(n_samples=2200)

    # Save to data/plants.csv
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved {len(df):,} rows to: {OUTPUT_FILE}")

    print("\n--- Class Distribution ---")
    dist = df["outcome"].value_counts()
    dist_pct = df["outcome"].value_counts(normalize=True) * 100
    for label in ["Thrives", "Survives", "Doomed"]:
        count = dist.get(label, 0)
        pct = dist_pct.get(label, 0.0)
        print(f"  {label:<10}: {count:>5} samples ({pct:.1f}%)")


if __name__ == "__main__":
    main()
