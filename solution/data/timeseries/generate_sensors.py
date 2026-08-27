"""Generate the synthetic environmental-sensor dataset for the time-series agent.

Deterministic: seeded RNG plus a fixed epoch, so re-running reproduces the file
byte for byte. That matters because the anomaly counts are asserted in tests and
quoted in the evaluation write-up.

The three provided datasets (structured/, unstructured/, multimodal/) are course
assets and must not change. This is a new, additional source that belongs to the
fourth agent alone.

Usage:
    python generate_sensors.py          # writes neighborhood_sensors.csv
"""

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

NEIGHBORHOODS = ["Ashford", "Huntington", "Kingsley", "Maplewood", "Rosedale"]

# 14 days of hourly readings per neighborhood = 1,680 rows.
EPOCH = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
HOURS = 24 * 14
SEED = 20260301

# Per-neighborhood baselines. Maplewood is deliberately the noisiest and dirtiest
# so the evaluation prompt about air quality there has something to find.
BASELINES = {
    #                pm25   noise_db  water_gal
    "Ashford":      (8.0,   44.0,     310.0),
    "Huntington":   (11.0,  48.0,     355.0),
    "Kingsley":     (9.5,   46.0,     330.0),
    "Maplewood":    (16.0,  57.0,     420.0),
    "Rosedale":     (10.0,  45.0,     340.0),
}

# (neighborhood, hour offset, field, multiplier) — injected anomalies the agent
# is expected to flag. Kept few and large so detection is unambiguous.
ANOMALIES = [
    ("Maplewood", 78, "pm25", 6.5),
    ("Maplewood", 79, "pm25", 5.8),
    ("Maplewood", 201, "pm25", 4.9),
    ("Ashford", 150, "noise_db", 1.9),
    ("Rosedale", 96, "water_gal", 4.2),
    ("Kingsley", 300, "pm25", 5.1),
]


def diurnal(hour: int, amplitude: float) -> float:
    """Daily rhythm: quiet overnight, peaking mid-afternoon."""
    return amplitude * math.sin((hour % 24 - 6) / 24 * 2 * math.pi)


def main() -> None:
    rng = random.Random(SEED)
    anomaly_index = {(n, h, f): m for n, h, f, m in ANOMALIES}
    rows = []

    for neighborhood in NEIGHBORHOODS:
        pm_base, noise_base, water_base = BASELINES[neighborhood]

        for hour in range(HOURS):
            ts = EPOCH + timedelta(hours=hour)

            pm25 = pm_base + diurnal(hour, pm_base * 0.25) + rng.gauss(0, pm_base * 0.08)
            noise = noise_base + diurnal(hour, 6.0) + rng.gauss(0, 1.5)
            water = water_base + diurnal(hour, water_base * 0.30) + rng.gauss(0, water_base * 0.06)

            values = {"pm25": pm25, "noise_db": noise, "water_gal": water}
            flagged = []
            for field in values:
                multiplier = anomaly_index.get((neighborhood, hour, field))
                if multiplier:
                    values[field] *= multiplier
                    flagged.append(field)

            rows.append(
                {
                    "neighborhood": neighborhood,
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "pm25": f"{max(values['pm25'], 0.0):.2f}",
                    "noise_db": f"{max(values['noise_db'], 0.0):.2f}",
                    "water_gal": f"{max(values['water_gal'], 0.0):.2f}",
                    "sensor_id": f"{neighborhood[:3].upper()}-ENV-{hour % 3 + 1:02d}",
                    # Ground truth for the tests only. The agent never reads this
                    # column; it has to rediscover the anomalies statistically.
                    "injected_anomaly": ",".join(flagged),
                }
            )

    out = Path(__file__).parent / "neighborhood_sensors.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        # LF, not the csv module's default CRLF — the provided course datasets are
        # LF and a mixed repo makes diffs and checksums noisy.
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    injected = sum(1 for r in rows if r["injected_anomaly"])
    print(f"Wrote {out} — {len(rows)} rows, {injected} injected anomalies.")


if __name__ == "__main__":
    main()
