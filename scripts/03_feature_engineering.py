import pandas as pd
import numpy as np
from pathlib import Path

def time_to_minutes(ts):
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) + int(parts[1]) / 60
    elif len(parts) == 3:
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
    return None

def main():
    base = Path(__file__).parent.parent
    df = pd.read_csv(base / "outputs" / "cleaned_data.csv")
    df["insight_minute"] = df["insight_timestamp"].apply(time_to_minutes)
    df["advice_word_count"] = df["meaningful_advice"].str.split().str.len()
    df["highlights_word_count"] = df["review_highlights"].str.split().str.len()
    discovered_dummies = pd.get_dummies(df["how_discovered"], prefix="disc")
    genre_dummies = pd.get_dummies(df["genre"].str.replace("/", "_"), prefix="genre")
    features = pd.concat([
        df[[
            "average_rating", "number_of_reviews",
            "suggested_pct", "insight_minute",
            "advice_word_count", "highlights_word_count"
        ]],
        discovered_dummies, genre_dummies
    ], axis=1)
    features.to_csv(base / "outputs" / "features.csv", index=False)

if __name__ == "__main__":
    main()
