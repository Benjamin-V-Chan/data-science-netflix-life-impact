import pandas as pd
import re
from pathlib import Path

def parse_suggested(val):
    m = re.match(r"(\d+)%\s*Y", val)
    return (int(m.group(1)) if m else None, bool(m))

def split_insight(val):
    parts = val.split("–", 1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else (None)

def main():
    base = Path(__file__).parent.parent
    raw = pd.read_pickle(base / "outputs" / "raw_data.pkl")
    df = raw.rename(columns={
        "Movie Title": "movie_title",
        "Genre": "genre",
        "Release Year": "release_year",
        "Average Rating": "average_rating",
        "Number of Reviews": "number_of_reviews",
        "Review Highlights": "review_highlights",
        "Minute of Life-Changing Insight": "minute_of_insight",
        "How Discovered": "how_discovered",
        "Meaningful Advice Taken": "meaningful_advice",
        "Suggested to Friends/Family (Y/N %)": "suggested_raw"
    })
    df = df.dropna(subset=["movie_title", "genre", "average_rating"])
    df["release_year"] = df["release_year"].astype(int)
    df["average_rating"] = df["average_rating"].astype(float)
    df["number_of_reviews"] = df["number_of_reviews"].astype(int)
    df[["suggested_pct", "suggested_flag"]] = df["suggested_raw"].apply(lambda x: pd.Series(parse_suggested(x)))
    df[["insight_timestamp", "insight_event"]] = df["minute_of_insight"].apply(lambda x: pd.Series(split_insight(x)))
    out = base / "outputs"
    df.to_csv(out / "cleaned_data.csv", index=False)

if __name__ == "__main__":
    main()
