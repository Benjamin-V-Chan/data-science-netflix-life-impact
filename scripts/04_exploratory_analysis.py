import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_fig(fig, name):
    fig.savefig(name, bbox_inches="tight")

def main():
    base = Path(__file__).parent.parent
    features = pd.read_csv(base / "outputs" / "features.csv")
    figures = base / "outputs" / "figures"
    figures.mkdir(exist_ok=True)
    
    # Rating distribution
    fig = plt.figure()
    plt.hist(features["average_rating"], bins=10)
    plt.title("Distribution of Average Ratings")
    save_fig(fig, figures / "rating_distribution.png")
    
    # Genre counts (extract from columns)
    genre_cols = [c for c in features.columns if c.startswith("genre_")]
    counts = features[genre_cols].sum().sort_values(ascending=False)
    fig = plt.figure()
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Movie Count by Genre")
    save_fig(fig, figures / "genre_counts.png")
    
    # Suggested % vs Rating
    fig = plt.figure()
    plt.scatter(features["average_rating"], features["suggested_pct"])
    plt.xlabel("Average Rating")
    plt.ylabel("Suggested %")
    plt.title("Suggested % vs Rating")
    save_fig(fig, figures / "suggested_vs_rating.png")
  
    # Mean suggested % by discovery channel
    disc_cols = [c for c in features.columns if c.startswith("disc_")]
    means = {
        c.replace("disc_", ""): (features[c] * features["suggested_pct"]).sum() / features[c].sum()
        for c in disc_cols
    }
    fig = plt.figure()
    plt.bar(means.keys(), means.values())
    plt.xticks(rotation=45, ha="right")
    plt.title("Mean Suggested % by Discovery Channel")
    save_fig(fig, figures / "suggested_by_discovery.png")

if __name__ == "__main__":
    main()
