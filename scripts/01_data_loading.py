import pandas as pd
from pathlib import Path

def main():
    data_path = Path(__file__).parent.parent / "data" / "Netflix Life Impact Dataset (NLID).csv"
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    out = Path(__file__).parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    df.to_pickle(out / "raw_data.pkl")

if __name__ == "__main__":
    main()
