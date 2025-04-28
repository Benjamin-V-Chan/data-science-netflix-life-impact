import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

def main():
    base = Path(__file__).parent.parent
    feats = pd.read_csv(base / "outputs" / "features.csv")
    senti = pd.read_csv(base / "outputs" / "sentiment_topics.csv")
    X = feats.join(senti.filter(like="topic_")).dropna()
    y = senti.loc[X.index, "suggested_pct"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    # feature importances
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:20]
    fig = plt.figure()
    plt.barh(imp.index, imp.values)
    plt.title("Top 20 Feature Importances")
    fig.savefig(base / "outputs" / "figures" / "feature_importances.png", bbox_inches="tight")
 
    # save model & metrics
    joblib.dump(model, base / "outputs" / "model.pkl")
    with open(base / "outputs" / "metrics.json", "w") as f:
        json.dump({"rmse": rmse, "r2": r2}, f)

if __name__ == "__main__":
    main()
