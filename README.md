# data-science-netflix-life-impact

## Project Overview

This project analyzes the Netflix Life Impact Dataset (NLID) to explore how transformative films leave lasting emotional, intellectual, and behavioral imprints on audiences. Through data preprocessing, feature engineering, exploratory analysis, text sentiment and topic modeling, and predictive modeling, we uncover insights into viewer engagement and the spread of meaningful advice.

## Folder Structure

```
project-root/
├── data/
│   └── Netflix Life Impact Dataset (NLID).csv
├── scripts/
│   ├── 01_data_loading.py
│   ├── 02_preprocessing.py
│   ├── 03_feature_engineering.py
│   ├── 04_exploratory_analysis.py
│   ├── 05_text_analysis.py
│   └── 06_modeling.py
└── outputs/
    ├── raw_data.pkl
    ├── cleaned_data.csv
    ├── features.csv
    ├── figures/
    │   ├── rating_distribution.png
    │   ├── genre_counts.png
    │   ├── suggested_vs_rating.png
    │   ├── suggested_by_discovery.png
    │   └── feature_importances.png
    ├── sentiment_topics.csv
    ├── topic_words.csv
    ├── model.pkl
    └── metrics.json
```

## Usage

1. **Setup the Project:**
   - Clone the repository.
   - Ensure you have Python installed.
   - Install required dependencies using the requirements.txt file.
   ```bash
   pip install -r requirements.txt
   ```

2. **Load Raw Data:**
   ```bash
   python scripts/01_data_loading.py
   ```

3. **Preprocess Data:**
   ```bash
   python scripts/02_preprocessing.py
   ```

4. **Engineer Features:**
   ```bash
   python scripts/03_feature_engineering.py
   ```

5. **Exploratory Analysis:**
   ```bash
   python scripts/04_exploratory_analysis.py
   ```

6. **Text Analysis (Sentiment & Topics):**
   ```bash
   python scripts/05_text_analysis.py
   ```

7. **Modeling (Predict Suggested %):**
   ```bash
   python scripts/06_modeling.py
   ```

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- textblob
- joblib

## Acknowledgments

dataset name: Netflix Life Impact Dataset (NLID)  
dataset author: Towhidul Islam  
dataset source: https://www.kaggle.com/datasets/towhid121/netflix-life-impact-dataset-nlid