import pandas as pd
from pathlib import Path
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def main():
    base = Path(__file__).parent.parent
    df = pd.read_csv(base / "outputs" / "cleaned_data.csv")
    # sentiment
    df["sentiment_polarity"] = df["meaningful_advice"].apply(lambda t: TextBlob(t).sentiment.polarity)
    df["sentiment_subjectivity"] = df["meaningful_advice"].apply(lambda t: TextBlob(t).sentiment.subjectivity)
    # TF-IDF + NMF topics
    vec = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf = vec.fit_transform(df["meaningful_advice"])
    nmf = NMF(n_components=5, random_state=0)
    topics = nmf.fit_transform(tfidf)
    topic_df = pd.DataFrame(topics, columns=[f"topic_{i}" for i in range(5)])
    # save
    df_sent = pd.concat([df, topic_df], axis=1)
    df_sent.to_csv(base / "outputs" / "sentiment_topics.csv", index=False)
    # save topic words
    topic_words = {}
    for i, comp in enumerate(nmf.components_):
        top = [vec.get_feature_names_out()[j] for j in comp.argsort()[-10:]][::-1]
        topic_words[f"topic_{i}"] = top
    pd.DataFrame(topic_words).to_csv(base / "outputs" / "topic_words.csv", index=False)

if __name__ == "__main__":
    main()
