import pandas as pd
from textblob import TextBlob

def sentiment_counts(df):
    """Overall sentiment distribution."""
    return df['sentiment'].value_counts().reset_index()

def sentiment_by_entity(df, top_n=10):
    """Sentiment distribution grouped by entity/brand."""
    top_entities = df['entity'].value_counts().head(top_n).index
    filtered = df[df['entity'].isin(top_entities)]
    grouped = filtered.groupby(['entity', 'sentiment']).size().reset_index(name='count')
    return grouped

def add_polarity_score(df):
    """Add TextBlob polarity score to each tweet."""
    def get_polarity(text):
        if not isinstance(text, str) or text.strip() == "":
            return 0.0
        return TextBlob(text).sentiment.polarity

    df = df.copy()
    df['polarity'] = df['processed_text'].apply(get_polarity)
    return df

def sentiment_trend_by_entity(df, entity_name):
    """Filter tweets for a specific entity."""
    return df[df['entity'].str.lower() == entity_name.lower()]

def top_words_per_sentiment(df, sentiment_label, n=30):
    """Get most frequent words for a given sentiment label."""
    from collections import Counter
    subset = df[df['sentiment'] == sentiment_label]['processed_text']
    all_words = ' '.join(subset).split()
    return Counter(all_words).most_common(n)


