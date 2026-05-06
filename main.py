from src.preprocess import load_data, preprocess
from src.analyze import (
    sentiment_counts,
    sentiment_by_entity,
    add_polarity_score,
    top_words_per_sentiment
)
from src.visualize import (
    plot_sentiment_distribution,
    plot_sentiment_pie,
    plot_stacked_bar_by_entity,
    plot_heatmap,
    plot_wordcloud,
    plot_polarity_distribution,
    plot_top_entities
)

#  1. Load 
print("Loading data...")
train_df, val_df = load_data(
    'data/twitter_training.csv'
)

df = train_df.copy()   # use training set for full analysis

# 2. Preprocess 
print("Preprocessing...")
df = preprocess(df)
print(f"Clean dataset shape: {df.shape}")
print(df['sentiment'].value_counts())

# 3. Add polarity scores 
print("Computing polarity scores (TextBlob)...")
df = add_polarity_score(df)

# 4 . Aggregate
counts    = sentiment_counts(df)
by_entity = sentiment_by_entity(df, top_n=10)

# 5. Visualize 
print("Generating visualizations...")
plot_sentiment_distribution(counts)
plot_sentiment_pie(counts)
plot_top_entities(df)
plot_stacked_bar_by_entity(by_entity)
plot_heatmap(by_entity)
plot_polarity_distribution(df)

for label in ['Positive', 'Negative', 'Neutral']:
    plot_wordcloud(df, label)

print("Done! All charts saved in /outputs")