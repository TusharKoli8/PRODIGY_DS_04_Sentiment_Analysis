import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import os

COLORS = {
    'Positive': "#2ecc4b",
    'Negative': '#e74c3c',
    'Neutral':  '#3498db',
    'Irrelevant': '#95a5a6'
}
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_sentiment_distribution(counts_df, save=True):
    """Bar chart of overall sentiment distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts_df['sentiment'],
        counts_df['count'],
        color=[COLORS.get(s, '#999') for s in counts_df['sentiment']],
        edgecolor='white', linewidth=0.8
    )
    ax.bar_label(bars, padding=3, fontsize=11, fontweight='bold')
    ax.set_title('Overall Sentiment Distribution', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Number of Tweets', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/sentiment_distribution.png', dpi=150)
    plt.show()

def plot_sentiment_pie(counts_df, save=True):
    """Pie chart showing percentage of each sentiment."""
    fig, ax = plt.subplots(figsize=(7, 7))
    wedge_colors = [COLORS.get(s, '#999') for s in counts_df['sentiment']]
    ax.pie(
        counts_df['count'],
        labels=counts_df['sentiment'],
        autopct='%1.1f%%',
        colors=wedge_colors,
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    ax.set_title('Sentiment Share (%)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/sentiment_pie.png', dpi=150)
    plt.show()

def plot_stacked_bar_by_entity(grouped_df, save=True):
    """Stacked bar chart of sentiment per entity/brand."""
    pivot = grouped_df.pivot(index='entity', columns='sentiment', values='count').fillna(0)
    pivot = pivot[[c for c in ['Positive', 'Negative', 'Neutral', 'Irrelevant'] if c in pivot.columns]]
    ax = pivot.plot(
        kind='bar', stacked=True, figsize=(14, 6),
        color=[COLORS.get(c, '#999') for c in pivot.columns],
        edgecolor='white', linewidth=0.5
    )
    ax.set_title('Sentiment Distribution per Brand/Entity', fontsize=15, fontweight='bold')
    ax.set_xlabel('Entity', fontsize=12)
    ax.set_ylabel('Tweet Count', fontsize=12)
    ax.legend(title='Sentiment', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/stacked_bar_entity.png', dpi=150)
    plt.show()

def plot_heatmap(grouped_df, save=True):
    """Heatmap of sentiment counts per entity."""
    pivot = grouped_df.pivot(index='entity', columns='sentiment', values='count').fillna(0)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5, ax=ax)
    ax.set_title('Sentiment Heatmap by Entity', fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/heatmap.png', dpi=150)
    plt.show()

def plot_wordcloud(df, sentiment_label, save=True):
    """Word cloud for a specific sentiment."""
    text = ' '.join(df[df['sentiment'] == sentiment_label]['processed_text'].dropna())
    if not text.strip():
        print(f"No text found for sentiment: {sentiment_label}")
        return
    wc = WordCloud(
        width=900, height=450,
        background_color='white',
        colormap='RdYlGn' if sentiment_label == 'Positive' else 'Reds',
        max_words=100
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud — {sentiment_label} Tweets', fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save:
        fname = sentiment_label.lower()
        plt.savefig(f'{OUTPUT_DIR}/wordcloud_{fname}.png', dpi=150)
    plt.show()

def plot_polarity_distribution(df, save=True):
    """Histogram of TextBlob polarity scores by sentiment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for sentiment, color in COLORS.items():
        subset = df[df['sentiment'] == sentiment]['polarity']
        if not subset.empty:
            subset.plot.kde(ax=ax, label=sentiment, color=color, linewidth=2)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Neutral (0)')
    ax.set_title('Polarity Score Distribution by Sentiment', fontsize=15, fontweight='bold')
    ax.set_xlabel('Polarity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(title='Sentiment')
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/polarity_distribution.png', dpi=150)
    plt.show()

def plot_top_entities(df, top_n=15, save=True):
    """Bar chart of most discussed brands/entities."""
    top = df['entity'].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    top.plot(kind='bar', color='#5b8dee', edgecolor='white', ax=ax)
    ax.set_title(f'Top {top_n} Most Discussed Entities', fontsize=15, fontweight='bold')
    ax.set_xlabel('Entity', fontsize=12)
    ax.set_ylabel('Tweet Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/top_entities.png', dpi=150)
    plt.show()