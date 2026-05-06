#  Twitter Sentiment Analysis 

Analyze and visualize sentiment patterns in social media (Twitter) data to understand
public opinion and attitudes toward specific topics and brands.

## Objective

- Classify tweets as **Positive**, **Negative**, **Neutral**, or **Irrelevant**
- Identify which brands/entities receive the most positive or negative attention
- Visualize sentiment distribution at both macro (all tweets) and micro (per entity) levels
- Use NLP techniques to extract meaningful patterns from raw text


##  Dataset

**Source:** [Prodigy InfoTech Data Science Datasets — Task 4](https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%204)


## 🛠️ Tech Stack

| Tool | Purpose |
| Python 3.9+ | Core language |
| Pandas | Data loading & manipulation |
| NLTK | Stopword removal, lemmatization |
| TextBlob | Polarity scoring |
| Matplotlib / Seaborn | Static charts |
| WordCloud | Word frequency visualization |
| Scikit-learn | (Optional) ML classification |


## Run

pip install -r requirements.txt

## Run the analysis
python main.py


##  Visualizations Generated

| Chart | File | Description |

| Bar chart | 'sentiment_distribution.png' | Count of each sentiment label |
| Pie chart | 'sentiment_pie.png' | Percentage share of sentiments |
| Top entities | 'top_entities.png' | Most discussed brands |
| Stacked bar | 'stacked_bar_entity.png' | Sentiment split per brand |
| Heatmap | 'heatmap.png'| Sentiment intensity by entity |
| Word clouds | 'wordcloud_positive/negative/neutral.png' | Most common words per sentiment |
| KDE plot | 'polarity_distribution.png' | TextBlob polarity score distribution |


## Key Findings (sample - update after running)

- **X% of tweets** are Positive; **Y%** are Negative
- Brand **A** has the highest positive sentiment ratio
- Brand **B** generates the most negative discussions
- Positive tweets commonly include words like: 'love', 'great', 'awesome'
- Negative tweets commonly include words like: 'bad', 'hate', 'worst'
