import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data(train_path):
    """Load training CSV and split into train/validation."""
    df = pd.read_csv(train_path, header=None)
    df.columns = ['tweet_id', 'entity', 'sentiment', 'tweet_text']

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )

    print(f"Training set   : {len(train_df)} rows")
    print(f"Validation set : {len(val_df)} rows")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def clean_text(text):
    """Clean raw tweet text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # Remove URLs
    text = re.sub(r'@\w+', '', text)                     # Remove mentions
    text = re.sub(r'#\w+', '', text)                     # Remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)                 # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra spaces
    return text

def remove_stopwords_and_lemmatize(text):
    """Remove stopwords and apply lemmatization."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def preprocess(df):
    """Full preprocessing pipeline."""
    df = df.dropna(subset=['tweet_text', 'sentiment'])
    df = df[df['tweet_text'] != 'Not Available'].copy()
    df['cleaned_text'] = df['tweet_text'].apply(clean_text)
    df['processed_text'] = df['cleaned_text'].apply(remove_stopwords_and_lemmatize)
    df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
    # Keep standard labels only
    df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral', 'Irrelevant'])]
    return df




