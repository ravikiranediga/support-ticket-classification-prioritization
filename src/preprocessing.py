"""
Text Preprocessing Module
Handles all text cleaning, tokenization, and normalization
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data if missing
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Initialize once
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercase
    - Replace {product_purchased} placeholder
    - Remove special characters
    - Normalize whitespace
    """
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\{product_purchased\}', ' product_placeholder ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Clean text
    2. Tokenize
    3. Remove stopwords
    4. Lemmatize
    5. Filter short tokens
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]
    return ' '.join(tokens)


def combine_ticket(subject: str, description: str) -> str:
    """
    Combine subject and description.
    Subject is weighted more heavily (2x) as it carries more signal.
    """
    subj = preprocess_text(subject)
    desc = preprocess_text(description)
    return f"{subj} {subj} {desc}".strip()


def batch_preprocess(subjects, descriptions):
    """Process multiple tickets at once"""
    return [
        combine_ticket(subj, desc)
        for subj, desc in zip(subjects, descriptions)
    ]
