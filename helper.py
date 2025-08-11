import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import numpy as np
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# Load SBERT model once globally
SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

SAFE_DIV = 1e-4

# --- Preprocessing ---
def preprocess(text: str) -> str:
    """Clean and preprocess question text."""
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()

    replacements = {
        '%': ' percent',
        '$': ' dollar ',
        '₹': ' rupee ',
        '€': ' euro ',
        '@': ' at ',
        '[math]': ''
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Replace large numbers with shorthand notation
    text = re.sub(r'([0-9]+)000000000', r'\1b', text)
    text = re.sub(r'([0-9]+)000000', r'\1m', text)
    text = re.sub(r'([0-9]+)000', r'\1k', text)

    contractions = {
        
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    
    }
    # Decontract
    words = text.split()
    decontracted = [contractions.get(w, w) for w in words]
    text = ' '.join(decontracted)

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove non-word characters
    text = re.sub(r'\W+', ' ', text).strip()

    return text

# --- Feature Extraction ---

def safe_divide(num: float, den: float) -> float:
    return num / (den + SAFE_DIV)

def common_words(q1: str, q2: str) -> int:
    w1 = set(q1.split())
    w2 = set(q2.split())
    return len(w1 & w2)

def total_words(q1: str, q2: str) -> int:
    w1 = set(q1.split())
    w2 = set(q2.split())
    return len(w1) + len(w2)

def fetch_token_features(q1: str, q2: str) -> list:
    token_features = [0.0]*8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return token_features

    q1_words = set(t for t in q1_tokens if t not in STOP_WORDS)
    q2_words = set(t for t in q2_tokens if t not in STOP_WORDS)

    q1_stops = set(t for t in q1_tokens if t in STOP_WORDS)
    q2_stops = set(t for t in q2_tokens if t in STOP_WORDS)

    common_word_count = len(q1_words & q2_words)
    common_stop_count = len(q1_stops & q2_stops)
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = safe_divide(common_word_count, min(len(q1_words), len(q2_words)))
    token_features[1] = safe_divide(common_word_count, max(len(q1_words), len(q2_words)))
    token_features[2] = safe_divide(common_stop_count, min(len(q1_stops), len(q2_stops)))
    token_features[3] = safe_divide(common_stop_count, max(len(q1_stops), len(q2_stops)))
    token_features[4] = safe_divide(common_token_count, min(len(q1_tokens), len(q2_tokens)))
    token_features[5] = safe_divide(common_token_count, max(len(q1_tokens), len(q2_tokens)))
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

def fetch_length_features(q1: str, q2: str) -> list:
    length_features = [0.0]*3
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    substrings = list(distance.lcsubstrings(q1, q2))
    length_features[2] = safe_divide(len(substrings[0]) if substrings else 0, min(len(q1), len(q2)))

    return length_features

def fetch_fuzzy_features(q1: str, q2: str) -> list:
    if not q1.strip() or not q2.strip():
        return [0.0]*4
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ]

# --- Main Function ---

def create_feature_vector(q1_raw: str, q2_raw: str) -> np.ndarray:
    """Create a full feature vector for a question pair."""
    q1 = preprocess(q1_raw)
    q2 = preprocess(q2_raw)

    base_feats = [
        len(q1), len(q2),
        len(q1.split()), len(q2.split()),
        common_words(q1, q2),
        total_words(q1, q2),
        round(safe_divide(common_words(q1, q2), total_words(q1, q2)), 2)
    ]

    base_feats.extend(fetch_token_features(q1, q2))
    base_feats.extend(fetch_length_features(q1, q2))
    base_feats.extend(fetch_fuzzy_features(q1, q2))

    # Add SBERT embeddings
    q1_emb = SBERT_MODEL.encode([q1], convert_to_numpy=True)
    q2_emb = SBERT_MODEL.encode([q2], convert_to_numpy=True)

    feature_vector = np.hstack((np.array(base_feats).reshape(1, -1), q1_emb, q2_emb))

    return feature_vector
