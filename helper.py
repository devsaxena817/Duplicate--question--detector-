import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import numpy as np
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# ------------------------------
# Global setup
# ------------------------------
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# Load SBERT model globally
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

SAFE_DIV = 1e-4


# ------------------------------
# Helper functions
# ------------------------------
def safe_divide(num, den):
    """Safely divide two numbers, avoiding ZeroDivisionError."""
    return num / (den + SAFE_DIV)


def test_common_words(q1, q2):
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    return len(w1 & w2)


def test_total_words(q1, q2):
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    return len(w1) + len(w2)


def test_fetch_token_features(q1, q2):
    token_features = [0.0] * 8

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


def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    # Longest common substring ratio
    strs = list(distance.lcsubstrings(q1, q2))
    if strs:
        length_features[2] = safe_divide(len(strs[0]), min(len(q1), len(q2)))
    else:
        length_features[2] = 0.0

    return length_features


def test_fetch_fuzzy_features(q1, q2):
    if not q1.strip() or not q2.strip():
        return [0.0, 0.0, 0.0, 0.0]

    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]


def preprocess(q):
    q = str(q).lower().strip()

    # Replace currency and symbols
    replacements = {
        '%': ' percent',
        '$': ' dollar ',
        '₹': ' rupee ',
        '€': ' euro ',
        '@': ' at ',
        '[math]': ''
    }
    for k, v in replacements.items():
        q = q.replace(k, v)

    # Replace large numbers with shorthand
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Expand contractions
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
    for k, v in contractions.items():
        q = q.replace(k, v)
    # Remove HTML tags
    q = BeautifulSoup(q, "html.parser").get_text()

    # Remove non-word characters
    q = re.sub(r'\W+', ' ', q).strip()

    return q


# ------------------------------
# Main feature vector function
# ------------------------------
def query_point_creator(q1, q2):
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query = [
        len(q1), len(q2),
        len(q1.split()), len(q2.split()),
        test_common_words(q1, q2),
        test_total_words(q1, q2),
        round(safe_divide(test_common_words(q1, q2), test_total_words(q1, q2)), 2)
    ]

    input_query.extend(test_fetch_token_features(q1, q2))
    input_query.extend(test_fetch_length_features(q1, q2))
    input_query.extend(test_fetch_fuzzy_features(q1, q2))

    # SBERT embeddings
    q1_emb = sbert_model.encode([q1], convert_to_numpy=True)
    q2_emb = sbert_model.encode([q2], convert_to_numpy=True)

    return np.hstack((np.array(input_query).reshape(1, -1), q1_emb, q2_emb))
