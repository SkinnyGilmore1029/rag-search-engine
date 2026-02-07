import json
import string
from pathlib import Path
from nltk.stem import PorterStemmer


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Paths relative to project root
Movie_path = PROJECT_ROOT / "data" / "movies.json"
STOP_WORD_PATH = PROJECT_ROOT / "data" / "stopwords.txt"

# Optional cache for movies
_MOVIES_CACHE = None

stop_words = {
    word
    for line in STOP_WORD_PATH.read_text(encoding="utf-8").splitlines()
    if (word := line.strip())
}


stemmer = PorterStemmer()

def load_movies_json() -> list[dict]:
    global _MOVIES_CACHE
    if _MOVIES_CACHE is None:
        with Movie_path.open("r", encoding="utf-8") as f:
            _MOVIES_CACHE = json.load(f)["movies"]
    return _MOVIES_CACHE


def clean_words(word: str) -> str:
    cleaned_word = word.lower()
    cleaned_word = cleaned_word.translate(str.maketrans("", "", string.punctuation))
    return cleaned_word

def make_tokens(text: str) -> list[str]:
    cleaned = clean_words(text)
    return [
        stemmer.stem(token)
        for token in cleaned.split()
        if token and token not in stop_words
    ]

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    return bool(set(query_tokens) & set(title_tokens))

