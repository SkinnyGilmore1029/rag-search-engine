import json
from pathlib import Path
import string

PROJECT_ROOT = Path(__file__).parent.parent

Movie_path = Path("data/movies.json")
STOP_WORD_PATH = Path("data/stopwords.txt")
stop_words = {
    word
    for line in STOP_WORD_PATH.read_text(encoding="utf-8").splitlines()
    if (word := line.strip())
}



def load_movies_json() -> dict:
    return json.loads(Movie_path.read_text(encoding='utf-8'))["movies"]

def clean_words(word: str) -> str:
    cleaned_word = word.lower()
    cleaned_word = cleaned_word.translate(str.maketrans("", "", string.punctuation))
    return cleaned_word

def make_tokens(word: str) -> list[str]:
    words_for_token = clean_words(word)
    tokens = words_for_token.split() # split makes list of words
    search_words = []
    for token in tokens:
        if token:
            search_words.append(token)
    filtered_search_words = []
    for word in search_words:
        if word not in stop_words:
            filtered_search_words.append(word)
    return filtered_search_words

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

