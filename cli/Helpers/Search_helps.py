import json
import string
from pickle import dump
from pathlib import Path
from nltk.stem import PorterStemmer


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Paths relative to project root
Movie_path = PROJECT_ROOT / "data" / "movies.json"
STOP_WORD_PATH = PROJECT_ROOT / "data" / "stopwords.txt"

# Optional cache for movies
_MOVIES_CACHE = None

# Save Path for the inverted index
PATH_FOR_INDEX = Path("cache/index.pkl")
PATH_FOR_DOCMAP = Path("cache/docmap.pkl")

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



class InvertedIndex:
    def __init__(self, index:dict[str, set[int]], docmap:dict[int, object]):
        self.index = index
        self.docmap = docmap
        
    def __add_document(self, doc_id:int, text:str) -> None:
        text_tokens = make_tokens(text)
        for token in text_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
    
    def get_documents(self, term:str) -> list[int]:
        key = term.lower()
        if key not in self.index: # <- index is a dict[str, set[int]]
            print(f"Term '{term}' not found in index.")
            return []
        return sorted(self.index[key])  # Return sorted list of document IDs for consistency
    
    def build(self) -> None:
        all_movies = load_movies_json()
        for movie in all_movies:
            movie_info = f"{movie['title']} {movie['description']}"
            self.__add_document(movie['id'], movie_info)
            self.docmap[movie['id']] = movie
    
    def save(self) -> None:
        # Create parent directory (cache/)
        PATH_FOR_INDEX.parent.mkdir(parents=True, exist_ok=True)

        with PATH_FOR_INDEX.open("wb") as f:
            dump(self.index, f)

        with PATH_FOR_DOCMAP.open("wb") as f:
            dump(self.docmap, f)