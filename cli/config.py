from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Paths relative to project root
Movie_path = PROJECT_ROOT / "data" / "movies.json"
STOP_WORD_PATH = PROJECT_ROOT / "data" / "stopwords.txt"


# Save Path for the inverted index
PATH_FOR_INDEX = Path("cache/index.pkl")
PATH_FOR_DOCMAP = Path("cache/docmap.pkl")
PATH_FOR_FREQUENCIES = Path("cache/term_frequencies.pkl")
stop_words = {
    word
    for line in STOP_WORD_PATH.read_text(encoding="utf-8").splitlines()
    if (word := line.strip())
}


BM25_K1 = 1.5