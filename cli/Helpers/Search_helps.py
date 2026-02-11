import json
import string
import math
from collections import Counter
from pickle import dump, load
from nltk.stem import PorterStemmer
from .config import (Movie_path, 
                     PATH_FOR_INDEX, 
                     PATH_FOR_DOCMAP, 
                     PATH_FOR_FREQUENCIES, 
                     PATH_FOR_DOCLENGTHS,
                     stop_words,
                     BM25_K1,
                     BM25_B)


# Optional cache for movies
_MOVIES_CACHE = None



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
    def __init__(self, index:dict[str, set[int]], docmap:dict[int, object], term_frequencies:dict[int, Counter]):
        self.index = index
        self.docmap = docmap
        self.term_frequencies = term_frequencies
        self.doc_lengths: dict = {}
        
    def __add_document(self, doc_id:int, text:str) -> None:
        text_tokens = make_tokens(text)
        self.term_frequencies[doc_id] = Counter()
        for token in text_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.doc_lengths[doc_id] = self.doc_lengths.get(doc_id, 0) + 1
            self.term_frequencies[doc_id][token] += 1
            self.index[token].add(doc_id)
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
         
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
    
    def get_tf(self, doc_id:int, term:str) -> int:
        text_tokens = make_tokens(term)
        #Term does not exist in the document
        if self.term_frequencies.get(doc_id) is None:
            return 0
        #To many tokens in the term
        if len(text_tokens) > 1:
            raise Exception("There are to many tokens in the term. Please only input one term.")
        
        #Term does not exist in the document
        if len(text_tokens) == 0:
            return 0

        return self.term_frequencies[doc_id][text_tokens[0]]
    
    def get_bm25_idf(self, term: str) -> float:
        # IDF calculation for BM25
        #log((N - df + 0.5) / (df + 0.5) + 1)
         
        #total number of documents
        N = len(self.docmap)
        
        #document frequency for the term
        df = len(self.get_documents(term))
        
        #Tokenize the term
        text_tokens = make_tokens(term)
        if len(text_tokens) > 1:
            raise Exception("There are to many tokens in the term. Please only input one term.")
        
        if df == 0:
            return 0.0
        
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id:int, term:str, k1:float=BM25_K1, b:float=BM25_B) -> float:
        raw_term_freq = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        # Apply to term frequency
        tf_component = (raw_term_freq * (k1 + 1)) / (raw_term_freq + k1 * length_norm)
        
        if raw_term_freq == 0:
            return 0.0
        return tf_component
    
    def bm25(self, doc_id:int, term:str) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf
    
    def bm25_search(self, query: str, limit: int = 5):
        # Tokenize the query
        query_tokens = make_tokens(query)

        # Dictionary to store total BM25 scores per document
        scores: dict[int, float] = {}

        # For each query token
        for token in query_tokens:
            # Get documents containing this token
            doc_ids = self.get_documents(token)

            for doc_id in doc_ids:
                # Initialize score if not already present
                if doc_id not in scores:
                    scores[doc_id] = 0.0

                # Add BM25 score for this token
                scores[doc_id] += self.bm25(doc_id, token)

        # Sort documents by total score (highest first)
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top `limit` results
        return ranked_docs[:limit]

    
    def save(self) -> None:
        # Create parent directory (cache/)
        PATH_FOR_INDEX.parent.mkdir(parents=True, exist_ok=True)
        PATH_FOR_DOCMAP.parent.mkdir(parents=True, exist_ok=True)
        PATH_FOR_FREQUENCIES.parent.mkdir(parents=True, exist_ok=True)
        PATH_FOR_DOCLENGTHS.parent.mkdir(parents=True, exist_ok=True)

        with PATH_FOR_INDEX.open("wb") as f:
            dump(self.index, f)

        with PATH_FOR_FREQUENCIES.open("wb") as f:
            dump(self.term_frequencies, f)

        with PATH_FOR_DOCMAP.open("wb") as f:
            dump(self.docmap, f)
            
        with PATH_FOR_DOCLENGTHS.open("wb") as f:
            dump(self.doc_lengths, f)
            
    def load(self) -> None:
        if (
            PATH_FOR_INDEX.exists() and 
            PATH_FOR_DOCMAP.exists() and 
            PATH_FOR_FREQUENCIES.exists() and 
            PATH_FOR_DOCLENGTHS
            ):
            
            with PATH_FOR_INDEX.open("rb") as f:
                self.index = load(f)
            with PATH_FOR_FREQUENCIES.open("rb") as f:
                self.term_frequencies = load(f)
            with PATH_FOR_DOCMAP.open("rb") as f:
                self.docmap = load(f)
                
            with PATH_FOR_DOCLENGTHS.open("rb") as f:
                self.doc_lengths = load(f)
        else:
            raise FileNotFoundError("Index files and docmap files not found. Please build the index and the docmap first.")