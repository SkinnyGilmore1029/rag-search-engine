from .Search_helps import make_tokens, InvertedIndex
from .config import BM25_K1
import math

def get_loaded_index() -> InvertedIndex | None:
    """
    Create an InvertedIndex instance and try to load it from disk.
    Returns the loaded InvertedIndex if successful, otherwise None.
    """
    inverted_index = InvertedIndex({}, {}, {})  # Create the instance
    try:
        inverted_index.load()  # Attempt to load index and docmap from disk
        return inverted_index
    except FileNotFoundError:
        print("Index files not found. Please build the index first using the 'build' command.")
        return None

def Search_Command(query: str) -> list[dict]:
    """Handles the 'search' command for the CLI using cached title tokens."""
    
    inverted_index = get_loaded_index()
    if inverted_index is None:
        return []
    Movie_data = inverted_index.docmap.values()  # Get the list of movies from the docmap
    
    # Precompute tokens for each movie title (cache)
    for movie in Movie_data:
        if "tokens" not in movie:  # only compute once
            movie["tokens"] = make_tokens(movie["title"])
    
    # Tokenize query once
    query_tokens:list[str] = make_tokens(query) 
    
    results: list[dict] = []
    
    # Search through the inverted index for each query token using docmap
    for token in query_tokens:
        doc_ids: list[int] = inverted_index.get_documents(token)
        for doc_id in doc_ids:
            movie = inverted_index.docmap.get(doc_id)
            if movie and movie not in results:
                results.append(movie)
                if len(results) >= 5:  # stop after first 5 matches
                    break
  
    return results

def Term_Frequency_Command(doc_id: int, term: str) -> int:
    """Handles the 'tf' command for the CLI to get term frequency."""
  
    inverted_index = get_loaded_index()
    if inverted_index is None:
        return 0
    tf = inverted_index.get_tf(doc_id, term)
    return tf

def Inverse_Document_Frequency_Command(term: str) -> float:
    """Handles the 'idf' command for the CLI to get inverse document frequency."""
    inverted_index = get_loaded_index()
    if inverted_index is None:
        return 0.0
    
    text_tokens = make_tokens(term)
    
    if text_tokens is None or len(text_tokens) == 0:
        print("No valid tokens found in the term.")
        return 0.0
    
    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.get_documents(text_tokens[0]))
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

def TF_IDF_Command(doc_id: int, term: str) -> float:
    """Handles the 'tf-idf' command for the CLI to get TF-IDF score."""
    tf = Term_Frequency_Command(doc_id, term)
    idf = Inverse_Document_Frequency_Command(term)
    return tf * idf

def bm25_idf_command(term: str) -> float:
    """Handles the 'bm25-idf' command for the CLI to get BM25 IDF score."""
    inverted_index = get_loaded_index()
    if inverted_index is None:
        return 0.0
    
    text_tokens = make_tokens(term)
    
    if text_tokens is None or len(text_tokens) == 0:
        print("No valid tokens found in the term.")
        return 0.0
    
    return inverted_index.get_bm25_idf(text_tokens[0])


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    """Handles the 'bm25-tf' command for the CLI to get BM25 TF score."""
    inverted_index = get_loaded_index()
    if inverted_index is None:
        return 0.0
    
    text_tokens = make_tokens(term)
    
    if text_tokens is None or len(text_tokens) == 0:
        print("No valid tokens found in the term.")
        return 0.0
    
    return inverted_index.get_bm25_tf(doc_id, text_tokens[0])


def bm25search(query: str, limit: int = 5) -> list[dict]:
    """Handles the 'bm25search' command for the CLI to perform a BM25 search."""
    inverted_index = get_loaded_index()
    if inverted_index is None:
        return []

    query_tokens = make_tokens(query)
    if not query_tokens:
        print("No valid tokens found in the query.")
        return []

    # Calculate BM25 scores for all documents
    doc_scores = {}
    for token in query_tokens:
        doc_ids = inverted_index.get_documents(token)
        for doc_id in doc_ids:
            score = inverted_index.bm25(doc_id, token)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

    # Sort documents by score and return top results
    sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    top_docs = sorted_docs[:limit]

    results = []
    for doc_id, score in top_docs:
        movie = inverted_index.docmap.get(doc_id)
        if movie:
            # Create a new dict with the score included
            results.append({
                "id": movie["id"],
                "title": movie["title"],
                "score": score
            })

    return results
