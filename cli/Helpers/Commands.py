from .Search_helps import load_movies_json, make_tokens, has_matching_token, InvertedIndex
import math

def Search_Command(query: str) -> list[dict]:
    """Handles the 'search' command for the CLI using cached title tokens."""
    
    # Load movies from json
    # Movie_data = load_movies_json()  # dictionary of movies
    
    #
    inverted_index = InvertedIndex({}, {},{})  # Create an instance of the InvertedIndex class
    try:
        inverted_index.load()  # Load the index and docmap from disk
    except FileNotFoundError:
        print("Index files not found. Please build the index first using the 'build' command.")
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
    
    """
    # Search through cached tokens
    for movie in Movie_data:
        title_tokens = movie["tokens"]
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= 5:  # stop after first 5 matches
                break
    """
 
    return results

def Term_Frequency_Command(doc_id: int, term: str) -> int:
    """Handles the 'tf' command for the CLI to get term frequency."""
    
    inverted_index = InvertedIndex({}, {}, {})  # Create an instance of the InvertedIndex class
    try:
        inverted_index.load()  # Load the index and docmap from disk
    except FileNotFoundError:
        print("Index files not found. Please build the index first using the 'build' command.")
        return 0
    
    tf = inverted_index.get_tf(doc_id, term)
    return tf

def Inverse_Document_Frequency_Command(term: str) -> float:
    """Handles the 'idf' command for the CLI to get inverse document frequency."""
    # This function can be implemented similarly to Term_Frequency_Command
    inverted_index = InvertedIndex({}, {}, {})  # Create an instance of the InvertedIndex class
    try:
        inverted_index.load()  # Load the index and docmap from disk
    except FileNotFoundError:
        print("Index files not found. Please build the index first using the 'build' command.")
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