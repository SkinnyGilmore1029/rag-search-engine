from .Search_helps import load_movies_json, make_tokens, has_matching_token, InvertedIndex

def Search_Command(query: str) -> list[dict]:
    """Handles the 'search' command for the CLI using cached title tokens."""
    
    # Load movies from json
    # Movie_data = load_movies_json()  # dictionary of movies
    
    #
    inverted_index = InvertedIndex({}, {})
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

    
